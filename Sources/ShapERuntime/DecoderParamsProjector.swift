import Foundation
@preconcurrency import MLX

import Foundation
@preconcurrency import MLX

public enum DecoderParamsProjectorError: Error, Equatable, LocalizedError {
    case invalidTensorShape(name: String, expected: String, actual: [Int])
    case invalidScalar(name: String, expected: String, actual: Float)
    case invalidParamShape(name: String, shape: [Int])
    case duplicateParamName(String)
    case missingProjection(name: String)
    case projectionShapeMismatch(name: String, tensor: String, expected: [Int], actual: [Int])
    case latentContextMismatch(flattenedWidth: Int, latentCtx: Int)
    case projectionDepthMismatch(name: String, expected: Int, actual: Int)
    case consumedLatentContextExceedsAvailable(consumed: Int, latentCtx: Int)
    case incompleteLatentCoverage(consumed: Int, latentCtx: Int)
    case unsupportedLatentWarp(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidTensorShape(name, expected, actual):
            return "Tensor '\(name)' has shape \(actual) but expected \(expected)."
        case let .invalidScalar(name, expected, actual):
            return "Scalar '\(name)' has value \(actual) but expected \(expected)."
        case let .invalidParamShape(name, shape):
            return "Decoder parameter '\(name)' has invalid shape \(shape)."
        case let .duplicateParamName(name):
            return "Decoder parameter '\(name)' appears more than once."
        case let .missingProjection(name):
            return "Decoder parameter '\(name)' is missing its projection weights."
        case let .projectionShapeMismatch(name, tensor, expected, actual):
            return "Decoder projection '\(name)' tensor '\(tensor)' has shape \(actual) but expected \(expected)."
        case let .latentContextMismatch(flattenedWidth, latentCtx):
            return "Flattened latent width \(flattenedWidth) is not divisible by latent_ctx \(latentCtx)."
        case let .projectionDepthMismatch(name, expected, actual):
            return "Decoder projection '\(name)' expects latent depth \(expected), but the input latent provides \(actual)."
        case let .consumedLatentContextExceedsAvailable(consumed, latentCtx):
            return "Decoder projections consume \(consumed) latent vectors, but latent_ctx is only \(latentCtx)."
        case let .incompleteLatentCoverage(consumed, latentCtx):
            return "Decoder projections consume \(consumed) latent vectors, but latent_ctx is \(latentCtx). Every latent vector must map to a parameter."
        case let .unsupportedLatentWarp(name):
            return "Unsupported decoder latent warp '\(name)'."
        }
    }
}

/// Projects a bottleneck latent into weights for the NeRSTF renderer.
///
/// Note: this type intentionally does not conform to Sendable because it stores
/// MLXArray values. Keep MLXArray use on the same executor/stream unless your
/// surrounding code provides explicit synchronization.
public struct DecoderParamsProjector {
    private struct FlatParamShape {
        let name: String
        let shape: [Int]
        let vectors: Int
        let channels: Int
    }

    private struct ProjectionWeights {
        let weightVCD: MLXArray
        let biasVC: MLXArray
        let lnWeightC: MLXArray
        let lnBiasC: MLXArray
        let depth: Int
    }

    public let artifact: DecoderParamsArtifact
    private let flatParamShapes: [FlatParamShape]
    private let projectionWeights: [String: ProjectionWeights]

    public init(artifact: DecoderParamsArtifact) throws {
        self.artifact = artifact

        guard artifact.latentCtx > 0 else {
            throw DecoderParamsProjectorError.invalidScalar(
                name: "latent_ctx",
                expected: "> 0",
                actual: Float(artifact.latentCtx)
            )
        }

        let lnEps = Float(artifact.lnEps)
        guard lnEps.isFinite, lnEps > 0 else {
            throw DecoderParamsProjectorError.invalidScalar(
                name: "ln_eps",
                expected: "finite and > 0",
                actual: lnEps
            )
        }

        var flatParams: [FlatParamShape] = []
        var weightDict: [String: ProjectionWeights] = [:]
        var seenParamNames = Set<String>()
        var consumed = 0

        for param in artifact.paramShapes {
            guard seenParamNames.insert(param.name).inserted else {
                throw DecoderParamsProjectorError.duplicateParamName(param.name)
            }

            let total = try Self.validatedElementCount(name: param.name, shape: param.shape)
            let channels = param.shape[param.shape.count - 1]
            let vectors = total / channels

            guard vectors > 0 else {
                throw DecoderParamsProjectorError.invalidParamShape(name: param.name, shape: param.shape)
            }

            let (newConsumed, overflow) = consumed.addingReportingOverflow(vectors)
            guard !overflow else {
                throw DecoderParamsProjectorError.consumedLatentContextExceedsAvailable(
                    consumed: Int.max,
                    latentCtx: artifact.latentCtx
                )
            }
            consumed = newConsumed

            guard let proj = artifact.projections[param.name] else {
                throw DecoderParamsProjectorError.missingProjection(name: param.name)
            }

            let w = try proj.weightVCD.asMLX().asType(.float32)
            let b = try proj.biasVC.asMLX().asType(.float32)
            let lnW = try proj.lnWeightC.asMLX().asType(.float32)
            let lnB = try proj.lnBiasC.asMLX().asType(.float32)

            try Self.validateProjectionShapes(
                name: param.name,
                vectors: vectors,
                channels: channels,
                weightVCD: w,
                biasVC: b,
                lnWeightC: lnW,
                lnBiasC: lnB
            )

            flatParams.append(
                FlatParamShape(
                    name: param.name,
                    shape: param.shape,
                    vectors: vectors,
                    channels: channels
                )
            )

            weightDict[param.name] = ProjectionWeights(
                weightVCD: w,
                biasVC: b,
                lnWeightC: lnW,
                lnBiasC: lnB,
                depth: w.shape[2]
            )
        }

        if consumed > artifact.latentCtx {
            throw DecoderParamsProjectorError.consumedLatentContextExceedsAvailable(
                consumed: consumed,
                latentCtx: artifact.latentCtx
            )
        }
        if consumed != artifact.latentCtx {
            throw DecoderParamsProjectorError.incompleteLatentCoverage(
                consumed: consumed,
                latentCtx: artifact.latentCtx
            )
        }

        self.flatParamShapes = flatParams
        self.projectionWeights = weightDict
    }

    /// Project a latent [B, D] into a dictionary of parameter tensors.
    public func project(latent: MLXArray) throws -> [String: MLXArray] {
        guard latent.ndim == 2 else {
            throw DecoderParamsProjectorError.invalidTensorShape(
                name: "latent",
                expected: "[batch, d_latent]",
                actual: latent.shape
            )
        }

        let batchSize = latent.shape[0]
        let dLatent = latent.shape[1]

        guard batchSize > 0, dLatent > 0 else {
            throw DecoderParamsProjectorError.invalidTensorShape(
                name: "latent",
                expected: "positive [batch, d_latent]",
                actual: latent.shape
            )
        }

        guard dLatent % artifact.latentCtx == 0 else {
            throw DecoderParamsProjectorError.latentContextMismatch(
                flattenedWidth: dLatent,
                latentCtx: artifact.latentCtx
            )
        }

        let depth = dLatent / artifact.latentCtx

        let unwarped = try unwarp(latent.asType(.float32), artifact: artifact)
            .reshaped(batchSize, artifact.latentCtx, depth)

        var outputs: [String: MLXArray] = [:]
        var startVector = 0
        let lnEps = Float(artifact.lnEps)

        for param in flatParamShapes {
            guard let proj = projectionWeights[param.name] else {
                throw DecoderParamsProjectorError.missingProjection(name: param.name)
            }

            guard proj.depth == depth else {
                throw DecoderParamsProjectorError.projectionDepthMismatch(
                    name: param.name,
                    expected: proj.depth,
                    actual: depth
                )
            }

            let endVector = startVector + param.vectors

            // xBVD: [B, vectors, depth]
            let xBVD = unwarped[.ellipsis, startVector..<endVector, 0..<depth]

            // h: [B, vectors, channels]
            var h = MLX.einsum("bvd,vcd->bvc", xBVD, proj.weightVCD)

            // Manual LayerNorm matching the reference logic.
            let mean = MLX.mean(h, axis: -1, keepDims: true)
            let centered = h - mean
            let variance = MLX.mean(MLX.square(centered), axis: -1, keepDims: true)
            h = centered * MLX.rsqrt(variance + lnEps)

            h = h * proj.lnWeightC.reshaped(1, 1, param.channels)
            h = h + proj.lnBiasC.reshaped(1, 1, param.channels)
            h = h + proj.biasVC.reshaped(1, param.vectors, param.channels)

            let outShape = [batchSize] + param.shape
            outputs[param.name] = h.reshaped(outShape).asType(.float32)

            startVector = endVector
        }

        return outputs
    }

    private func unwarp(_ x: MLXArray, artifact: DecoderParamsArtifact) throws -> MLXArray {
        switch artifact.latentWarpName {
        case "", "identity", "none":
            return x.asType(.float32)

        case "tan2":
            let coeff1Double = Double(artifact.latentWarpCoeff1)
            guard coeff1Double.isFinite, coeff1Double != 0 else {
                throw DecoderParamsProjectorError.invalidScalar(
                    name: "latent_warp_coeff1",
                    expected: "finite and non-zero for tan2",
                    actual: Float(coeff1Double)
                )
            }

            let scaleDouble = tan(tan(1.0) * coeff1Double)
            guard scaleDouble.isFinite else {
                throw DecoderParamsProjectorError.invalidScalar(
                    name: "latent_warp_scale",
                    expected: "finite",
                    actual: Float(scaleDouble)
                )
            }

            let coeff1 = Float(coeff1Double)
            let scale = Float(scaleDouble)
            let xc = x.asType(.float32) * scale

            // Do not clip here. The Torch reference clamps pred_xstart inside the
            // denoiser path when clip_denoised is true. This unwarp should only
            // apply the tan2 inverse used by the decoder artifact.
            return MLX.atan(MLX.atan(xc) / coeff1).asType(.float32)

        default:
            throw DecoderParamsProjectorError.unsupportedLatentWarp(artifact.latentWarpName)
        }
    }

    private static func validatedElementCount(name: String, shape: [Int]) throws -> Int {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw DecoderParamsProjectorError.invalidParamShape(name: name, shape: shape)
        }

        var total = 1
        for dim in shape {
            let result = total.multipliedReportingOverflow(by: dim)
            guard !result.overflow else {
                throw DecoderParamsProjectorError.invalidParamShape(name: name, shape: shape)
            }
            total = result.partialValue
        }

        return total
    }

    private static func validateProjectionShapes(
        name: String,
        vectors: Int,
        channels: Int,
        weightVCD w: MLXArray,
        biasVC b: MLXArray,
        lnWeightC lnW: MLXArray,
        lnBiasC lnB: MLXArray
    ) throws {
        guard w.shape.count == 3,
              w.shape[0] == vectors,
              w.shape[1] == channels,
              w.shape[2] > 0 else {
            throw DecoderParamsProjectorError.projectionShapeMismatch(
                name: name,
                tensor: "weightVCD",
                expected: [vectors, channels, -1],
                actual: w.shape
            )
        }

        guard b.shape == [vectors, channels] else {
            throw DecoderParamsProjectorError.projectionShapeMismatch(
                name: name,
                tensor: "biasVC",
                expected: [vectors, channels],
                actual: b.shape
            )
        }

        guard lnW.shape == [channels] else {
            throw DecoderParamsProjectorError.projectionShapeMismatch(
                name: name,
                tensor: "lnWeightC",
                expected: [channels],
                actual: lnW.shape
            )
        }

        guard lnB.shape == [channels] else {
            throw DecoderParamsProjectorError.projectionShapeMismatch(
                name: name,
                tensor: "lnBiasC",
                expected: [channels],
                actual: lnB.shape
            )
        }
    }
}
