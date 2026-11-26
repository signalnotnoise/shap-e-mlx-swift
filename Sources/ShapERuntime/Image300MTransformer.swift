import Foundation
import MLX
import MLXNN

// MARK: - Errors

public enum Image300MTransformerError: Error, Equatable, LocalizedError {
    case invalidTensorShape(name: String, expected: String, actual: [Int])
    case unsupportedTimestepDataType(TensorDataType)
    case batchSizeMismatch(expected: Int, actual: Int, tensorName: String)
    case sequenceLengthExceedsContext(sequenceLength: Int, contextLength: Int)
    case modelWidthNotDivisible(width: Int, heads: Int)
    case invalidConfiguration(String)
    case invalidConditionShape(name: String, expected: String, actual: [Int])
    case invalidLayerCount(Int)

    public var errorDescription: String? {
        switch self {
        case let .invalidTensorShape(name, expected, actual):
            return "Tensor '\(name)' has shape \(actual) but expected \(expected)."
        case let .unsupportedTimestepDataType(dataType):
            return "Image300M timesteps must be float32, int32, or int64. Received \(dataType.rawValue)."
        case let .batchSizeMismatch(expected, actual, tensorName):
            return "Tensor '\(tensorName)' has batch size \(actual) but expected \(expected)."
        case let .sequenceLengthExceedsContext(sequenceLength, contextLength):
            return "Input sequence length \(sequenceLength) exceeds image300m context length \(contextLength)."
        case let .modelWidthNotDivisible(width, heads):
            return "Image300M model width \(width) must be divisible by attention head count \(heads)."
        case let .invalidConfiguration(message):
            return "Invalid Image300M configuration: \(message)"
        case let .invalidConditionShape(name, expected, actual):
            return "Condition '\(name)' has shape \(actual) but expected \(expected)."
        case let .invalidLayerCount(count):
            return "Image300M layer count must be > 0. Inferred \(count)."
        }
    }
}

// MARK: - Image300M Transformer

/// MLXArray execution path for the Image300M diffusion transformer.
///
/// Important:
/// - This class intentionally does not conform to `Sendable`. Stored `MLXArray` weights are lazy graph
///   objects and should not be passed across concurrency domains without external synchronization.
/// - The forward path supports CLIP conditioning as either [B, clipFeatureDim] or [B, L, clipFeatureDim].
/// - Output is [B, outputChannels, T].

// Conforms as @unchecked Sendable because it stores MLXArray graph objects that are not Sendable; 
// see the note below about not passing across concurrency domains without external synchronization.
public final class Image300MTransformer: DiffusionTransformer, @unchecked Sendable {

    public struct Configuration: Equatable, Sendable {
        public let dModel: Int
        public let nLayers: Int
        public let nHeads: Int
        public let mlpHiddenDim: Int
        public let inputChannels: Int
        public let outputChannels: Int
        public let clipFeatureDim: Int
        public let tokenCond: Bool
        public let timeTokenCond: Bool
        public let usePosEmb: Bool

        public init(
            dModel: Int,
            nLayers: Int,
            nHeads: Int,
            mlpHiddenDim: Int,
            inputChannels: Int,
            outputChannels: Int,
            clipFeatureDim: Int,
            tokenCond: Bool = true,
            timeTokenCond: Bool = true,
            usePosEmb: Bool = true
        ) {
            self.dModel = dModel
            self.nLayers = nLayers
            self.nHeads = nHeads
            self.mlpHiddenDim = mlpHiddenDim
            self.inputChannels = inputChannels
            self.outputChannels = outputChannels
            self.clipFeatureDim = clipFeatureDim
            self.tokenCond = tokenCond
            self.timeTokenCond = timeTokenCond
            self.usePosEmb = usePosEmb
        }
    }

    private struct TransformerLayer {
        let ln1Weight: MLXArray
        let ln1Bias: MLXArray
        let attentionQKVWeight: MLXArray
        let attentionQKVBias: MLXArray
        let attentionOutputWeight: MLXArray
        let attentionOutputBias: MLXArray
        let ln2Weight: MLXArray
        let ln2Bias: MLXArray
        let mlpFCWeight: MLXArray
        let mlpFCBias: MLXArray
        let mlpProjectionWeight: MLXArray
        let mlpProjectionBias: MLXArray
    }

    public let artifact: Image300MArtifact
    public var nCtx: Int { artifact.nCtx }
    public let configuration: Configuration

    private let inputProjectionWeight: MLXArray
    private let inputProjectionBias: MLXArray
    private let clipEmbeddingLNWeight: MLXArray
    private let clipEmbeddingLNBias: MLXArray
    private let clipEmbeddingWeight: MLXArray
    private let clipEmbeddingBias: MLXArray
    private let layerNormPreWeight: MLXArray
    private let layerNormPreBias: MLXArray
    private let layerNormPostWeight: MLXArray
    private let layerNormPostBias: MLXArray
    private let outputProjectionWeight: MLXArray
    private let outputProjectionBias: MLXArray
    private let positionEmbedding: MLXArray
    private let positionEmbeddingRows: Int
    private let timeEmbeddingFCWeight: MLXArray
    private let timeEmbeddingFCBias: MLXArray
    private let timeEmbeddingProjectionWeight: MLXArray
    private let timeEmbeddingProjectionBias: MLXArray
    private let layers: [TransformerLayer]

    public convenience init(
        artifact: Image300MArtifact,
        nHeads: Int = 8,
        tokenCond: Bool = true,
        timeTokenCond: Bool = true,
        usePosEmb: Bool = true
    ) throws {
        try self.init(
            artifact: artifact,
            configuration: try Self.inferConfiguration(
                from: artifact,
                nHeads: nHeads,
                tokenCond: tokenCond,
                timeTokenCond: timeTokenCond,
                usePosEmb: usePosEmb
            )
        )
    }

    public init(artifact: Image300MArtifact, configuration: Configuration) throws {
        guard configuration.dModel > 0 else { throw Image300MTransformerError.invalidConfiguration("dModel must be > 0.") }
        guard configuration.nHeads > 0 else { throw Image300MTransformerError.invalidConfiguration("nHeads must be > 0.") }
        guard configuration.nLayers > 0 else { throw Image300MTransformerError.invalidLayerCount(configuration.nLayers) }
        guard configuration.dModel % configuration.nHeads == 0 else {
            throw Image300MTransformerError.modelWidthNotDivisible(width: configuration.dModel, heads: configuration.nHeads)
        }
        guard configuration.inputChannels > 0,
              configuration.outputChannels > 0,
              configuration.clipFeatureDim > 0,
              configuration.mlpHiddenDim > 0 else {
            throw Image300MTransformerError.invalidConfiguration("inputChannels, outputChannels, clipFeatureDim, and mlpHiddenDim must all be > 0.")
        }

        self.artifact = artifact
        self.configuration = configuration

        self.inputProjectionWeight = try artifact.requireTensor(.inputProjectionWeight, expectedShape: [configuration.dModel, configuration.inputChannels], dataType: .float32).asMLX().asType(.float32)
        self.inputProjectionBias = try artifact.requireTensor(.inputProjectionBias, expectedShape: [configuration.dModel], dataType: .float32).asMLX().asType(.float32)
        self.clipEmbeddingLNWeight = try artifact.requireTensor(.clipEmbeddingLNWeight, expectedShape: [configuration.clipFeatureDim], dataType: .float32).asMLX().asType(.float32)
        self.clipEmbeddingLNBias = try artifact.requireTensor(.clipEmbeddingLNBias, expectedShape: [configuration.clipFeatureDim], dataType: .float32).asMLX().asType(.float32)
        self.clipEmbeddingWeight = try artifact.requireTensor(.clipEmbeddingLinearWeight, expectedShape: [configuration.dModel, configuration.clipFeatureDim], dataType: .float32).asMLX().asType(.float32)
        self.clipEmbeddingBias = try artifact.requireTensor(.clipEmbeddingLinearBias, expectedShape: [configuration.dModel], dataType: .float32).asMLX().asType(.float32)
        self.layerNormPreWeight = try artifact.requireTensor(.layerNormPreWeight, expectedShape: [configuration.dModel], dataType: .float32).asMLX().asType(.float32)
        self.layerNormPreBias = try artifact.requireTensor(.layerNormPreBias, expectedShape: [configuration.dModel], dataType: .float32).asMLX().asType(.float32)
        self.layerNormPostWeight = try artifact.requireTensor(.layerNormPostWeight, expectedShape: [configuration.dModel], dataType: .float32).asMLX().asType(.float32)
        self.layerNormPostBias = try artifact.requireTensor(.layerNormPostBias, expectedShape: [configuration.dModel], dataType: .float32).asMLX().asType(.float32)
        self.outputProjectionWeight = try artifact.requireTensor(.outputProjectionWeight, expectedShape: [configuration.outputChannels, configuration.dModel], dataType: .float32).asMLX().asType(.float32)
        self.outputProjectionBias = try artifact.requireTensor(.outputProjectionBias, expectedShape: [configuration.outputChannels], dataType: .float32).asMLX().asType(.float32)

        let posT = try artifact.requireTensor(.positionEmbedding, dataType: .float32)
        guard posT.shape.count == 2, posT.shape[1] == configuration.dModel else {
            throw Image300MTransformerError.invalidTensorShape(name: posT.name, expected: "[context, \(configuration.dModel)]", actual: posT.shape)
        }
        self.positionEmbeddingRows = posT.shape[0]
        self.positionEmbedding = try posT.asMLX().asType(.float32)

        self.timeEmbeddingFCWeight = try artifact.requireTensor(.timeEmbeddingFCWeight, expectedShape: [configuration.mlpHiddenDim, configuration.dModel], dataType: .float32).asMLX().asType(.float32)
        self.timeEmbeddingFCBias = try artifact.requireTensor(.timeEmbeddingFCBias, expectedShape: [configuration.mlpHiddenDim], dataType: .float32).asMLX().asType(.float32)
        self.timeEmbeddingProjectionWeight = try artifact.requireTensor(.timeEmbeddingProjectionWeight, expectedShape: [configuration.dModel, configuration.mlpHiddenDim], dataType: .float32).asMLX().asType(.float32)
        self.timeEmbeddingProjectionBias = try artifact.requireTensor(.timeEmbeddingProjectionBias, expectedShape: [configuration.dModel], dataType: .float32).asMLX().asType(.float32)
        self.layers = try Self.loadLayers(from: artifact, configuration: configuration)
    }

    public static func inferConfiguration(
        from artifact: Image300MArtifact,
        nHeads: Int = 16,
        tokenCond: Bool = true,
        timeTokenCond: Bool = true,
        usePosEmb: Bool = true
    ) throws -> Configuration {
        let inputW = try artifact.requireTensor(.inputProjectionWeight, dataType: .float32)
        guard inputW.shape.count == 2 else {
            throw Image300MTransformerError.invalidTensorShape(name: inputW.name, expected: "[d_model, input_channels]", actual: inputW.shape)
        }
        let clipW = try artifact.requireTensor(.clipEmbeddingLinearWeight, dataType: .float32)
        guard clipW.shape.count == 2 else {
            throw Image300MTransformerError.invalidTensorShape(name: clipW.name, expected: "[d_model, clip_feature_dim]", actual: clipW.shape)
        }
        let outputW = try artifact.requireTensor(.outputProjectionWeight, dataType: .float32)
        guard outputW.shape.count == 2 else {
            throw Image300MTransformerError.invalidTensorShape(name: outputW.name, expected: "[output_channels, d_model]", actual: outputW.shape)
        }
        let timeW = try artifact.requireTensor(.timeEmbeddingFCWeight, dataType: .float32)
        guard timeW.shape.count == 2 else {
            throw Image300MTransformerError.invalidTensorShape(name: timeW.name, expected: "[mlp_hidden_dim, d_model]", actual: timeW.shape)
        }
        let layerCount = inferLayerCount(from: artifact.state.keys)
        guard layerCount > 0 else { throw Image300MTransformerError.invalidLayerCount(layerCount) }

        return Configuration(
            dModel: inputW.shape[0],
            nLayers: layerCount,
            nHeads: nHeads,
            mlpHiddenDim: timeW.shape[0],
            inputChannels: inputW.shape[1],
            outputChannels: outputW.shape[0],
            clipFeatureDim: clipW.shape[1],
            tokenCond: tokenCond,
            timeTokenCond: timeTokenCond,
            usePosEmb: usePosEmb
        )
    }

    public func predict(x latent: RuntimeTensor, t timesteps: RuntimeTensor, embeddings clipEmbeddings: RuntimeTensor) throws -> RuntimeTensor {
        try validateRuntimeInputs(latent: latent, timesteps: timesteps, clipEmbeddings: clipEmbeddings)

        let batchSize = latent.shape[0]
        let sequenceLength = latent.shape[2]
        let hidden = try predictInternal(
            x: try latent.asMLX().asType(.float32),
            t: try timesteps.asMLX().asType(.float32),
            embeddings: try clipEmbeddings.asMLX().asType(.float32)
        )

        guard hidden.shape == [batchSize, configuration.outputChannels, sequenceLength] else {
            throw Image300MTransformerError.invalidTensorShape(name: "image300m_output", expected: "[\(batchSize), \(configuration.outputChannels), \(sequenceLength)]", actual: hidden.shape)
        }

        MLX.eval(hidden)
        let outFloats = hidden.asArray(Float.self)
        let outData = outFloats.withUnsafeBufferPointer { Data(buffer: $0) }
        return try RuntimeTensor(name: "image300m_output", shape: [batchSize, configuration.outputChannels, sequenceLength], dataType: .float32, data: outData)
    }

    public func predictInternal(x xMLX: MLXArray, t tMLX: MLXArray, embeddings clipMLX: MLXArray) throws -> MLXArray {
        try validateMLXInputs(x: xMLX, t: tMLX, clip: clipMLX)

        let batchSize = xMLX.shape[0]
        let sequenceLength = xMLX.shape[2]
        let tsEmb = TimestepEmbedding.compute(timesteps: tMLX.asType(.float32), dimension: configuration.dModel).asType(.float32)
        let tEmbed = timeEmbed(tsEmb)
        let clipEmbed = clipConditionEmbed(clipMLX.asType(.float32))

        return try forwardWithConditioning(
            latentNCT: xMLX.asType(.float32),
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            conditions: [
                ("time", tEmbed, configuration.timeTokenCond),
                ("clip", clipEmbed, configuration.tokenCond)
            ]
        )
    }

    private func validateRuntimeInputs(latent: RuntimeTensor, timesteps: RuntimeTensor, clipEmbeddings: RuntimeTensor) throws {
        guard latent.shape.count == 3 else {
            throw Image300MTransformerError.invalidTensorShape(name: latent.name, expected: "[batch, channels, tokens]", actual: latent.shape)
        }
        let batchSize = latent.shape[0]
        let inputChannels = latent.shape[1]
        let sequenceLength = latent.shape[2]
        guard inputChannels == configuration.inputChannels else {
            throw Image300MTransformerError.invalidTensorShape(name: latent.name, expected: "[batch, \(configuration.inputChannels), tokens]", actual: latent.shape)
        }
        guard sequenceLength <= artifact.nCtx else {
            throw Image300MTransformerError.sequenceLengthExceedsContext(sequenceLength: sequenceLength, contextLength: artifact.nCtx)
        }
        guard sequenceLength <= positionEmbeddingRows else {
            throw Image300MTransformerError.sequenceLengthExceedsContext(sequenceLength: sequenceLength, contextLength: positionEmbeddingRows)
        }
        guard timesteps.shape == [batchSize] else {
            throw Image300MTransformerError.invalidTensorShape(name: timesteps.name, expected: "[\(batchSize)]", actual: timesteps.shape)
        }
        switch timesteps.dataType {
        case .float32, .int32, .int64:
            break
        default:
            throw Image300MTransformerError.unsupportedTimestepDataType(timesteps.dataType)
        }
        guard isValidClipShape(clipEmbeddings.shape, batchSize: batchSize) else {
            throw Image300MTransformerError.invalidTensorShape(name: clipEmbeddings.name, expected: "[\(batchSize), \(configuration.clipFeatureDim)] or [\(batchSize), tokens, \(configuration.clipFeatureDim)]", actual: clipEmbeddings.shape)
        }
    }

    private func validateMLXInputs(x: MLXArray, t: MLXArray, clip: MLXArray) throws {
        guard x.ndim == 3 else {
            throw Image300MTransformerError.invalidTensorShape(name: "x", expected: "[batch, \(configuration.inputChannels), tokens]", actual: x.shape)
        }
        let batchSize = x.shape[0]
        let inputChannels = x.shape[1]
        let sequenceLength = x.shape[2]
        guard inputChannels == configuration.inputChannels else {
            throw Image300MTransformerError.invalidTensorShape(name: "x", expected: "[batch, \(configuration.inputChannels), tokens]", actual: x.shape)
        }
        guard sequenceLength <= artifact.nCtx else {
            throw Image300MTransformerError.sequenceLengthExceedsContext(sequenceLength: sequenceLength, contextLength: artifact.nCtx)
        }
        guard sequenceLength <= positionEmbeddingRows else {
            throw Image300MTransformerError.sequenceLengthExceedsContext(sequenceLength: sequenceLength, contextLength: positionEmbeddingRows)
        }
        guard t.shape == [batchSize] else {
            throw Image300MTransformerError.invalidTensorShape(name: "t", expected: "[\(batchSize)]", actual: t.shape)
        }
        guard isValidClipShape(clip.shape, batchSize: batchSize) else {
            throw Image300MTransformerError.invalidTensorShape(name: "embeddings", expected: "[\(batchSize), \(configuration.clipFeatureDim)] or [\(batchSize), tokens, \(configuration.clipFeatureDim)]", actual: clip.shape)
        }
    }

    private func isValidClipShape(_ shape: [Int], batchSize: Int) -> Bool {
        if shape.count == 2 { return shape[0] == batchSize && shape[1] == configuration.clipFeatureDim }
        if shape.count == 3 { return shape[0] == batchSize && shape[2] == configuration.clipFeatureDim }
        return false
    }

    private func timeEmbed(_ tsEmb: MLXArray) -> MLXArray {
        let h = MLX.matmul(tsEmb, timeEmbeddingFCWeight.transposed()) + timeEmbeddingFCBias
        let act = Self.gelu(h)
        return MLX.matmul(act, timeEmbeddingProjectionWeight.transposed()) + timeEmbeddingProjectionBias
    }

    private func clipConditionEmbed(_ clip: MLXArray) -> MLXArray {
        let clipOutLN = Self.layerNorm(clip, weight: clipEmbeddingLNWeight, bias: clipEmbeddingLNBias)
        return MLX.matmul(clipOutLN, clipEmbeddingWeight.transposed()) + clipEmbeddingBias
    }

    private func forwardWithConditioning(
        latentNCT: MLXArray,
        batchSize: Int,
        sequenceLength: Int,
        conditions: [(name: String, embedding: MLXArray, asToken: Bool)]
    ) throws -> MLXArray {
        let d = configuration.dModel
        let heads = configuration.nHeads
        let headDim = d / heads

        var h = MLX.matmul(latentNCT.transposed(axes: [0, 2, 1]), inputProjectionWeight.transposed()) + inputProjectionBias

        for condition in conditions where !condition.asToken {
            let emb = condition.embedding
            if emb.ndim == 2 {
                guard emb.shape == [batchSize, d] else {
                    throw Image300MTransformerError.invalidConditionShape(name: condition.name, expected: "[\(batchSize), \(d)]", actual: emb.shape)
                }
                h = h + emb.expandedDimensions(axes: [1])
            } else if emb.ndim == 3 {
                guard emb.shape[0] == batchSize, emb.shape[2] == d else {
                    throw Image300MTransformerError.invalidConditionShape(name: condition.name, expected: "[\(batchSize), tokens, \(d)]", actual: emb.shape)
                }
                if emb.shape[1] == 1 || emb.shape[1] == sequenceLength {
                    h = h + emb
                } else {
                    throw Image300MTransformerError.invalidConditionShape(name: condition.name, expected: "[\(batchSize), 1, \(d)] or [\(batchSize), \(sequenceLength), \(d)]", actual: emb.shape)
                }
            } else {
                throw Image300MTransformerError.invalidConditionShape(name: condition.name, expected: "[B, d] or [B, tokens, d]", actual: emb.shape)
            }
        }

        if configuration.usePosEmb {
            h = h + positionEmbedding[0..<sequenceLength].expandedDimensions(axes: [0])
        }

        let extraTokenArrays: [MLXArray] = try conditions.compactMap { condition in
            guard condition.asToken else { return nil }
            let emb = condition.embedding
            if emb.ndim == 2 {
                guard emb.shape == [batchSize, d] else {
                    throw Image300MTransformerError.invalidConditionShape(name: condition.name, expected: "[\(batchSize), \(d)]", actual: emb.shape)
                }
                return emb.expandedDimensions(axes: [1])
            }
            if emb.ndim == 3 {
                guard emb.shape[0] == batchSize, emb.shape[2] == d else {
                    throw Image300MTransformerError.invalidConditionShape(name: condition.name, expected: "[\(batchSize), tokens, \(d)]", actual: emb.shape)
                }
                return emb
            }
            throw Image300MTransformerError.invalidConditionShape(name: condition.name, expected: "[B, d] or [B, tokens, d]", actual: emb.shape)
        }

        if !extraTokenArrays.isEmpty {
            h = MLX.concatenated(extraTokenArrays + [h], axis: 1)
        }

        let totalSeq = h.shape[1]
        h = Self.layerNorm(h, weight: layerNormPreWeight, bias: layerNormPreBias)
        let attentionScale = Float(1.0 / sqrt(sqrt(Double(headDim))))

        for layer in layers {
            let ln1 = Self.layerNorm(h, weight: layer.ln1Weight, bias: layer.ln1Bias)
            let qkv = MLX.matmul(ln1, layer.attentionQKVWeight.transposed()) + layer.attentionQKVBias
            let qkvR = qkv.reshaped(batchSize, totalSeq, heads, 3 * headDim)

            let q = qkvR[0..., 0..., 0..., 0..<headDim].transposed(axes: [0, 2, 1, 3])
            let k = qkvR[0..., 0..., 0..., headDim..<(2 * headDim)].transposed(axes: [0, 2, 1, 3])
            let v = qkvR[0..., 0..., 0..., (2 * headDim)..<(3 * headDim)].transposed(axes: [0, 2, 1, 3])

            let attnWeights = MLX.softmax(
                MLX.matmul(q * attentionScale, (k * attentionScale).transposed(axes: [0, 1, 3, 2])),
                axis: -1
            )
            let attnOut = MLX.matmul(attnWeights, v)
                .transposed(axes: [0, 2, 1, 3])
                .reshaped(batchSize, totalSeq, d)

            h = h + MLX.matmul(attnOut, layer.attentionOutputWeight.transposed()) + layer.attentionOutputBias

            let ln2 = Self.layerNorm(h, weight: layer.ln2Weight, bias: layer.ln2Bias)
            let mlpHidden = MLX.matmul(ln2, layer.mlpFCWeight.transposed()) + layer.mlpFCBias
            let activated = Self.gelu(mlpHidden)
            let mlpOut = MLX.matmul(activated, layer.mlpProjectionWeight.transposed()) + layer.mlpProjectionBias
            h = h + mlpOut
        }

        h = Self.layerNorm(h, weight: layerNormPostWeight, bias: layerNormPostBias)

        if !extraTokenArrays.isEmpty {
            let nExtra = extraTokenArrays.reduce(0) { $0 + $1.shape[1] }
            h = h[0..., nExtra..<(nExtra + sequenceLength), 0...]
        }

        let projected = MLX.matmul(h, outputProjectionWeight.transposed()) + outputProjectionBias
        return projected.transposed(axes: [0, 2, 1]).asType(.float32)
    }

    private static func layerNorm(_ x: MLXArray, weight: MLXArray, bias: MLXArray, epsilon: Float = 1e-5) -> MLXArray {
        let mean = MLX.mean(x, axes: [-1], keepDims: true)
        let variance = MLX.mean(MLX.square(x - mean), axes: [-1], keepDims: true)
        return (x - mean) * MLX.rsqrt(variance + epsilon) * weight + bias
    }

    private static func gelu(_ x: MLXArray) -> MLXArray {
        let half = MLXArray(Float(0.5))
        let one = MLXArray(Float(1.0))
        let div = MLXArray(Float(sqrt(2.0)))
        
        return x * half * (one + MLX.erf(x / div))
    }

    private static func quickGELU(_ x: MLXArray) -> MLXArray {
        return x * MLX.sigmoid(Float(1.702) * x)
    }

    private static func silu(_ x: MLXArray) -> MLXArray {
        return x * MLX.sigmoid(x)
    }

    private static func inferLayerCount(from keys: Dictionary<String, RuntimeTensor>.Keys) -> Int {
        let prefix = "wrapped.backbone.resblocks.layers."
        var maxIndex = -1
        for key in keys where key.hasPrefix(prefix) {
            let remainder = key.dropFirst(prefix.count)
            if let end = remainder.firstIndex(of: "."), let idx = Int(remainder[..<end]) {
                maxIndex = max(maxIndex, idx)
            }
        }
        return maxIndex + 1
    }

    private static func loadLayers(from artifact: Image300MArtifact, configuration: Configuration) throws -> [TransformerLayer] {
        let d = configuration.dModel
        let h = configuration.mlpHiddenDim
        return try (0..<configuration.nLayers).map { i in
            TransformerLayer(
                ln1Weight: try artifact.requireTensor(.resblock(i, .ln1Weight), expectedShape: [d], dataType: .float32).asMLX().asType(.float32),
                ln1Bias: try artifact.requireTensor(.resblock(i, .ln1Bias), expectedShape: [d], dataType: .float32).asMLX().asType(.float32),
                attentionQKVWeight: try artifact.requireTensor(.resblock(i, .attentionQKVWeight), expectedShape: [3 * d, d], dataType: .float32).asMLX().asType(.float32),
                attentionQKVBias: try artifact.requireTensor(.resblock(i, .attentionQKVBias), expectedShape: [3 * d], dataType: .float32).asMLX().asType(.float32),
                attentionOutputWeight: try artifact.requireTensor(.resblock(i, .attentionOutputWeight), expectedShape: [d, d], dataType: .float32).asMLX().asType(.float32),
                attentionOutputBias: try artifact.requireTensor(.resblock(i, .attentionOutputBias), expectedShape: [d], dataType: .float32).asMLX().asType(.float32),
                ln2Weight: try artifact.requireTensor(.resblock(i, .ln2Weight), expectedShape: [d], dataType: .float32).asMLX().asType(.float32),
                ln2Bias: try artifact.requireTensor(.resblock(i, .ln2Bias), expectedShape: [d], dataType: .float32).asMLX().asType(.float32),
                mlpFCWeight: try artifact.requireTensor(.resblock(i, .mlpFCWeight), expectedShape: [h, d], dataType: .float32).asMLX().asType(.float32),
                mlpFCBias: try artifact.requireTensor(.resblock(i, .mlpFCBias), expectedShape: [h], dataType: .float32).asMLX().asType(.float32),
                mlpProjectionWeight: try artifact.requireTensor(.resblock(i, .mlpOutputWeight), expectedShape: [d, h], dataType: .float32).asMLX().asType(.float32),
                mlpProjectionBias: try artifact.requireTensor(.resblock(i, .mlpOutputBias), expectedShape: [d], dataType: .float32).asMLX().asType(.float32)
            )
        }
    }
}

