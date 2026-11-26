import Foundation
import MLX

// MARK: - DiffusionStep
//
// Swift port of shap_e/src/mlx_shap_e/models/diffusion/diffusion_step.py
//
// Covers three operations that the Karras sampler calls on every denoising step:
//   1. split_vector_forward  — reshape flat latent, run Text300M, flatten [eps, var]
//   2. extractInto           — index a 1-D schedule array by timestep, broadcast to latent shape
//   3. pMeanVariance         — learned-range variance + posterior mean reconstruction

// MARK: - Output type

/// The four tensors returned by one denoising step.
public struct DiffusionStepOutput: @unchecked Sendable {
    /// Posterior mean μ̃(x_t, x_0)  — shape [B, C, T]
    public let mean: MLXArray
    /// Posterior variance σ²         — shape [B, C, T]
    public let variance: MLXArray
    /// Log  posterior variance ln σ² — shape [B, C, T]
    public let logVariance: MLXArray
    /// Predicted x_0 from eps        — shape [B, C, T]
    public let predXStart: MLXArray

    public init(mean: MLXArray, variance: MLXArray, logVariance: MLXArray, predXStart: MLXArray) {
        self.mean = mean
        self.variance = variance
        self.logVariance = logVariance
        self.predXStart = predXStart
    }
}

// MARK: - DiffusionStep

public enum DiffusionStep {

    // MARK: 1. split_vector_forward

    /// Reshape a flat latent `[B, D]` into `[B, C, T]`, run the Text300M transformer,
    /// then split the `[B, 2C, T]` output into eps and var and return them flat as `[B, 2D]`.
    public static func splitVectorForward(
        transformer: Text300MTransformer,
        xFlat: RuntimeTensor,       // [B, D]
        timesteps: RuntimeTensor,   // [B]
        embeddings: RuntimeTensor   // [B, clipDim]
    ) throws -> RuntimeTensor {
        let outMLX = try splitVectorForward(
            transformer: transformer,
            xFlat: try xFlat.asMLX(),
            t: try timesteps.asMLX().asType(.float32),
            embeddings: try embeddings.asMLX().asType(.float32)
        )
        let outFloats = outMLX.asArray(Float.self)
        let outData   = outFloats.withUnsafeBufferPointer { Data(buffer: $0) }
        return try RuntimeTensor(
            name: "split_vector_output",
            shape: outMLX.shape,
            dataType: .float32,
            data: outData
        )
    }

    /// MLXArray native path for splitVectorForward.
    public static func splitVectorForward(
        transformer: any DiffusionTransformer,
        xFlat: MLXArray,            // [B, D]
        t: MLXArray,                // [B]
        embeddings: MLXArray        // [B, clipDim]
    ) throws -> MLXArray {
        let batchSize = xFlat.shape[0]
        let dLatent   = xFlat.shape[1]
        let nCtx      = transformer.nCtx
        let channels  = dLatent / nCtx

        // [B, D] → [B, T, C] → [B, C, T]
        let xNCT = xFlat.reshaped(batchSize, nCtx, channels).transposed(axes: [0, 2, 1])

        // Run the transformer → [B, 2C, T]
        let yNCT = try transformer.predictInternal(x: xNCT, t: t, embeddings: embeddings)

        // Split eps / var along the channel axis
        let half   = yNCT.shape[1] / 2
        let eps    = yNCT[0..., 0..<half, 0...]            // [B, C, T]
        let varOut = yNCT[0..., half...,  0...]            // [B, C, T]

        // Flatten: [B, C, T] → [B, T, C] → [B, D] then concat → [B, 2D]
        let epsFlat = eps.transposed(axes: [0, 2, 1]).reshaped(batchSize, dLatent)
        let varFlat = varOut.transposed(axes: [0, 2, 1]).reshaped(batchSize, dLatent)
        return MLX.concatenated([epsFlat, varFlat], axis: 1)  // [B, 2D]
    }

    // MARK: 2. _extract_into

    /// Index a 1-D schedule `arr` at integer timesteps and broadcast to `broadcastShape`.
    static func extractInto(
        arr: MLXArray,              // 1-D schedule, e.g. alphas_cumprod [T_max]
        timesteps: MLXArray,        // int32 [B]
        broadcastShape: [Int]       // target shape, typically [B, C, T] of the latent
    ) -> MLXArray {
        // Index: arr[timesteps] → [B]
        var result = arr.asType(.float32)[timesteps.asType(.int32)]
        // Expand trailing dims until rank matches
        while result.ndim < broadcastShape.count {
            result = result.expandedDimensions(axes: [-1])
        }
        return MLX.broadcast(result, to: broadcastShape).asType(.float32)
    }

    // MARK: 3. p_mean_variance (learned-range)

    /// Compute the posterior mean, variance and predicted x_0 for one denoising step.
    public static func pMeanVariance(
        diffusion: RuntimeDiffusionConfig,
        x: RuntimeTensor,               // [B, C, T]
        modelOutput: RuntimeTensor,     // [B, 2C, T]
        timesteps: RuntimeTensor,       // [B]  int or float
        modelMeanType: String = "epsilon",
        clipDenoised: Bool = true
    ) throws -> DiffusionStepOutput {
        // This is now just a bridge to the MLXInternal version
        let outMLX = try pMeanVariance(
            diffusion:     diffusion,
            x:             try x.asMLX().asType(.float32),
            modelOutput:   try modelOutput.asMLX().asType(.float32),
            timesteps:     try timesteps.asMLX().asType(.int32),
            modelMeanType: modelMeanType,
            clipDenoised:  clipDenoised
        )
        return outMLX // DiffusionStepOutput now uses MLXArray, so we don't need to re-wrap
    }

    /// MLXArray native path for pMeanVariance.
    public static func pMeanVariance(
        diffusion: RuntimeDiffusionConfig,
        x: MLXArray,                    // [B, C, T]
        modelOutput: MLXArray,          // [B, 2C, T]
        timesteps: MLXArray,            // [B]
        modelMeanType: String = "epsilon",
        clipDenoised: Bool = true
    ) throws -> DiffusionStepOutput {
        guard x.ndim == 3 else {
            throw DiffusionStepError.invalidInputShape(
                name: "x", expected: "[B, C, T]", actual: x.shape
            )
        }
        let batchSize = x.shape[0]
        let channels  = x.shape[1]
        let nCtx      = x.shape[2]

        guard modelOutput.shape == [batchSize, channels * 2, nCtx] else {
            throw DiffusionStepError.invalidInputShape(
                name: "modelOutput",
                expected: "[\(batchSize), \(channels * 2), \(nCtx)]",
                actual: modelOutput.shape
            )
        }

        // Split model output into eps and raw variance signal
        let half          = channels
        let eps           = modelOutput[0..., 0..<half, 0...]        // [B, C, T]
        let modelVarValues = modelOutput[0..., half...,  0...]       // [B, C, T]

        // --- Learned-range variance ---
        let minLog =  extractInto(
            arr: try diffusion.posteriorLogVarianceClipped.asMLX(),
            timesteps: timesteps,
            broadcastShape: x.shape
        )
        let betasMLX = try diffusion.betas.asMLX().asType(.float32)
        let maxLog =  extractInto(
            arr: MLX.log(betasMLX),
            timesteps: timesteps,
            broadcastShape: x.shape
        )
        let frac             = (modelVarValues + Float(1.0)) / Float(2.0)
        let modelLogVariance = frac * maxLog + (Float(1.0) - frac) * minLog
        let modelVariance    = MLX.exp(modelLogVariance)

        // --- Predicted x_0 ---
        var predXStart: MLXArray
        switch modelMeanType {
        case "x_start":
            predXStart = eps
        case "epsilon":
            let sqrtRecip = try extractInto(
                arr: diffusion.sqrtRecipAlphasCumprod.asMLX(),
                timesteps: timesteps,
                broadcastShape: x.shape
            )
            let sqrtRecipm1 = try extractInto(
                arr: diffusion.sqrtRecipm1AlphasCumprod.asMLX(),
                timesteps: timesteps,
                broadcastShape: x.shape
            )
            predXStart = sqrtRecip * x - sqrtRecipm1 * eps
        default:
            throw DiffusionStepError.unsupportedModelMeanType(modelMeanType)
        }

        if clipDenoised {
            predXStart = MLX.clip(predXStart, min: Float(-1.0), max: Float(1.0))
        }

        // --- Posterior mean ---
        let coef1 = try extractInto(
            arr: diffusion.posteriorMeanCoef1.asMLX(),
            timesteps: timesteps,
            broadcastShape: x.shape
        )
        let coef2 = try extractInto(
            arr: diffusion.posteriorMeanCoef2.asMLX(),
            timesteps: timesteps,
            broadcastShape: x.shape
        )
        let modelMean = coef1 * predXStart + coef2 * x

        return DiffusionStepOutput(
            mean:         modelMean,
            variance:     modelVariance,
            logVariance:  modelLogVariance,
            predXStart:   predXStart
        )
    }
}

// MARK: - Errors

public enum DiffusionStepError: Error, LocalizedError {
    case invalidInputShape(name: String, expected: String, actual: [Int])
    case unsupportedModelMeanType(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidInputShape(name, expected, actual):
            return "DiffusionStep: tensor '\(name)' has shape \(actual), expected \(expected)."
        case let .unsupportedModelMeanType(t):
            return "DiffusionStep: unsupported modelMeanType '\(t)'. Use 'epsilon' or 'x_start'."
        }
    }
}
