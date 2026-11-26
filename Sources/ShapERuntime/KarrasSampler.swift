import Foundation
import MLX

// MARK: - KarrasSampler
//
// Swift port of the MLX production path in:
//   shap_e/src/mlx_shap_e/models/diffusion/karras_sampler.py

// MARK: - Sampler output

public struct KarrasSampleResult: @unchecked Sendable {
    /// Final denoised latent [B, D], channel-unscaled (already evaluated on GPU)
    public let x: MLXArray
    /// Predicted x_0 from the last denoising step [B, D], channel-unscaled
    public let predXStart: MLXArray
    /// Full sigma schedule [steps+1]
    public let sigmas: MLXArray
}

// MARK: - Sigma schedule

public enum KarrasSigmas {
    /// Karras et al. (2022) decreasing sigma schedule with terminal zero appended.
    /// Matches: `get_sigmas_karras_mlx`
    public static func build(
        steps: Int,
        sigmaMin: Float,
        sigmaMax: Float,
        rho: Float = 7.0
    ) -> MLXArray {
        let ramp      = MLX.linspace(Float(0), Float(1), count: steps).asType(.float32)
        let minInvRho = pow(Double(sigmaMin), 1.0 / Double(rho))
        let maxInvRho = pow(Double(sigmaMax), 1.0 / Double(rho))
        // sigmas[i] = (maxInvRho + ramp[i]*(minInvRho - maxInvRho))^rho
        let sigmas = MLX.pow(
            maxInvRho + ramp * Float(minInvRho - maxInvRho),
            rho
        )
        let zero = MLXArray([Float(0)])
        return MLX.concatenated([sigmas, zero], axis: 0)   // [steps+1]
    }
}

// MARK: - GaussianToKarrasDenoiser

public protocol DiffusionTransformer: Sendable {
    var nCtx: Int { get }
    func predictInternal(
        x xMLX: MLXArray,
        t tMLX: MLXArray,
        embeddings clipMLX: MLXArray
    ) throws -> MLXArray
}

public enum KarrasSamplerError: Error, Equatable, LocalizedError {
    case invalidConfig(name: String, message: String)
    case invalidTensorShape(name: String, expected: String, actual: [Int])
    case invalidChannelLayout(dLatent: Int, nCtx: Int)

    public var errorDescription: String? {
        switch self {
        case let .invalidConfig(name, message):
            return "Invalid Karras sampler config '\(name)': \(message)."
        case let .invalidTensorShape(name, expected, actual):
            return "Tensor '\(name)' has shape \(actual) but expected \(expected)."
        case let .invalidChannelLayout(dLatent, nCtx):
            return "Latent width \(dLatent) is not divisible by nCtx \(nCtx)."
        }
    }
}

public final class GaussianToKarrasDenoiser {

    private let transformer: any DiffusionTransformer
    private let diffusion: RuntimeDiffusionConfig
    private let nCtx: Int

    private let alphasAsc: MLXArray        // float32 [T], alphas_cumprod reversed into ascending alpha order
    private let timestepsAsc: MLXArray     // float32 [T], timestep indices paired with alphasAsc
    private let alphaHi: Float             // max alpha_cumprod, usually near 1 at t=0
    private let alphaLo: Float             // min alpha_cumprod, usually near 0 at t=T-1
    private let lastT: Int

    private let channelBiases: MLXArray?   // [1, D] or nil
    private let channelScales: MLXArray?   // [1, D] or nil

    public init(
        transformer: any DiffusionTransformer,
        diffusion: RuntimeDiffusionConfig,
        nCtx: Int = 1024
    ) throws {
        guard nCtx > 0 else {
            throw KarrasSamplerError.invalidConfig(name: "nCtx", message: "must be greater than zero")
        }

        self.transformer = transformer
        self.diffusion = diffusion
        self.nCtx = nCtx

        let alphas = try diffusion.alphasCumprod.asMLX().asType(.float32).asArray(Float.self)
        let T = alphas.count
        guard T >= 2 else {
            throw KarrasSamplerError.invalidTensorShape(
                name: "alphasCumprod",
                expected: "at least 2 values",
                actual: [T]
            )
        }

        var alphasRev = [Float](repeating: 0, count: T)
        var timestepsRev = [Float](repeating: 0, count: T)
        for i in 0..<T {
            alphasRev[i] = alphas[T - 1 - i]
            timestepsRev[i] = Float(T - 1 - i)
        }

        // Important fix: alphasRev is ascending if the original alphas_cumprod is descending.
        // Therefore first is alphaLo and last is alphaHi.
        self.alphasAsc = MLXArray(alphasRev)
        self.timestepsAsc = MLXArray(timestepsRev)
        self.alphaLo = alphasRev.first!
        self.alphaHi = alphasRev.last!
        self.lastT = T - 1

        if let b = diffusion.channelBiases {
            let arr = try b.asMLX().asType(.float32).flattened()
            self.channelBiases = arr.reshaped(1, arr.shape[0])
        } else {
            self.channelBiases = nil
        }

        if let s = diffusion.channelScales {
            let arr = try s.asMLX().asType(.float32).flattened()
            self.channelScales = arr.reshaped(1, arr.shape[0])
        } else {
            self.channelScales = nil
        }
    }

    // MARK: sigma -> integer timestep

    func sigmaToT(sigmas: MLXArray) -> MLXArray {
        let s = sigmas.asType(.float32)
        let alphaCumprod = (Float(1.0) / (s * s + Float(1.0))).asType(.float32)   // [B]

        // Count how many ascending alpha values are <= alphaCumprod[i].
        let ac = alphaCumprod.expandedDimensions(axes: [1])    // [B, 1]
        let asc = alphasAsc.expandedDimensions(axes: [0])      // [1, T]
        let geMask = ac .>= asc                                // [B, T] bool

        var idx = MLX.sum(geMask.asType(.int32), axes: [1]) - 1 // [B]
        let maxIdx = alphasAsc.shape[0] - 2
        idx = MLX.clip(idx, min: 0, max: maxIdx).asType(.int32)

        let x0 = alphasAsc[idx]
        let x1 = alphasAsc[idx + 1]
        let y0 = timestepsAsc[idx]
        let y1 = timestepsAsc[idx + 1]

        let denom = x1 - x0
        let eps = MLXArray(Float(1e-12))
        let frac = MLX.where(
            MLX.abs(denom) .> eps,
            (alphaCumprod - x0) / denom,
            MLX.zeros(like: alphaCumprod)
        )

        var t = y0 + frac * (y1 - y0)

        // Clamp out-of-table sigma-derived alpha values to the endpoint timesteps.
        t = MLX.where(alphaCumprod .> MLXArray(alphaHi), MLXArray(Float(0)), t)
        t = MLX.where(alphaCumprod .<= MLXArray(alphaLo), MLXArray(Float(lastT)), t)

        return t.asType(.int32)
    }

    // MARK: One denoise step -> pred_xstart [B, D]

    func denoise(
        x: MLXArray,
        sigmas: MLXArray,
        embeddings: MLXArray,
        clipDenoised: Bool = true
    ) throws -> MLXArray {
        guard x.ndim == 2 else {
            throw KarrasSamplerError.invalidTensorShape(
                name: "x",
                expected: "[batch, d_latent]",
                actual: x.shape
            )
        }

        let B = x.shape[0]
        let D = x.shape[1]

        guard sigmas.ndim == 1, sigmas.shape[0] == B else {
            throw KarrasSamplerError.invalidTensorShape(
                name: "sigmas",
                expected: "[batch] matching x.shape[0]",
                actual: sigmas.shape
            )
        }

        guard D > 0, D % nCtx == 0 else {
            throw KarrasSamplerError.invalidChannelLayout(dLatent: D, nCtx: nCtx)
        }

        let tInt = sigmaToT(sigmas: sigmas)

        let cIn = appendDims(Float(1.0) / MLX.sqrt(sigmas * sigmas + Float(1.0)), x.ndim)
        let xIn = (x * cIn).asType(.float32)

        // Run transformer natively -> [B, 2D].
        let modelOutputMLX = try DiffusionStep.splitVectorForward(
            transformer: transformer,
            xFlat: xIn,
            t: tInt,
            embeddings: embeddings
        )

        guard modelOutputMLX.ndim == 2,
              modelOutputMLX.shape[0] == B,
              modelOutputMLX.shape[1] == 2 * D else {
            throw KarrasSamplerError.invalidTensorShape(
                name: "modelOutputMLX",
                expected: "[\(B), \(2 * D)]",
                actual: modelOutputMLX.shape
            )
        }

        let channels = D / nCtx
        let xInNCT = xIn
            .reshaped(B, nCtx, channels)
            .transposed(axes: [0, 2, 1])

        let epsFlat = modelOutputMLX[0..., 0..<D]
        let varFlat = modelOutputMLX[0..., D..<(2 * D)]

        let epsNCT = epsFlat
            .reshaped(B, nCtx, channels)
            .transposed(axes: [0, 2, 1])

        let varNCT = varFlat
            .reshaped(B, nCtx, channels)
            .transposed(axes: [0, 2, 1])

        let yNCT = MLX.concatenated([epsNCT, varNCT], axis: 1)

        // This is where the original Torch clamp belongs:
        // denoised = denoised.clamp(-1, 1) inside the Gaussian/Karras denoiser path.
        let stepOut = try DiffusionStep.pMeanVariance(
            diffusion: diffusion,
            x: xInNCT,
            modelOutput: yNCT,
            timesteps: tInt,
            modelMeanType: diffusion.modelMeanType,
            clipDenoised: clipDenoised
        )

        return stepOut.predXStart
            .transposed(axes: [0, 2, 1])
            .reshaped(B, D)
            .asType(.float32)
    }

    // MARK: Channel unscaling

    func unscale(_ x: MLXArray) throws -> MLXArray {
        guard x.ndim == 2 else {
            throw KarrasSamplerError.invalidTensorShape(
                name: "x",
                expected: "[batch, d_latent]",
                actual: x.shape
            )
        }

        let D = x.shape[1]
        var out = x.asType(.float32)

        if let bias = channelBiases {
            guard bias.shape == [1, D] else {
                throw KarrasSamplerError.invalidTensorShape(
                    name: "channelBiases",
                    expected: "[1, \(D)]",
                    actual: bias.shape
                )
            }
            out = out - bias
        }

        if let scale = channelScales {
            guard scale.shape == [1, D] else {
                throw KarrasSamplerError.invalidTensorShape(
                    name: "channelScales",
                    expected: "[1, \(D)]",
                    actual: scale.shape
                )
            }
            out = out / scale
        }

        return out.asType(.float32)
    }

    private func appendDims(_ x: MLXArray, _ targetDims: Int) -> MLXArray {
        var out = x
        while out.ndim < targetDims {
            out = out.expandedDimensions(axes: [-1])
        }
        return out
    }
}

// MARK: - KarrasSampler

public enum KarrasSampler {

    public struct Config: Sendable {
        public let steps: Int
        public let sigmaMin: Float
        public let sigmaMax: Float
        public let rho: Float
        public let sChurn: Float
        public let sTmin: Float
        public let sTmax: Float
        public let sNoise: Float
        public let guidanceScale: Float
        public let clipDenoised: Bool
        public let seed: Int
        /// How often to `MLX.eval(x)` during sampling **without** mesh extraction.
        /// `0` disables periodic sync (only the terminal eval runs). Default `4`.
        public let evalInterval: Int
        public let progressInterval: Int

        public init(
            steps: Int = 64,
            sigmaMin: Float = 0.002,
            sigmaMax: Float = 80.0,
            rho: Float = 7.0,
            sChurn: Float = 0.0,
            sTmin: Float = 0.0,
            sTmax: Float = .infinity,
            sNoise: Float = 1.0,
            guidanceScale: Float = 0.0,
            clipDenoised: Bool = true,
            seed: Int = 0,
            evalInterval: Int = 4,
            progressInterval: Int = 5
        ) {
            self.steps = steps
            self.sigmaMin = sigmaMin
            self.sigmaMax = sigmaMax
            self.rho = rho
            self.sChurn = sChurn
            self.sTmin = sTmin
            self.sTmax = sTmax
            self.sNoise = sNoise
            self.guidanceScale = guidanceScale
            self.clipDenoised = clipDenoised
            self.seed = seed
            self.evalInterval = evalInterval
            self.progressInterval = progressInterval
        }
    }

    public static func sample(
        denoiser: GaussianToKarrasDenoiser,
        embeddings: RuntimeTensor,
        uncondEmbeddings: RuntimeTensor? = nil,
        shape: (batchSize: Int, dLatent: Int),
        config: Config,
        /// Optional initial noise `[B, D]`. When set, used instead of MLXRandom so Swift
        /// can match Python's `np.random.default_rng(seed).standard_normal * sigma_max`.
        initX: MLXArray? = nil,
        progress: ((_ step: Int, _ totalSteps: Int, _ denoisedLatent: MLXArray?) -> Void)? = nil
    ) throws -> KarrasSampleResult {
        try validate(config: config, shape: shape)

        let B = shape.batchSize
        let D = shape.dLatent

        let sigmas = KarrasSigmas.build(
            steps: config.steps,
            sigmaMin: config.sigmaMin,
            sigmaMax: config.sigmaMax,
            rho: config.rho
        )

        let sigmaValues = sigmas.asArray(Float.self)
        let nSteps = sigmaValues.count
        guard nSteps >= 2 else {
            throw KarrasSamplerError.invalidTensorShape(
                name: "sigmas",
                expected: "at least 2 values, usually steps plus final zero",
                actual: [nSteps]
            )
        }

        // Private state, not MLXRandom.seed. seed() writes the process-wide PRNG,
        // so overlapping sample() calls would reseeds each other's draws.
        // One draw from RandomState(seed:) matches seed() followed by one normal().
        var noiseState = MLXRandom.RandomState(seed: UInt64(config.seed))
        var x: MLXArray
        if let initX {
            guard initX.shape == [B, D] else {
                throw KarrasSamplerError.invalidTensorShape(
                    name: "initX",
                    expected: "[\(B), \(D)]",
                    actual: initX.shape
                )
            }
            x = initX.asType(.float32)
        } else if B == 1 {
            // Not bit-identical to Python's NumPy RNG. Parity passes initX instead.
            x = MLXRandom.normal([1, D], key: noiseState) * config.sigmaMax
        } else {
            // Distinct seed per batch row (base, base+1, …) so batchSize > 1
            // yields different shapes instead of near-identical draws from one stream.
            var rows: [MLXArray] = []
            rows.reserveCapacity(B)
            for i in 0..<B {
                let rowState = MLXRandom.RandomState(seed: UInt64(config.seed) &+ UInt64(i))
                rows.append(MLXRandom.normal([1, D], key: rowState) * config.sigmaMax)
                noiseState = rowState
            }
            x = MLX.concatenated(rows, axis: 0)
        }

        var embMLX = try embeddings.asMLX().asType(.float32)
        guard embMLX.ndim >= 1, embMLX.shape[0] == B else {
            throw KarrasSamplerError.invalidTensorShape(
                name: "embeddings",
                expected: "batch dimension \(B)",
                actual: embMLX.shape
            )
        }

        let useCFG = config.guidanceScale != 0.0 && config.guidanceScale != 1.0
        if useCFG {
            let uncondMLX: MLXArray
            if let uncondEmbeddings {
                uncondMLX = try uncondEmbeddings.asMLX().asType(.float32)
                guard uncondMLX.shape == embMLX.shape else {
                    throw KarrasSamplerError.invalidTensorShape(
                        name: "uncondEmbeddings",
                        expected: "same shape as embeddings, \(embMLX.shape)",
                        actual: uncondMLX.shape
                    )
                }
            } else {
                uncondMLX = MLX.zeros(like: embMLX)
            }

            // Match the Torch code: x_t is duplicated as [cond, uncond], and the denoiser
            // clamps each half before CFG combines them. Do not clip the guided result again.
            embMLX = MLX.concatenated([embMLX, uncondMLX], axis: 0)
        }

        let sIn = MLX.ones([B], type: Float.self)
        var denoised = MLX.zeros([B, D], type: Float.self)
        let sqrtTwo = Float(sqrt(2.0))
        let nStepsMinusOne = nSteps - 1
        for i in 0..<nStepsMinusOne {
            let sI = sigmaValues[i]
            let sNext = sigmaValues[i + 1]

            let inRange = sI >= config.sTmin && sI <= config.sTmax
            let gamma = inRange
                ? min(config.sChurn / Float(nStepsMinusOne), sqrtTwo - Float(1.0))
                : Float(0.0)
            let sigmaHat = sI * (gamma + Float(1.0))

            if gamma > 0.0 {
                let noise = MLXRandom.normal([B, D], key: noiseState) * config.sNoise
                let delta = max(Double(sigmaHat * sigmaHat - sI * sI), 0.0)
                let noiseMag = Float(delta.squareRoot())
                x = x + noise * noiseMag
            }

            let sigmaHatBatch = MLXArray(sigmaHat) * sIn

            denoised = try singleDenoise(
                denoiser: denoiser,
                x: x,
                sigmas: sigmaHatBatch,
                embeddings: embMLX,
                useCFG: useCFG,
                guidanceScale: config.guidanceScale,
                clipDenoised: config.clipDenoised,
                B: B,
                D: D
            )

            let d = (x - denoised) / appendDims(sigmaHatBatch, targetDims: x.ndim)
            let dt = sNext - sigmaHat

            if sNext == Float(0.0) {
                x = x + d * dt
            } else {
                let x2 = x + d * dt
                let sigmaNextBatch = MLXArray(sNext) * sIn

                let denoised2 = try singleDenoise(
                    denoiser: denoiser,
                    x: x2,
                    sigmas: sigmaNextBatch,
                    embeddings: embMLX,
                    useCFG: useCFG,
                    guidanceScale: config.guidanceScale,
                    clipDenoised: config.clipDenoised,
                    B: B,
                    D: D
                )

                let d2 = (x2 - denoised2) / appendDims(sigmaNextBatch, targetDims: x2.ndim)
                let dPrime = (d + d2) * Float(0.5)
                x = x + dPrime * dt
            }

            
            try Task.checkCancellation()

            let step = i + 1
            let isLast = step == nStepsMinusOne
            let evalEvery = config.evalInterval
            let shouldEval = isLast || (evalEvery > 0 && step.isMultiple(of: evalEvery))
            let shouldExtract = progress != nil
                && config.progressInterval > 0
                && i % config.progressInterval == 0

            if shouldExtract {
                let unscaled = try denoiser.unscale(denoised)
                MLX.eval(x, unscaled)
                progress?(step, nStepsMinusOne, unscaled)
            } else {
                // Materialize diffusion without marching-cubes / projector work.
                if shouldEval {
                    MLX.eval(x)
                }
                progress?(step, nStepsMinusOne, nil)
            }
        }

        let xOut = try denoiser.unscale(x)
        let pxOut = try denoiser.unscale(denoised)

        MLX.eval(xOut, pxOut, sigmas)

        return KarrasSampleResult(x: xOut, predXStart: pxOut, sigmas: sigmas)
    }

    private static func singleDenoise(
        denoiser: GaussianToKarrasDenoiser,
        x: MLXArray,
        sigmas: MLXArray,
        embeddings: MLXArray,
        useCFG: Bool,
        guidanceScale: Float,
        clipDenoised: Bool,
        B: Int,
        D: Int
    ) throws -> MLXArray {
        if useCFG {
            let xCat = MLX.concatenated([x, x], axis: 0)
            let sCat = MLX.concatenated([sigmas, sigmas], axis: 0)

            let x0Cat = try denoiser.denoise(
                x: xCat,
                sigmas: sCat,
                embeddings: embeddings,
                clipDenoised: clipDenoised
            )

            guard x0Cat.ndim == 2, x0Cat.shape == [2 * B, D] else {
                throw KarrasSamplerError.invalidTensorShape(
                    name: "x0Cat",
                    expected: "[\(2 * B), \(D)]",
                    actual: x0Cat.shape
                )
            }

            let cond = x0Cat[0..<B]
            let uncond = x0Cat[B..<(2 * B)]
            let guided = uncond + guidanceScale * (cond - uncond)

            return guided.asType(.float32)
        } else {
            let x0 = try denoiser.denoise(
                x: x,
                sigmas: sigmas,
                embeddings: embeddings,
                clipDenoised: clipDenoised
            )

            guard x0.ndim == 2, x0.shape == [B, D] else {
                throw KarrasSamplerError.invalidTensorShape(
                    name: "x0",
                    expected: "[\(B), \(D)]",
                    actual: x0.shape
                )
            }

            return x0.asType(.float32)
        }
    }

    private static func appendDims(_ x: MLXArray, targetDims: Int) -> MLXArray {
        var out = x
        while out.ndim < targetDims {
            out = out.expandedDimensions(axes: [-1])
        }
        return out
    }

    private static func validate(config: Config, shape: (batchSize: Int, dLatent: Int)) throws {
        guard shape.batchSize > 0 else {
            throw KarrasSamplerError.invalidConfig(name: "shape.batchSize", message: "must be greater than zero")
        }
        guard shape.dLatent > 0 else {
            throw KarrasSamplerError.invalidConfig(name: "shape.dLatent", message: "must be greater than zero")
        }
        guard config.steps > 0 else {
            throw KarrasSamplerError.invalidConfig(name: "steps", message: "must be greater than zero")
        }
        guard config.sigmaMin.isFinite, config.sigmaMin > 0 else {
            throw KarrasSamplerError.invalidConfig(name: "sigmaMin", message: "must be finite and greater than zero")
        }
        guard config.sigmaMax.isFinite, config.sigmaMax >= config.sigmaMin else {
            throw KarrasSamplerError.invalidConfig(name: "sigmaMax", message: "must be finite and >= sigmaMin")
        }
        guard config.rho.isFinite, config.rho > 0 else {
            throw KarrasSamplerError.invalidConfig(name: "rho", message: "must be finite and greater than zero")
        }
        guard config.sChurn.isFinite, config.sChurn >= 0 else {
            throw KarrasSamplerError.invalidConfig(name: "sChurn", message: "must be finite and >= 0")
        }
        guard config.sTmin.isFinite else {
            throw KarrasSamplerError.invalidConfig(name: "sTmin", message: "must be finite")
        }
        guard config.sTmax.isFinite || config.sTmax == Float.infinity else {
            throw KarrasSamplerError.invalidConfig(name: "sTmax", message: "must be finite or +infinity")
        }
        guard config.sTmin <= config.sTmax else {
            throw KarrasSamplerError.invalidConfig(name: "sTmin/sTmax", message: "sTmin must be <= sTmax")
        }
        guard config.sNoise.isFinite, config.sNoise >= 0 else {
            throw KarrasSamplerError.invalidConfig(name: "sNoise", message: "must be finite and >= 0")
        }
        guard config.guidanceScale.isFinite else {
            throw KarrasSamplerError.invalidConfig(name: "guidanceScale", message: "must be finite")
        }
        guard config.seed >= 0 else {
            throw KarrasSamplerError.invalidConfig(name: "seed", message: "must be >= 0 because UInt64(seed) would trap")
        }
    }
}
