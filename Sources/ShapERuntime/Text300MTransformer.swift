import Foundation
import MLX
import MLXNN

// MARK: - Errors

public enum Text300MTransformerError: Error, Equatable, LocalizedError {
    case invalidTensorShape(name: String, expected: String, actual: [Int])
    case unsupportedTimestepDataType(TensorDataType)
    case batchSizeMismatch(expected: Int, actual: Int, tensorName: String)
    case sequenceLengthExceedsContext(sequenceLength: Int, contextLength: Int)
    case modelWidthNotDivisible(width: Int, heads: Int)

    public var errorDescription: String? {
        switch self {
        case let .invalidTensorShape(name, expected, actual):
            return "Tensor '\(name)' has shape \(actual) but expected \(expected)."
        case let .unsupportedTimestepDataType(dataType):
            return "Text300M timesteps must be float32, int32, or int64. Received \(dataType.rawValue)."
        case let .batchSizeMismatch(expected, actual, tensorName):
            return "Tensor '\(tensorName)' has batch size \(actual) but expected \(expected)."
        case let .sequenceLengthExceedsContext(sequenceLength, contextLength):
            return "Input sequence length \(sequenceLength) exceeds text300m context length \(contextLength)."
        case let .modelWidthNotDivisible(width, heads):
            return "Text300M model width \(width) must be divisible by attention head count \(heads)."
        }
    }
}

// MARK: - RuntimeTensor → MLXArray bridge

public extension RuntimeTensor {
    /// One-time eager cast from a raw byte-buffer RuntimeTensor into a native MLXArray.
    /// All shape information is preserved. Supported dtypes: float32, int32, int64.
    func asMLX() throws -> MLXArray {
        switch dataType {
        case .float32:
            return MLXArray(try float32Values(), shape)
        case .int32:
            return MLXArray(try int32Values(), shape)
        case .int64:
            return MLXArray(try int64Values(), shape)
        }
    }

    /// Lazy cast of RuntimeTensor to MLXArray without copying data.
    /// Note: if the tensor data is moved to another device, the MLXArray will be invalid.
    func asMLXLazy() -> MLXArray {
        switch dataType {
        case .float32:
            return MLXArray(data, shape, dtype: .float32)
        case .int32:
            return MLXArray(data, shape, dtype: .int32)
        case .int64:
            return MLXArray(data, shape, dtype: .int64)
        }
    }
}

// MARK: - Timestep embedding (MLX native)

/// Sinusoidal timestep embedding, matching shap_e.models.generation.util.timestep_embedding.
public enum TimestepEmbedding {
    /// - Parameters:
    ///   - timesteps: 1-D MLXArray of shape [B] containing float32 timestep values.
    ///   - dimension:  Output embedding dimension.
    ///   - maxPeriod: Maximum sinusoidal period (default 10 000).
    /// - Returns: MLXArray of shape [B, dimension] in float32.
    public static func compute(
        timesteps: MLXArray,
        dimension: Int,
        maxPeriod: Double = 10_000
    ) -> MLXArray {
        precondition(dimension > 0, "TimestepEmbedding: dimension must be positive.")
        let half = dimension / 2
        // freqs: [half]
        let indices = MLXArray(Array(0..<half).map(Float.init))
        let freqs = MLX.exp(-Float(log(maxPeriod)) * indices / Float(half))
        // args: [B, half]
        let args = timesteps.asType(.float32).expandedDimensions(axes: [1]) * freqs.expandedDimensions(axes: [0])
        // embedding: [B, dim] (or [B, dim-1] + zero pad when dim is odd)
        var embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
        if dimension % 2 != 0 {
            let pad = MLX.zeros([embedding.shape[0], 1], type: Float.self)
            embedding = MLX.concatenated([embedding, pad], axis: -1)
        }
        return embedding
    }
}

// MARK: - Text300M Transformer (MLXArray execution path)

public final class Text300MTransformer: DiffusionTransformer, @unchecked Sendable {

    // MARK: Configuration

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

    // MARK: Per-layer weight bundle (all MLXArray, loaded once at init)

    private struct TransformerLayer: @unchecked Sendable {
        let ln1Weight: MLXArray
        let ln1Bias: MLXArray
        let attentionQKVWeight: MLXArray   // [3*d, d]
        let attentionQKVBias: MLXArray     // [3*d]
        let attentionOutputWeight: MLXArray // [d, d]
        let attentionOutputBias: MLXArray  // [d]
        let ln2Weight: MLXArray
        let ln2Bias: MLXArray
        let mlpFCWeight: MLXArray          // [hidden, d]
        let mlpFCBias: MLXArray            // [hidden]
        let mlpProjectionWeight: MLXArray  // [d, hidden]
        let mlpProjectionBias: MLXArray    // [d]
    }

    // MARK: Public properties

    public let artifact: Text300MArtifact
    public var nCtx: Int { artifact.nCtx }
    public let configuration: Configuration

    // MARK: Top-level weights (MLXArray)

    private let inputProjectionWeight: MLXArray    // [d, inputC]
    private let inputProjectionBias: MLXArray      // [d]
    private let clipEmbeddingWeight: MLXArray      // [d, clipDim]
    private let clipEmbeddingBias: MLXArray        // [d]
    private let layerNormPreWeight: MLXArray       // [d]
    private let layerNormPreBias: MLXArray         // [d]
    private let layerNormPostWeight: MLXArray      // [d]
    private let layerNormPostBias: MLXArray        // [d]
    private let outputProjectionWeight: MLXArray   // [outputC, d]
    private let outputProjectionBias: MLXArray     // [outputC]
    private let positionEmbedding: MLXArray        // [ctx, d]
    private let positionEmbeddingRows: Int
    private let timeEmbeddingFCWeight: MLXArray    // [hidden, d]
    private let timeEmbeddingFCBias: MLXArray      // [hidden]
    private let timeEmbeddingProjectionWeight: MLXArray // [d, hidden]
    private let timeEmbeddingProjectionBias: MLXArray   // [d]
    private let layers: [TransformerLayer]

    // MARK: Initializers

    /// Convenience init: infers all Configuration fields from the artifact's weight shapes.
    public convenience init(
        artifact: Text300MArtifact,
        nHeads: Int = 16,
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

    /// Designated init: performs the one-time eager cast from RuntimeTensor → MLXArray for
    /// every weight in the network. The hot-path forward pass never touches the artifact again.
    public init(artifact: Text300MArtifact, configuration: Configuration) throws {
        guard configuration.dModel % configuration.nHeads == 0 else {
            throw Text300MTransformerError.modelWidthNotDivisible(
                width: configuration.dModel,
                heads: configuration.nHeads
            )
        }

        self.artifact = artifact
        self.configuration = configuration

        // --- Top-level projections ---
        self.inputProjectionWeight = try artifact.requireTensor(
            .inputProjectionWeight,
            expectedShape: [configuration.dModel, configuration.inputChannels],
            dataType: .float32
        ).asMLX()
        self.inputProjectionBias = try artifact.requireTensor(
            .inputProjectionBias,
            expectedShape: [configuration.dModel],
            dataType: .float32
        ).asMLX()

        self.clipEmbeddingWeight = try artifact.requireTensor(
            .clipEmbeddingWeight,
            expectedShape: [configuration.dModel, configuration.clipFeatureDim],
            dataType: .float32
        ).asMLX()
        self.clipEmbeddingBias = try artifact.requireTensor(
            .clipEmbeddingBias,
            expectedShape: [configuration.dModel],
            dataType: .float32
        ).asMLX()

        self.layerNormPreWeight = try artifact.requireTensor(
            .layerNormPreWeight,
            expectedShape: [configuration.dModel],
            dataType: .float32
        ).asMLX()
        self.layerNormPreBias = try artifact.requireTensor(
            .layerNormPreBias,
            expectedShape: [configuration.dModel],
            dataType: .float32
        ).asMLX()

        self.layerNormPostWeight = try artifact.requireTensor(
            .layerNormPostWeight,
            expectedShape: [configuration.dModel],
            dataType: .float32
        ).asMLX()
        self.layerNormPostBias = try artifact.requireTensor(
            .layerNormPostBias,
            expectedShape: [configuration.dModel],
            dataType: .float32
        ).asMLX()

        self.outputProjectionWeight = try artifact.requireTensor(
            .outputProjectionWeight,
            expectedShape: [configuration.outputChannels, configuration.dModel],
            dataType: .float32
        ).asMLX()
        self.outputProjectionBias = try artifact.requireTensor(
            .outputProjectionBias,
            expectedShape: [configuration.outputChannels],
            dataType: .float32
        ).asMLX()

        // --- Position embedding ---
        let posT = try artifact.requireTensor(.positionEmbedding, dataType: .float32)
        guard posT.shape.count == 2, posT.shape[1] == configuration.dModel else {
            throw Text300MTransformerError.invalidTensorShape(
                name: posT.name,
                expected: "[context, \(configuration.dModel)]",
                actual: posT.shape
            )
        }
        self.positionEmbeddingRows = posT.shape[0]
        self.positionEmbedding = try posT.asMLX()

        // --- Time embedding MLP ---
        self.timeEmbeddingFCWeight = try artifact.requireTensor(
            .timeEmbeddingFCWeight,
            expectedShape: [configuration.mlpHiddenDim, configuration.dModel],
            dataType: .float32
        ).asMLX()
        self.timeEmbeddingFCBias = try artifact.requireTensor(
            .timeEmbeddingFCBias,
            expectedShape: [configuration.mlpHiddenDim],
            dataType: .float32
        ).asMLX()
        self.timeEmbeddingProjectionWeight = try artifact.requireTensor(
            .timeEmbeddingProjectionWeight,
            expectedShape: [configuration.dModel, configuration.mlpHiddenDim],
            dataType: .float32
        ).asMLX()
        self.timeEmbeddingProjectionBias = try artifact.requireTensor(
            .timeEmbeddingProjectionBias,
            expectedShape: [configuration.dModel],
            dataType: .float32
        ).asMLX()

        // --- Per-layer weights ---
        self.layers = try Self.loadLayers(from: artifact, configuration: configuration)
    }

    // MARK: Configuration inference

    public static func inferConfiguration(
        from artifact: Text300MArtifact,
        nHeads: Int = 16,
        tokenCond: Bool = true,
        timeTokenCond: Bool = true,
        usePosEmb: Bool = true
    ) throws -> Configuration {
        let inputW = try artifact.requireTensor(.inputProjectionWeight, dataType: .float32)
        guard inputW.shape.count == 2 else {
            throw Text300MTransformerError.invalidTensorShape(
                name: inputW.name, expected: "[d_model, input_channels]", actual: inputW.shape
            )
        }
        let clipW = try artifact.requireTensor(.clipEmbeddingWeight, dataType: .float32)
        guard clipW.shape.count == 2 else {
            throw Text300MTransformerError.invalidTensorShape(
                name: clipW.name, expected: "[d_model, clip_feature_dim]", actual: clipW.shape
            )
        }
        let outputW = try artifact.requireTensor(.outputProjectionWeight, dataType: .float32)
        guard outputW.shape.count == 2 else {
            throw Text300MTransformerError.invalidTensorShape(
                name: outputW.name, expected: "[output_channels, d_model]", actual: outputW.shape
            )
        }
        let timeW = try artifact.requireTensor(.timeEmbeddingFCWeight, dataType: .float32)
        guard timeW.shape.count == 2 else {
            throw Text300MTransformerError.invalidTensorShape(
                name: timeW.name, expected: "[mlp_hidden_dim, d_model]", actual: timeW.shape
            )
        }

        return Configuration(
            dModel: inputW.shape[0],
            nLayers: inferLayerCount(from: artifact.state.keys),
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

    // MARK: Public forward pass

    /// Run the full Text300M diffusion forward pass.
    ///
    /// - Parameters:
    ///   - latent:        `RuntimeTensor` of shape [B, inputChannels, T] (float32) — the noisy latent.
    ///   - timesteps:     `RuntimeTensor` of shape [B] (float32 / int32 / int64) — diffusion step indices.
    ///   - clipEmbeddings:`RuntimeTensor` of shape [B, clipFeatureDim] (float32) — CLIP conditioning vectors.
    /// - Returns:          `RuntimeTensor` of shape [B, outputChannels, T] (float32) — predicted [eps, var].
    public func predict(
        x latent: RuntimeTensor,
        t timesteps: RuntimeTensor,
        embeddings clipEmbeddings: RuntimeTensor
    ) throws -> RuntimeTensor {
        // --- Input validation ---
        guard latent.shape.count == 3 else {
            throw Text300MTransformerError.invalidTensorShape(
                name: latent.name, expected: "[batch, channels, tokens]", actual: latent.shape
            )
        }
        let batchSize      = latent.shape[0]
        let inputChannels  = latent.shape[1]
        let sequenceLength = latent.shape[2]

        guard inputChannels == configuration.inputChannels else {
            throw Text300MTransformerError.invalidTensorShape(
                name: latent.name,
                expected: "[batch, \(configuration.inputChannels), tokens]",
                actual: latent.shape
            )
        }
        guard sequenceLength <= artifact.nCtx else {
            throw Text300MTransformerError.sequenceLengthExceedsContext(
                sequenceLength: sequenceLength, contextLength: artifact.nCtx
            )
        }
        guard sequenceLength <= positionEmbeddingRows else {
            throw Text300MTransformerError.sequenceLengthExceedsContext(
                sequenceLength: sequenceLength, contextLength: positionEmbeddingRows
            )
        }
        guard timesteps.shape == [batchSize] else {
            throw Text300MTransformerError.invalidTensorShape(
                name: timesteps.name, expected: "[\(batchSize)]", actual: timesteps.shape
            )
        }
        guard clipEmbeddings.shape == [batchSize, configuration.clipFeatureDim] else {
            throw Text300MTransformerError.invalidTensorShape(
                name: clipEmbeddings.name,
                expected: "[\(batchSize), \(configuration.clipFeatureDim)]",
                actual: clipEmbeddings.shape
            )
        }

        // --- Execute internally staying in MLXArray space ---
        let xMLX = try latent.asMLX().asType(.float32)
        let tMLX = try timesteps.asMLX().asType(.float32)
        let cMLX = try clipEmbeddings.asMLX().asType(.float32)

        let hidden = try predictInternal(x: xMLX, t: tMLX, embeddings: cMLX)

        // --- Convert result back to RuntimeTensor (One final synchronization) ---
        let outFloats = hidden.asArray(Float.self)
        let outData   = outFloats.withUnsafeBufferPointer { Data(buffer: $0) }
        return try RuntimeTensor(
            name: "text300m_output",
            shape: [batchSize, configuration.outputChannels, sequenceLength],
            dataType: .float32,
            data: outData
        )
    }

    /// Native MLXArray execution path for Text300M. Avoids bridging overhead.
    public func predictInternal(
        x xMLX: MLXArray,          // [B, C, T]
        t tMLX: MLXArray,          // [B]
        embeddings clipMLX: MLXArray // [B, clipDim]
    ) throws -> MLXArray {
        let batchSize      = xMLX.shape[0]
        let sequenceLength = xMLX.shape[2]

        // --- Time embedding ---
        // Python: t_embed = time_embed(timestep_embedding(t, d_model))
        let tsEmb  = TimestepEmbedding.compute(timesteps: tMLX, dimension: configuration.dModel) // [B, d]
        let tEmbed = timeEmbed(tsEmb)                                                              // [B, d]

        // --- CLIP conditioning ---
        // Python: clip_out = sqrt(clip_dim) * embeddings; clip_embed = clip_embed(clip_out)
        let clipScale = Float(sqrt(Double(configuration.clipFeatureDim)))
        let clipOut   = clipMLX * clipScale                                                        // [B, clipDim]
        let clipEmbed = MLX.matmul(clipOut, clipEmbeddingWeight.transposed()) + clipEmbeddingBias  // [B, d]

        // --- Transformer forward ---
        return forwardWithConditioning(
            latentNCT: xMLX,
            batchSize:      batchSize,
            sequenceLength: sequenceLength,
            conditions: [
                (clipEmbed, configuration.tokenCond),
                (tEmbed,    configuration.timeTokenCond),
            ]
        )
    }

    // MARK: Time embedding MLP (QuickGELU activation)

    /// Two-layer MLP that maps the sinusoidal timestep tokens into model-width space.
    /// Activation: QuickGELU: x * sigmoid(1.702 * x)
    private func timeEmbed(_ tsEmb: MLXArray) -> MLXArray {
        // Python: c_proj(gelu(c_fc(x)))
        let h = MLX.matmul(tsEmb, timeEmbeddingFCWeight.transposed()) + timeEmbeddingFCBias
        let act = Self.gelu(h)
        return MLX.matmul(act, timeEmbeddingProjectionWeight.transposed()) + timeEmbeddingProjectionBias
    }

    // MARK: Core forward (WrappedText300M._forward_with_cond)

    private func forwardWithConditioning(
        latentNCT: MLXArray,                  // [B, C, T]
        batchSize: Int,
        sequenceLength: Int,
        conditions: [(MLXArray, Bool)]        // (embedding [B, d], asToken)
    ) -> MLXArray {
        let d = configuration.dModel

        // NCT -> NTC, then project input channels to d_model
        // Python: h = input_proj(x_nct.transpose(0,2,1))  [B, T, d]
        var h = MLX.matmul(
            latentNCT.transposed(axes: [0, 2, 1]),     // [B, T, C]
            inputProjectionWeight.transposed()           // [C, d]
        ) + inputProjectionBias                          // [B, T, d]

        // Add non-token conditions broadcast over the sequence dimension
        // Python: for emb, as_token in cond_as_token: if not as_token: h += emb[:,None,:]
        for (emb, asToken) in conditions where !asToken {
            h = h + emb.expandedDimensions(axes: [1])   // broadcasts [B, 1, d]
        }

        // Add position embedding
        // Python: h += pos_emb[:T][None,:,:]
        if configuration.usePosEmb {
            h = h + positionEmbedding[0..<sequenceLength].expandedDimensions(axes: [0]) // [1, T, d]
        }

        // Prepend token-conditioning as extra leading tokens
        // Python: extra_tokens = [emb[:,None,:] for emb,as_token in ... if as_token]
        //         h = concat(extra_tokens + [h], axis=1)
        let extraTokenArrays = conditions.compactMap { (emb, asToken) -> MLXArray? in
            asToken ? emb.expandedDimensions(axes: [1]) : nil  // [B, 1, d]
        }
        if !extraTokenArrays.isEmpty {
            h = MLX.concatenated(extraTokenArrays + [h], axis: 1)  // [B, T + nExtra, d]
        }
        let totalSeq = h.shape[1]

        // Pre-norm
        h = Self.layerNorm(h, weight: layerNormPreWeight, bias: layerNormPreBias)

        // Transformer residual blocks
        let heads   = configuration.nHeads
        let headDim = d / heads
        let scale   = Float(1.0 / sqrt(sqrt(Double(headDim))))

        for layer in layers {
            // --- Self-attention ---
            let ln1 = Self.layerNorm(h, weight: layer.ln1Weight, bias: layer.ln1Bias)

            // QKV projection: [B, T, 3d]
            let qkv = MLX.matmul(ln1, layer.attentionQKVWeight.transposed()) + layer.attentionQKVBias
            // Reshape to [B, T, H, 3*headDim] then split Q, K, V
            let qkvR = qkv.reshaped(batchSize, totalSeq, heads, 3 * headDim)
            let q = qkvR[0..., 0..., 0..., 0..<headDim].transposed(axes: [0, 2, 1, 3])            // [B,H,T,hd]
            let k = qkvR[0..., 0..., 0..., headDim..<(2 * headDim)].transposed(axes: [0, 2, 1, 3])
            let v = qkvR[0..., 0..., 0..., (2 * headDim)...].transposed(axes: [0, 2, 1, 3])

            // Scaled dot-product attention
            let attnW = MLX.softmax(
                MLX.matmul(q * scale, (k * scale).transposed(axes: [0, 1, 3, 2])),
                axis: -1
            )                                                                                        // [B,H,T,T]
            let attnOut = MLX.matmul(attnW, v)                                                      // [B,H,T,hd]
                .transposed(axes: [0, 2, 1, 3])                                                     // [B,T,H,hd]
                .reshaped(batchSize, totalSeq, d)                                                    // [B,T,d]

            // Attention output projection + residual
            h = h + MLX.matmul(attnOut, layer.attentionOutputWeight.transposed()) + layer.attentionOutputBias

            // --- Feed-forward MLP ---
            let ln2      = Self.layerNorm(h, weight: layer.ln2Weight, bias: layer.ln2Bias)
            let mlpHidden = MLX.matmul(ln2, layer.mlpFCWeight.transposed()) + layer.mlpFCBias      // [B,T,hidden]
            // Standard GELU
            let activated = Self.gelu(mlpHidden)
            let mlpOut    = MLX.matmul(activated, layer.mlpProjectionWeight.transposed()) + layer.mlpProjectionBias

            h = h + mlpOut
        }

        // Post-norm
        h = Self.layerNorm(h, weight: layerNormPostWeight, bias: layerNormPostBias)

        // Strip the leading extra tokens that were prepended for conditioning
        // Python: if extra_tokens: h = h[:, n_extra:, :]
        if !extraTokenArrays.isEmpty {
            let nExtra = extraTokenArrays.reduce(0) { $0 + $1.shape[1] }
            h = h[0..., nExtra...]  // [B, T, d]
        }

        // Output projection: [B, T, d] -> [B, T, outputC]
        let projected = MLX.matmul(h, outputProjectionWeight.transposed()) + outputProjectionBias

        // NTC -> NCT
        return projected.transposed(axes: [0, 2, 1])   // [B, outputC, T]
    }

    // MARK: Layer normalisation (no MLXNN dependency)

    private static func layerNorm(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        epsilon: Float = 1e-5
    ) -> MLXArray {
        let mean  = MLX.mean(x, axes: [-1], keepDims: true)
        let vari  = MLX.mean(MLX.square(x - mean), axes: [-1], keepDims: true)
        let xNorm = (x - mean) * MLX.rsqrt(vari + epsilon)
        return xNorm * weight + bias
    }

    // MARK: Activation Functions
    
    private static func gelu(_ x: MLXArray) -> MLXArray {
        let half = MLXArray(Float(0.5))
        let one  = MLXArray(Float(1.0))
        let div  = MLXArray(Float(sqrt(2.0)))
        return x * half * (one + MLX.erf(x / div))
    }

    // MARK: Weight loading helpers

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

    private static func loadLayers(
        from artifact: Text300MArtifact,
        configuration: Configuration
    ) throws -> [TransformerLayer] {
        let d = configuration.dModel
        let h = configuration.mlpHiddenDim
        return try (0..<configuration.nLayers).map { i in
            TransformerLayer(
                ln1Weight: try artifact.requireTensor(
                    .resblock(i, .ln1Weight), expectedShape: [d], dataType: .float32
                ).asMLX(),
                ln1Bias: try artifact.requireTensor(
                    .resblock(i, .ln1Bias), expectedShape: [d], dataType: .float32
                ).asMLX(),
                attentionQKVWeight: try artifact.requireTensor(
                    .resblock(i, .attentionQKVWeight), expectedShape: [3 * d, d], dataType: .float32
                ).asMLX(),
                attentionQKVBias: try artifact.requireTensor(
                    .resblock(i, .attentionQKVBias), expectedShape: [3 * d], dataType: .float32
                ).asMLX(),
                attentionOutputWeight: try artifact.requireTensor(
                    .resblock(i, .attentionOutputWeight), expectedShape: [d, d], dataType: .float32
                ).asMLX(),
                attentionOutputBias: try artifact.requireTensor(
                    .resblock(i, .attentionOutputBias), expectedShape: [d], dataType: .float32
                ).asMLX(),
                ln2Weight: try artifact.requireTensor(
                    .resblock(i, .ln2Weight), expectedShape: [d], dataType: .float32
                ).asMLX(),
                ln2Bias: try artifact.requireTensor(
                    .resblock(i, .ln2Bias), expectedShape: [d], dataType: .float32
                ).asMLX(),
                mlpFCWeight: try artifact.requireTensor(
                    .resblock(i, .mlpFCWeight), expectedShape: [h, d], dataType: .float32
                ).asMLX(),
                mlpFCBias: try artifact.requireTensor(
                    .resblock(i, .mlpFCBias), expectedShape: [h], dataType: .float32
                ).asMLX(),
                mlpProjectionWeight: try artifact.requireTensor(
                    .resblock(i, .mlpOutputWeight), expectedShape: [d, h], dataType: .float32
                ).asMLX(),
                mlpProjectionBias: try artifact.requireTensor(
                    .resblock(i, .mlpOutputBias), expectedShape: [d], dataType: .float32
                ).asMLX()
            )
        }
    }
}
