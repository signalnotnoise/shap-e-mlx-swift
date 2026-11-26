import Foundation
import MLX

public enum ClipTextEmbedderError: Error, Equatable, LocalizedError {
    case tokenizerUnavailable
    case inconsistentTokenRowLength(expected: Int, actual: Int)
    case sequenceLengthExceedsContextLength(sequenceLength: Int, contextLength: Int)
    case tokenIDOutOfRange(tokenID: Int, vocabularySize: Int)
    case invalidTensorShape(name: String, expected: String, actual: [Int])
    case transformerWidthNotDivisible(width: Int, heads: Int)

    public var errorDescription: String? {
        switch self {
        case .tokenizerUnavailable:
            return "ClipTextEmbedder was constructed without a tokenizer, so prompt strings cannot be embedded directly."
        case let .inconsistentTokenRowLength(expected, actual):
            return "All CLIP token rows must share a common length. Expected \(expected) tokens but found \(actual)."
        case let .sequenceLengthExceedsContextLength(sequenceLength, contextLength):
            return "CLIP token sequence length \(sequenceLength) exceeds artifact context length \(contextLength)."
        case let .tokenIDOutOfRange(tokenID, vocabularySize):
            return "CLIP token id \(tokenID) is out of range for vocabulary size \(vocabularySize)."
        case let .invalidTensorShape(name, expected, actual):
            return "Tensor '\(name)' has shape \(actual) but expected \(expected)."
        case let .transformerWidthNotDivisible(width, heads):
            return "CLIP transformer width \(width) must be divisible by head count \(heads)."
        }
    }
}

public final class ClipTextEmbedder: @unchecked Sendable {
    private struct TransformerLayer: @unchecked Sendable {
        let ln1Weight: MLXArray
        let ln1Bias: MLXArray
        let attentionInProjectionWeight: MLXArray
        let attentionInProjectionBias: MLXArray
        let attentionOutputWeight: MLXArray
        let attentionOutputBias: MLXArray
        let ln2Weight: MLXArray
        let ln2Bias: MLXArray
        let mlpFCWeight: MLXArray
        let mlpFCBias: MLXArray
        let mlpProjectionWeight: MLXArray
        let mlpProjectionBias: MLXArray
        let mlpHiddenDim: Int
    }

    public let artifact: ClipTextArtifact
    public let tokenizer: ClipSimpleTokenizer?

    private let tokenEmbeddingWeight: MLXArray
    private let positionalEmbedding: MLXArray
    private let layerNormFinalWeight: MLXArray
    private let layerNormFinalBias: MLXArray
    private let textProjection: MLXArray
    private let textProjectionDim: Int
    private let layers: [TransformerLayer]

    public init(artifact: ClipTextArtifact, tokenizer: ClipSimpleTokenizer? = nil) throws {
        let width = artifact.transformerWidth
        guard width % artifact.transformerHeads == 0 else {
            throw ClipTextEmbedderError.transformerWidthNotDivisible(
                width: width,
                heads: artifact.transformerHeads
            )
        }

        self.artifact = artifact
        self.tokenizer = tokenizer

        let tokenEmbeddingTensor = try artifact.requireTensor(.tokenEmbeddingWeight, dataType: .float32)
        guard tokenEmbeddingTensor.shape.count == 2, tokenEmbeddingTensor.shape[1] == width else {
            throw ClipTextEmbedderError.invalidTensorShape(
                name: tokenEmbeddingTensor.name,
                expected: "[vocab, \(width)]",
                actual: tokenEmbeddingTensor.shape
            )
        }
        self.tokenEmbeddingWeight = try tokenEmbeddingTensor.asMLX()

        let positionalEmbeddingTensor = try artifact.requireTensor(
            .positionalEmbedding,
            expectedShape: [artifact.contextLength, width],
            dataType: .float32
        )
        self.positionalEmbedding = try positionalEmbeddingTensor.asMLX()

        self.layerNormFinalWeight = try artifact.requireTensor(
            .layerNormFinalWeight,
            expectedShape: [width],
            dataType: .float32
        ).asMLX()
        self.layerNormFinalBias = try artifact.requireTensor(
            .layerNormFinalBias,
            expectedShape: [width],
            dataType: .float32
        ).asMLX()

        let textProjectionTensor = try artifact.requireTensor(.textProjection, dataType: .float32)
        guard textProjectionTensor.shape.count == 2, textProjectionTensor.shape[0] == width else {
            throw ClipTextEmbedderError.invalidTensorShape(
                name: textProjectionTensor.name,
                expected: "[\(width), output_dim]",
                actual: textProjectionTensor.shape
            )
        }
        self.textProjection = try textProjectionTensor.asMLX()
        self.textProjectionDim = textProjectionTensor.shape[1]

        self.layers = try Self.loadLayers(from: artifact, width: width)
    }

    public convenience init(artifact: ClipTextArtifact, vocabularyURL: URL) throws {
        try self.init(artifact: artifact, tokenizer: ClipSimpleTokenizer(vocabularyURL: vocabularyURL))
    }

    public func embedText(_ prompts: [String]) throws -> RuntimeTensor {
        guard let tokenizer else {
            throw ClipTextEmbedderError.tokenizerUnavailable
        }
        let tokenRows = try tokenizer.tokenize(prompts, contextLength: artifact.contextLength)
        return try embedTokenIDs(tokenRows, endOfTextToken: Int32(tokenizer.endOfTextToken))
    }

    public func embedTokenIDs(_ tokenRows: [[Int32]], endOfTextToken: Int32) throws -> RuntimeTensor {
        if tokenRows.isEmpty {
            return try Self.makeFloatTensor(
                name: "clip_text_embedding",
                shape: [0, textProjectionDim],
                values: []
            )
        }

        let batchSize = tokenRows.count
        let sequenceLength = tokenRows[0].count
        guard sequenceLength <= artifact.contextLength else {
            throw ClipTextEmbedderError.sequenceLengthExceedsContextLength(
                sequenceLength: sequenceLength,
                contextLength: artifact.contextLength
            )
        }

        for row in tokenRows.dropFirst() where row.count != sequenceLength {
            throw ClipTextEmbedderError.inconsistentTokenRowLength(
                expected: sequenceLength,
                actual: row.count
            )
        }

        let width = artifact.transformerWidth
        let vocabularySize = tokenEmbeddingWeight.shape[0]

        // Validate token IDs
        for row in tokenRows {
            for tokenID in row {
                guard Int(tokenID) >= 0, Int(tokenID) < vocabularySize else {
                    throw ClipTextEmbedderError.tokenIDOutOfRange(
                        tokenID: Int(tokenID),
                        vocabularySize: vocabularySize
                    )
                }
            }
        }

        let flatTokens = tokenRows.flatMap { $0 }
        let tokensMLX = MLXArray(flatTokens, [batchSize, sequenceLength])
        
        // 1. Embed tokens
        var hidden = tokenEmbeddingWeight[tokensMLX]
        
        // 2. Add position embeddings
        hidden = hidden + positionalEmbedding[0..<sequenceLength].expandedDimensions(axes: [0])
        
        let heads = artifact.transformerHeads
        let headDim = width / heads
        let scale = Float(1.0 / sqrt(Double(headDim)))
        
        // Causal mask: [1, 1, T, T]
        var maskData = [Float](repeating: 0, count: sequenceLength * sequenceLength)
        for i in 0..<sequenceLength {
            for j in 0..<sequenceLength {
                if j > i {
                    maskData[i * sequenceLength + j] = -1e9
                }
            }
        }
        let causalMask = MLXArray(maskData, [1, 1, sequenceLength, sequenceLength])

        // 3. Transformer Layers
        for layer in layers {
            let ln1 = Self.layerNorm(hidden, weight: layer.ln1Weight, bias: layer.ln1Bias)
            
            // QKV projection
            let qkv = MLX.matmul(ln1, layer.attentionInProjectionWeight.transposed()) + layer.attentionInProjectionBias
            
            // Split Q, K, V linearly across the channel dimension using slicing
            let qSlice = qkv[0..., 0..., 0..<width]
            let kSlice = qkv[0..., 0..., width..<(2 * width)]
            let vSlice = qkv[0..., 0..., (2 * width)...]
            
            let q = qSlice.reshaped(batchSize, sequenceLength, heads, headDim).transposed(axes: [0, 2, 1, 3])
            let k = kSlice.reshaped(batchSize, sequenceLength, heads, headDim).transposed(axes: [0, 2, 1, 3])
            let v = vSlice.reshaped(batchSize, sequenceLength, heads, headDim).transposed(axes: [0, 2, 1, 3])
            
            // Scaled dot-product attention with causal mask
            let attnScores = MLX.matmul(q * scale, k.transposed(axes: [0, 1, 3, 2]))
            let attnW = MLX.softmax(attnScores + causalMask, axis: -1)
            
            let attnOut = MLX.matmul(attnW, v)
                .transposed(axes: [0, 2, 1, 3])
                .reshaped(batchSize, sequenceLength, width)
                
            let projectedAttention = MLX.matmul(attnOut, layer.attentionOutputWeight.transposed()) + layer.attentionOutputBias
            hidden = hidden + projectedAttention
            
            // MLP
            let ln2 = Self.layerNorm(hidden, weight: layer.ln2Weight, bias: layer.ln2Bias)
            let mlpHidden = MLX.matmul(ln2, layer.mlpFCWeight.transposed()) + layer.mlpFCBias
            let mlpActivated = Self.quickGELU(mlpHidden)
            let mlpOutput = MLX.matmul(mlpActivated, layer.mlpProjectionWeight.transposed()) + layer.mlpProjectionBias
            hidden = hidden + mlpOutput
        }

        // Final LayerNorm
        hidden = Self.layerNorm(hidden, weight: layerNormFinalWeight, bias: layerNormFinalBias)
        
        // Pool based on EOT token indices
        let poolIndices = Self.selectPoolIndices(tokenRows: tokenRows, eotToken: endOfTextToken)
        
        var pooledRows: [MLXArray] = []
        for b in 0..<batchSize {
            let t = poolIndices[b]
            let row = hidden[b, t]
            pooledRows.append(row.expandedDimensions(axes: [0]))
        }
        let pooled = MLX.concatenated(pooledRows, axis: 0)
        
        // Project to text projection dim
        var embeddings = MLX.matmul(pooled, textProjection)
        
        // Normalize rows (L2 norm)
        let norm = MLX.sqrt(MLX.sum(MLX.square(embeddings), axes: [-1], keepDims: true))
        embeddings = embeddings / MLX.maximum(norm, MLXArray(Float(1e-12)))
        
        // Evaluate the MLX graph to force computation
        MLX.eval(embeddings)
        
        return try Self.makeFloatTensor(
            name: "clip_text_embedding",
            shape: [batchSize, textProjectionDim],
            values: embeddings.asArray(Float.self)
        )
    }

    public static func selectPoolIndices(tokenRows: [[Int32]], eotToken: Int32) -> [Int] {
        tokenRows.map { row in
            if let lastEOT = row.lastIndex(of: eotToken) {
                return lastEOT
            }
            if let lastNonZero = row.lastIndex(where: { $0 != 0 }) {
                return lastNonZero
            }
            return max(row.count - 1, 0)
        }
    }

    private static func loadLayers(from artifact: ClipTextArtifact, width: Int) throws -> [TransformerLayer] {
        try (0..<artifact.transformerLayers).map { index in
            let prefix = { (component: ClipTextArtifact.TensorPath.ResBlockComponent) in
                ClipTextArtifact.TensorPath.resblock(index, component)
            }

            return TransformerLayer(
                ln1Weight: try artifact.requireTensor(prefix(.ln1Weight), expectedShape: [width], dataType: .float32).asMLX(),
                ln1Bias: try artifact.requireTensor(prefix(.ln1Bias), expectedShape: [width], dataType: .float32).asMLX(),
                attentionInProjectionWeight: try artifact.requireTensor(prefix(.attentionInProjectionWeight), expectedShape: [3 * width, width], dataType: .float32).asMLX(),
                attentionInProjectionBias: try artifact.requireTensor(prefix(.attentionInProjectionBias), expectedShape: [3 * width], dataType: .float32).asMLX(),
                attentionOutputWeight: try artifact.requireTensor(prefix(.attentionOutputWeight), expectedShape: [width, width], dataType: .float32).asMLX(),
                attentionOutputBias: try artifact.requireTensor(prefix(.attentionOutputBias), expectedShape: [width], dataType: .float32).asMLX(),
                ln2Weight: try artifact.requireTensor(prefix(.ln2Weight), expectedShape: [width], dataType: .float32).asMLX(),
                ln2Bias: try artifact.requireTensor(prefix(.ln2Bias), expectedShape: [width], dataType: .float32).asMLX(),
                mlpFCWeight: try artifact.requireTensor(prefix(.mlpFCWeight), dataType: .float32).asMLX(),
                mlpFCBias: try artifact.requireTensor(prefix(.mlpFCBias), dataType: .float32).asMLX(),
                mlpProjectionWeight: try artifact.requireTensor(prefix(.mlpProjectionWeight), dataType: .float32).asMLX(),
                mlpProjectionBias: try artifact.requireTensor(prefix(.mlpProjectionBias), expectedShape: [width], dataType: .float32).asMLX(),
                mlpHiddenDim: try artifact.requireTensor(prefix(.mlpFCWeight), dataType: .float32).shape[0]
            )
        }
    }

    private static func layerNorm(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        epsilon: Float = 1e-5
    ) -> MLXArray {
        let mean = MLX.mean(x, axes: [-1], keepDims: true)
        let vari = MLX.mean(MLX.square(x - mean), axes: [-1], keepDims: true)
        let xNorm = (x - mean) * MLX.rsqrt(vari + epsilon)
        return xNorm * weight + bias
    }

    private static func quickGELU(_ x: MLXArray) -> MLXArray {
        return x * MLX.sigmoid(x * 1.702)
    }

    private static func makeFloatTensor(name: String, shape: [Int], values: [Float]) throws -> RuntimeTensor {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return try RuntimeTensor(name: name, shape: shape, dataType: .float32, data: data)
    }
}
