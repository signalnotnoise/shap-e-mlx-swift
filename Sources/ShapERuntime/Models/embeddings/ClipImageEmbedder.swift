import Foundation
import MLX

public enum ClipImageEmbedderError: Error, Equatable, LocalizedError {
    case invalidTensorShape(name: String, expected: String, actual: [Int])
    case transformerWidthNotDivisible(width: Int, heads: Int)
    case missingTensor(name: String)

    public var errorDescription: String? {
        switch self {
        case let .invalidTensorShape(name, expected, actual):
            return "Tensor '\(name)' has shape \(actual) but expected \(expected)."
        case let .transformerWidthNotDivisible(width, heads):
            return "CLIP transformer width \(width) must be divisible by head count \(heads)."
        case let .missingTensor(name):
            return "Missing tensor '\(name)' in ClipVisionArtifact."
        }
    }
}

public final class ClipImageEmbedder: @unchecked Sendable {
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
    }

    public let artifact: ClipVisionArtifact

    private let classEmbedding: MLXArray
    private let positionalEmbedding: MLXArray
    private let conv1Weight: MLXArray
    private let layerNormPreWeight: MLXArray
    private let layerNormPreBias: MLXArray
    private let layerNormPostWeight: MLXArray
    private let layerNormPostBias: MLXArray
    private let proj: MLXArray
    private let layers: [TransformerLayer]

    public init(artifact: ClipVisionArtifact) throws {
        let width = artifact.transformerWidth
        guard width % artifact.transformerHeads == 0 else {
            throw ClipImageEmbedderError.transformerWidthNotDivisible(
                width: width,
                heads: artifact.transformerHeads
            )
        }

        self.artifact = artifact

        func requireMLX(_ name: String) throws -> MLXArray {
            guard let t = artifact.state[name] else {
                throw ClipImageEmbedderError.missingTensor(name: name)
            }
            return try t.asMLX().asType(.float32)
        }

        self.classEmbedding = try requireMLX("class_embedding")
        self.positionalEmbedding = try requireMLX("positional_embedding")
        self.conv1Weight = try requireMLX("conv1.weight")
        self.layerNormPreWeight = try requireMLX("ln_pre.weight")
        self.layerNormPreBias = try requireMLX("ln_pre.bias")
        self.layerNormPostWeight = try requireMLX("ln_post.weight")
        self.layerNormPostBias = try requireMLX("ln_post.bias")
        self.proj = try requireMLX("proj")

        var loadedLayers: [TransformerLayer] = []
        for i in 0..<artifact.transformerLayers {
            let prefix = "transformer.resblocks.\(i)"
            loadedLayers.append(TransformerLayer(
                ln1Weight: try requireMLX("\(prefix).ln_1.weight"),
                ln1Bias: try requireMLX("\(prefix).ln_1.bias"),
                attentionInProjectionWeight: try requireMLX("\(prefix).attn.in_proj_weight"),
                attentionInProjectionBias: try requireMLX("\(prefix).attn.in_proj_bias"),
                attentionOutputWeight: try requireMLX("\(prefix).attn.out_proj.weight"),
                attentionOutputBias: try requireMLX("\(prefix).attn.out_proj.bias"),
                ln2Weight: try requireMLX("\(prefix).ln_2.weight"),
                ln2Bias: try requireMLX("\(prefix).ln_2.bias"),
                mlpFCWeight: try requireMLX("\(prefix).mlp.c_fc.weight"),
                mlpFCBias: try requireMLX("\(prefix).mlp.c_fc.bias"),
                mlpProjectionWeight: try requireMLX("\(prefix).mlp.c_proj.weight"),
                mlpProjectionBias: try requireMLX("\(prefix).mlp.c_proj.bias")
            ))
        }
        self.layers = loadedLayers
    }

    /// Embeds a batch of images and extracts the grid hidden states for image300M conditioning.
    /// - Parameter images: Float32 MLXArray of shape [B, H, W, C] normalized to CLIP's expected input format.
    /// - Returns: A `RuntimeTensor` containing the grid embeddings of shape [B, 256, 1024].
    public func embedImagesGrid(_ images: MLXArray) throws -> RuntimeTensor {
        let batchSize = images.shape[0]
        let width = artifact.transformerWidth
        
        // CLIP expects [B, C, H, W] for Conv2D usually, but MLX's conv2d expects [B, H, W, C]
        // The original PyTorch conv1.weight has shape [width, 3, patch_size, patch_size].
        // We need to transpose it to [width, patch_size, patch_size, 3] and then
        // transpose again to [width, patch_size, patch_size, 3] ... Wait, MLX conv2d weight is
        // [O, H, W, I]. So torch's [O, I, H, W] -> [O, H, W, I].
        let convW = conv1Weight.transposed(axes: [0, 2, 3, 1]) // [width, patch_size, patch_size, 3]
        
        // Pass through conv2d with stride = patch_size
        var x = MLX.conv2d(images, convW, stride: [artifact.patchSize, artifact.patchSize]) // output: [B, grid_H, grid_W, width]
        
        let gridH = x.shape[1]
        let gridW = x.shape[2]
        
        // Reshape to [B, grid_H * grid_W, width]
        x = x.reshaped(batchSize, gridH * gridW, width)
        
        // Concat class embedding
        let cls = classEmbedding.reshaped(1, 1, width)
        let clsBroadcast = MLX.broadcast(cls, to: [batchSize, 1, width])
        x = MLX.concatenated([clsBroadcast, x], axis: 1) // [B, 1 + grid**2, width]
        
        // Add positional embedding
        x = x + positionalEmbedding.reshaped(1, x.shape[1], width)
        
        // Pre layer norm
        x = Self.layerNorm(x, weight: layerNormPreWeight, bias: layerNormPreBias)
        
        // Transformer Layers
        let sequenceLength = x.shape[1]
        let heads = artifact.transformerHeads
        let headDim = width / heads
        let scale = Float(1.0 / sqrt(Double(headDim)))
        
        for layer in layers {
            let ln1 = Self.layerNorm(x, weight: layer.ln1Weight, bias: layer.ln1Bias)
            
            // QKV projection
            let qkv = MLX.matmul(ln1, layer.attentionInProjectionWeight.transposed()) + layer.attentionInProjectionBias
            
            // Split Q, K, V linearly across the channel dimension
            let qSlice = qkv[0..., 0..., 0..<width]
            let kSlice = qkv[0..., 0..., width..<(2 * width)]
            let vSlice = qkv[0..., 0..., (2 * width)...]
            
            let q = qSlice.reshaped(batchSize, sequenceLength, heads, headDim).transposed(axes: [0, 2, 1, 3])
            let k = kSlice.reshaped(batchSize, sequenceLength, heads, headDim).transposed(axes: [0, 2, 1, 3])
            let v = vSlice.reshaped(batchSize, sequenceLength, heads, headDim).transposed(axes: [0, 2, 1, 3])
            
            // Scaled dot-product attention (no causal mask for Vision Transformer)
            let attnScores = MLX.matmul(q * scale, k.transposed(axes: [0, 1, 3, 2]))
            let attnW = MLX.softmax(attnScores, axis: -1)
            
            let attnOut = MLX.matmul(attnW, v)
                .transposed(axes: [0, 2, 1, 3])
                .reshaped(batchSize, sequenceLength, width)
                
            let projectedAttention = MLX.matmul(attnOut, layer.attentionOutputWeight.transposed()) + layer.attentionOutputBias
            x = x + projectedAttention
            
            // MLP
            let ln2 = Self.layerNorm(x, weight: layer.ln2Weight, bias: layer.ln2Bias)
            let mlpHidden = MLX.matmul(ln2, layer.mlpFCWeight.transposed()) + layer.mlpFCBias
            let mlpActivated = Self.quickGELU(mlpHidden)
            let mlpOutput = MLX.matmul(mlpActivated, layer.mlpProjectionWeight.transposed()) + layer.mlpProjectionBias
            x = x + mlpOutput
        }
        
        // Notice we do NOT apply the final layer norm and proj for embed_images_grid.
        // We just drop the first token (class token) and return the grid.
        let gridEmbeddings = x[0..., 1..., 0...] // [B, 256, 1024]
        
        MLX.eval(gridEmbeddings)
        
        let floats = gridEmbeddings.asArray(Float.self)
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        return try RuntimeTensor(
            name: "clip_vision_grid_embedding",
            shape: gridEmbeddings.shape,
            dataType: .float32,
            data: data
        )
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
}
