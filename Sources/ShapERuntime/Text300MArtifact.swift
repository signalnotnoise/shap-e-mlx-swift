import Foundation

public extension Text300MArtifact {
    struct TensorPath: Hashable, Sendable, ExpressibleByStringLiteral, CustomStringConvertible {
        public enum ResBlockComponent: String, CaseIterable, Sendable {
            case ln1Weight = "ln_1.weight"
            case ln1Bias = "ln_1.bias"
            case attentionQKVWeight = "attn.c_qkv.weight"
            case attentionQKVBias = "attn.c_qkv.bias"
            case attentionOutputWeight = "attn.c_proj.weight"
            case attentionOutputBias = "attn.c_proj.bias"
            case ln2Weight = "ln_2.weight"
            case ln2Bias = "ln_2.bias"
            case mlpFCWeight = "mlp.c_fc.weight"
            case mlpFCBias = "mlp.c_fc.bias"
            case mlpOutputWeight = "mlp.c_proj.weight"
            case mlpOutputBias = "mlp.c_proj.bias"
        }

        public let rawValue: String

        public init(_ rawValue: String) {
            self.rawValue = rawValue
        }

        public init(stringLiteral value: StringLiteralType) {
            self.rawValue = value
        }

        public var description: String {
            rawValue
        }

        public static let inputProjectionWeight: Self = "wrapped.input_proj.weight"
        public static let inputProjectionBias: Self = "wrapped.input_proj.bias"
        public static let clipEmbeddingWeight: Self = "wrapped.clip_embed.weight"
        public static let clipEmbeddingBias: Self = "wrapped.clip_embed.bias"
        public static let layerNormPreWeight: Self = "wrapped.ln_pre.weight"
        public static let layerNormPreBias: Self = "wrapped.ln_pre.bias"
        public static let layerNormPostWeight: Self = "wrapped.ln_post.weight"
        public static let layerNormPostBias: Self = "wrapped.ln_post.bias"
        public static let outputProjectionWeight: Self = "wrapped.output_proj.weight"
        public static let outputProjectionBias: Self = "wrapped.output_proj.bias"
        public static let positionEmbedding: Self = "wrapped.pos_emb"
        public static let timeEmbeddingFCWeight: Self = "wrapped.time_embed.c_fc.weight"
        public static let timeEmbeddingFCBias: Self = "wrapped.time_embed.c_fc.bias"
        public static let timeEmbeddingProjectionWeight: Self = "wrapped.time_embed.c_proj.weight"
        public static let timeEmbeddingProjectionBias: Self = "wrapped.time_embed.c_proj.bias"

        public static func resblock(_ index: Int, _ component: ResBlockComponent) -> Self {
            Self("wrapped.backbone.resblocks.layers.\(index).\(component.rawValue)")
        }
    }

    static var text300MFamilyName: String { "text300m" }

    func tensor(_ path: TensorPath) -> RuntimeTensor? {
        state[path.rawValue]
    }

    func tensor(named path: String) -> RuntimeTensor? {
        state[path]
    }

    func requireTensor(
        _ path: TensorPath,
        expectedShape: [Int]? = nil,
        dataType expectedType: TensorDataType? = nil
    ) throws -> RuntimeTensor {
        try requireTensor(named: path.rawValue, expectedShape: expectedShape, dataType: expectedType)
    }

    func requireTensor(
        named path: String,
        expectedShape: [Int]? = nil,
        dataType expectedType: TensorDataType? = nil
    ) throws -> RuntimeTensor {
        guard let tensor = state[path] else {
            throw RuntimeArtifactError.missingTensor(family: Self.text300MFamilyName, tensorName: path)
        }
        if let expectedShape, tensor.shape != expectedShape {
            throw RuntimeArtifactError.tensorShapeMismatch(
                name: path,
                expected: expectedShape,
                actual: tensor.shape
            )
        }
        if let expectedType, tensor.dataType != expectedType {
            throw RuntimeArtifactError.tensorTypeMismatch(
                name: path,
                expected: expectedType,
                actual: tensor.dataType
            )
        }
        return tensor
    }
}
