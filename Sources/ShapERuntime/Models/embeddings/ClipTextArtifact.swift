import Foundation

public struct ClipTextArtifact: Sendable {
    public let contextLength: Int
    public let transformerWidth: Int
    public let transformerHeads: Int
    public let transformerLayers: Int
    public let state: [String: RuntimeTensor]

    public init(
        contextLength: Int,
        transformerWidth: Int,
        transformerHeads: Int,
        transformerLayers: Int,
        state: [String: RuntimeTensor]
    ) {
        self.contextLength = contextLength
        self.transformerWidth = transformerWidth
        self.transformerHeads = transformerHeads
        self.transformerLayers = transformerLayers
        self.state = state
    }
}

public extension ClipTextArtifact {
    struct TensorPath: Hashable, Sendable, ExpressibleByStringLiteral, CustomStringConvertible {
        public enum ResBlockComponent: String, CaseIterable, Sendable {
            case ln1Weight = "ln_1.weight"
            case ln1Bias = "ln_1.bias"
            case attentionInProjectionWeight = "attn.in_proj_weight"
            case attentionInProjectionBias = "attn.in_proj_bias"
            case attentionOutputWeight = "attn.out_proj.weight"
            case attentionOutputBias = "attn.out_proj.bias"
            case ln2Weight = "ln_2.weight"
            case ln2Bias = "ln_2.bias"
            case mlpFCWeight = "mlp.c_fc.weight"
            case mlpFCBias = "mlp.c_fc.bias"
            case mlpProjectionWeight = "mlp.c_proj.weight"
            case mlpProjectionBias = "mlp.c_proj.bias"
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

        public static let tokenEmbeddingWeight: Self = "token_embedding.weight"
        public static let positionalEmbedding: Self = "positional_embedding"
        public static let layerNormFinalWeight: Self = "ln_final.weight"
        public static let layerNormFinalBias: Self = "ln_final.bias"
        public static let textProjection: Self = "text_projection"

        public static func resblock(_ index: Int, _ component: ResBlockComponent) -> Self {
            Self("transformer.resblocks.\(index).\(component.rawValue)")
        }
    }

    static var clipTextFamilyName: String { "clip_text" }

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
            throw RuntimeArtifactError.missingTensor(family: Self.clipTextFamilyName, tensorName: path)
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
