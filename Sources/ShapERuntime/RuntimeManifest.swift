import Foundation

public enum JSONValue: Codable, Equatable, Sendable {
    case string(String)
    case integer(Int)
    case double(Double)
    case bool(Bool)
    case array([JSONValue])
    case object([String: JSONValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .integer(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported JSON value in runtime manifest."
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .string(value):
            try container.encode(value)
        case let .integer(value):
            try container.encode(value)
        case let .double(value):
            try container.encode(value)
        case let .bool(value):
            try container.encode(value)
        case let .array(value):
            try container.encode(value)
        case let .object(value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }

    public var stringValue: String? {
        guard case let .string(value) = self else {
            return nil
        }
        return value
    }

    public var intValue: Int? {
        switch self {
        case let .integer(value):
            return value
        case let .double(value) where value.rounded() == value:
            return Int(value)
        default:
            return nil
        }
    }

    public var doubleValue: Double? {
        switch self {
        case let .double(value):
            return value
        case let .integer(value):
            return Double(value)
        default:
            return nil
        }
    }

    public var boolValue: Bool? {
        guard case let .bool(value) = self else {
            return nil
        }
        return value
    }

    public var arrayValue: [JSONValue]? {
        guard case let .array(value) = self else {
            return nil
        }
        return value
    }

    public var objectValue: [String: JSONValue]? {
        guard case let .object(value) = self else {
            return nil
        }
        return value
    }
}

public struct TensorEntry: Codable, Equatable, Sendable {
    public let name: String
    public let path: String
    public let shape: [Int]
    public let dtype: String

    public init(name: String, path: String, shape: [Int], dtype: String) {
        self.name = name
        self.path = path
        self.shape = shape
        self.dtype = dtype
    }
}

public struct ArtifactFamilyManifest: Codable, Equatable, Sendable {
    public let version: Int
    public let metadata: [String: JSONValue]
    public let tensors: [String: TensorEntry]

    public init(
        version: Int,
        metadata: [String: JSONValue] = [:],
        tensors: [String: TensorEntry] = [:]
    ) {
        self.version = version
        self.metadata = metadata
        self.tensors = tensors
    }
}

public struct RuntimeArtifactManifest: Codable, Equatable, Sendable {
    public static let currentVersion = 1

    public let artifactVersion: Int
    public let families: [String: ArtifactFamilyManifest]

    enum CodingKeys: String, CodingKey {
        case artifactVersion = "artifact_version"
        case families
    }

    public init(artifactVersion: Int, families: [String: ArtifactFamilyManifest]) {
        self.artifactVersion = artifactVersion
        self.families = families
    }
}
