import Foundation

public enum TensorDataType: String, Codable, Equatable, Sendable {
    case float32
    case int32
    case int64

    public var byteWidth: Int {
        switch self {
        case .float32, .int32:
            return 4
        case .int64:
            return 8
        }
    }
}

public struct RuntimeTensor: Equatable, Sendable, Codable {
    public let name: String
    public let shape: [Int]
    public let dataType: TensorDataType
    public let data: Data

    private enum CodingKeys: String, CodingKey {
        case name
        case shape
        case dataType
        case data
    }

    public init(name: String, shape: [Int], dataType: TensorDataType, data: Data) throws {
        guard shape.allSatisfy({ $0 >= 0 }) else {
            throw RuntimeArtifactError.invalidTensorShape(name: name, shape: shape)
        }

        self.name = name
        self.shape = shape
        self.dataType = dataType
        self.data = data

        let expectedByteCount = try Self.checkedElementCount(for: shape) * dataType.byteWidth
        guard data.count == expectedByteCount else {
            throw RuntimeArtifactError.invalidTensorByteCount(
                name: name,
                expected: expectedByteCount,
                actual: data.count
            )
        }
    }

    /// Decode through the throwing initializer so shape / byte-count invariants hold.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let name = try container.decode(String.self, forKey: .name)
        let shape = try container.decode([Int].self, forKey: .shape)
        let dataType = try container.decode(TensorDataType.self, forKey: .dataType)
        let data = try container.decode(Data.self, forKey: .data)
        try self.init(name: name, shape: shape, dataType: dataType, data: data)
    }

    public var elementCount: Int {
        (try? Self.checkedElementCount(for: shape)) ?? 0
    }

    public func float32Values() throws -> [Float] {
        guard dataType == .float32 else {
            throw RuntimeArtifactError.tensorTypeMismatch(
                name: name,
                expected: .float32,
                actual: dataType
            )
        }
        return copyScalars(Float.self)
    }

    public func int32Values() throws -> [Int32] {
        guard dataType == .int32 else {
            throw RuntimeArtifactError.tensorTypeMismatch(
                name: name,
                expected: .int32,
                actual: dataType
            )
        }
        return copyScalars(Int32.self)
    }

    public func int64Values() throws -> [Int64] {
        guard dataType == .int64 else {
            throw RuntimeArtifactError.tensorTypeMismatch(
                name: name,
                expected: .int64,
                actual: dataType
            )
        }
        return copyScalars(Int64.self)
    }

    private func copyScalars<T>(_ type: T.Type) -> [T] {
        var values = Array<T>(unsafeUninitializedCapacity: elementCount) { _, initializedCount in
            initializedCount = elementCount
        }
        _ = values.withUnsafeMutableBytes { destination in
            data.copyBytes(to: destination)
        }
        return values
    }

    private static func checkedElementCount(for shape: [Int]) throws -> Int {
        try shape.reduce(1) { partial, dimension in
            let next = partial.multipliedReportingOverflow(by: dimension)
            if next.overflow {
                throw RuntimeArtifactError.invalidTensorShape(name: "<shape>", shape: shape)
            }
            return next.partialValue
        }
    }
}
public extension RuntimeTensor {
    func zerosLike(name newName: String? = nil) throws -> RuntimeTensor {
        let zeroData = Data(count: elementCount * dataType.byteWidth)

        return try RuntimeTensor(
            name: newName ?? "\(name).zeros_like",
            shape: shape,
            dataType: dataType,
            data: zeroData
        )
    }
}
public enum TensorBlobLoader {
    public static func loadTensor(from bundleURL: URL, entry: TensorEntry, family: String) throws -> RuntimeTensor {
        guard let dataType = TensorDataType(rawValue: entry.dtype) else {
            throw RuntimeArtifactError.unsupportedTensorDataType(
                tensorName: entry.name,
                dtype: entry.dtype
            )
        }

        let tensorURL = bundleURL.appendingPathComponent(entry.path, isDirectory: false)
        guard FileManager.default.fileExists(atPath: tensorURL.path) else {
            throw RuntimeArtifactError.missingTensorBlob(
                family: family,
                tensorName: entry.name,
                url: tensorURL
            )
        }

        let data = try Data(contentsOf: tensorURL, options: [.mappedIfSafe])
        return try RuntimeTensor(
            name: entry.name,
            shape: entry.shape,
            dataType: dataType,
            data: data
        )
    }
}
