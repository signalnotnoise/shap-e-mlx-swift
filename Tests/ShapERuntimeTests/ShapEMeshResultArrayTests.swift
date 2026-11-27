import Foundation
import XCTest
@testable import ShapERuntime

final class ShapEMeshResultArrayTests: XCTestCase {
    private func tensor<T>(_ values: [T], _ type: TensorDataType, shape: [Int]) throws -> RuntimeTensor {
        try RuntimeTensor(name: "t", shape: shape, dataType: type,
                          data: values.withUnsafeBytes { Data($0) })
    }

    private func result(faces: RuntimeTensor) throws -> ShapEMeshResult {
        try ShapEMeshResult(meshResult: MeshResult(
            vertices: tensor([Float(0), 0, 0, 1, 0, 0, 0, 1, 0], .float32, shape: [3, 3]),
            faces: faces,
            vertexChannels: ["RGBA": tensor([Float](repeating: 0.5, count: 12), .float32, shape: [3, 4])],
            bbox: BoundingBox(minCorner: [0, 0, 0], maxCorner: [1, 1, 0]),
            timings: [:],
            artifactInfo: ArtifactVersionInfo(artifactVersion: 0, manifestPath: "test")
        ))
    }

    func testScalarAccessorsRoundTrip() throws {
        let r = try result(faces: tensor([Int32(0), 1, 2], .int32, shape: [1, 3]))
        XCTAssertEqual(try r.verticesArray().count, 9)
        XCTAssertEqual(try r.facesArray(), [0, 1, 2])
        XCTAssertEqual(try r.colorsArray()?.count, 12)
    }

    func testInt64FacesConvertWhenInRange() throws {
        let r = try result(faces: tensor([Int64(0), 1, 2], .int64, shape: [1, 3]))
        XCTAssertEqual(try r.facesArray(), [0, 1, 2])
    }

    func testInt64FacesOutOfRangeThrowInsteadOfTrapping() throws {
        let r = try result(faces: tensor([Int64(0), 1, Int64(Int32.max) + 1], .int64, shape: [1, 3]))
        XCTAssertThrowsError(try r.facesArray())
    }
}
