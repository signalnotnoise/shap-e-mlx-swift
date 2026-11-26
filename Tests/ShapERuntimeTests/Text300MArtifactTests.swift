import Foundation
import XCTest
@testable import ShapERuntime

final class Text300MArtifactTests: XCTestCase {
    private func float32Tensor(name: String, shape: [Int], values: [Float]) throws -> RuntimeTensor {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return try RuntimeTensor(name: name, shape: shape, dataType: .float32, data: data)
    }

    private func makeArtifact() throws -> Text300MArtifact {
        let state: [String: RuntimeTensor] = [
            Text300MArtifact.TensorPath.inputProjectionWeight.rawValue: try float32Tensor(
                name: "wrapped.input_proj.weight",
                shape: [2, 2],
                values: [1, 2, 3, 4]
            ),
            Text300MArtifact.TensorPath.resblock(2, .attentionQKVWeight).rawValue: try float32Tensor(
                name: "wrapped.backbone.resblocks.layers.2.attn.c_qkv.weight",
                shape: [3, 4],
                values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            ),
        ]
        return Text300MArtifact(nCtx: 1024, dLatent: 1024, state: state)
    }

    func testTensorPathBuildersMatchExpectedModelPaths() {
        XCTAssertEqual(Text300MArtifact.TensorPath.inputProjectionWeight.rawValue, "wrapped.input_proj.weight")
        XCTAssertEqual(Text300MArtifact.TensorPath.outputProjectionBias.rawValue, "wrapped.output_proj.bias")
        XCTAssertEqual(
            Text300MArtifact.TensorPath.resblock(2, .attentionQKVWeight).rawValue,
            "wrapped.backbone.resblocks.layers.2.attn.c_qkv.weight"
        )
        XCTAssertEqual(
            Text300MArtifact.TensorPath.resblock(7, .mlpOutputBias).rawValue,
            "wrapped.backbone.resblocks.layers.7.mlp.c_proj.bias"
        )
    }

    func testRequireTensorValidatesLookupShapeAndType() throws {
        let artifact = try makeArtifact()
        let tensor = try artifact.requireTensor(
            .inputProjectionWeight,
            expectedShape: [2, 2],
            dataType: .float32
        )

        XCTAssertEqual(tensor.shape, [2, 2])
        XCTAssertEqual(tensor.dataType, .float32)
        XCTAssertNotNil(artifact.tensor(.resblock(2, .attentionQKVWeight)))
    }

    func testRequireTensorReportsMissingAndShapeMismatch() throws {
        let artifact = try makeArtifact()

        XCTAssertThrowsError(
            try artifact.requireTensor(named: "wrapped.missing.weight")
        ) { error in
            guard case let RuntimeArtifactError.missingTensor(family, name) = error else {
                XCTFail("Expected missingTensor error, got \(error)")
                return
            }
            XCTAssertEqual(family, "text300m")
            XCTAssertEqual(name, "wrapped.missing.weight")
        }

        XCTAssertThrowsError(
            try artifact.requireTensor(
                .inputProjectionWeight,
                expectedShape: [4, 4],
                dataType: .float32
            )
        ) { error in
            guard case let RuntimeArtifactError.tensorShapeMismatch(name, expected, actual) = error else {
                XCTFail("Expected tensorShapeMismatch error, got \(error)")
                return
            }
            XCTAssertEqual(name, "wrapped.input_proj.weight")
            XCTAssertEqual(expected, [4, 4])
            XCTAssertEqual(actual, [2, 2])
        }
    }
}
