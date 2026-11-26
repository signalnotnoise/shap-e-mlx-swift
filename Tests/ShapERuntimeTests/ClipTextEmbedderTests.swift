import Foundation
import XCTest
@testable import ShapERuntime

final class ClipTextEmbedderTests: XCTestCase {
    private func float32Tensor(name: String, shape: [Int], values: [Float]) throws -> RuntimeTensor {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return try RuntimeTensor(name: name, shape: shape, dataType: .float32, data: data)
    }

    private func makeArtifact() throws -> ClipTextArtifact {
        let state: [String: RuntimeTensor] = [
            ClipTextArtifact.TensorPath.tokenEmbeddingWeight.rawValue: try float32Tensor(
                name: ClipTextArtifact.TensorPath.tokenEmbeddingWeight.rawValue,
                shape: [4, 2],
                values: [
                    0, 0,
                    1, 0,
                    2, -1,
                    3, 5,
                ]
            ),
            ClipTextArtifact.TensorPath.positionalEmbedding.rawValue: try float32Tensor(
                name: ClipTextArtifact.TensorPath.positionalEmbedding.rawValue,
                shape: [4, 2],
                values: [
                    0, 0,
                    1, 1,
                    1, 0,
                    0, 0,
                ]
            ),
            ClipTextArtifact.TensorPath.layerNormFinalWeight.rawValue: try float32Tensor(
                name: ClipTextArtifact.TensorPath.layerNormFinalWeight.rawValue,
                shape: [2],
                values: [1, 1]
            ),
            ClipTextArtifact.TensorPath.layerNormFinalBias.rawValue: try float32Tensor(
                name: ClipTextArtifact.TensorPath.layerNormFinalBias.rawValue,
                shape: [2],
                values: [0, 0]
            ),
            ClipTextArtifact.TensorPath.textProjection.rawValue: try float32Tensor(
                name: ClipTextArtifact.TensorPath.textProjection.rawValue,
                shape: [2, 2],
                values: [
                    1, 2,
                    3, 4,
                ]
            ),
        ]

        return ClipTextArtifact(
            contextLength: 4,
            transformerWidth: 2,
            transformerHeads: 1,
            transformerLayers: 0,
            state: state
        )
    }

    func testSelectPoolIndicesPrefersEndOfTextOverLargestTokenID() {
        let indices = ClipTextEmbedder.selectPoolIndices(
            tokenRows: [
                [1, 3, 2, 0],
                [1, 2, 3, 2],
            ],
            eotToken: 3
        )

        XCTAssertEqual(indices, [1, 2])
    }

    func testSelectPoolIndicesFallsBackToLastNonzeroWhenEndOfTextMissing() {
        let indices = ClipTextEmbedder.selectPoolIndices(
            tokenRows: [
                [1, 2, 0, 0],
                [0, 0, 7, 8],
            ],
            eotToken: 3
        )

        XCTAssertEqual(indices, [1, 3])
    }

    func testEmbedTokenIDsProjectsAndNormalizesPooledRepresentation() throws {
        let embedder = try ClipTextEmbedder(artifact: makeArtifact())

        let embedding = try embedder.embedTokenIDs(
            [
                [1, 2, 3, 0],
                [1, 2, 0, 0],
            ],
            endOfTextToken: 3
        )

        XCTAssertEqual(embedding.shape, [2, 2])

        let values = try embedding.float32Values()
        XCTAssertEqual(values.count, 4)
        XCTAssertEqual(values[0], 0.70710677, accuracy: 1e-5)
        XCTAssertEqual(values[1], 0.70710677, accuracy: 1e-5)
        XCTAssertEqual(values[2], -0.70710677, accuracy: 1e-5)
        XCTAssertEqual(values[3], -0.70710677, accuracy: 1e-5)
    }
}
