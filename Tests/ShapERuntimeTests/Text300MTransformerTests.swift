import Foundation
import MLX
import XCTest
@testable import ShapERuntime

final class Text300MTransformerTests: XCTestCase {

    /// Force MLX onto the CPU stream so `swift test` never tries to load the Metal shader library.
    /// In production the GPU path is enabled automatically; this guard is test-only.
    override class func setUp() {
        super.setUp()
        // Use the non-deprecated path: pin the default stream onto the CPU device.
        _ = MLX.Stream.defaultStream(.cpu)
    }

    private func float32Tensor(name: String, shape: [Int], values: [Float]) throws -> RuntimeTensor {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return try RuntimeTensor(name: name, shape: shape, dataType: .float32, data: data)
    }

    private func zeroValues(count: Int) -> [Float] {
        [Float](repeating: 0, count: count)
    }

    private func makeZeroLayerArtifact() throws -> Text300MArtifact {
        let state: [String: RuntimeTensor] = [
            Text300MArtifact.TensorPath.inputProjectionWeight.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.inputProjectionWeight.rawValue,
                shape: [2, 2],
                values: [1, 0, 0, 1]
            ),
            Text300MArtifact.TensorPath.inputProjectionBias.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.inputProjectionBias.rawValue,
                shape: [2],
                values: [0, 0]
            ),
            Text300MArtifact.TensorPath.clipEmbeddingWeight.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.clipEmbeddingWeight.rawValue,
                shape: [2, 2],
                values: zeroValues(count: 4)
            ),
            Text300MArtifact.TensorPath.clipEmbeddingBias.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.clipEmbeddingBias.rawValue,
                shape: [2],
                values: [0, 0]
            ),
            Text300MArtifact.TensorPath.layerNormPreWeight.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.layerNormPreWeight.rawValue,
                shape: [2],
                values: [1, 1]
            ),
            Text300MArtifact.TensorPath.layerNormPreBias.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.layerNormPreBias.rawValue,
                shape: [2],
                values: [0, 0]
            ),
            Text300MArtifact.TensorPath.layerNormPostWeight.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.layerNormPostWeight.rawValue,
                shape: [2],
                values: [1, 1]
            ),
            Text300MArtifact.TensorPath.layerNormPostBias.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.layerNormPostBias.rawValue,
                shape: [2],
                values: [0, 0]
            ),
            Text300MArtifact.TensorPath.outputProjectionWeight.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.outputProjectionWeight.rawValue,
                shape: [1, 2],
                values: [1, 0]
            ),
            Text300MArtifact.TensorPath.outputProjectionBias.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.outputProjectionBias.rawValue,
                shape: [1],
                values: [0]
            ),
            Text300MArtifact.TensorPath.positionEmbedding.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.positionEmbedding.rawValue,
                shape: [3, 2],
                values: zeroValues(count: 6)
            ),
            Text300MArtifact.TensorPath.timeEmbeddingFCWeight.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.timeEmbeddingFCWeight.rawValue,
                shape: [2, 2],
                values: zeroValues(count: 4)
            ),
            Text300MArtifact.TensorPath.timeEmbeddingFCBias.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.timeEmbeddingFCBias.rawValue,
                shape: [2],
                values: [0, 0]
            ),
            Text300MArtifact.TensorPath.timeEmbeddingProjectionWeight.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.timeEmbeddingProjectionWeight.rawValue,
                shape: [2, 2],
                values: zeroValues(count: 4)
            ),
            Text300MArtifact.TensorPath.timeEmbeddingProjectionBias.rawValue: try float32Tensor(
                name: Text300MArtifact.TensorPath.timeEmbeddingProjectionBias.rawValue,
                shape: [2],
                values: [0, 0]
            ),
        ]
        return Text300MArtifact(nCtx: 3, dLatent: 2, state: state)
    }

    private func makeSingleLayerZeroArtifact() throws -> Text300MArtifact {
        var artifact = try makeZeroLayerArtifact()
        var state = artifact.state
        let layerNames: [(Text300MArtifact.TensorPath.ResBlockComponent, [Int])] = [
            (.ln1Weight, [2]),
            (.ln1Bias, [2]),
            (.attentionQKVWeight, [6, 2]),
            (.attentionQKVBias, [6]),
            (.attentionOutputWeight, [2, 2]),
            (.attentionOutputBias, [2]),
            (.ln2Weight, [2]),
            (.ln2Bias, [2]),
            (.mlpFCWeight, [2, 2]),
            (.mlpFCBias, [2]),
            (.mlpOutputWeight, [2, 2]),
            (.mlpOutputBias, [2]),
        ]

        for (component, shape) in layerNames {
            state[Text300MArtifact.TensorPath.resblock(0, component).rawValue] = try float32Tensor(
                name: Text300MArtifact.TensorPath.resblock(0, component).rawValue,
                shape: shape,
                values: zeroValues(count: shape.reduce(1, *))
            )
        }

        artifact = Text300MArtifact(nCtx: artifact.nCtx, dLatent: artifact.dLatent, state: state)
        return artifact
    }

    func testTimestepEmbeddingMatchesReferenceValues() {
        // New API: timesteps is MLXArray; extract flat [Float] via asArray before asserting.
        let mlx = TimestepEmbedding.compute(
            timesteps: MLXArray([Float(0), Float(1)]),
            dimension: 5
        )
        let values = mlx.asArray(Float.self)    // triggers CPU-side lazy evaluation

        XCTAssertEqual(values.count, 10)
        XCTAssertEqual(values[0], 1,            accuracy: 1e-6)
        XCTAssertEqual(values[1], 1,            accuracy: 1e-6)
        XCTAssertEqual(values[2], 0,            accuracy: 1e-6)
        XCTAssertEqual(values[3], 0,            accuracy: 1e-6)
        XCTAssertEqual(values[4], 0,            accuracy: 1e-6)
        XCTAssertEqual(values[5], 0.5403023,   accuracy: 1e-6)
        XCTAssertEqual(values[6], 0.99995,     accuracy: 1e-6)
        XCTAssertEqual(values[7], 0.84147096,  accuracy: 1e-6)
        XCTAssertEqual(values[8], 0.009999833, accuracy: 1e-6)
        XCTAssertEqual(values[9], 0,            accuracy: 1e-6)
    }

    func testForwardWithoutTransformerLayersProjectsExpectedChannel() throws {
        let transformer = try Text300MTransformer(
            artifact: makeZeroLayerArtifact(),
            configuration: .init(
                dModel: 2,
                nLayers: 0,
                nHeads: 1,
                mlpHiddenDim: 2,
                inputChannels: 2,
                outputChannels: 1,
                clipFeatureDim: 2,
                tokenCond: false,
                timeTokenCond: false,
                usePosEmb: true
            )
        )

        let latent = try float32Tensor(
            name: "latent",
            shape: [1, 2, 3],
            values: [1, 1, 1, -1, -1, -1]
        )
        let timesteps = try float32Tensor(name: "timesteps", shape: [1], values: [0])
        let embeddings = try float32Tensor(name: "embeddings", shape: [1, 2], values: [0, 0])

        let output = try transformer.predict(x: latent, t: timesteps, embeddings: embeddings)
        let values = try output.float32Values()

        XCTAssertEqual(output.shape, [1, 1, 3])
        for value in values {
            XCTAssertEqual(value, 0.999995, accuracy: 1e-5)
        }
    }

    func testForwardWithTokenConditioningAndZeroWeightsPreservesResidualPath() throws {
        let transformer = try Text300MTransformer(
            artifact: makeSingleLayerZeroArtifact(),
            configuration: .init(
                dModel: 2,
                nLayers: 1,
                nHeads: 1,
                mlpHiddenDim: 2,
                inputChannels: 2,
                outputChannels: 1,
                clipFeatureDim: 2,
                tokenCond: true,
                timeTokenCond: true,
                usePosEmb: true
            )
        )

        let latent = try float32Tensor(
            name: "latent",
            shape: [1, 2, 2],
            values: [5, 7, 11, 13]
        )
        let timesteps = try float32Tensor(name: "timesteps", shape: [1], values: [3])
        let embeddings = try float32Tensor(name: "embeddings", shape: [1, 2], values: [2, 4])

        let output = try transformer.predict(x: latent, t: timesteps, embeddings: embeddings)
        let values = try output.float32Values()

        XCTAssertEqual(output.shape, [1, 1, 2])
        XCTAssertEqual(values[0], -0.999995, accuracy: 1e-5)
        XCTAssertEqual(values[1], -0.999995, accuracy: 1e-5)
    }
}
