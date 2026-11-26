import XCTest
@testable import ShapERuntime

final class RuntimeArtifactBundleTests: XCTestCase {
    func testLoadsManifestAndArtifactInfo() throws {
        let bundleURL = RuntimeBundleFixture.bundleURL
        let bundle = try RuntimeArtifactBundle(bundleURL: bundleURL)

        XCTAssertEqual(bundle.manifest.artifactVersion, 1)
        XCTAssertEqual(
            Set(bundle.manifest.families.keys),
            Set(["clip_text", "decoder_params", "diffusion", "renderer", "text300m"])
        )
        XCTAssertEqual(try bundle.familyVersion("clip_text"), 1)
        XCTAssertEqual(bundle.artifactInfo.familyVersions["renderer"], 1)
        XCTAssertEqual(bundle.artifactInfo.manifestPath, bundleURL.appendingPathComponent("manifest.json").path)

        let clipFamily = try bundle.requireFamily("clip_text")
        XCTAssertEqual(clipFamily.metadata["context_length"]?.intValue, 77)
        XCTAssertEqual(clipFamily.metadata["transformer_width"]?.intValue, 8)
        XCTAssertEqual(
            clipFamily.metadata["tokenizer"]?.objectValue?["type"]?.stringValue,
            "clip_bpe"
        )
    }

    func testLoadsFloat32TensorBlobFromRendererFamily() throws {
        let bundle = try RuntimeArtifactBundle(bundleURL: RuntimeBundleFixture.bundleURL)
        let tensors = try bundle.loadFamilyTensors("renderer")
        let bboxMin = try XCTUnwrap(tensors["bbox_min"])
        let bboxMax = try XCTUnwrap(tensors["bbox_max"])

        XCTAssertEqual(bboxMin.shape, [3])
        XCTAssertEqual(bboxMin.dataType, .float32)
        XCTAssertEqual(try bboxMin.float32Values(), [-1.0, -1.0, -1.0])
        XCTAssertEqual(try bboxMax.float32Values(), [1.0, 1.0, 1.0])
    }

    func testLoadsTypedText300MArtifact() throws {
        let bundle = try RuntimeArtifactBundle(bundleURL: RuntimeBundleFixture.bundleURL)
        let artifact = try bundle.loadText300MArtifact()

        XCTAssertEqual(artifact.nCtx, 1024)
        XCTAssertEqual(artifact.dLatent, 1024)
        XCTAssertEqual(artifact.state["wrapped.output_proj.weight"]?.shape, [2, 2])
        XCTAssertEqual(
            try artifact.requireTensor(.outputProjectionWeight, expectedShape: [2, 2], dataType: .float32).shape,
            [2, 2]
        )
    }

    func testLoadsTypedDiffusionConfig() throws {
        let bundle = try RuntimeArtifactBundle(bundleURL: RuntimeBundleFixture.bundleURL)
        let config = try bundle.loadDiffusionConfig()

        XCTAssertEqual(config.numTimesteps, 2)
        XCTAssertEqual(config.modelMeanType, "epsilon")
        XCTAssertEqual(try config.alphasCumprod.float32Values(), [0.9, 0.8])
        XCTAssertEqual(try config.channelBiases?.float32Values(), [0.01, 0.02])
        XCTAssertEqual(try config.channelScales?.float32Values(), [1.5, 1.6])
    }

    func testLoadsTypedDecoderParamsArtifact() throws {
        let bundle = try RuntimeArtifactBundle(bundleURL: RuntimeBundleFixture.bundleURL)
        let artifact = try bundle.loadDecoderParamsArtifact()

        XCTAssertEqual(artifact.paramShapes, [DecoderParamShape(name: "nerstf.mlp.0.weight", shape: [2, 3])])
        XCTAssertEqual(artifact.latentCtx, 16)
        XCTAssertEqual(artifact.latentWarpName, "identity")
        XCTAssertEqual(artifact.latentWarpCoeff1, 1.0, accuracy: 1e-12)
        XCTAssertEqual(artifact.lnEps, 1e-5, accuracy: 1e-12)
        XCTAssertEqual(artifact.projections["nerstf.mlp.0.weight"]?.weightVCD.shape, [2, 3, 4])
        XCTAssertEqual(artifact.projections["nerstf.mlp.0.weight"]?.biasVC.shape, [2, 3])
    }

    func testLoadsTypedRendererArtifact() throws {
        let bundle = try RuntimeArtifactBundle(bundleURL: RuntimeBundleFixture.bundleURL)
        let artifact = try bundle.loadRendererArtifact()

        XCTAssertEqual(artifact.gridSize, 32)
        XCTAssertEqual(artifact.directionDependentShape, true)
        XCTAssertEqual(artifact.separateNerfChannels, true)
        XCTAssertEqual(artifact.separateCoarseChannels, false)
        XCTAssertEqual(artifact.channelActivation, "sigmoid")
        XCTAssertEqual(artifact.sdfActivation, "tanh")
        XCTAssertEqual(artifact.textureChannels, ["R", "G", "B"])
        XCTAssertEqual(artifact.bboxMin.shape, [3])
        XCTAssertEqual(artifact.staticBiases[0]?.shape, [2])
        XCTAssertEqual(artifact.staticWeights[4]?.shape, [2, 2])
    }

    func testMissingFamilyReportsAvailableFamilies() throws {
        let bundle = try RuntimeArtifactBundle(bundleURL: RuntimeBundleFixture.bundleURL)

        do {
            _ = try bundle.requireFamily("missing_family")
            XCTFail("Expected missing family error.")
        } catch let RuntimeArtifactError.familyNotFound(name, _, available) {
            XCTAssertEqual(name, "missing_family")
            XCTAssertTrue(available.contains("renderer"))
            XCTAssertTrue(available.contains("text300m"))
        }
    }
}
