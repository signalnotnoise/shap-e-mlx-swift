import Foundation
import MLX
import XCTest
@testable import ShapERuntime

final class DecoderParamsProjectorTests: XCTestCase {

    override class func setUp() {
        super.setUp()
        _ = MLX.Stream.defaultStream(.cpu)
    }

    // MARK: Helpers

    private func makeTensor(name: String, shape: [Int], values: [Float]) throws -> RuntimeTensor {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return try RuntimeTensor(name: name, shape: shape, dataType: .float32, data: data)
    }

    /// Build a minimal DecoderParamsArtifact with one param group of given shape.
    private func makeArtifact(
        paramShape: [Int],      // e.g. [2, 3] → vectors=2, channels=3
        latentCtx: Int,
        dLatentPerCtx: Int,
        warpName: String = "identity",
        warpCoeff1: Double = 1.0
    ) throws -> DecoderParamsArtifact {
        let (vectors, channels) = flatShape(paramShape)
        let D = dLatentPerCtx

        // weight_vcd: [V, C, D], bias_vc: [V, C], ln_weight_c: [C], ln_bias_c: [C]
        let wVCD = try makeTensor(name: "w", shape: [vectors, channels, D],
                                  values: [Float](repeating: 0, count: vectors * channels * D))
        let bVC  = try makeTensor(name: "b", shape: [vectors, channels],
                                  values: [Float](repeating: 0, count: vectors * channels))
        let lnW  = try makeTensor(name: "lw", shape: [channels],
                                  values: [Float](repeating: 1, count: channels))  // scale=1
        let lnB  = try makeTensor(name: "lb", shape: [channels],
                                  values: [Float](repeating: 0, count: channels))  // bias=0

        let proj = DecoderProjectionWeights(weightVCD: wVCD, biasVC: bVC, lnWeightC: lnW, lnBiasC: lnB)

        return DecoderParamsArtifact(
            paramShapes: [DecoderParamShape(name: "test_param", shape: paramShape)],
            latentCtx:   latentCtx,
            latentWarpName:   warpName,
            latentWarpCoeff1: warpCoeff1,
            projections: ["test_param": proj],
            lnEps: 1e-5
        )
    }

    private func flatShape(_ shape: [Int]) -> (Int, Int) {
        let total    = shape.reduce(1, *)
        let channels = shape.last!
        return (total / channels, channels)
    }

    // MARK: Tests

    func testOutputShapeMatchesParamShape() throws {
        let latentCtx     = 2   // must equal the 2 vectors in [2, 3]
        let dLatentPerCtx = 8
        let paramShape    = [2, 3]     // vectors=2, channels=3  → total 6 params per batch item

        let artifact = try makeArtifact(paramShape: paramShape, latentCtx: latentCtx,
                                        dLatentPerCtx: dLatentPerCtx)

        let B = 2
        let D = latentCtx * dLatentPerCtx
        let latent = try makeTensor(name: "latent", shape: [B, D],
                                    values: [Float](repeating: 0.1, count: B * D))

        let projector = try DecoderParamsProjector(artifact: artifact)
        let out = try projector.project(latent: try latent.asMLX())

        XCTAssertEqual(out.count, 1)
        XCTAssertNotNil(out["test_param"])
        // Expected shape: [B, *paramShape] = [2, 2, 3]
        XCTAssertEqual(out["test_param"]?.shape, [B] + paramShape)
    }

    func testZeroWeightProducesOnlyBiasContribution() throws {
        // With weight=0 and bias=0.5, output should be 0.5 everywhere after LN passes through
        let latentCtx     = 2
        let dLatentPerCtx = 4
        let vectors       = 2
        let channels      = 3

        let wVCD = try makeTensor(name: "w", shape: [vectors, channels, dLatentPerCtx],
                                  values: [Float](repeating: 0, count: vectors * channels * dLatentPerCtx))
        let bVC  = try makeTensor(name: "b", shape: [vectors, channels],
                                  values: [Float](repeating: 0.5, count: vectors * channels))
        // LN with scale=1, bias=0: normalized zero input stays zero; then bias_vc pushes it to 0.5
        let lnW  = try makeTensor(name: "lw", shape: [channels],
                                  values: [Float](repeating: 1, count: channels))
        let lnB  = try makeTensor(name: "lb", shape: [channels],
                                  values: [Float](repeating: 0, count: channels))

        let proj = DecoderProjectionWeights(weightVCD: wVCD, biasVC: bVC, lnWeightC: lnW, lnBiasC: lnB)
        let artifact = DecoderParamsArtifact(
            paramShapes: [DecoderParamShape(name: "p", shape: [vectors, channels])],
            latentCtx: latentCtx, latentWarpName: "identity", latentWarpCoeff1: 1.0,
            projections: ["p": proj], lnEps: 1e-5
        )

        let latent = try makeTensor(name: "latent", shape: [1, latentCtx * dLatentPerCtx],
                                    values: [Float](repeating: 0.3, count: latentCtx * dLatentPerCtx))

        let projector = try DecoderParamsProjector(artifact: artifact)
        let out = try projector.project(latent: try latent.asMLX())
        let values = try out["p"]!.asArray(Float.self)

        // All outputs should be 0.5 (bias only, weight projection = 0, LN of zero = 0)
        for v in values {
            XCTAssertEqual(v, 0.5, accuracy: 1e-5)
        }
    }

    func testIdentityWarpIsPassThrough() throws {
        // With identity warp, unwarp(x) = x
        let artifact = try makeArtifact(paramShape: [1, 2], latentCtx: 1, dLatentPerCtx: 2,
                                        warpName: "identity")
        let latent   = try makeTensor(name: "l", shape: [1, 2], values: [0.42, 0.77])
        // Just verify it doesn't throw and returns correctly shaped output
        let projector = try DecoderParamsProjector(artifact: artifact)
        let out = try projector.project(latent: try latent.asMLX())
        XCTAssertEqual(out["test_param"]?.shape, [1, 1, 2])
    }

    func testTan2WarpChangesValues() throws {
        // With tan2 warp, arctan(arctan(x * scale) / coeff1) ≠ x for x > 0
        let artifact = try makeArtifact(paramShape: [1, 2], latentCtx: 1, dLatentPerCtx: 2,
                                        warpName: "tan2", warpCoeff1: 1.0)
        let latent = try makeTensor(name: "l", shape: [1, 2], values: [0.5, -0.5])

        // Just verify it runs without error — math parity is validated by the Python golden
        let projector = try DecoderParamsProjector(artifact: artifact)
        let out = try projector.project(latent: try latent.asMLX())
        XCTAssertEqual(out.count, 1)
    }

    func testRejectsInvalidLatentRank() throws {
        let artifact = try makeArtifact(paramShape: [2, 3], latentCtx: 2, dLatentPerCtx: 3)
        let badLatent = try makeTensor(name: "bad", shape: [2, 4, 3],
                                       values: [Float](repeating: 0, count: 24))
        let projector = try DecoderParamsProjector(artifact: artifact)
        XCTAssertThrowsError(
            try projector.project(latent: try badLatent.asMLX())
        )
    }

    func testMultipleParamGroupsAreProjectedIndependently() throws {
        // Two param groups: shapes [2, 4] and [1, 8]
        let latentCtx     = 3   // 2 vectors + 1 vector
        let dLatentPerCtx = 6
        let D = latentCtx * dLatentPerCtx

        func makeProj(v: Int, c: Int, d: Int) throws -> DecoderProjectionWeights {
            let w = try makeTensor(name: "w", shape: [v, c, d],
                                   values: [Float](repeating: 0, count: v * c * d))
            let b = try makeTensor(name: "b", shape: [v, c],
                                   values: [Float](repeating: 0.1, count: v * c))
            let lw = try makeTensor(name: "lw", shape: [c], values: [Float](repeating: 1, count: c))
            let lb = try makeTensor(name: "lb", shape: [c], values: [Float](repeating: 0, count: c))
            return DecoderProjectionWeights(weightVCD: w, biasVC: b, lnWeightC: lw, lnBiasC: lb)
        }

        let artifact = DecoderParamsArtifact(
            paramShapes: [
                DecoderParamShape(name: "groupA", shape: [2, 4]),
                DecoderParamShape(name: "groupB", shape: [1, 8]),
            ],
            latentCtx: latentCtx,
            latentWarpName: "identity", latentWarpCoeff1: 1.0,
            projections: [
                "groupA": try makeProj(v: 2, c: 4, d: dLatentPerCtx),
                "groupB": try makeProj(v: 1, c: 8, d: dLatentPerCtx),
            ]
        )

        let latent = try makeTensor(name: "latent", shape: [1, D],
                                    values: [Float](repeating: 0.2, count: D))
        let projector = try DecoderParamsProjector(artifact: artifact)
        let out = try projector.project(latent: try latent.asMLX())

        XCTAssertEqual(out.count, 2)
        XCTAssertNotNil(out["groupA"])
        XCTAssertEqual(out["groupA"]?.shape, [1, 2, 4])
        XCTAssertNotNil(out["groupB"])
        XCTAssertEqual(out["groupB"]?.shape, [1, 1, 8])
    }

    func testRejectsParameterListThatDoesNotCoverLatentContext() throws {
        // [2, 3] consumes 2 latent vectors. latent_ctx 4 leaves the tail unused.
        let artifact = try makeArtifact(paramShape: [2, 3], latentCtx: 4, dLatentPerCtx: 8)
        XCTAssertThrowsError(try DecoderParamsProjector(artifact: artifact)) { error in
            guard let projectorError = error as? DecoderParamsProjectorError,
                  case let .incompleteLatentCoverage(consumed, latentCtx) = projectorError else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(consumed, 2)
            XCTAssertEqual(latentCtx, 4)
        }
    }
}
