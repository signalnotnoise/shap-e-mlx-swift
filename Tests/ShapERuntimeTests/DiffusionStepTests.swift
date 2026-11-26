import Foundation
import MLX
import XCTest
@testable import ShapERuntime

final class DiffusionStepTests: XCTestCase {

    override class func setUp() {
        super.setUp()
        _ = MLX.Stream.defaultStream(.cpu)
    }

    // MARK: - Helpers

    private func makeTensor(name: String, shape: [Int], values: [Float]) throws -> RuntimeTensor {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return try RuntimeTensor(name: name, shape: shape, dataType: .float32, data: data)
    }

    private func makeSchedule(count: Int, value: Float = 0.5) throws -> RuntimeTensor {
        try makeTensor(name: "schedule", shape: [count], values: [Float](repeating: value, count: count))
    }

    /// Minimal RuntimeDiffusionConfig with uniform schedules — good enough for shape/math checks.
    private func makeConfig(steps: Int = 4) throws -> RuntimeDiffusionConfig {
        let alphas = try makeSchedule(count: steps, value: 0.9)
        let betas  = try makeSchedule(count: steps, value: 0.1)
        let postLogVar = try makeSchedule(count: steps, value: -2.0)
        let sqrtRecip  = try makeSchedule(count: steps, value: 1.1)
        let sqrtRecip1 = try makeSchedule(count: steps, value: 0.3)
        let coef1      = try makeSchedule(count: steps, value: 0.6)
        let coef2      = try makeSchedule(count: steps, value: 0.4)

        return RuntimeDiffusionConfig(
            alphasCumprod: alphas,
            betas: betas,
            posteriorLogVarianceClipped: postLogVar,
            sqrtRecipAlphasCumprod: sqrtRecip,
            sqrtRecipm1AlphasCumprod: sqrtRecip1,
            posteriorMeanCoef1: coef1,
            posteriorMeanCoef2: coef2,
            numTimesteps: steps,
            modelMeanType: "epsilon"
        )
    }

    // MARK: - extractInto

    func testExtractIntoIndexesAndBroadcasts() {
        // arr = [0, 10, 20, 30], timesteps = [1, 3], broadcastShape = [2, 2, 2]
        let arr       = MLXArray([Float(0), Float(10), Float(20), Float(30)])
        let timesteps = MLXArray([Int32(1), Int32(3)])
        let result    = DiffusionStep.extractInto(
            arr: arr, timesteps: timesteps, broadcastShape: [2, 2, 2]
        )
        let values = result.asArray(Float.self)

        // batch 0 → arr[1]=10, batch 1 → arr[3]=30, broadcast over [2,2]
        XCTAssertEqual(result.shape, [2, 2, 2])
        XCTAssertEqual(values[0], 10, accuracy: 1e-6)   // batch 0, first spread element
        XCTAssertEqual(values[4], 30, accuracy: 1e-6)   // batch 1, first spread element
    }

    // MARK: - pMeanVariance

    func testPMeanVarianceOutputShapesMatchInput() throws {
        let config = try makeConfig(steps: 4)

        // x: [1, 2, 3], model_output: [1, 4, 3] (2C channels)
        let B = 1; let C = 2; let T = 3
        let x = try makeTensor(name: "x", shape: [B, C, T],
                               values: [Float](repeating: 0.5, count: B * C * T))
        let y = try makeTensor(name: "y", shape: [B, C * 2, T],
                               values: [Float](repeating: 0.0, count: B * C * 2 * T))
        let t = try makeTensor(name: "t", shape: [B],
                               values: [Float(1)])    // timestep index 1

        let out = try DiffusionStep.pMeanVariance(
            diffusion: config,
            x: x,
            modelOutput: y,
            timesteps: t,
            modelMeanType: "epsilon",
            clipDenoised: true
        )

        XCTAssertEqual(out.mean.shape,        [B, C, T])
        XCTAssertEqual(out.variance.shape,    [B, C, T])
        XCTAssertEqual(out.logVariance.shape, [B, C, T])
        XCTAssertEqual(out.predXStart.shape,  [B, C, T])
    }

    func testPMeanVarianceEpsilonPathPredXStartFormula() throws {
        // With x=1, eps=0, sqrtRecip=1.1, sqrtRecipm1=0.3:
        //   pred_xstart = 1.1 * 1.0 - 0.3 * 0.0 = 1.1 → clip → 1.0
        let config = try makeConfig(steps: 4)
        let B = 1; let C = 1; let T = 1

        let x = try makeTensor(name: "x", shape: [B, C, T], values: [1.0])
        // model output = [eps=0, var=0] concatenated along channel axis
        let y = try makeTensor(name: "y", shape: [B, C * 2, T], values: [0.0, 0.0])
        let t = try makeTensor(name: "t", shape: [B], values: [Float(0)])

        let out = try DiffusionStep.pMeanVariance(
            diffusion: config, x: x, modelOutput: y, timesteps: t,
            modelMeanType: "epsilon", clipDenoised: true
        )

        let predXStart = out.predXStart.asArray(Float.self)[0]
        // 1.1 * 1.0 - 0.3 * 0.0 = 1.1, clamped to 1.0
        XCTAssertEqual(predXStart, 1.0, accuracy: 1e-5)
    }

    func testPMeanVarianceXStartPathPassesThroughEpsChannel() throws {
        // With modelMeanType="x_start", pred_xstart = eps channel directly
        let config = try makeConfig(steps: 4)
        let B = 1; let C = 1; let T = 1

        let x = try makeTensor(name: "x", shape: [B, C, T], values: [0.8])
        // eps channel = 0.42, var channel = 0.0
        let y = try makeTensor(name: "y", shape: [B, C * 2, T], values: [0.42, 0.0])
        let t = try makeTensor(name: "t", shape: [B], values: [Float(0)])

        let out = try DiffusionStep.pMeanVariance(
            diffusion: config, x: x, modelOutput: y, timesteps: t,
            modelMeanType: "x_start", clipDenoised: false
        )

        let predXStart = out.predXStart.asArray(Float.self)[0]
        XCTAssertEqual(predXStart, 0.42, accuracy: 1e-5)
    }

    func testPMeanVariancePosteriorMeanFormula() throws {
        // With pred_xstart=0.5 (clipped), x=0.5, coef1=0.6, coef2=0.4:
        //   mean = 0.6 * 0.5 + 0.4 * 0.5 = 0.5
        let config = try makeConfig(steps: 4)
        let B = 1; let C = 1; let T = 1

        let x = try makeTensor(name: "x", shape: [B, C, T], values: [0.5])
        // eps=0 → pred_xstart = 1.1*0.5 - 0.3*0 = 0.55 ... but clipped to [−1,1]
        // However: mean = coef1*0.55 + coef2*0.5 = 0.6*0.55 + 0.4*0.5 = 0.33 + 0.2 = 0.53
        let y = try makeTensor(name: "y", shape: [B, C * 2, T], values: [0.0, 0.0])
        let t = try makeTensor(name: "t", shape: [B], values: [Float(0)])

        let out = try DiffusionStep.pMeanVariance(
            diffusion: config, x: x, modelOutput: y, timesteps: t,
            modelMeanType: "epsilon", clipDenoised: false
        )

        let mean = out.mean.asArray(Float.self)[0]
        // pred_xstart = 1.1*0.5 - 0.3*0 = 0.55
        // mean = 0.6*0.55 + 0.4*0.5 = 0.33 + 0.20 = 0.53
        XCTAssertEqual(mean, 0.53, accuracy: 1e-5)
    }

    func testPMeanVarianceRejectsWrongChannelCount() throws {
        let config = try makeConfig()
        let x = try makeTensor(name: "x", shape: [1, 2, 3], values: [Float](repeating: 0, count: 6))
        // wrong: should be [1, 4, 3] but passing [1, 3, 3]
        let y = try makeTensor(name: "y", shape: [1, 3, 3], values: [Float](repeating: 0, count: 9))
        let t = try makeTensor(name: "t", shape: [1], values: [Float(0)])

        XCTAssertThrowsError(
            try DiffusionStep.pMeanVariance(diffusion: config, x: x, modelOutput: y, timesteps: t)
        )
    }
}
