import CoreGraphics
import MLX
import XCTest
@testable import ShapERuntime

final class ImageProcessorTests: XCTestCase {

    override class func setUp() {
        super.setUp()
        _ = MLX.Stream.defaultStream(.cpu)
    }

    func testWideImageIsCenterCroppedRatherThanPadded() throws {
        // 448×224 solid red. Aspect-fit would letterbox the top and bottom white.
        // CLIP Resize+CenterCrop covers the 224 canvas with the source.
        let image = try solidImage(width: 448, height: 224, red: 255, green: 0, blue: 0)
        let processed = try ImageProcessor.processCLIPImage(image)
        MLX.eval(processed)
        XCTAssertEqual(processed.shape, [1, 224, 224, 3])

        let green = processed[0..., 0..., 0..., 1]
        let maxGreen = green.max().item(Float.self)
        // Normalized red has a negative green channel. Padded white is about +2.
        XCTAssertLessThan(maxGreen, 0)
    }

    func testImageOrientationIsPreserved() throws {
        let image = try quadrantImage()
        let processed = try ImageProcessor.processCLIPImage(image)
        MLX.eval(processed)

        func pixel(row: Int, column: Int) -> [Float] {
            (0..<3).map { channel in
                processed[0, row, column, channel].item(Float.self)
            }
        }

        let topLeft = pixel(row: 20, column: 20)       // red
        let topRight = pixel(row: 20, column: 203)     // green
        let bottomLeft = pixel(row: 203, column: 20)   // blue
        let bottomRight = pixel(row: 203, column: 203) // white

        XCTAssertGreaterThan(topLeft[0], topLeft[1])
        XCTAssertGreaterThan(topLeft[0], topLeft[2])
        XCTAssertGreaterThan(topRight[1], topRight[0])
        XCTAssertGreaterThan(topRight[1], topRight[2])
        XCTAssertGreaterThan(bottomLeft[2], bottomLeft[0])
        XCTAssertGreaterThan(bottomLeft[2], bottomLeft[1])
        XCTAssertGreaterThan(bottomRight[0], 1.5)
        XCTAssertGreaterThan(bottomRight[1], 1.5)
        XCTAssertGreaterThan(bottomRight[2], 1.5)
    }

    func testTallImageIsCenterCroppedRatherThanPadded() throws {
        let image = try solidImage(width: 224, height: 448, red: 0, green: 0, blue: 255)
        let processed = try ImageProcessor.processCLIPImage(image)
        MLX.eval(processed)

        let red = processed[0..., 0..., 0..., 0]
        let maxRed = red.max().item(Float.self)
        // Normalized blue has a negative red channel. Padded white is about +1.9.
        XCTAssertLessThan(maxRed, 0)
    }

    private func quadrantImage() throws -> CGImage {
        let size = 224
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw ImageProcessor.ImageProcessorError.contextCreationFailed
        }
        // CoreGraphics uses a bottom-left origin in this context. The upper
        // half in image coordinates is therefore y=112..<224 here.
        context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 112, width: 112, height: 112))
        context.setFillColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1))
        context.fill(CGRect(x: 112, y: 112, width: 112, height: 112))
        context.setFillColor(CGColor(red: 0, green: 0, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: 112, height: 112))
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 112, y: 0, width: 112, height: 112))
        guard let image = context.makeImage() else {
            throw ImageProcessor.ImageProcessorError.contextCreationFailed
        }
        return image
    }

    private func solidImage(width: Int, height: Int, red: CGFloat, green: CGFloat, blue: CGFloat) throws -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw ImageProcessor.ImageProcessorError.contextCreationFailed
        }
        context.setFillColor(CGColor(red: red / 255, green: green / 255, blue: blue / 255, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let image = context.makeImage() else {
            throw ImageProcessor.ImageProcessorError.contextCreationFailed
        }
        return image
    }
}
