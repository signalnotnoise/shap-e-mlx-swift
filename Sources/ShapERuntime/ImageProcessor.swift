import Foundation
import CoreGraphics
import MLX

public enum ImageProcessor {
    
    public enum ImageProcessorError: Error, LocalizedError {
        case contextCreationFailed
        case unsupportedImageFormat
        
        public var errorDescription: String? {
            switch self {
            case .contextCreationFailed:
                return "Failed to create CoreGraphics context for image resizing."
            case .unsupportedImageFormat:
                return "The provided CGImage format is unsupported."
            }
        }
    }
    
    /// Processes a CGImage for CLIP Vision input.
    ///
    /// Matches torchvision `Resize(224)` + `CenterCrop(224)`: the shorter side
    /// becomes 224 and the longer side is center-cropped. Returns float32
    /// `[1, 224, 224, 3]` after CLIP mean/std normalization.
    public static func processCLIPImage(_ image: CGImage) throws -> MLXArray {
        let targetSize = 224
        
        // 1. Scale the shorter side to 224, then center-crop.
        let width = CGFloat(image.width)
        let height = CGFloat(image.height)
        
        let scale = max(CGFloat(targetSize) / width, CGFloat(targetSize) / height)
        let scaledWidth = width * scale
        let scaledHeight = height * scale
        
        let xOffset = (CGFloat(targetSize) - scaledWidth) / 2.0
        let yOffset = (CGFloat(targetSize) - scaledHeight) / 2.0
        
        
        // 2. Render to a 224x224 raw RGBA buffer
        let bytesPerPixel = 4
        let bytesPerRow = targetSize * bytesPerPixel
        let totalBytes = targetSize * bytesPerRow
        
        // White only covers a subpixel fringe. The scaled image covers the 224 canvas.
        var pixelData = [UInt8](repeating: 255, count: totalBytes)
        
        guard let context = CGContext(
            data: &pixelData,
            width: targetSize,
            height: targetSize,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else {
            throw ImageProcessorError.contextCreationFailed
        }
        
        // Do not flip this bitmap context: drawing a CGImage into it already
        // yields the buffer in top-to-bottom raster order expected by CLIP.
        // Adding a flip here produces a vertically reversed tensor.
        
        // Negative offset on the long axis center-crops. The short axis lands on 0.
        let drawRect = CGRect(
            x: xOffset,
            y: yOffset,
            width: scaledWidth,
            height: scaledHeight
        )
        
        // High quality interpolation
        context.interpolationQuality = .high
        context.draw(image, in: drawRect)
        
        // 3. Create MLXArray directly from the UInt8 buffer (shape [1, 224, 224, 4] for RGBA)
        let rawArray = MLXArray(pixelData, [1, targetSize, targetSize, 4])
        
        // 4. Slice to drop the Alpha channel and scale to [0, 1] via the GPU/accelerator
        let rgbArray = rawArray[0..., 0..., 0..., 0..<3]
        let imageArray = rgbArray.asType(Float32.self) / 255.0
        
        // 5. Apply CLIP Normalization
        // mean: [0.48145466, 0.4578275, 0.40821073], std: [0.26862954, 0.26130258, 0.27577711]
        let mean = MLXArray([Float(0.48145466), Float(0.4578275), Float(0.40821073)], [1, 1, 1, 3])
        let std = MLXArray([Float(0.26862954), Float(0.26130258), Float(0.27577711)], [1, 1, 1, 3])
        
        let normalized = (imageArray - mean) / std
        return normalized
    }
}
