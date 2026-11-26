import CoreGraphics
import Foundation
import MLX
import ShapERuntime

/// Swift side of the image-conditioning parity harness.
///
/// The Python dump is the independent upstream Torch oracle. This mode keeps
/// preprocessing, CLIP-vision embedding, Image300M forward evaluation, and an
/// optional production mesh smoke separate so a failure identifies the stage
/// that diverged instead of only reporting a bad final mesh.
enum ImageParityDump {
    struct Options {
        var steps: Int
        var guidance: Float
        var grid: Int
        var queryBatch: Int
        var sigmaMin: Float
        var sigmaMax: Float
        var skipMesh: Bool
    }

    private static func runtimeTensor(
        name: String,
        shape: [Int],
        floats: [Float]
    ) throws -> RuntimeTensor {
        var data = Data(count: floats.count * MemoryLayout<Float>.size)
        data.withUnsafeMutableBytes { destination in
            floats.withUnsafeBufferPointer { source in
                guard let dst = destination.baseAddress, let src = source.baseAddress else { return }
                dst.copyMemory(
                    from: src,
                    byteCount: floats.count * MemoryLayout<Float>.size
                )
            }
        }
        return try RuntimeTensor(
            name: name,
            shape: shape,
            dataType: .float32,
            data: data
        )
    }

    private static func loadArray(_ path: URL) throws -> MLXArray {
        let (shape, floats) = try loadNpyFloat32(path: path)
        return MLXArray(floats, shape)
    }

    private static func makeCGImage(fromRGB path: URL) throws -> CGImage {
        let (shape, floats) = try loadNpyFloat32(path: path)
        guard shape.count == 3, shape[2] == 3, shape[0] > 0, shape[1] > 0 else {
            throw NpyError.invalidFile("Expected raw RGB shape [height, width, 3], got \(shape)")
        }
        let height = shape[0]
        let width = shape[1]
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0..<(width * height) {
            for channel in 0..<3 {
                let value = min(max(floats[pixel * 3 + channel], 0), 255)
                rgba[pixel * 4 + channel] = UInt8(value.rounded())
            }
        }
        let data = Data(rgba) as CFData
        guard let provider = CGDataProvider(data: data),
              let image = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(
                    rawValue: CGImageAlphaInfo.premultipliedLast.rawValue
                        | CGBitmapInfo.byteOrder32Big.rawValue
                ),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              )
        else {
            throw ImageProcessor.ImageProcessorError.contextCreationFailed
        }
        return image
    }

    static func run(bundleURL: URL, parityDir: URL, options: Options) throws {
        try FileManager.default.createDirectory(
            at: parityDir,
            withIntermediateDirectories: true
        )
        print("Image parity dir: \(parityDir.path)")
        print("Bundle:           \(bundleURL.path)")

        Memory.cacheLimit = 768 * 1024 * 1024
        let bundle = try RuntimeArtifactBundle(bundleURL: bundleURL)

        // Stage 1: production CGImage preprocessing. The Python oracle saves
        // the same source pixels and its CLIP-preprocessed NHWC equivalent.
        print("Stage 1/4: Swift CLIP preprocessing...")
        let cgImage = try makeCGImage(
            fromRGB: parityDir.appendingPathComponent("python_image_rgb.npy")
        )
        let swiftClipInput = try ImageProcessor.processCLIPImage(cgImage)
        try writeMLXArray(
            parityDir.appendingPathComponent("swift_clip_input.npy"),
            swiftClipInput
        )
        print("  clip_input \(swiftClipInput.shape)")

        // Stage 2: CLIP vision grid from the exact shared preprocessed tensor.
        // This isolates the vision transformer from CoreGraphics/PIL resampling.
        print("Stage 2/4: Swift CLIP vision grid...")
        let visionArtifact = try bundle.loadClipVisionArtifact()
        let imageEmbedder = try ClipImageEmbedder(artifact: visionArtifact)
        let sharedClipInput = try loadArray(
            parityDir.appendingPathComponent("python_clip_input.npy")
        )
        let swiftGrid = try imageEmbedder.embedImagesGrid(sharedClipInput)
        try writeMLXArray(
            parityDir.appendingPathComponent("swift_image_grid.npy"),
            try swiftGrid.asMLX()
        )
        print("  image_grid \(swiftGrid.shape)")

        // Stage 3: Image300M forward. Run once with the Torch grid to isolate
        // the diffusion transformer, and once with the Swift grid to measure
        // the accumulated component chain.
        print("Stage 3/4: Swift Image300M forward...")
        let imageArtifact = try bundle.loadImage300MArtifact()
        let transformer = try Image300MTransformer(artifact: imageArtifact)

        let latentPath = parityDir.appendingPathComponent("python_image_latent.npy")
        let (latentShape, latentFloats) = try loadNpyFloat32(path: latentPath)
        let latent = try runtimeTensor(
            name: "image_latent",
            shape: latentShape,
            floats: latentFloats
        )
        let timestepPath = parityDir.appendingPathComponent("python_image_timestep.npy")
        let (timestepShape, timestepFloats) = try loadNpyFloat32(path: timestepPath)
        let timestep = try runtimeTensor(
            name: "image_timestep",
            shape: timestepShape,
            floats: timestepFloats
        )
        let gridPath = parityDir.appendingPathComponent("python_image_grid.npy")
        let (gridShape, gridFloats) = try loadNpyFloat32(path: gridPath)
        let pythonGrid = try runtimeTensor(
            name: "clip_vision_grid_embedding",
            shape: gridShape,
            floats: gridFloats
        )

        let outputFromPythonGrid = try transformer.predict(
            x: latent,
            t: timestep,
            embeddings: pythonGrid
        )
        try writeMLXArray(
            parityDir.appendingPathComponent("swift_image300m_output_from_python_grid.npy"),
            try outputFromPythonGrid.asMLX()
        )
        let outputFromSwiftGrid = try transformer.predict(
            x: latent,
            t: timestep,
            embeddings: swiftGrid
        )
        try writeMLXArray(
            parityDir.appendingPathComponent("swift_image300m_output.npy"),
            try outputFromSwiftGrid.asMLX()
        )
        print("  image300m output \(outputFromSwiftGrid.shape)")

        guard !options.skipMesh else {
            print("Stage 4/4: skipped mesh smoke")
            return
        }

        // Stage 4: production components from the Swift-processed image,
        // through sampling, decoder projection, SDF extraction, and CPU and
        // Metal topology checks on the same field.
        print("Stage 4/4: Swift image-to-mesh smoke...")
        let productionGrid = try imageEmbedder.embedImagesGrid(swiftClipInput)
        let denoiser = try GaussianToKarrasDenoiser(
            transformer: transformer,
            diffusion: try bundle.loadDiffusionConfig(),
            nCtx: imageArtifact.nCtx
        )
        let latentMLX = MLXArray(latentFloats, latentShape)
        // Karras latent storage is token-major [B, tokens*channels], while the
        // shared component input is [B, channels, tokens].
        let initX = latentMLX
            .transposed(axes: [0, 2, 1])
            .reshaped(1, imageArtifact.dLatent) * options.sigmaMax
        let sample = try KarrasSampler.sample(
            denoiser: denoiser,
            embeddings: productionGrid,
            uncondEmbeddings: nil,
            shape: (batchSize: 1, dLatent: imageArtifact.dLatent),
            config: KarrasSampler.Config(
                steps: options.steps,
                sigmaMin: options.sigmaMin,
                sigmaMax: options.sigmaMax,
                guidanceScale: options.guidance,
                clipDenoised: true,
                seed: 0,
                progressInterval: 0
            ),
            initX: initX,
            progress: nil
        )
        try writeMLXArray(
            parityDir.appendingPathComponent("swift_image_latent.npy"),
            sample.x
        )
        try writeMLXArray(
            parityDir.appendingPathComponent("swift_image_pred_xstart.npy"),
            sample.predXStart
        )

        let projector = try DecoderParamsProjector(
            artifact: try bundle.loadDecoderParamsArtifact()
        )
        let params = try projector.project(latent: sample.x)
        let renderer = try STFMeshRenderer(
            artifact: try bundle.loadRendererArtifact()
        )
        let (fields, _) = try renderer.extractSDFFields(
            dynamicParams: params,
            queryBatchSize: options.queryBatch,
            gridSizeOverride: options.grid
        )
        let field = fields[0]
        try writeMLXArray(
            parityDir.appendingPathComponent("swift_image_sdf_field.npy"),
            field
        )

        func meshStats(_ vertices: MLXArray, _ faces: MLXArray) -> [String: Any] {
            let stats = MeshTopology.stats(
                vertexCount: vertices.shape[0],
                faces: faces.asArray(Int32.self)
            )
            return [
                "vertexCount": stats.vertexCount,
                "faceCount": stats.faceCount,
                "boundaryEdgeCount": stats.boundaryEdgeCount,
                "nonManifoldEdgeCount": stats.nonManifoldEdgeCount,
                "invalidFaceCount": stats.invalidFaceCount,
                "isClosed": stats.isClosed,
            ]
        }

        let (cpuVertices, cpuFaces) = try renderer.meshFromSDFField(
            field: field,
            useMetalMarchingCubes: false
        )
        eval(cpuVertices, cpuFaces)
        try writeMLXArray(
            parityDir.appendingPathComponent("swift_image_verts.npy"),
            cpuVertices
        )
        let cpuFaceValues = cpuFaces.asArray(Int32.self)
        try writeNpyInt32(
            path: parityDir.appendingPathComponent("swift_image_faces.npy"),
            shape: cpuFaces.shape,
            values: cpuFaceValues
        )
        if cpuVertices.shape[0] > 0 {
            try MeshExport.writeOBJ(
                path: parityDir.appendingPathComponent("swift_image_mesh_cpu.obj"),
                vertices: cpuVertices.asArray(Float.self),
                faces: cpuFaceValues,
                comment: "Swift image parity CPU mesh"
            )
        }

        var report: [String: Any] = [
            "steps": options.steps,
            "grid": options.grid,
            "guidance": options.guidance,
            "sigmaMin": options.sigmaMin,
            "sigmaMax": options.sigmaMax,
            "sdfMin": MLX.min(field).item(Float.self),
            "sdfMax": MLX.max(field).item(Float.self),
            "cpu": meshStats(cpuVertices, cpuFaces),
        ]
        do {
            let (metalVertices, metalFaces) = try renderer.meshFromSDFField(
                field: field,
                useMetalMarchingCubes: true
            )
            eval(metalVertices, metalFaces)
            report["metal"] = meshStats(metalVertices, metalFaces)
            if metalVertices.shape[0] > 0 {
                try MeshExport.writeOBJ(
                    path: parityDir.appendingPathComponent("swift_image_mesh_metal.obj"),
                    vertices: metalVertices.asArray(Float.self),
                    faces: metalFaces.asArray(Int32.self),
                    comment: "Swift image parity Metal mesh"
                )
            }
        } catch {
            report["metalError"] = String(describing: error)
            print("  Metal mesh failed: \(error)")
        }

        let reportData = try JSONSerialization.data(
            withJSONObject: report,
            options: [.prettyPrinted, .sortedKeys]
        )
        try reportData.write(
            to: parityDir.appendingPathComponent("swift_image_mesh_summary.json")
        )
        print("  image mesh \(String(data: reportData, encoding: .utf8) ?? "<unprintable>")")
    }
}
