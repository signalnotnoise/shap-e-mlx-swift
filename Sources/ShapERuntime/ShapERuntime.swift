import Foundation
import CoreGraphics
import MLX

// MARK: - ShapEMeshResult

/// An end-to-end result containing the generated 3D output.
/// Provides both standard Swift collections (`[Float]`/`[Int32]`) and native `MLXArray` references.
public struct ShapEMeshResult: Sendable {
    /// Native `RuntimeTensor` mapping to the shape coordinates.
    public let meshResult: MeshResult

    /// Retrieve the vertices as an `MLXArray` instance for continued MLX-graph evaluation or custom math.
    public func mlxVertices() throws -> MLXArray {
        try meshResult.vertices.asMLX().asType(Float.self)
    }

    /// Retrieve the faces as an `MLXArray` instance.
    public func mlxFaces() throws -> MLXArray {
        try meshResult.faces.asMLX().asType(Int32.self)
    }
    
    /// Retrieve specific texture channels generically as an MLXArray.
    public func mlxTextureChannel(_ name: String) throws -> MLXArray? {
        try meshResult.vertexChannels?[name]?.asMLX().asType(Float.self)
    }

    /// Extract standard Swift sequence type for vertices `[V, 3]`. Implicitly evaluates the tensor locally.
    public func verticesArray() throws -> [Float] {
        try mlxVertices().asArray(Float.self)
    }

    /// Extract standard Swift sequence type for geometry triangle faces `[F, 3]`. Implicitly evaluates.
    public func facesArray() throws -> [Int32] {
        try mlxFaces().asArray(Int32.self)
    }
    
    /// Extract standard Swift sequence type for vertex RGBA colors `[V, 4]`.
    public func colorsArray() throws -> [Float]? {
        guard let rgba = try mlxTextureChannel("RGBA") else {
            return nil
        }
        return rgba.asArray(Float.self)
    }
}

public enum ShapERuntimeError: Error, LocalizedError, Sendable {
    case imagePipelineUnavailable
    /// iOS/visionOS: IOGPU kills the process if a Metal command buffer
    /// completes after the scene leaves the foreground.
    case gpuNotForeground
    case invalidBatchSize(Int)

    public var errorDescription: String? {
        switch self {
        case .imagePipelineUnavailable:
            return "Image generation pipeline failed to load (clip_vision / image300m)."
        case .gpuNotForeground:
            return "Generation stopped because the app left the foreground. Keep Loftform on screen while generating."
        case .invalidBatchSize(let batchSize):
            return "batchSize must be greater than zero, got \(batchSize)."
        }
    }
}

// MARK: - ShapERuntime

public final class ShapERuntime: @unchecked Sendable {
    
    public let bundle: RuntimeArtifactBundle
    public let tokenizer: ClipSimpleTokenizer
    public let embedder: ClipTextEmbedder
    public let transformer: Text300MTransformer
    public let denoiser: GaussianToKarrasDenoiser

    /// Image weights (~2.3 GB) are loaded on first `generate(image:)` call.
    private let imagePipelineLock = NSLock()
    private var imageEmbedder: ClipImageEmbedder?
    private var imageTransformer: Image300MTransformer?
    private var imageDenoiser: GaussianToKarrasDenoiser?
    private var imageArtifact: Image300MArtifact?
    
    public let projector: DecoderParamsProjector
    
    /// Internal flag to ensure MLX Metal kernels are initialized once.
    private static let _gpuInitialized: Void = {
        if let path = Bundle.main.resourcePath {
            setenv("MLX_METAL_PATH", path, 1)
        } else {
            // Fallback for standalone binaries: check the executable's directory
            let execPath = Bundle.main.bundleURL.appendingPathComponent("swift_ShapERuntime.bundle").path
            if FileManager.default.fileExists(atPath: execPath) {
                setenv("MLX_METAL_PATH", execPath, 1)
            }
        }
    }()
    
    // Kept artifacts for shapes
    public let textArtifact: Text300MArtifact
    private let decoderArtifact: DecoderParamsArtifact
    
    public let renderer: STFMeshRenderer

    public init(modelBundleURL: URL, bpeVocabularyURL: URL) throws {
        _ = Self._gpuInitialized
        let startInit = CFAbsoluteTimeGetCurrent()
        ShapELogger.debug("[ShapE] Initializing runtime...")
        ShapELogger.debug("[ShapE] MLX Device: \(MLX.Device.defaultDevice())")
        
        ShapELogger.debug("[ShapE] Loading bundle manifest from \(modelBundleURL.lastPathComponent)...")
        let startBundle = CFAbsoluteTimeGetCurrent()
        self.bundle = try RuntimeArtifactBundle(bundleURL: modelBundleURL)
        ShapELogger.debug("[ShapE] Bundle manifest loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startBundle))s")
        
        // 1. Initialize tokenizer and embedder
        ShapELogger.debug("[ShapE] Loading tokenizer and embedder...")
        let startEmbedder = CFAbsoluteTimeGetCurrent()
        self.tokenizer = try ClipSimpleTokenizer(vocabularyURL: bpeVocabularyURL)
        self.embedder = try ClipTextEmbedder(
            artifact: try bundle.loadClipTextArtifact(),
            tokenizer: tokenizer
        )
        ShapELogger.debug("[ShapE] Tokenizer and embedder loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startEmbedder))s")

        // 2. Initialize diffusion transformer (infers config from artifact weights)
        ShapELogger.debug("[ShapE] Loading diffusion transformer...")
        let startTransformer = CFAbsoluteTimeGetCurrent()
        self.textArtifact = try bundle.loadText300MArtifact()
        self.transformer = try Text300MTransformer(artifact: textArtifact)
        ShapELogger.debug("[ShapE] Diffusion transformer loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startTransformer))s")

        // 3. Initialize Karras Denoiser
        ShapELogger.debug("[ShapE] Loading denoiser...")
        let startDenoiser = CFAbsoluteTimeGetCurrent()
        self.denoiser = try GaussianToKarrasDenoiser(
            transformer: transformer,
            diffusion: try bundle.loadDiffusionConfig(),
            nCtx: textArtifact.nCtx
        )
        ShapELogger.debug("[ShapE] Denoiser loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startDenoiser))s")

        ShapELogger.debug("[ShapE] Skipping clip_vision + image300m until first image generate")

        // 4. Load parameter projection info
        ShapELogger.debug("[ShapE] Loading decoder params projector...")
        let startProjector = CFAbsoluteTimeGetCurrent()
        self.decoderArtifact = try bundle.loadDecoderParamsArtifact()
        self.projector = try DecoderParamsProjector(artifact: decoderArtifact)
        ShapELogger.debug("[ShapE] Projector loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startProjector))s")

        // 5. Initialize the renderer
        ShapELogger.debug("[ShapE] Loading mesh renderer...")
        let startRenderer = CFAbsoluteTimeGetCurrent()
        self.renderer = try STFMeshRenderer(
            artifact: try bundle.loadRendererArtifact()
        )
        ShapELogger.debug("[ShapE] Mesh renderer loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startRenderer))s")
        
        ShapELogger.debug("[ShapE] Runtime initialization completed in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startInit))s")
    }

    /// Loads CLIP vision + image300m once. Text-only init skips ~2.3 GB of weights.
    private func ensureImagePipeline() throws {
        imagePipelineLock.lock()
        defer { imagePipelineLock.unlock() }
        if imageEmbedder != nil {
            return
        }

        ShapELogger.debug("[ShapE] Loading image pipeline (clip_vision + image300m)...")
        let start = CFAbsoluteTimeGetCurrent()
        let embedder = try ClipImageEmbedder(
            artifact: try bundle.loadClipVisionArtifact()
        )
        let artifact = try bundle.loadImage300MArtifact()
        let transformer = try Image300MTransformer(artifact: artifact)
        let denoiser = try GaussianToKarrasDenoiser(
            transformer: transformer,
            diffusion: try bundle.loadDiffusionConfig(),
            nCtx: artifact.nCtx
        )
        imageEmbedder = embedder
        imageArtifact = artifact
        imageTransformer = transformer
        imageDenoiser = denoiser
        ShapELogger.debug("[ShapE] Image pipeline loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - start))s")
    }

    public typealias ShapEProgressCallback = (_ phase: String, _ step: Int, _ totalSteps: Int, _ partialMeshes: [ShapEMeshResult]?) -> Void

    /// Fire a progress tick then yield so SwiftUI can paint before the next
    /// synchronous CLIP / diffusion / extract chunk.
    private func reportProgress(
        _ progress: ShapEProgressCallback?,
        _ phase: String,
        step: Int,
        total: Int,
        meshes: [ShapEMeshResult]? = nil
    ) async throws {
        progress?(phase, step, total, meshes)
        try Task.checkCancellation()
        await Task.yield()
    }

    /// Generates one mesh from a text prompt.
    ///
    /// `seed` defaults to 0. Call the `batchSize` overload for more than one sample.
    @concurrent
    public func generate(
        prompt: String,
        guidanceScale: Float = 15.0,
        numInferenceSteps: Int = 64,
        sigmaMin: Float = 0.002,
        sigmaMax: Float = 160.0,
        gridSize: Int = 128,
        queryBatchSize: Int = 65536,
        seed: Int = 0
    ) async throws -> ShapEMeshResult {
        let meshes = try await generate(
            prompt: prompt,
            batchSize: 1,
            guidanceScale: guidanceScale,
            numInferenceSteps: numInferenceSteps,
            sigmaMin: sigmaMin,
            sigmaMax: sigmaMax,
            gridSize: gridSize,
            queryBatchSize: queryBatchSize,
            seed: seed,
            useMetalMarchingCubes: false,
            progressInterval: 0,
            progress: nil
        )
        guard let mesh = meshes.first else {
            throw MeshAssemblyError.emptyExtraction
        }
        return mesh
    }

    /// Generates one or more meshes from a text prompt.
    ///
    /// `batchSize` has no default so `generate(prompt:)` keeps returning one
    /// `ShapEMeshResult`. `seed` defaults to 0. Each batch row uses `seed + row`.
    ///
    /// - Parameter progressInterval: How often the sampler attaches an unscaled latent
    ///   for intermediate mesh previews. `0` disables partial extraction entirely
    ///   (step progress is still reported). Default `5`.
    @concurrent
    public func generate(
        prompt: String,
        batchSize: Int,
        guidanceScale: Float = 15.0,
        numInferenceSteps: Int = 64,
        sigmaMin: Float = 0.002,
        sigmaMax: Float = 160.0,
        gridSize: Int = 128,
        queryBatchSize: Int = 65536,
        seed: Int = 0,
        useMetalMarchingCubes: Bool = false,
        progressInterval: Int = 5,
        progress: ShapEProgressCallback? = nil
    ) async throws -> [ShapEMeshResult] {
        guard batchSize > 0 else { throw ShapERuntimeError.invalidBatchSize(batchSize) }
        let startGenerate = CFAbsoluteTimeGetCurrent()
        ShapELogger.debug("[ShapE] Starting generation for prompt: '\(prompt)'")
        
        // Synchronize and flush prior tasks from the MLX stream queue
        MLX.eval()

        try await reportProgress(progress, "embedding", step: 0, total: 1)
        // 1. Tokenize and encode the text prompt
        ShapELogger.debug("[ShapE] Tokenizing and embedding text...")
        let startEmbed = CFAbsoluteTimeGetCurrent()
        let prompts = Array(repeating: prompt, count: batchSize)
        let embeddings = try embedder.embedText(prompts)
        let uncondPrompts = Array(repeating: "", count: batchSize)
        let uncondEmbeddings = try embedder.embedText(uncondPrompts)

        let actualBatchSize = embeddings.shape[0]
        let latentShape = (batchSize: actualBatchSize, dLatent: textArtifact.dLatent)
        ShapELogger.debug("[ShapE] Text embedding completed in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startEmbed))s")
        try await reportProgress(progress, "embedding", step: 1, total: 1)

        let config = KarrasSampler.Config(
            steps: numInferenceSteps,
            sigmaMin: sigmaMin,
            sigmaMax: sigmaMax,
            guidanceScale: guidanceScale,
            clipDenoised: true,
            seed: seed,
            progressInterval: progressInterval
        )

        // 2. Denoise and generate the terminal latent
        ShapELogger.debug("[ShapE] Starting diffusion process (\(numInferenceSteps) steps)...")
        let startDiffusion = CFAbsoluteTimeGetCurrent()

        let samplerProgress: ((Int, Int, MLXArray?) -> Void)? = progress.map { p in
            return { step, total, unscaled in
                var partial: [ShapEMeshResult]? = nil

                // Always report phase "sampling" so clients can track step progress.
                // Only extract intermediate meshes when the sampler provides an unscaled latent
                // (every `progressInterval` steps when progressInterval > 0).
                // Partials always use CPU marching cubes to avoid GPU contention with MLX.
                if let currentUnscaledDenoised = unscaled {
                    do {
                        let dynamicParams = try self.projector.project(latent: currentUnscaledDenoised)
                        let meshBatch = try self.renderer.extractMeshes(
                            dynamicParams: dynamicParams,
                            queryBatchSize: queryBatchSize,
                            includeVertexColors: true,
                            gridSizeOverride: min(48, gridSize),
                            artifactInfo: ArtifactVersionInfo(artifactVersion: 0, manifestPath: "runtime"),
                            useMetalMarchingCubes: false
                        )
                        let output = meshBatch.meshes.map { ShapEMeshResult(meshResult: $0) }
                        for r in output {
                            MLX.eval(try r.mlxVertices(), try r.mlxFaces())
                            _ = try r.colorsArray()
                        }
                        partial = output
                        Memory.clearCache()
                    } catch {
                        ShapELogger.debug("[ShapE] Partial render failed: \(error)")
                    }
                }

                p("sampling", step, total, partial)
            }
        }

        let result = try KarrasSampler.sample(
            denoiser: denoiser,
            embeddings: embeddings,
            uncondEmbeddings: uncondEmbeddings,
            shape: latentShape,
            config: config,
            progress: samplerProgress
        )

        let diffEnd = CFAbsoluteTimeGetCurrent()
        ShapELogger.debug("[ShapE] Diffusion completed in \(String(format: "%.2f", diffEnd - startDiffusion))s")

        try await reportProgress(progress, "projecting", step: 0, total: 1)
        // 3. Project Latent Space onto Decoded Network Space params
        ShapELogger.debug("[ShapE] Projecting latent space...")
        let startProject = CFAbsoluteTimeGetCurrent()
        let dynamicParams = try projector.project(latent: result.x)
        MLX.eval(Array(dynamicParams.values))
        try await reportProgress(progress, "projecting", step: 1, total: 1)
        ShapELogger.debug("[ShapE] Projection completed in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startProject))s")

        // 4. Query volumetric parameters via Marching Cubes (with sub-progress)
        try await reportProgress(progress, "volume", step: 0, total: 1)
        ShapELogger.debug("[ShapE] Extracting mesh via Marching Cubes (grid size: \(gridSize))...")
        let startExtract = CFAbsoluteTimeGetCurrent()
        let extractProgress: STFMeshRenderer.MeshExtractProgress? = progress.map { p in
            { phase, step, total in
                p(phase, step, total, nil)
            }
        }
        let meshBatch = try renderer.extractMeshes(
            dynamicParams: dynamicParams,
            queryBatchSize: queryBatchSize,
            includeVertexColors: true,
            gridSizeOverride: gridSize,
            artifactInfo: ArtifactVersionInfo(artifactVersion: 0, manifestPath: "runtime"),
            useMetalMarchingCubes: useMetalMarchingCubes,
            progress: extractProgress
        )
        let extEnd = CFAbsoluteTimeGetCurrent()
        ShapELogger.debug("[ShapE] Mesh extraction completed in \(String(format: "%.2f", extEnd - startExtract))s")

        guard !meshBatch.meshes.isEmpty else {
            throw MeshAssemblyError.emptyExtraction
        }

        // 5. Build presentation model and force resolution 
        ShapELogger.debug("[ShapE] Pre-warming geometry arrays...")
        try await reportProgress(progress, "resolving", step: 0, total: 1)
        let startResolve = CFAbsoluteTimeGetCurrent()
        let output = meshBatch.meshes.map { ShapEMeshResult(meshResult: $0) }
        
        // Prewarm MLX compilation trees by fetching standard matrices natively
        for result in output {
            MLX.eval(try result.mlxVertices(), try result.mlxFaces())
            if let rgbs = try result.colorsArray() {
                // Implicit access evaluates graph memory safely.
                _ = rgbs
            }
        }
        try await reportProgress(progress, "resolving", step: 1, total: 1)
        ShapELogger.debug("[ShapE] Array resolution completed in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startResolve))s")

        ShapELogger.debug("[ShapE] Total generation time: \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startGenerate))s")
        return output
    }

    /// Generates a mesh from a CGImage.
    @concurrent
    public func generate(
        image: CGImage,
        batchSize: Int = 1,
        guidanceScale: Float = 3.0,
        numInferenceSteps: Int = 64,
        sigmaMin: Float = 0.002,
        sigmaMax: Float = 160.0,
        gridSize: Int = 128,
        queryBatchSize: Int = 65536,
        seed: Int = 0,
        useMetalMarchingCubes: Bool = false,
        progressInterval: Int = 5,
        progress: ShapEProgressCallback? = nil
    ) async throws -> [ShapEMeshResult] {
        let mlxImage = try ImageProcessor.processCLIPImage(image)
        return try await generate(
            image: mlxImage,
            batchSize: batchSize,
            guidanceScale: guidanceScale,
            numInferenceSteps: numInferenceSteps,
            sigmaMin: sigmaMin,
            sigmaMax: sigmaMax,
            gridSize: gridSize,
            queryBatchSize: queryBatchSize,
            seed: seed,
            useMetalMarchingCubes: useMetalMarchingCubes,
            progressInterval: progressInterval,
            progress: progress
        )
    }

    /// Generates a mesh from an image.
    ///
    /// - Parameter progressInterval: `0` disables intermediate mesh extracts.
    @concurrent
    public func generate(
        image: MLXArray, // Expected [1, 224, 224, 3] normalized
        batchSize: Int = 1,
        guidanceScale: Float = 3.0,
        numInferenceSteps: Int = 64,
        sigmaMin: Float = 0.002,
        sigmaMax: Float = 160.0,
        gridSize: Int = 128,
        queryBatchSize: Int = 65536,
        seed: Int = 0,
        useMetalMarchingCubes: Bool = false,
        progressInterval: Int = 5,
        progress: ShapEProgressCallback? = nil
    ) async throws -> [ShapEMeshResult] {
        guard batchSize > 0 else { throw ShapERuntimeError.invalidBatchSize(batchSize) }
        let startGenerate = CFAbsoluteTimeGetCurrent()
        ShapELogger.debug("[ShapE] Starting image generation...")
        try await reportProgress(progress, "embedding", step: 0, total: 1)
        try ensureImagePipeline()
        guard
            let imageEmbedder,
            let imageArtifact,
            let imageDenoiser
        else {
            throw ShapERuntimeError.imagePipelineUnavailable
        }

        MLX.eval()

        ShapELogger.debug("[ShapE] Embedding image...")
        let startEmbed = CFAbsoluteTimeGetCurrent()
        let batchedImage = batchSize > 1 ? MLX.concatenated(Array(repeating: image, count: batchSize), axis: 0) : image
        let embeddings = try imageEmbedder.embedImagesGrid(batchedImage)

        // The official Shap-E implementation uses zeros for unconditional image embeddings, not a solid black image.
        let uncondEmbeddings: RuntimeTensor? = nil
        let actualBatchSize = batchedImage.shape[0]
        let latentShape = (batchSize: actualBatchSize, dLatent: imageArtifact.dLatent)

        ShapELogger.debug("[ShapE] Image embedding completed in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startEmbed))s")
        try await reportProgress(progress, "embedding", step: 1, total: 1)

        let config = KarrasSampler.Config(
            steps: numInferenceSteps,
            sigmaMin: sigmaMin,
            sigmaMax: sigmaMax,
            guidanceScale: guidanceScale,
            clipDenoised: true,
            seed: seed,
            progressInterval: progressInterval
        )

        ShapELogger.debug("[ShapE] Starting diffusion process (\(numInferenceSteps) steps)...")
        let startDiffusion = CFAbsoluteTimeGetCurrent()

        let samplerProgress: ((Int, Int, MLXArray?) -> Void)? = progress.map { p in
            return { step, total, unscaled in
                var partial: [ShapEMeshResult]? = nil
                // Sampler already channel-unscales. Partials use CPU MC to avoid GPU contention.
                if let currentUnscaledDenoised = unscaled {
                    do {
                        let dynamicParams = try self.projector.project(latent: currentUnscaledDenoised)
                        let meshBatch = try self.renderer.extractMeshes(
                            dynamicParams: dynamicParams,
                            queryBatchSize: queryBatchSize,
                            includeVertexColors: true,
                            gridSizeOverride: min(48, gridSize),
                            artifactInfo: ArtifactVersionInfo(artifactVersion: 0, manifestPath: "runtime"),
                            useMetalMarchingCubes: false
                        )
                        let output = meshBatch.meshes.map { ShapEMeshResult(meshResult: $0) }
                        for r in output {
                            MLX.eval(try r.mlxVertices(), try r.mlxFaces())
                            _ = try r.colorsArray()
                        }
                        partial = output
                        Memory.clearCache()
                    } catch {
                        ShapELogger.debug("[ShapE] Partial render failed: \(error)")
                    }
                }

                p("sampling", step, total, partial)
            }
        }

        let result = try KarrasSampler.sample(
            denoiser: imageDenoiser,
            embeddings: embeddings,
            uncondEmbeddings: uncondEmbeddings,
            shape: latentShape,
            config: config,
            progress: samplerProgress
        )

        let diffEnd = CFAbsoluteTimeGetCurrent()
        ShapELogger.debug("[ShapE] Diffusion completed in \(String(format: "%.2f", diffEnd - startDiffusion))s")

        try await reportProgress(progress, "projecting", step: 0, total: 1)
        ShapELogger.debug("[ShapE] Projecting latent space...")
        let startProject = CFAbsoluteTimeGetCurrent()
        let dynamicParams = try projector.project(latent: result.x)
        MLX.eval(Array(dynamicParams.values))
        try await reportProgress(progress, "projecting", step: 1, total: 1)
        ShapELogger.debug("[ShapE] Projection completed in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startProject))s")

        try await reportProgress(progress, "volume", step: 0, total: 1)
        ShapELogger.debug("[ShapE] Extracting mesh via Marching Cubes (grid size: \(gridSize))...")
        let startExtract = CFAbsoluteTimeGetCurrent()
        let extractProgress: STFMeshRenderer.MeshExtractProgress? = progress.map { p in
            { phase, step, total in
                p(phase, step, total, nil)
            }
        }
        let meshBatch = try renderer.extractMeshes(
            dynamicParams: dynamicParams,
            queryBatchSize: queryBatchSize,
            includeVertexColors: true,
            gridSizeOverride: gridSize,
            artifactInfo: ArtifactVersionInfo(artifactVersion: 0, manifestPath: "runtime"),
            useMetalMarchingCubes: useMetalMarchingCubes,
            progress: extractProgress
        )
        let extEnd = CFAbsoluteTimeGetCurrent()
        ShapELogger.debug("[ShapE] Mesh extraction completed in \(String(format: "%.2f", extEnd - startExtract))s")

        guard !meshBatch.meshes.isEmpty else {
            throw MeshAssemblyError.emptyExtraction
        }

        ShapELogger.debug("[ShapE] Pre-warming geometry arrays...")
        try await reportProgress(progress, "resolving", step: 0, total: 1)
        let startResolve = CFAbsoluteTimeGetCurrent()
        let output = meshBatch.meshes.map { ShapEMeshResult(meshResult: $0) }
        
        for result in output {
            MLX.eval(try result.mlxVertices(), try result.mlxFaces())
            if let rgbs = try result.colorsArray() {
                _ = rgbs
            }
        }
        try await reportProgress(progress, "resolving", step: 1, total: 1)
        ShapELogger.debug("[ShapE] Array resolution completed in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startResolve))s")

        ShapELogger.debug("[ShapE] Total generation time: \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - startGenerate))s")
        return output
    }
}

public enum MeshAssemblyError: Error, LocalizedError {
    case emptyExtraction
    
    public var errorDescription: String? {
        switch self {
        case .emptyExtraction:
            return "Marching cubes sequence evaluated and returned 0 mesh blocks without exception faults."
        }
    }
}
