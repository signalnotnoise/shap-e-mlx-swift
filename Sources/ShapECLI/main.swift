import Foundation
import MLX
import ShapERuntime

// MARK: - Minimal .npy (C-order, little-endian) writer for parity dumps

enum NpyError: Error, LocalizedError {
    case invalidFile(String)
    case missingShape(String)

    var errorDescription: String? {
        switch self {
        case .invalidFile(let m), .missingShape(let m): return m
        }
    }
}

func writeNpyFloat32(path: URL, shape: [Int], floats: [Float]) throws {
    var headerBody = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
    if shape.isEmpty {
        headerBody += ")"
    } else if shape.count == 1 {
        headerBody += "\(shape[0]),)"
    } else {
        headerBody += shape.map(String.init).joined(separator: ", ") + ")"
    }
    headerBody += ", }"
    let prefix = 10
    var header = headerBody
    let pad = (16 - ((prefix + header.utf8.count + 1) % 16)) % 16
    header += String(repeating: " ", count: pad) + "\n"
    let hlen = header.utf8.count

    var raw = Data(count: floats.count * MemoryLayout<Float>.size)
    raw.withUnsafeMutableBytes { dst in
        floats.withUnsafeBufferPointer { src in
            guard let d = dst.baseAddress, let s = src.baseAddress else { return }
            d.copyMemory(from: s, byteCount: floats.count * MemoryLayout<Float>.size)
        }
    }

    var data = Data()
    data.append(0x93)
    data.append(contentsOf: Array("NUMPY".utf8))
    data.append(contentsOf: [0x01, 0x00])
    data.append(UInt8(hlen & 0xFF))
    data.append(UInt8((hlen >> 8) & 0xFF))
    data.append(contentsOf: Array(header.utf8))
    data.append(raw)
    try data.write(to: path)
}

func writeNpyInt32(path: URL, shape: [Int], values: [Int32]) throws {
    var headerBody = "{'descr': '<i4', 'fortran_order': False, 'shape': ("
    if shape.isEmpty {
        headerBody += ")"
    } else if shape.count == 1 {
        headerBody += "\(shape[0]),)"
    } else {
        headerBody += shape.map(String.init).joined(separator: ", ") + ")"
    }
    headerBody += ", }"
    let prefix = 10
    var header = headerBody
    let pad = (16 - ((prefix + header.utf8.count + 1) % 16)) % 16
    header += String(repeating: " ", count: pad) + "\n"
    let hlen = header.utf8.count

    var raw = Data(count: values.count * MemoryLayout<Int32>.size)
    raw.withUnsafeMutableBytes { dst in
        values.withUnsafeBufferPointer { src in
            guard let d = dst.baseAddress, let s = src.baseAddress else { return }
            d.copyMemory(from: s, byteCount: values.count * MemoryLayout<Int32>.size)
        }
    }

    var data = Data()
    data.append(0x93)
    data.append(contentsOf: Array("NUMPY".utf8))
    data.append(contentsOf: [0x01, 0x00])
    data.append(UInt8(hlen & 0xFF))
    data.append(UInt8((hlen >> 8) & 0xFF))
    data.append(contentsOf: Array(header.utf8))
    data.append(raw)
    try data.write(to: path)
}

func writeMLXArray(_ path: URL, _ array: MLXArray) throws {
    eval(array)
    let shape = array.shape
    let floats = array.asArray(Float.self)
    try writeNpyFloat32(path: path, shape: shape, floats: floats)
}

func loadNpyFloat32(path: URL) throws -> (shape: [Int], data: [Float]) {
    let data = try Data(contentsOf: path)
    guard data.count > 10 else { throw NpyError.invalidFile("too short") }
    let hlen = Int(data[8]) | (Int(data[9]) << 8)
    let headerStart = 10
    let headerEnd = headerStart + hlen
    guard headerEnd <= data.count else { throw NpyError.invalidFile("header overflow") }
    let header = String(data: data.subdata(in: headerStart..<headerEnd), encoding: .ascii) ?? ""
    guard let shapeRange = header.range(of: "shape': (") ?? header.range(of: "shape\": (") else {
        throw NpyError.missingShape(header)
    }
    let after = header[shapeRange.upperBound...]
    guard let close = after.firstIndex(of: ")") else { throw NpyError.missingShape(header) }
    let inner = after[..<close]
        .replacingOccurrences(of: " ", with: "")
        .trimmingCharacters(in: CharacterSet(charactersIn: ","))
    let shape: [Int]
    if inner.isEmpty {
        shape = []
    } else {
        shape = inner.split(separator: ",").compactMap { Int($0) }
    }
    let payload = data.subdata(in: headerEnd..<data.count)
    let count = payload.count / 4
    var floats = [Float](repeating: 0, count: count)
    _ = floats.withUnsafeMutableBytes { dst in
        payload.copyBytes(to: dst)
    }
    return (shape, floats)
}

/// Match Python `re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")`.
func sanitizeParamKeyPython(_ key: String) -> String {
    var out = ""
    var lastUnderscore = false
    for ch in key {
        if ch.isLetter || ch.isNumber {
            out.append(ch)
            lastUnderscore = false
        } else if !lastUnderscore {
            out.append("_")
            lastUnderscore = true
        }
    }
    return out.trimmingCharacters(in: CharacterSet(charactersIn: "_"))
}

// MARK: - Case JSON (matches RuntimeGoldenCase / dump_mlx_swift_parity)

struct ParityCase: Codable {
    var prompt: String
    var seed: Int
    var guidance_scale: Double
    var steps: Int
    var sigma_min: Double
    var sigma_max: Double
    var mesh_grid_size: Int
    var query_batch_size: Int
}

func resolveRepoRoot() -> URL {
    // Prefer walking up from CWD looking for runtime_bundle or shap_e/parity
    var url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    for _ in 0..<8 {
        let bundle = url.appendingPathComponent("runtime_bundle")
        let parity = url.appendingPathComponent("shap_e/parity")
        if FileManager.default.fileExists(atPath: bundle.path)
            || FileManager.default.fileExists(atPath: parity.path)
        {
            return url
        }
        let parent = url.deletingLastPathComponent()
        if parent.path == url.path { break }
        url = parent
    }
    return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
}

func resolveParityDir(cli: String?) -> URL {
    if let cli {
        return URL(fileURLWithPath: cli, isDirectory: true).standardizedFileURL
    }
    let root = resolveRepoRoot()
    let candidates = [
        root.appendingPathComponent("shap_e/parity"),
        URL(fileURLWithPath: "shap_e/parity", isDirectory: true),
        URL(fileURLWithPath: "parity", isDirectory: true),
        URL(fileURLWithPath: "../parity", isDirectory: true),
    ]
    for c in candidates {
        let u = c.standardizedFileURL
        if FileManager.default.fileExists(atPath: u.path) { return u }
    }
    let u = root.appendingPathComponent("shap_e/parity")
    try? FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
    return u
}

func resolveBundle(cli: String?) -> URL? {
    if let cli {
        let u = URL(fileURLWithPath: cli, isDirectory: true)
        if FileManager.default.fileExists(atPath: u.appendingPathComponent("manifest.json").path) {
            return u
        }
    }
    let root = resolveRepoRoot()
    let candidates = [
        root.appendingPathComponent("runtime_bundle"),
        URL(fileURLWithPath: "runtime_bundle", isDirectory: true),
        URL(fileURLWithPath: "../runtime_bundle", isDirectory: true),
        URL(fileURLWithPath: "../../runtime_bundle", isDirectory: true),
    ]
    return candidates.map(\.standardizedFileURL).first {
        FileManager.default.fileExists(atPath: $0.appendingPathComponent("manifest.json").path)
    }
}

func resolveBPE(cli: String?) -> URL? {
    if let cli {
        let u = URL(fileURLWithPath: cli)
        if FileManager.default.fileExists(atPath: u.path) { return u }
    }
    let root = resolveRepoRoot()
    let candidates = [
        root.appendingPathComponent(".mlx_venv/lib/python3.12/site-packages/clip/bpe_simple_vocab_16e6.txt.gz"),
        root.appendingPathComponent(".mlx_venv/lib/python3.14/site-packages/clip/bpe_simple_vocab_16e6.txt.gz"),
        root.appendingPathComponent(".mlx_venv/lib/python3.11/site-packages/clip/bpe_simple_vocab_16e6.txt.gz"),
        URL(fileURLWithPath: "bpe_simple_vocab_16e6.txt.gz"),
    ]
    // Also scan site-packages for any python version
    let venvLib = root.appendingPathComponent(".mlx_venv/lib")
    if let pyDirs = try? FileManager.default.contentsOfDirectory(at: venvLib, includingPropertiesForKeys: nil) {
        for py in pyDirs where py.lastPathComponent.hasPrefix("python") {
            let bpe = py.appendingPathComponent("site-packages/clip/bpe_simple_vocab_16e6.txt.gz")
            if FileManager.default.fileExists(atPath: bpe.path) { return bpe }
        }
    }
    return candidates.map(\.standardizedFileURL).first {
        FileManager.default.fileExists(atPath: $0.path)
    }
}

func loadCase(parityDir: URL) throws -> ParityCase {
    let path = parityDir.appendingPathComponent("case.json")
    if FileManager.default.fileExists(atPath: path.path) {
        let data = try Data(contentsOf: path)
        return try JSONDecoder().decode(ParityCase.self, from: data)
    }
    // Fallback matches tests/fixtures/runtime_golden_a_red_chair_steps2.json
    return ParityCase(
        prompt: "a red chair",
        seed: 7,
        guidance_scale: 15.0,
        steps: 2,
        sigma_min: 0.001,
        sigma_max: 160.0,
        mesh_grid_size: 16,
        query_batch_size: 2048
    )
}

func runParityDump(bundleURL: URL, bpeURL: URL, parityDir: URL, useMetalMC: Bool) throws {
    try FileManager.default.createDirectory(at: parityDir, withIntermediateDirectories: true)
    let caseConfig = try loadCase(parityDir: parityDir)
    print("Parity dir: \(parityDir.path)")
    print("Bundle:     \(bundleURL.path)")
    print("BPE:        \(bpeURL.path)")
    print(
        "Case: prompt=\(caseConfig.prompt) seed=\(caseConfig.seed) steps=\(caseConfig.steps) "
            + "grid=\(caseConfig.mesh_grid_size) sigma=[\(caseConfig.sigma_min), \(caseConfig.sigma_max)]"
    )

    Memory.cacheLimit = 512 * 1024 * 1024

    print("Loading ShapERuntime...")
    let runtime = try ShapERuntime(modelBundleURL: bundleURL, bpeVocabularyURL: bpeURL)

    // Stage 1: tokenize
    print("Stage 1/6: tokenize...")
    let tokenRows = try runtime.tokenizer.tokenize(
        [caseConfig.prompt],
        contextLength: runtime.embedder.artifact.contextLength
    )
    let tokenFlat = tokenRows[0]
    try writeNpyInt32(
        path: parityDir.appendingPathComponent("swift_token_ids.npy"),
        shape: [1, tokenFlat.count],
        values: tokenFlat
    )
    print("  token_ids [1, \(tokenFlat.count)]")

    // Stage 2: CLIP embedding
    print("Stage 2/6: CLIP embed...")
    let embeddings = try runtime.embedder.embedText([caseConfig.prompt])
    // Match Python karras_sample_heun_mlx: uncond branch is zeros_like(embeddings), not embed("").
    // Production generate() still uses empty-string embedding; parity dumps follow MLX Python.
    let embMLX = try embeddings.asMLX().asType(.float32)
    eval(embMLX)
    let uncondZeros = try RuntimeTensor(
        name: "uncond_zeros",
        shape: embeddings.shape,
        dataType: .float32,
        data: Data(count: embeddings.elementCount * 4)
    )
    try writeMLXArray(parityDir.appendingPathComponent("swift_clip_embedding.npy"), embMLX)
    print("  clip_embedding \(embMLX.shape) (CFG uncond = zeros, matching Python MLX)")

    // Stage 3: Karras sample — share NumPy init_x from Python dump when present
    print("Stage 3/6: Karras sample (\(caseConfig.steps) steps)...")
    let config = KarrasSampler.Config(
        steps: caseConfig.steps,
        sigmaMin: Float(caseConfig.sigma_min),
        sigmaMax: Float(caseConfig.sigma_max),
        guidanceScale: Float(caseConfig.guidance_scale),
        clipDenoised: true,
        seed: caseConfig.seed
    )
    let shape = (batchSize: 1, dLatent: runtime.textArtifact.dLatent)
    var initX: MLXArray? = nil
    let initPath = parityDir.appendingPathComponent("python_init_x.npy")
    if FileManager.default.fileExists(atPath: initPath.path) {
        let (initShape, initData) = try loadNpyFloat32(path: initPath)
        initX = MLXArray(initData).reshaped(initShape)
        print("  using shared python_init_x.npy \(initShape)")
    } else {
        print("  WARNING: python_init_x.npy missing; using MLXRandom (latent will not match Python)")
    }
    let sample = try KarrasSampler.sample(
        denoiser: runtime.denoiser,
        embeddings: embeddings,
        uncondEmbeddings: uncondZeros,
        shape: shape,
        config: config,
        initX: initX,
        progress: nil
    )
    eval(sample.x, sample.predXStart, sample.sigmas)
    try writeMLXArray(parityDir.appendingPathComponent("swift_latent.npy"), sample.x)
    try writeMLXArray(parityDir.appendingPathComponent("swift_pred_xstart.npy"), sample.predXStart)
    try writeMLXArray(parityDir.appendingPathComponent("swift_sigmas.npy"), sample.sigmas)
    print("  latent \(sample.x.shape) sigmas \(sample.sigmas.shape)")

    // Stage 4: decoder params
    print("Stage 4/6: decoder params...")
    let params = try runtime.projector.project(latent: sample.x)
    for (key, value) in params.sorted(by: { $0.key < $1.key }) {
        eval(value)
        let safe = sanitizeParamKeyPython(key)
        try writeMLXArray(
            parityDir.appendingPathComponent("swift_params_\(safe).npy"),
            value
        )
        print("  params \(key) -> \(value.shape)")
    }

    // Stage 5: SDF field
    print("Stage 5/6: SDF field (grid \(caseConfig.mesh_grid_size))...")
    let (fields, _) = try runtime.renderer.extractSDFFields(
        dynamicParams: params,
        queryBatchSize: caseConfig.query_batch_size,
        gridSizeOverride: caseConfig.mesh_grid_size
    )
    eval(fields)
    let field0 = fields[0]
    try writeMLXArray(parityDir.appendingPathComponent("swift_sdf_field.npy"), field0)
    print("  sdf_field \(field0.shape)")

    // Stage 6: mesh (CPU MC by default)
    print("Stage 6/6: mesh extract (metal=\(useMetalMC))...")
    let meshBatch = try runtime.renderer.extractMeshes(
        dynamicParams: params,
        queryBatchSize: caseConfig.query_batch_size,
        includeVertexColors: false,
        gridSizeOverride: caseConfig.mesh_grid_size,
        artifactInfo: ArtifactVersionInfo(artifactVersion: 0, manifestPath: "parity"),
        useMetalMarchingCubes: useMetalMC
    )
    guard let mesh = meshBatch.meshes.first else {
        print("  empty mesh")
        try writeNpyFloat32(
            path: parityDir.appendingPathComponent("swift_verts.npy"),
            shape: [0, 3],
            floats: []
        )
        try writeNpyInt32(
            path: parityDir.appendingPathComponent("swift_faces.npy"),
            shape: [0, 3],
            values: []
        )
        return
    }
    let verts = try mesh.vertices.float32Values()
    let faces = try mesh.faces.int32Values()
    let vShape = mesh.vertices.shape
    let fShape = mesh.faces.shape
    try writeNpyFloat32(
        path: parityDir.appendingPathComponent("swift_verts.npy"),
        shape: vShape,
        floats: verts
    )
    try writeNpyInt32(
        path: parityDir.appendingPathComponent("swift_faces.npy"),
        shape: fShape,
        values: faces
    )
    print("  verts \(vShape) faces \(fShape)")
    print("Wrote swift_*.npy to \(parityDir.path)")
    print("Next: python -m cli.check.compare_mlx_swift_parity")
}

// MARK: - Mesh A/B (CPU vs Metal on same SDF)

func resolveOutDir(cli: String?, defaultName: String) -> URL {
    if let cli {
        let u = URL(fileURLWithPath: cli, isDirectory: true).standardizedFileURL
        try? FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        return u
    }
    let root = resolveRepoRoot()
    let u = root.appendingPathComponent(defaultName)
    try? FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
    return u
}

func runMeshAB(
    bundleURL: URL,
    bpeURL: URL,
    outDir: URL,
    prompt: String,
    seed: Int,
    steps: Int,
    guidance: Float,
    grid: Int,
    queryBatch: Int,
    sigmaMin: Float,
    sigmaMax: Float,
    skipMetal: Bool,
    skipCPU: Bool,
    uncondZeros: Bool
) throws {
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let uncondMode = uncondZeros ? "zeros (Python Karras)" : "embed(\"\") (production generate)"
    print("=== Shap-E mesh A/B (same SDF → CPU MC vs Metal MC) ===")
    print("out: \(outDir.path)")
    print("prompt=\(prompt) seed=\(seed) steps=\(steps) grid=\(grid) guidance=\(guidance)")
    print("CFG uncond=\(uncondMode)")

    Memory.cacheLimit = 768 * 1024 * 1024
    let runtime = try ShapERuntime(modelBundleURL: bundleURL, bpeVocabularyURL: bpeURL)

    print("1/4 CLIP embed...")
    let embeddings = try runtime.embedder.embedText([prompt])
    let uncondEmbeddings: RuntimeTensor? = uncondZeros
        ? nil
        : try runtime.embedder.embedText([""])

    print("2/4 Karras sample (\(steps) steps)...")
    let config = KarrasSampler.Config(
        steps: steps,
        sigmaMin: sigmaMin,
        sigmaMax: sigmaMax,
        guidanceScale: guidance,
        clipDenoised: true,
        seed: seed
    )
    let sample = try KarrasSampler.sample(
        denoiser: runtime.denoiser,
        embeddings: embeddings,
        uncondEmbeddings: uncondEmbeddings,
        shape: (batchSize: 1, dLatent: runtime.textArtifact.dLatent),
        config: config,
        initX: nil,
        progress: { step, total, _ in
            if step == 0 || step + 1 == total || (step + 1) % max(1, total / 8) == 0 {
                print("  diffusion \(step + 1)/\(total)")
            }
        }
    )
    eval(sample.x)

    print("3/4 decoder + SDF (grid \(grid))...")
    let params = try runtime.projector.project(latent: sample.x)
    let (fields, _) = try runtime.renderer.extractSDFFields(
        dynamicParams: params,
        queryBatchSize: queryBatch,
        gridSizeOverride: grid
    )
    eval(fields)
    let field0 = fields[0]
    try writeMLXArray(outDir.appendingPathComponent("sdf_field.npy"), field0)
    let fMin = MLX.min(field0).item(Float.self)
    let fMax = MLX.max(field0).item(Float.self)
    print("  sdf range [\(fMin), \(fMax)] shape \(field0.shape)")

    print("4/4 marching cubes A/B on THIS sdf...")
    struct RunResult {
        var label: String
        var verts: [Float]
        var faces: [Int32]
        var stats: MeshTopologyStats
        var seconds: Double
    }
    var results: [RunResult] = []

    func runMC(label: String, metal: Bool) throws -> RunResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        let (v, f) = try runtime.renderer.meshFromSDFField(
            field: field0,
            useMetalMarchingCubes: metal
        )
        eval(v, f)
        let secs = CFAbsoluteTimeGetCurrent() - t0
        let verts = v.asArray(Float.self)
        let faces = f.asArray(Int32.self)
        let vCount = v.shape[0]
        let stats = MeshTopology.stats(vertexCount: vCount, faces: faces)
        print("  \(label): \(stats.summaryLine)  (\(String(format: "%.2f", secs))s)")
        return RunResult(label: label, verts: verts, faces: faces, stats: stats, seconds: secs)
    }

    if !skipCPU {
        let cpu = try runMC(label: "CPU", metal: false)
        results.append(cpu)
        if !cpu.verts.isEmpty {
            try MeshExport.writeOBJ(
                path: outDir.appendingPathComponent("mesh_cpu.obj"),
                vertices: cpu.verts,
                faces: cpu.faces,
                comment: "CPU MarchingCubes prompt=\(prompt) seed=\(seed) grid=\(grid)"
            )
            print("  wrote mesh_cpu.obj")
        }
    }

    if !skipMetal {
        do {
            let metal = try runMC(label: "Metal", metal: true)
            results.append(metal)
            if !metal.verts.isEmpty {
                try MeshExport.writeOBJ(
                    path: outDir.appendingPathComponent("mesh_metal.obj"),
                    vertices: metal.verts,
                    faces: metal.faces,
                    comment: "Metal MarchingCubes prompt=\(prompt) seed=\(seed) grid=\(grid)"
                )
                print("  wrote mesh_metal.obj")
            }
        } catch {
            print("  Metal MC FAILED: \(error)")
            print("  (CPU result still valid for model-quality diagnosis)")
        }
    }

    if let cpu = results.first(where: { $0.label == "CPU" }),
       let metal = results.first(where: { $0.label == "Metal" })
    {
        print(MeshTopology.compare(
            labelA: "CPU",
            statsA: cpu.stats,
            labelB: "Metal",
            statsB: metal.stats
        ))
    } else if let only = results.first {
        if only.stats.boundaryEdgeCount > 0 {
            print(
                "VERDICT: single path \(only.label) has boundary_edges=\(only.stats.boundaryEdgeCount) "
                    + "→ open surface (holes/cracks possible). Compare to Python on same seed if both CPU&Metal open."
            )
        } else {
            print("VERDICT: \(only.label) mesh is closed (no boundary edges). Visual holes may be color/shading.")
        }
    }

    // JSON report for later sessions
    var report: [String: Any] = [
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "grid": grid,
        "guidance": guidance,
        "cfg_uncond": uncondZeros ? "zeros" : "embed_empty",
        "sdf_min": fMin,
        "sdf_max": fMax,
    ]
    for r in results {
        report[r.label.lowercased()] = [
            "vertexCount": r.stats.vertexCount,
            "faceCount": r.stats.faceCount,
            "boundaryEdgeCount": r.stats.boundaryEdgeCount,
            "nonManifoldEdgeCount": r.stats.nonManifoldEdgeCount,
            "invalidFaceCount": r.stats.invalidFaceCount,
            "isClosed": r.stats.isClosed,
            "seconds": r.seconds,
        ]
    }
    let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    try json.write(to: outDir.appendingPathComponent("ab_report.json"))
    print("Wrote ab_report.json + sdf_field.npy + mesh_*.obj under \(outDir.path)")
    print("Open OBJs in Blender: wireframe + Select → Select All by Trait → Non Manifold")
}

// MARK: - CLI

struct Args {
    var parity = false
    var imageParity = false
    var skipMesh = false
    var abMesh = false
    var metalMC = false
    var skipMetal = false
    var skipCPU = false
    var uncondZeros = false
    var bundle: String?
    var bpe: String?
    var parityDir: String?
    var outDir: String?
    var prompt: String = "a ford mustang"
    var seed: Int = 859_729
    var steps: Int = 64
    var guidance: Float = 15.0
    var grid: Int = 128
    var queryBatch: Int = 65536
    var sigmaMin: Float = 0.002
    var sigmaMax: Float = 160.0
}

func parseArgs() -> Args {
    var args = Args()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        switch a {
        case "--parity":
            args.parity = true
        case "--image-parity":
            args.imageParity = true
        case "--skip-mesh":
            args.skipMesh = true
        case "--ab-mesh":
            args.abMesh = true
        case "--metal-mc":
            args.metalMC = true
        case "--skip-metal":
            args.skipMetal = true
        case "--skip-cpu":
            args.skipCPU = true
        case "--uncond-zeros":
            args.uncondZeros = true
        case "--bundle":
            args.bundle = it.next()
        case "--bpe":
            args.bpe = it.next()
        case "--parity-dir":
            args.parityDir = it.next()
        case "--out-dir":
            args.outDir = it.next()
        case "--prompt":
            args.prompt = it.next() ?? args.prompt
        case "--seed":
            args.seed = Int(it.next() ?? "") ?? args.seed
        case "--steps":
            args.steps = Int(it.next() ?? "") ?? args.steps
        case "--guidance":
            args.guidance = Float(it.next() ?? "") ?? args.guidance
        case "--grid":
            args.grid = Int(it.next() ?? "") ?? args.grid
        case "--query-batch":
            args.queryBatch = Int(it.next() ?? "") ?? args.queryBatch
        case "-h", "--help":
            print(
                """
                ShapECLI

                Modes:
                  --parity              Dump Swift tensors for MLX↔Swift parity
                  --image-parity        Dump Swift image tensors for Torch↔Swift parity
                  --ab-mesh             Same SDF → CPU vs Metal MC; OBJ + boundary stats

                Shared:
                  --bundle PATH         runtime_bundle directory
                  --bpe PATH            CLIP BPE vocab .txt.gz

                Parity:
                  --parity-dir PATH     Output directory (default shap_e/parity)
                  --metal-mc            Use Metal marching cubes for mesh stage

                Image parity:
                  --steps N             Mesh-smoke diffusion steps (default: 64)
                  --guidance F          Mesh-smoke guidance (default: 15; pass 3)
                  --grid N              Mesh-smoke grid (default: 128; pass 64)
                  --skip-mesh           Stop after component outputs

                Mesh A/B (--ab-mesh):
                  --out-dir PATH        Output (default shap_e/mesh_ab)
                  --prompt TEXT         default: "a ford mustang"
                  --seed N              default: 859729 (matches Vision UI example)
                  --steps N             default: 64
                  --guidance F          default: 15
                  --grid N              default: 128
                  --skip-metal          CPU only
                  --skip-cpu            Metal only
                  --uncond-zeros        CFG uncond = zeros (Python Karras / L2 parity).
                                        Default is embed("") to match production generate().

                Example:
                  ShapECLI --ab-mesh --prompt "a ford mustang" --seed 859729 --steps 64 --grid 128
                """
            )
            exit(0)
        default:
            fputs("Unknown argument: \(a)\n", stderr)
            exit(2)
        }
    }
    return args
}

let args = parseArgs()
do {
    guard let bundle = resolveBundle(cli: args.bundle) else {
        fputs("runtime_bundle not found. Pass --bundle PATH\n", stderr)
        exit(1)
    }
    if args.imageParity {
        let parityDir = resolveParityDir(cli: args.parityDir)
        try ImageParityDump.run(
            bundleURL: bundle,
            parityDir: parityDir,
            options: ImageParityDump.Options(
                steps: args.steps,
                guidance: args.guidance,
                grid: args.grid,
                queryBatch: args.queryBatch,
                sigmaMin: args.sigmaMin,
                sigmaMax: args.sigmaMax,
                skipMesh: args.skipMesh
            )
        )
        exit(0)
    }

    guard let bpe = resolveBPE(cli: args.bpe) else {
        fputs("CLIP BPE not found. Pass --bpe PATH to bpe_simple_vocab_16e6.txt.gz\n", stderr)
        exit(1)
    }

    if args.abMesh {
        let out = resolveOutDir(cli: args.outDir, defaultName: "shap_e/mesh_ab")
        try runMeshAB(
            bundleURL: bundle,
            bpeURL: bpe,
            outDir: out,
            prompt: args.prompt,
            seed: args.seed,
            steps: args.steps,
            guidance: args.guidance,
            grid: args.grid,
            queryBatch: args.queryBatch,
            sigmaMin: args.sigmaMin,
            sigmaMax: args.sigmaMax,
            skipMetal: args.skipMetal,
            skipCPU: args.skipCPU,
            uncondZeros: args.uncondZeros
        )
    } else if args.parity {
        let parityDir = resolveParityDir(cli: args.parityDir)
        try runParityDump(
            bundleURL: bundle,
            bpeURL: bpe,
            parityDir: parityDir,
            useMetalMC: args.metalMC
        )
    } else {
        fputs("No mode selected. Use --ab-mesh, --parity, or --image-parity (see --help).\n", stderr)
        exit(2)
    }
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
