import Foundation
import MLX
#if canImport(UIKit)
import UIKit
#endif

/// IOGPU terminates the process if we `MLX.eval` after the scene backgrounds.
private func throwIfMetalNotPermitted() throws {
    try Task.checkCancellation()
#if os(iOS) || os(visionOS)
    let state: UIApplication.State
    if Thread.isMainThread {
        state = UIApplication.shared.applicationState
    } else {
        state = DispatchQueue.main.sync { UIApplication.shared.applicationState }
    }
    if state == .background {
        throw ShapERuntimeError.gpuNotForeground
    }
#endif
}

// MARK: - STFMeshRenderer
//
// Swift port of MLXSTFMeshRenderer (MLX path only) from:
//   shap_e/src/mlx_shap_e/models/rendering/stf_mesh_renderer.py
//
// Uses existing RuntimeCoreTypes.MeshResult / MeshBatchResult.
// Covers:
//   • Grid point generation
//   • NeRF positional encoding
//   • Swish activation
//   • NeRSTF forward pass (layers 0-3 dynamic, 4-6 static)
//   • SDF grid evaluation + border padding + marching cubes

public final class STFMeshRenderer: @unchecked Sendable {

    // Config
    public let gridSize: Int
    public let directionDependentShape: Bool
    public let separateNerfChannels: Bool
    public let separateCoarseChannels: Bool
    public let channelActivation: String
    public let sdfActivation: String
    public let textureChannelNames: [String]

    // Weights (loaded once at init)
    private let bboxMin: MLXArray    // [3]
    private let bboxMax: MLXArray    // [3]
    private let staticWeights: [Int: MLXArray]   // layers 4, 5, 6
    private let staticBiases:  [Int: MLXArray]   // layers 0 – 6

    // Cache. Locked because extractMeshes can run on overlapping generate() calls.
    private let volumeCacheLock = NSLock()
    private var volumePointsCache: [Int: MLXArray] = [:]

    // Direction encoding dim: 3 + 2 * 3 * 8 = 51
    private let dirEncDim: Int = 3 + 2 * 3 * 8

    public init(artifact: RendererArtifact) throws {
        self.gridSize                 = artifact.gridSize
        self.directionDependentShape  = artifact.directionDependentShape
        self.separateNerfChannels     = artifact.separateNerfChannels
        self.separateCoarseChannels   = artifact.separateCoarseChannels
        self.channelActivation        = artifact.channelActivation
        self.sdfActivation            = artifact.sdfActivation
        self.textureChannelNames      = artifact.textureChannels

        guard artifact.bboxMin.shape == [3] else {
            throw STFRendererError.invalidPositionsShape(artifact.bboxMin.shape)
        }
        guard artifact.bboxMax.shape == [3] else {
            throw STFRendererError.invalidPositionsShape(artifact.bboxMax.shape)
        }

        self.bboxMin = try artifact.bboxMin.asMLX().asType(.float32)
        self.bboxMax = try artifact.bboxMax.asMLX().asType(.float32)

        var sw: [Int: MLXArray] = [:]
        var sb: [Int: MLXArray] = [:]
        for idx in [4, 5, 6] {
            guard let wT = artifact.staticWeights[idx] else {
                throw STFRendererError.missingStaticWeight(layerIndex: idx)
            }
            sw[idx] = try wT.asMLX().asType(.float32)
        }
        for idx in [0, 1, 2, 3, 4, 5, 6] {
            guard let bT = artifact.staticBiases[idx] else {
                throw STFRendererError.missingStaticBias(layerIndex: idx)
            }
            sb[idx] = try bT.asMLX().asType(.float32)
        }
        self.staticWeights = sw
        self.staticBiases  = sb
    }

    // MARK: - SDF evaluation

    /// Evaluate signed distance at positions [N, 3]. Returns [N, 1] float32.
    public func signedDistance(positions: MLXArray, dynamicParams: [String: MLXArray]) throws -> MLXArray {
        ShapELogger.debug("[Debug] signedDistance: positions.shape=\(positions.shape)")
        let h = try nerstfForward(positions: positions, dynamicParams: dynamicParams)
        // Use [.ellipsis, 0..<1] to correctly slice the channel axis regardless of batch rank
        return applyActivation(h[.ellipsis, 0..<1], name: sdfActivation).asType(DType.float32)
    }

    /// Evaluate texture channels at positions [N, 3]. Returns [N, C] float32.
    public func textureChannels(positions: MLXArray, dynamicParams: [String: MLXArray]) throws -> MLXArray {
        let h = try nerstfForward(positions: positions, dynamicParams: dynamicParams)
        let (start, end) = stfChannelSlice()
        // Use [.ellipsis, start..<end] to correctly slice the channel axis
        let ch = applyActivation(h[.ellipsis, start..<end], name: channelActivation)
        return MLX.clip(ch.asType(DType.float32), min: Float(0.0), max: Float(1.0))
    }

    // MARK: - Mesh extraction

    public func extractMesh(
        dynamicParams: [String: MLXArray],
        queryBatchSize: Int = 65536,
        includeVertexColors: Bool = false,
        gridSizeOverride: Int? = nil,
        artifactInfo: ArtifactVersionInfo = ArtifactVersionInfo(artifactVersion: 1, manifestPath: "stf_renderer")
    ) throws -> MeshResult {
        let result = try extractMeshes(
            dynamicParams: dynamicParams,
            queryBatchSize: queryBatchSize,
            includeVertexColors: includeVertexColors,
            gridSizeOverride: gridSizeOverride,
            artifactInfo: artifactInfo
        )
        guard result.meshes.count == 1 else {
            throw STFRendererError.unexpectedBatchSize(result.meshes.count)
        }
        return result.meshes[0]
    }

    /// Evaluate unpadded SDF volumes for each batch item.
    ///
    /// Progress during final mesh assembly: phase is one of
    /// `volume` (SDF grid), `meshing` (marching cubes), `colors` (vertex textures).
    public typealias MeshExtractProgress = (_ phase: String, _ step: Int, _ totalSteps: Int) -> Void

    /// Matches: `extract_sdf_fields` in stf_mesh_renderer.py
    /// - Returns: float32 `[B, G, G, G]` (before the -1 border pad used by MC) and batch size.
    public func extractSDFFields(
        dynamicParams: [String: MLXArray],
        queryBatchSize: Int = 65536,
        gridSizeOverride: Int? = nil,
        progress: MeshExtractProgress? = nil
    ) throws -> (fields: MLXArray, batchSize: Int) {
        let activeGrid = gridSizeOverride ?? gridSize
        guard activeGrid >= 2 else { throw STFRendererError.invalidGridSize(activeGrid) }
        guard queryBatchSize > 0 else { throw STFRendererError.invalidQueryBatchSize(queryBatchSize) }

        let batchSize = try dynamicBatchSize(dynamicParams)
        let points = volumeQueryPoints(gridSize: activeGrid) // [G^3, 3]
        let pointCount = points.shape[0]
        let chunkCount = max(1, Int(ceil(Double(pointCount) / Double(max(queryBatchSize, 1)))))

        var sdfBatches: [MLXArray] = []
        var i = 0
        var chunkIndex = 0
        while i < pointCount {
            let j = min(i + queryBatchSize, pointCount)
            chunkIndex += 1
            try throwIfMetalNotPermitted()
            var chunk = try signedDistance(positions: points[i..<j], dynamicParams: dynamicParams)

            if chunk.ndim == 2 {
                chunk = chunk.expandedDimensions(axes: [0])
            }
            if chunk.shape[0] == 1 && batchSize > 1 {
                chunk = MLX.broadcast(chunk, to: [batchSize, chunk.shape[1], chunk.shape[2]])
            }

            MLX.eval(chunk)
            progress?("volume", chunkIndex, chunkCount)
            sdfBatches.append(chunk)
            i = j
        }

        let sdfValues: MLXArray = sdfBatches.isEmpty
            ? MLX.zeros([batchSize, 0, 1], type: Float.self)
            : MLX.concatenated(sdfBatches, axis: 1)

        let fields = sdfValues
            .squeezed(axes: [-1])
            .reshaped(batchSize, activeGrid, activeGrid, activeGrid)
            .asType(.float32)
        return (fields: fields, batchSize: batchSize)
    }

    /// Marching-cubes a single unpadded SDF field `[G,G,G]` (adds -1 border pad).
    ///
    /// Use the **same** `field` with `useMetalMarchingCubes` true/false to A/B Metal vs CPU
    /// without re-running diffusion or SDF eval.
    public func meshFromSDFField(
        field: MLXArray,
        useMetalMarchingCubes: Bool = false
    ) throws -> (verts: MLXArray, faces: MLXArray) {
        guard field.ndim == 3 else {
            throw STFRendererError.invalidPositionsShape(field.shape)
        }
        let padded = MLX.padded(
            field,
            widths: [IntOrPair(1), IntOrPair(1), IntOrPair(1)],
            mode: .constant,
            value: MLXArray(Float(-1.0))
        ).asType(.float32)

        let vMin = MLX.min(padded).item(Float.self)
        let vMax = MLX.max(padded).item(Float.self)
        guard vMin <= 0.0 && vMax >= 0.0 else {
            return (
                MLX.zeros([0, 3], type: Float.self),
                MLX.zeros([0, 3], type: Int32.self)
            )
        }

        let bboxSizeArr = bboxMax - bboxMin
        if useMetalMarchingCubes {
            return try MetalMarchingCubes.extract(
                sdf: padded,
                bboxMin: bboxMin,
                bboxSize: bboxSizeArr
            )
        }
        return MarchingCubes.extract(
            sdf: padded,
            bboxMin: bboxMin,
            bboxSize: bboxSizeArr
        )
    }

    /// Batched mesh extraction for `batchSize` items.
    ///
    /// Matches: `extract_meshes` in stf_mesh_renderer.py
    ///
    /// - Parameter progress: Optional sub-progress for long final extracts
    ///   (`volume` → `meshing` → `colors`). Omit for silent partial previews.
    public func extractMeshes(
        dynamicParams: [String: MLXArray],
        queryBatchSize: Int = 65536,
        includeVertexColors: Bool = false,
        gridSizeOverride: Int? = nil,
        artifactInfo: ArtifactVersionInfo = ArtifactVersionInfo(artifactVersion: 1, manifestPath: "stf_renderer"),
        useMetalMarchingCubes: Bool = false,
        progress: MeshExtractProgress? = nil
    ) throws -> MeshBatchResult {
        let activeGrid = gridSizeOverride ?? gridSize
        guard activeGrid >= 2 else { throw STFRendererError.invalidGridSize(activeGrid) }
        guard queryBatchSize > 0 else { throw STFRendererError.invalidQueryBatchSize(queryBatchSize) }

        let (fields, batchSize) = try extractSDFFields(
            dynamicParams: dynamicParams,
            queryBatchSize: queryBatchSize,
            gridSizeOverride: gridSizeOverride,
            progress: progress
        )

        var meshes: [MeshResult] = []

        for b in 0..<batchSize {
            progress?("meshing", b + 1, max(batchSize, 1))

            let (verts, faces) = try meshFromSDFField(
                field: fields[b],
                useMetalMarchingCubes: useMetalMarchingCubes
            )
            guard faces.shape[0] > 0 else {
                meshes.append(try emptyMeshResult(artifactInfo: artifactInfo))
                continue
            }

            var vertChannels: [String: RuntimeTensor]? = nil
            if includeVertexColors && verts.shape[0] > 0 {
                let paramsI = dynamicParams.mapValues { $0[b..<(b+1)] }
                ShapELogger.debug("[Debug] extractMeshes: evaluating texture for \(verts.shape[0]) vertices in chunks of \(queryBatchSize)")
                
                let vertCount = verts.shape[0]
                let colorChunkCount = max(1, Int(ceil(Double(vertCount) / Double(max(queryBatchSize, 1)))))
                var chunks: [MLXArray] = []
                var i = 0
                var colorChunkIndex = 0
                while i < vertCount {
                    let j = min(i + queryBatchSize, vertCount)
                    colorChunkIndex += 1
                    try throwIfMetalNotPermitted()
                    let chunk = try textureChannels(positions: verts[i..<j], dynamicParams: paramsI)
                    MLX.eval(chunk)
                    progress?("colors", colorChunkIndex, colorChunkCount)
                    chunks.append(chunk)
                    //TODO: really investigate Memory.clearCache()
                    i = j
                }
                
                let ch = MLX.concatenated(chunks, axis: 0) // [V, C]
                var chDict: [String: RuntimeTensor] = [:]
                
                if textureChannelNames == ["R", "G", "B"] {
                    let r = ch[.ellipsis, 0..<1]
                    let g = ch[.ellipsis, 1..<2]
                    let b = ch[.ellipsis, 2..<3]
                    let alpha = MLX.ones(r.shape, type: Float.self)
                    
                    let rgba = MLX.concatenated([r, g, b, alpha], axis: 1).flattened().asType(.float32)
                    let rgbaArray = rgba.asArray(Float.self)
                    chDict["RGBA"] = try RuntimeTensor(
                        name: "RGBA", shape: [rgbaArray.count], dataType: .float32,
                        data: rgbaArray.withUnsafeBufferPointer { Data(buffer: $0) }
                    )
                } else {
                    for (ci, name) in textureChannelNames.enumerated() {
                        let col = ch[.ellipsis, ci..<(ci+1)].flattened().asType(DType.float32)
                        let fc  = col.asArray(Float.self)
                        chDict[name] = try RuntimeTensor(name: name, shape: [fc.count], dataType: .float32,
                                                         data: fc.withUnsafeBufferPointer { buffer in Data(buffer: buffer) })
                    }
                }
                vertChannels = chDict
            }

            let vf  = verts.asType(.float32).asArray(Float.self)
            let fi  = faces.asType(.int32).asArray(Int32.self)
            let vT  = try RuntimeTensor(name: "verts", shape: verts.shape, dataType: .float32,
                                        data: vf.withUnsafeBufferPointer { Data(buffer: $0) })
            let fT  = try RuntimeTensor(name: "faces", shape: faces.shape, dataType: .int32,
                                        data: fi.withUnsafeBufferPointer { Data(buffer: $0) })

            let bboxMinF = bboxMin.asArray(Float.self)
            let bboxMaxF = bboxMax.asArray(Float.self)
            let bbox = BoundingBox(minCorner: bboxMinF, maxCorner: bboxMaxF)
            meshes.append(MeshResult(
                vertices: vT, faces: fT,
                vertexChannels: vertChannels,
                bbox: bbox,
                timings: [:],
                artifactInfo: artifactInfo
            ))
        }

        let bboxMinF = bboxMin.asArray(Float.self)
        let bboxMaxF = bboxMax.asArray(Float.self)
        let bbox = BoundingBox(minCorner: bboxMinF, maxCorner: bboxMaxF)
        return MeshBatchResult(meshes: meshes, bbox: bbox, timings: [:], artifactInfo: artifactInfo)
    }

    // MARK: - NeRSTF forward pass

    /// Core NeRSTF MLP: posenc → 4 dynamic meta-linear layers → dir concat → 3 static linear layers.
    ///
    /// Matches: `_nerstf_forward_mx`
    func nerstfForward(positions: MLXArray, dynamicParams: [String: MLXArray]) throws -> MLXArray {
        let batchSize  = try dynamicBatchSize(dynamicParams)
        var pos        = positions.asType(.float32)
        // If batchSize > 1 and positions are NOT batch-aware, we need to broadcast positions
        // so that each batch item gets the full point grid.
        let squeezeOut = pos.ndim == 2 && batchSize == 1
        ShapELogger.debug("[Debug] nerstfForward: pos.shape=\(pos.shape), batchSize=\(batchSize)")

        if pos.ndim == 2 {
            // [V, 3] -> [1, V, 3]
            pos = pos.expandedDimensions(axes: [0])
            if batchSize > 1 {
                pos = MLX.broadcast(pos, to: [batchSize, pos.shape[1], 3])
            }
        } else if pos.ndim == 3 {
            if pos.shape[0] == 1 && batchSize > 1 {
                pos = MLX.broadcast(pos, to: [batchSize, pos.shape[1], 3])
            } else if pos.shape[0] != batchSize {
                throw STFRendererError.batchMismatch(posBatch: pos.shape[0], paramsBatch: batchSize)
            }
        } else {
            throw STFRendererError.invalidPositionsShape(pos.shape)
        }

        // pos is now [B, N, 3]
        var h = posEncNerf(pos, minDeg: 0, maxDeg: 15)

        for idx in [0, 1, 2, 3] {
            guard let w = dynamicParams["nerstf.mlp.\(idx).weight"],
                  let b = staticBiases[idx] else {
                throw STFRendererError.missingDynamicWeight(layerIndex: idx)
            }
            ShapELogger.debug("[Debug] nerstfForward layer \(idx): h.shape=\(h.shape), w.shape=\(w.shape), b.shape=\(b.shape)")
            // Ensure bias matches the 3D rank of h and w
            h = metaLinear(h: h, w: w, b: b.reshaped(1, 1, b.shape[0]))
            h = swish(h)
        }

        let N      = h.shape[1]
        let dirEnc = MLX.zeros([batchSize, N, dirEncDim], type: Swift.Float.self)
        h = MLX.concatenated([h, dirEnc], axis: -1).asType(.float32)

        for idx in [4, 5, 6] {
            let w = staticWeights[idx]!
            let b = staticBiases[idx]!
            ShapELogger.debug("[Debug] nerstfForward static layer \(idx): h.shape=\(h.shape), w.shape=\(w.shape), b.shape=\(b.shape)")
            h = MLX.matmul(h, w.transposed(axes: [1, 0])) + b.reshaped(1, 1, b.shape[0])
            if idx < 6 { h = swish(h) }
        }

        h = h.asType(.float32)
        return squeezeOut ? h[0] : h
    }

    // MARK: - Math helpers

    /// NeRF positional encoding. Matches: `_posenc_nerf_mx`
    func posEncNerf(_ x: MLXArray, minDeg: Int, maxDeg: Int) -> MLXArray {
        guard minDeg != maxDeg else { return x.asType(.float32) }
        let degs    = (minDeg..<maxDeg).map { Float(pow(2.0, Double($0))) }
        let scales  = MLXArray(degs)                              // [D]
        let numDegs = maxDeg - minDeg
        let lastDim = x.shape.last!

        // x: [..., lastDim] → [..., 1, lastDim] * [numDegs, 1] → [..., numDegs*lastDim]
        let xExp = x.expandedDimensions(axes: [-2])
        var scaleShape = [Int](repeating: 1, count: xExp.ndim)
        scaleShape[xExp.ndim - 2] = numDegs
        let scalesR = scales.reshaped(scaleShape)
        let leadingShape = Array(x.shape.dropLast())
        let xb = (xExp * scalesR).reshaped(leadingShape + [numDegs * lastDim])
        let emb = MLX.sin(MLX.concatenated([xb, xb + Float.pi / 2.0], axis: -1))
        return MLX.concatenated([x, emb], axis: -1).asType(.float32)
    }

    /// Swish: x * sigmoid(clip(x, -60, 60)). Matches: `_swish_mx`
    func swish(_ x: MLXArray) -> MLXArray {
        let xc = MLX.clip(x, min: Float(-60.0), max: Float(60.0))
        return x * (Float(1.0) / (Float(1.0) + MLX.exp(-xc)))
    }

    /// Meta-linear: matmul(h, wT) + b. Matches: `_meta_linear_mx`
    func metaLinear(h: MLXArray, w: MLXArray, b: MLXArray) -> MLXArray {
        // h: [B, N, D_in], w: [B, D_out, D_in]
        // result: [B, N, D_out]
        
        let wT = w.transposed(axes: [0, 2, 1]) // [B, D_in, D_out]
        
        // MLX matmul handles [B, N, Di] @ [B, Di, Do] correctly
        let res = MLX.matmul(h, wT)
        
        // Ensure bias is broadcastable: [1, 1, Do]
        let bias = b.ndim == 1 ? b.reshaped(1, 1, b.shape[0]) : b
        return res + bias
    }

    func applyActivation(_ x: MLXArray, name: String) -> MLXArray {
        switch name {
        case "sigmoid":  return Float(1.0) / (Float(1.0) + MLX.exp(-x))
        case "tanh":     return MLX.tanh(x)
        default:         return x
        }
    }

    // MARK: - Volume grid

    /// Dense [G^3, 3] world-space query grid. Matches: `_volume_query_points_mx`
    public func volumeQueryPoints(gridSize G: Int) -> MLXArray {
        precondition(G > 1, "gridSize must be greater than 1")
        volumeCacheLock.lock()
        defer { volumeCacheLock.unlock() }
        if let cached = volumePointsCache[G] { return cached }

        
        let squared = G * G
        let total =  squared * G
        let indices = MLX.arange(total, dtype: .int32)

        let zs = (indices % G).asType(.float32)
        let ys = (indices.floorDivide(G) % G).asType(.float32)
        let xs = (indices.floorDivide(squared)).asType(.float32)

        let grid = MLX.stacked([xs, ys, zs], axis: 1)

        let scale = (bboxMax - bboxMin)/MLXArray(Float(G - 1))
        let pts = grid  * scale + bboxMin

        volumePointsCache[G] = pts
        return pts
    }

    // MARK: - Utilities

    private func stfChannelSlice() -> (Int, Int) {
        if directionDependentShape { return separateCoarseChannels ? (3, 6) : (2, 5) }
        return (0, 3)
    }

    private func dynamicBatchSize(_ params: [String: MLXArray]) throws -> Int {
        for v in params.values where v.ndim > 0 { return v.shape[0] }
        throw STFRendererError.emptyDynamicParams
    }

    private func emptyMeshResult(artifactInfo: ArtifactVersionInfo) throws -> MeshResult {
        let emptyVerts = try RuntimeTensor(name: "verts", shape: [0, 3], dataType: .float32, data: Data())
        let emptyFaces = try RuntimeTensor(name: "faces", shape: [0, 3], dataType: .int32,   data: Data())
        let bboxMinF = bboxMin.asArray(Float.self)
        let bboxMaxF = bboxMax.asArray(Float.self)
        let bbox = BoundingBox(minCorner: bboxMinF, maxCorner: bboxMaxF)
        return MeshResult(vertices: emptyVerts, faces: emptyFaces, vertexChannels: nil,
                          bbox: bbox, timings: [:], artifactInfo: artifactInfo)
    }
}

// MARK: - Errors

public enum STFRendererError: Error, LocalizedError {
    case missingStaticWeight(layerIndex: Int)
    case missingStaticBias(layerIndex: Int)
    case missingDynamicWeight(layerIndex: Int)
    case invalidGridSize(Int)
    case invalidQueryBatchSize(Int)
    case unexpectedBatchSize(Int)
    case batchMismatch(posBatch: Int, paramsBatch: Int)
    case invalidPositionsShape([Int])
    case emptyDynamicParams

    public var errorDescription: String? {
        switch self {
        case let .missingStaticWeight(i):   return "STFMeshRenderer: missing static weight for layer \(i)."
        case let .missingStaticBias(i):     return "STFMeshRenderer: missing static bias for layer \(i)."
        case let .missingDynamicWeight(i):  return "STFMeshRenderer: missing 'nerstf.mlp.\(i).weight'."
        case let .invalidGridSize(g):       return "STFMeshRenderer: gridSize must be ≥ 2, got \(g)."
        case let .invalidQueryBatchSize(n): return "STFMeshRenderer: queryBatchSize must be > 0, got \(n)."
        case let .unexpectedBatchSize(b):   return "STFMeshRenderer: extractMesh needs B=1, got \(b)."
        case let .batchMismatch(p, d):      return "STFMeshRenderer: positions batch \(p) ≠ params batch \(d)."
        case let .invalidPositionsShape(s): return "STFMeshRenderer: positions must be [N,3] or [B,N,3], got \(s)."
        case .emptyDynamicParams:           return "STFMeshRenderer: dynamicParams is empty."
        }
    }
}
