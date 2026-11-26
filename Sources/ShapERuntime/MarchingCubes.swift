import Foundation
import MLX

// MARK: - MarchingCubes
//
// Swift port of shap_e/src/mlx_shap_e/models/rendering/marching_cubes.py
//
// Deterministic MLX marching-cubes that mirrors Shap-E's edge indexing and
// linear interpolation math exactly.
//
// Public entry point:
//   MarchingCubes.extract(sdf:bboxMin:bboxSize:) → (verts [V,3], faces [F,3])

public enum MarchingCubes {

    // MARK: - Public entry point

    /// Extract a triangle mesh from a signed-distance field at isosurface level 0.
    ///
    /// - Parameter sdf:      3-D SDF grid [X, Y, Z] float32 (already border-padded).
    /// - Parameter bboxMin:  World-space minimum corner [3] float32.
    /// - Parameter bboxSize: World-space extent (bboxMax - bboxMin) [3] float32.
    /// - Returns: (verts [V, 3], faces [F, 3] int32).  Empty arrays if no isosurface.
    public static func extract(
        sdf: MLXArray,
        bboxMin: MLXArray,
        bboxSize: MLXArray
    ) -> (verts: MLXArray, faces: MLXArray) {
        guard sdf.ndim == 3 else { return emptyMesh() }
        let (X, Y, Z) = (sdf.shape[0], sdf.shape[1], sdf.shape[2])
        guard X >= 2 && Y >= 2 && Z >= 2 else { return emptyMesh() }

        let lut = LookupTable.shared

        // ---- Bitmasks --------------------------------------------------------
        // For each cube corner mark positive/negative; pack 8 bits per cube.
        let positive = (sdf .> Float(0.0)).asType(Int32.self)   // [X,Y,Z]
        var bits = positive[0..<(X-1), 0..<(Y-1), 0..<(Z-1)]              // face [0]
        bits = bits + (positive[1..<X,   0..<(Y-1), 0..<(Z-1)] * MLXArray(Int32(2)))     // face [1]
        bits = bits + (positive[0..<(X-1), 1..<Y,   0..<(Z-1)] * MLXArray(Int32(4)))     // face [2]
        bits = bits + (positive[1..<X,     1..<Y,   0..<(Z-1)] * MLXArray(Int32(8)))     // face [3]
        bits = bits + (positive[0..<(X-1), 0..<(Y-1), 1..<Z]   * MLXArray(Int32(16)))    // face [4]
        bits = bits + (positive[1..<X,     0..<(Y-1), 1..<Z]   * MLXArray(Int32(32)))    // face [5]
        bits = bits + (positive[0..<(X-1), 1..<Y,     1..<Z]   * MLXArray(Int32(64)))    // face [6]
        bits = bits + (positive[1..<X,     1..<Y,     1..<Z]   * MLXArray(Int32(128)))   // face [7]

        // ---- Corner coordinates grid [X,Y,Z,3] --------------------------------
        let xRange = MLX.arange(X, dtype: .float32)
        let yRange = MLX.arange(Y, dtype: .float32)
        let zRange = MLX.arange(Z, dtype: .float32)
        let xs = MLX.broadcast(xRange.reshaped(X, 1, 1), to: [X, Y, Z])
        let ys = MLX.broadcast(yRange.reshaped(1, Y, 1), to: [X, Y, Z])
        let zs = MLX.broadcast(zRange.reshaped(1, 1, Z), to: [X, Y, Z])
        let cornerCoords = MLX.stacked([xs, ys, zs], axis: -1)    // [X,Y,Z,3]

        // ---- Edge midpoints ---------------------------------------------------
        // 12 edge types; concatenated into one flat array.
        let midX = ((cornerCoords[0..<(X-1)] + cornerCoords[1..<X]) / Float(2.0)).reshaped(-1, 3)
        let midY = ((cornerCoords[0..., 0..<(Y-1)] + cornerCoords[0..., 1..<Y]) / Float(2.0)).reshaped(-1, 3)
        let midZ = ((cornerCoords[0..., 0..., 0..<(Z-1)] + cornerCoords[0..., 0..., 1..<Z]) / Float(2.0)).reshaped(-1, 3)
        let edgeMidpoints = MLX.concatenated([midX, midY, midZ], axis: 0)   // [E, 3]

        // ---- Cube → flat edge index mapping ----------------------------------
        let cubeCoords = cornerCoords[0..<(X-1), 0..<(Y-1), 0..<(Z-1)]
            .asType(.int32).reshaped(-1, 3)                                  // [C, 3]
        let edgeIndices = flatEdgeIndices(cubeCoords: cubeCoords,
                                          gridSize: (X, Y, Z))               // [C, 12]

        // ---- Look up triangles per cube ----------------------------------------
        let flatBits = bits.reshaped(-1).asType(Int32.self)                  // [C]
        let localTris  = lut.cases[flatBits]                                  // [C, 5, 3]
        let localMasks = lut.masks[flatBits]                                  // [C, 5]

        guard MLX.any(localMasks).item(Bool.self) else { return emptyMesh() }

        // ---- Global edge indices for each triangle vertex ----------------------
        let C   = cubeCoords.shape[0]
        let gTris = MLX.takeAlong(
            edgeIndices,
            localTris.reshaped(C, -1),
            axis: 1
        ).reshaped(C, 5, 3)                                                   // [C, 5, 3]

        // Compact: keep only triangles where mask == true
        let selectedTris = compactRowsByMask(gTris.reshaped(-1, 3),
                                              mask: localMasks.reshaped(-1))   // [T, 3]
        guard selectedTris.shape[0] > 0 else { return emptyMesh() }

        // ---- Build unique vertex list ------------------------------------------
        let flatSelected  = selectedTris.reshaped(-1)                         // [3T]
        let order         = MLX.argSort(flatSelected)
        let sortedUsed    = flatSelected[order]
        let isFirst       = MLXArray([true])
        let isNew         = MLX.concatenated([isFirst,
                                              sortedUsed[1...] .!= sortedUsed[0..<(sortedUsed.shape[0]-1)]],
                                             axis: 0)                          // [3T]
        let usedVertIdx   = compactVectorByMask(sortedUsed, mask: isNew)       // [V]
        let usedEdgeMids  = edgeMidpoints[usedVertIdx]                        // [V, 3]

        // remap flat face indices to compact vertex IDs
        let compactIds    = MLX.cumsum(isNew.asType(Int32.self)) - 1               // [3T]
        let flatFaces     = MLX.zeros(like: flatSelected)
        flatFaces[order]  = compactIds
        let faces         = flatFaces.reshaped(selectedTris.shape)             // [T, 3]

        // ---- Linear interpolation to exact isosurface positions ----------------
        let v1 = MLX.floor(usedEdgeMids).asType(Int32.self)                       // [V, 3]
        let v2 = MLX.ceil(usedEdgeMids).asType(Int32.self)                        // [V, 3]
        let s1 = sdf[v1[0..., 0], v1[0..., 1], v1[0..., 2]]                   // [V]
        let s2 = sdf[v2[0..., 0], v2[0..., 1], v2[0..., 2]]                   // [V]

        let gridSizeF = MLXArray([Float(X - 1), Float(Y - 1), Float(Z - 1)])
        let bmin      = bboxMin.asType(Float.self)
        let bsz       = bboxSize.asType(Float.self)
        let p1 = (v1.asType(Float.self) / gridSizeF) * bsz + bmin
        let p2 = (v2.asType(Float.self) / gridSizeF) * bsz + bmin

        let denom = s1 - s2
        let safeDenom = MLX.where(denom .== 0.0, MLXArray(Float(1e-5)), denom)
        let t    = (s1 / safeDenom).expandedDimensions(axes: [-1])           // [V, 1]
        let verts = (t * p2 + (Float(1.0) - t) * p1).asType(Float.self)        // [V, 3]

        ShapELogger.debug("[Debug] MarchingCubes: extracted \(verts.shape[0]) vertices, \(faces.shape[0]) faces")
        return (verts, faces.asType(Int32.self))
    }

    // MARK: - Edge index mapping

    /// Build the [C, 12] flat edge index table for all cubes.
    ///
    /// Matches: `_create_flat_edge_indices` in marching_cubes.py
    private static func flatEdgeIndices(
        cubeCoords: MLXArray,    // [C, 3]  int32
        gridSize: (Int, Int, Int)
    ) -> MLXArray {
        let (X, Y, Z) = gridSize
        let ci = cubeCoords[0..., 0]   // xs [C]
        let cj = cubeCoords[0..., 1]   // ys [C]
        let ck = cubeCoords[0..., 2]   // zs [C]

        let numXs    = (X - 1) * Y * Z
        let yOffset  = numXs
        let numYs    = X * (Y - 1) * Z
        let zOffset  = numXs + numYs

        let Z_arr = MLXArray(Int32(Z))
        let YZ_arr = MLXArray(Int32(Y * Z))
        let Ym1Z_arr = MLXArray(Int32((Y - 1) * Z))
        let YZm1_arr = MLXArray(Int32(Y * (Z - 1)))
        let Zm1_arr = MLXArray(Int32(Z - 1))
        let one = MLXArray(Int32(1))
        let yOff = MLXArray(Int32(yOffset))
        let zOff = MLXArray(Int32(zOffset))

        // Edge 0: x-direction edges
        let e0  = ci * YZ_arr + cj * Z_arr + ck
        let e1  = ci * YZ_arr + (cj + one) * Z_arr + ck
        let e2  = ci * YZ_arr + cj * Z_arr + (ck + one)
        let e3  = ci * YZ_arr + (cj + one) * Z_arr + (ck + one)
        // y-direction edges
        let e4  = yOff + ci * Ym1Z_arr + cj * Z_arr + ck
        let e5  = yOff + (ci + one) * Ym1Z_arr + cj * Z_arr + ck
        let e6  = yOff + ci * Ym1Z_arr + cj * Z_arr + (ck + one)
        let e7  = yOff + (ci + one) * Ym1Z_arr + cj * Z_arr + (ck + one)
        // z-direction edges
        let e8  = zOff + ci * YZm1_arr + cj * Zm1_arr + ck
        let e9  = zOff + (ci + one) * YZm1_arr + cj * Zm1_arr + ck
        let e10 = zOff + ci * YZm1_arr + (cj + one) * Zm1_arr + ck
        let e11 = zOff + (ci + one) * YZm1_arr + (cj + one) * Zm1_arr + ck

        return MLX.stacked([e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11], axis: -1)
    }

    // MARK: - Compact helpers

    /// Keep only rows where mask[i] == true, placing them in the first N positions.
    ///
    /// Matches: `_compact_rows_by_mask`
    private static func compactRowsByMask(_ values: MLXArray, mask: MLXArray) -> MLXArray {
        let maskI       = mask.asType(Int32.self)
        let numSelected = Int(MLX.sum(maskI).item(Int32.self))
        guard numSelected > 0 else { return values[0..<0] }

        let trueRank  = MLX.cumsum(maskI) - 1
        let falseRank = MLX.cumsum(MLXArray(Int32(1)) - maskI) - 1
        // dests is a 1D permutation of indices [0...N-1]
        let dests     = MLX.where(mask, trueRank, falseRank + MLXArray(Int32(numSelected))).asType(Int32.self)
        
        let out = MLX.zeros(like: values)
        // Permute rows using 1D index array. 
        // DO NOT broadcast dests to values.shape, keep it rank-1 to avoid (N,3,3) expansion.
        out[dests] = values
        return out[0..<numSelected]
    }

    /// Keep only elements where mask[i] == true.
    ///
    /// Matches: `_compact_by_mask`
    private static func compactVectorByMask(_ values: MLXArray, mask: MLXArray) -> MLXArray {
        let maskI      = mask.asType(Int32.self)
        let numSelected = Int(MLX.sum(maskI).item(Int32.self))
        guard numSelected > 0 else { return values[0..<0] }

        let trueRank  = MLX.cumsum(maskI) - 1
        let falseRank = MLX.cumsum(MLXArray(Int32(1)) - maskI) - 1
        let dests     = MLX.where(mask, trueRank, falseRank + MLXArray(Int32(numSelected))).asType(Int32.self)
        
        let out = MLX.zeros(like: values)
        out[dests] = values
        return out[0..<numSelected]
    }

    private static func emptyMesh() -> (MLXArray, MLXArray) {
        (MLX.zeros([0, 3], type: Float.self), MLX.zeros([0, 3], type: Int32.self))
    }
}

// MARK: - MC Lookup Table

/// Pre-computed case→triangle lookup table (256 cases × 5 tris × 3 edge indices).
final class LookupTable: @unchecked Sendable {
    let cases: MLXArray   // [256, 5, 3] int32
    let masks: MLXArray   // [256, 5]    bool

    static let shared: LookupTable = LookupTable()

    private init() {
        // Edge corners: the 12 edges of a unit cube, each defined by its two corner indices.
        let edgeCorners: [(Int, Int)] = [
            (0,1),(2,3),(4,5),(6,7),
            (0,2),(1,3),(4,6),(5,7),
            (0,4),(1,5),(2,6),(3,7)
        ]
        var edgeToIndex: [Int64: Int] = [:]
        for (i, (a, b)) in edgeCorners.enumerated() {
            let lo = min(a, b); let hi = max(a, b)
            edgeToIndex[Int64(lo) << 32 | Int64(hi)] = i
        }

        var casesArr  = [Int32](repeating: 0, count: 256 * 5 * 3)
        var masksArr  = [Bool](repeating: false, count: 256 * 5)

        for (caseIdx, tri_list) in MC_TABLE.enumerated() {
            for (triIdx, tri) in tri_list.enumerated() {
                // tri is an array of 6 ints: [c0,c1, c2,c3, c4,c5] — three edge definitions
                for vertIdx in 0..<3 {
                    let c1 = tri[vertIdx * 2]
                    let c2 = tri[vertIdx * 2 + 1]
                    let lo = min(c1, c2); let hi = max(c1, c2)
                    let edgeIdx = edgeToIndex[Int64(lo) << 32 | Int64(hi)] ?? 0
                    casesArr[caseIdx * 15 + triIdx * 3 + vertIdx] = Int32(edgeIdx)
                }
                masksArr[caseIdx * 5 + triIdx] = true
            }
        }

        self.cases = MLXArray(casesArr).reshaped(256, 5, 3)
        let maskInts = masksArr.map { $0 ? Int32(1) : Int32(0) }
        self.masks = MLXArray(maskInts).reshaped(256, 5).asType(.bool)
    }
}
