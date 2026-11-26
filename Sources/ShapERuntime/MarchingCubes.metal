#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    uint3 gridSize;
    float3 bboxMin;
    float3 bboxSize;
};

constant uint3 cornerOffsets[8] = {
    uint3(0,0,0), uint3(1,0,0), uint3(0,1,0), uint3(1,1,0),
    uint3(0,0,1), uint3(1,0,1), uint3(0,1,1), uint3(1,1,1)
};

constant int2 edgeCorners[12] = {
    int2(0,1), int2(2,3), int2(4,5), int2(6,7),
    int2(0,2), int2(1,3), int2(4,6), int2(5,7),
    int2(0,4), int2(1,5), int2(2,6), int2(3,7)
};

// Assuming the MLX array is stored in flat [X, Y, Z] layout
inline float getSDF(device const float* sdfBuffer, uint3 gridPos, uint3 gridSize) {
    return sdfBuffer[(gridPos.x * gridSize.y * gridSize.z) + (gridPos.y * gridSize.z) + gridPos.z];
}

// Global flat edge index matching CPU MarchingCubes.flatEdgeIndices / shap_e Python.
// Local edge 0–11 uses the same corner pairing as edgeCorners above.
inline int flatEdgeIndex(uint3 gid, int edgeIdx, uint3 gridSize) {
    int X = int(gridSize.x);
    int Y = int(gridSize.y);
    int Z = int(gridSize.z);
    int ci = int(gid.x);
    int cj = int(gid.y);
    int ck = int(gid.z);
    int YZ = Y * Z;
    int numXs = (X - 1) * Y * Z;
    int yOff = numXs;
    int zOff = numXs + X * (Y - 1) * Z;
    int Ym1Z = (Y - 1) * Z;
    int YZm1 = Y * (Z - 1);
    int Zm1 = Z - 1;

    switch (edgeIdx) {
        case 0:  return ci * YZ + cj * Z + ck;
        case 1:  return ci * YZ + (cj + 1) * Z + ck;
        case 2:  return ci * YZ + cj * Z + (ck + 1);
        case 3:  return ci * YZ + (cj + 1) * Z + (ck + 1);
        case 4:  return yOff + ci * Ym1Z + cj * Z + ck;
        case 5:  return yOff + (ci + 1) * Ym1Z + cj * Z + ck;
        case 6:  return yOff + ci * Ym1Z + cj * Z + (ck + 1);
        case 7:  return yOff + (ci + 1) * Ym1Z + cj * Z + (ck + 1);
        case 8:  return zOff + ci * YZm1 + cj * Zm1 + ck;
        case 9:  return zOff + (ci + 1) * YZm1 + cj * Zm1 + ck;
        case 10: return zOff + ci * YZm1 + (cj + 1) * Zm1 + ck;
        case 11: return zOff + (ci + 1) * YZm1 + (cj + 1) * Zm1 + ck;
        default: return -1;
    }
}

// -----------------------------------------------------------------------------
// PASS 1: Count Triangles
// -----------------------------------------------------------------------------
kernel void countTriangles(
    device const float* sdfBuffer [[buffer(0)]],
    device const int* numTrisTable [[buffer(1)]],  // 256 array
    device atomic_uint* totalTriangles [[buffer(2)]],
    device uint* voxelOffsets [[buffer(3)]],       // gridSize.x * gridSize.y * gridSize.z
    constant Uniforms& uniforms [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uniforms.gridSize.x - 1 || 
        gid.y >= uniforms.gridSize.y - 1 || 
        gid.z >= uniforms.gridSize.z - 1) {
        return;
    }

    int mask = 0;
    for (int i = 0; i < 8; i++) {
        uint3 pos = gid + cornerOffsets[i];
        float sdf = getSDF(sdfBuffer, pos, uniforms.gridSize);
        if (sdf > 0.0) {
            mask |= (1 << i);
        }
    }

    int tris = numTrisTable[mask];
    
    // Calculate global flat index for voxel
    uint voxelIdx = (gid.x * uniforms.gridSize.y * uniforms.gridSize.z) + (gid.y * uniforms.gridSize.z) + gid.z;
    
    if (tris > 0) {
        // Claim 'tris' spot in the global face array
        uint offset = atomic_fetch_add_explicit(totalTriangles, tris, memory_order_relaxed);
        voxelOffsets[voxelIdx] = offset;
    } else {
        voxelOffsets[voxelIdx] = 0xFFFFFFFF; // Invalid/Empty
    }
}

// -----------------------------------------------------------------------------
// PASS 2: Extract Mesh
// -----------------------------------------------------------------------------
kernel void extractMesh(
    device const float* sdfBuffer [[buffer(0)]],
    device const int* numTrisTable [[buffer(1)]],
    device const int* mcTable [[buffer(2)]], // 256 * 15 flat array
    device const uint* voxelOffsets [[buffer(3)]],
    device packed_float3* outVertices [[buffer(4)]],
    device packed_int3* outFaces [[buffer(5)]],
    constant Uniforms& uniforms [[buffer(6)]],
    device int* outEdgeIds [[buffer(7)]], // global flat edge index per unindexed vertex
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= uniforms.gridSize.x - 1 || 
        gid.y >= uniforms.gridSize.y - 1 || 
        gid.z >= uniforms.gridSize.z - 1) {
        return;
    }

    uint voxelIdx = (gid.x * uniforms.gridSize.y * uniforms.gridSize.z) + (gid.y * uniforms.gridSize.z) + gid.z;
    uint offset = voxelOffsets[voxelIdx];
    
    if (offset == 0xFFFFFFFF) { return; } // No triangles to extract here

    int mask = 0;
    float sdfVals[8];
    for (int i = 0; i < 8; i++) {
        uint3 pos = gid + cornerOffsets[i];
        float sdf = getSDF(sdfBuffer, pos, uniforms.gridSize);
        sdfVals[i] = sdf;
        if (sdf > 0.0) {
            mask |= (1 << i);
        }
    }

    int tris = numTrisTable[mask];
    
    for (int t = 0; t < tris; t++) {
        int3 face;
        for (int v = 0; v < 3; v++) {
            // Read which edge of the cube this vertex lies on
            int edgeIdx = mcTable[mask * 15 + t * 3 + v];
            int2 corners = edgeCorners[edgeIdx];
            
            int c1 = corners.x;
            int c2 = corners.y;
            
            float s1 = sdfVals[c1];
            float s2 = sdfVals[c2];
            
            uint3 p1_grid = gid + cornerOffsets[c1];
            uint3 p2_grid = gid + cornerOffsets[c2];
            
            float3 gridSizeF = float3(uniforms.gridSize.x - 1, uniforms.gridSize.y - 1, uniforms.gridSize.z - 1);
            
            float3 p1 = (float3(p1_grid) / gridSizeF) * uniforms.bboxSize + uniforms.bboxMin;
            float3 p2 = (float3(p2_grid) / gridSizeF) * uniforms.bboxSize + uniforms.bboxMin;
            
            // Linear interpolate the position based on the exact distance field crossing
            float t_val = s1 / (s1 - s2);
            if (isnan(t_val) || isinf(t_val)) {
                t_val = 0.5;
            }
            float3 vertPos = t_val * p2 + (1.0 - t_val) * p1;
            
            // Unindexed triangle stream; host welds by outEdgeIds (shared grid-edge identity).
            uint globalVertIdx = (offset + t) * 3 + v;
            outVertices[globalVertIdx] = vertPos;
            outEdgeIds[globalVertIdx] = flatEdgeIndex(gid, edgeIdx, uniforms.gridSize);
            face[v] = globalVertIdx;
        }
        
        outFaces[offset + t] = face;
    }
}
