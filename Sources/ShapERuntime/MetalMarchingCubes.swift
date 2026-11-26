import Foundation
import Metal
import MLX
import simd

public enum MetalMarchingCubesError: Error, LocalizedError {
    case deviceNotFound
    case libraryNotFound
    case functionNotFound(String)
    case bufferAllocationFailed
    case sdfLayoutMismatch(expected: Int, actual: Int)
    
    public var errorDescription: String? {
        switch self {
        case .deviceNotFound: return "No suitable Metal device found."
        case .libraryNotFound: return "Could not load the Metal library."
        case .functionNotFound(let name): return "Could not find Metal function: \(name)"
        case .bufferAllocationFailed: return "Failed to allocate Metal buffer."
        case .sdfLayoutMismatch(let expected, let actual):
            return "SDF host buffer size mismatch: expected \(expected) floats (X*Y*Z), got \(actual)."
        }
    }
}

public enum MetalMarchingCubes {
    private struct Uniforms {
        var gridSize: simd_uint4
        var bboxMin: simd_float4
        var bboxSize: simd_float4
    }
    
    nonisolated(unsafe) private static var device: MTLDevice? = MTLCreateSystemDefaultDevice()
    nonisolated(unsafe) private static var commandQueue: MTLCommandQueue? = device?.makeCommandQueue()
    nonisolated(unsafe) private static var countPipelineState: MTLComputePipelineState?
    nonisolated(unsafe) private static var extractPipelineState: MTLComputePipelineState?
    nonisolated(unsafe) private static var mcTableBuffer: MTLBuffer?
    nonisolated(unsafe) private static var numTrisTableBuffer: MTLBuffer?
    private static let setupLock = NSLock()
    
    private static func setup() throws {
        setupLock.lock()
        defer { setupLock.unlock() }
        guard let device = device else { throw MetalMarchingCubesError.deviceNotFound }
        if countPipelineState != nil { return }
        
        guard let library = try? device.makeDefaultLibrary(bundle: Bundle.module) else {
            throw MetalMarchingCubesError.libraryNotFound
        }
        guard let countFunction = library.makeFunction(name: "countTriangles"),
              let extractFunction = library.makeFunction(name: "extractMesh") else {
            throw MetalMarchingCubesError.functionNotFound("countTriangles or extractMesh")
        }
        
        countPipelineState = try device.makeComputePipelineState(function: countFunction)
        extractPipelineState = try device.makeComputePipelineState(function: extractFunction)
        
        // Generate marching cubes tables
        let casesArr = LookupTable.shared.cases.asArray(Int32.self)
        let masksArr = LookupTable.shared.masks.asArray(Bool.self)
        
        var numTrisArr = [Int32](repeating: 0, count: 256)
        for i in 0..<256 {
            var count: Int32 = 0
            for j in 0..<5 {
                if masksArr[i * 5 + j] { count += 1 }
            }
            numTrisArr[i] = count
        }
        
        mcTableBuffer = device.makeBuffer(bytes: casesArr, length: casesArr.count * 4, options: .storageModeShared)
        numTrisTableBuffer = device.makeBuffer(bytes: numTrisArr, length: 256 * 4, options: .storageModeShared)
    }
    
    /// Metal implementation of Marching Cubes.
    public static func extract(
        sdf: MLXArray,
        bboxMin: MLXArray,
        bboxSize: MLXArray
    ) throws -> (verts: MLXArray, faces: MLXArray) {
        try setup()
        guard let device = device, let commandQueue = commandQueue,
              let countPipelineState = countPipelineState,
              let extractPipelineState = extractPipelineState,
              let mcTableBuffer = mcTableBuffer,
              let numTrisTableBuffer = numTrisTableBuffer else {
            throw MetalMarchingCubesError.deviceNotFound
        }
        
        // 1. Pull SDF into CPU array and send to Metal
        // MLX arrays from operations like pad() might not be C-contiguous in memory.
        // Adding 0.0 forces the allocation of a fresh, densely packed C-contiguous array.
        let contiguousSDF = sdf + MLXArray(Float(0.0))
        let sdfFloats = contiguousSDF.asArray(Float.self)
        
        let (X, Y, Z) = (sdf.shape[0], sdf.shape[1], sdf.shape[2])
        let expectedSDFCount = X * Y * Z
        guard sdfFloats.count == expectedSDFCount else {
            throw MetalMarchingCubesError.sdfLayoutMismatch(
                expected: expectedSDFCount,
                actual: sdfFloats.count
            )
        }
        guard let sdfBuffer = device.makeBuffer(bytes: sdfFloats, length: sdfFloats.count * 4, options: .storageModeShared) else {
            throw MetalMarchingCubesError.bufferAllocationFailed
        }
        
        let bboxMinF = bboxMin.asArray(Float.self)
        let bboxSizeF = bboxSize.asArray(Float.self)
        
        var uniforms = Uniforms(
            gridSize: simd_uint4(UInt32(X), UInt32(Y), UInt32(Z), 0),
            bboxMin: simd_float4(bboxMinF[0], bboxMinF[1], bboxMinF[2], 0),
            bboxSize: simd_float4(bboxSizeF[0], bboxSizeF[1], bboxSizeF[2], 0)
        )
        
        let totalVoxels = X * Y * Z
        guard let voxelOffsetsBuffer = device.makeBuffer(length: totalVoxels * 4, options: .storageModePrivate),
              let totalTrianglesBuffer = device.makeBuffer(length: 4, options: .storageModeShared) else {
            throw MetalMarchingCubesError.bufferAllocationFailed
        }
        totalTrianglesBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        
        let gridSize = MTLSize(width: X, height: Y, depth: Z)
        let w = min(8, X)
        let h = min(8, Y)
        let d = min(8, Z)
        let tgSize = MTLSize(width: w, height: h, depth: d)
        
        // 2. Dispatch Pass 1: Count Triangles
        let countCB = commandQueue.makeCommandBuffer()!
        let countCE = countCB.makeComputeCommandEncoder()!
        countCE.setComputePipelineState(countPipelineState)
        countCE.setBuffer(sdfBuffer, offset: 0, index: 0)
        countCE.setBuffer(numTrisTableBuffer, offset: 0, index: 1)
        countCE.setBuffer(totalTrianglesBuffer, offset: 0, index: 2)
        countCE.setBuffer(voxelOffsetsBuffer, offset: 0, index: 3)
        countCE.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 4)
        countCE.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        countCE.endEncoding()
        
        countCB.commit()
        countCB.waitUntilCompleted()
        
        // 3. Read back Atomic Counters
        let totalTris = Int(totalTrianglesBuffer.contents().load(as: UInt32.self))
        if totalTris == 0 {
            return (MLX.zeros([0, 3], type: Float.self), MLX.zeros([0, 3], type: Int32.self))
        }
        
        let totalVerts = totalTris * 3
        // packed_float3 is 12 bytes per vertex. packed_int3 is 12 bytes per face.
        // edgeIds: one int32 global edge index per unindexed vertex (for topological welding).
        guard let vertsBuffer = device.makeBuffer(length: totalVerts * 12, options: .storageModeShared),
              let facesBuffer = device.makeBuffer(length: totalTris * 12, options: .storageModeShared),
              let edgeIdsBuffer = device.makeBuffer(length: totalVerts * 4, options: .storageModeShared) else {
            throw MetalMarchingCubesError.bufferAllocationFailed
        }
        
        // 4. Dispatch Pass 2: Extract Mesh
        let extractCB = commandQueue.makeCommandBuffer()!
        let extractCE = extractCB.makeComputeCommandEncoder()!
        extractCE.setComputePipelineState(extractPipelineState)
        extractCE.setBuffer(sdfBuffer, offset: 0, index: 0)
        extractCE.setBuffer(numTrisTableBuffer, offset: 0, index: 1)
        extractCE.setBuffer(mcTableBuffer, offset: 0, index: 2)
        extractCE.setBuffer(voxelOffsetsBuffer, offset: 0, index: 3)
        extractCE.setBuffer(vertsBuffer, offset: 0, index: 4)
        extractCE.setBuffer(facesBuffer, offset: 0, index: 5)
        extractCE.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 6)
        extractCE.setBuffer(edgeIdsBuffer, offset: 0, index: 7)
        extractCE.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        extractCE.endEncoding()
        
        extractCB.commit()
        extractCB.waitUntilCompleted()
        
        // 5. Read GPU vertex stream + global edge IDs for topological welding
        let finalVertsFloats = Array(UnsafeBufferPointer(
            start: vertsBuffer.contents().bindMemory(to: Float.self, capacity: totalVerts * 3),
            count: totalVerts * 3
        ))
        let edgeIds = Array(UnsafeBufferPointer(
            start: edgeIdsBuffer.contents().bindMemory(to: Int32.self, capacity: totalVerts),
            count: totalVerts
        ))
        
        // 6. Weld unindexed vertices by shared grid-edge identity (matches CPU MarchingCubes).
        // Adjacent cubes recompute the same edge independently; welding by edge ID is
        // topologically correct and immune to float ULP / NaN key issues.
        var uniqueVerts: [Float] = []
        uniqueVerts.reserveCapacity(totalVerts * 3)
        var edgeToVertex = [Int32: Int32]()
        edgeToVertex.reserveCapacity(totalVerts)
        
        var weldedFaces = [Int32](repeating: 0, count: totalTris * 3)
        var nextIdx: Int32 = 0
        
        for i in 0..<totalVerts {
            let edgeId = edgeIds[i]
            if let existing = edgeToVertex[edgeId] {
                weldedFaces[i] = existing
            } else {
                edgeToVertex[edgeId] = nextIdx
                weldedFaces[i] = nextIdx
                uniqueVerts.append(finalVertsFloats[i * 3])
                uniqueVerts.append(finalVertsFloats[i * 3 + 1])
                uniqueVerts.append(finalVertsFloats[i * 3 + 2])
                nextIdx += 1
            }
        }
        
        let vertsMLX = MLXArray(uniqueVerts).reshaped([Int(nextIdx), 3])
        let facesMLX = MLXArray(weldedFaces).reshaped([totalTris, 3])
        
        return (vertsMLX, facesMLX)
    }
}
