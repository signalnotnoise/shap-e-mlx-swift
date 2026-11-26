import Foundation

/// Topology metrics for diagnosing holes / open boundaries in triangle meshes.
public struct MeshTopologyStats: Sendable, Equatable {
    public var vertexCount: Int
    public var faceCount: Int
    /// Edges that appear in exactly one face (open boundary). Closed manifold → 0.
    public var boundaryEdgeCount: Int
    /// Edges that appear in 3+ faces (non-manifold).
    public var nonManifoldEdgeCount: Int
    /// Faces with out-of-range vertex indices.
    public var invalidFaceCount: Int

    public var isClosed: Bool { boundaryEdgeCount == 0 && invalidFaceCount == 0 }

    public var summaryLine: String {
        "V=\(vertexCount) F=\(faceCount) boundary_edges=\(boundaryEdgeCount) "
            + "nonmanifold_edges=\(nonManifoldEdgeCount) invalid_faces=\(invalidFaceCount) "
            + "closed=\(isClosed)"
    }
}

public enum MeshTopology {
    /// Undirected edge key with canonical endpoint order.
    private struct EdgeKey: Hashable {
        let a: Int32
        let b: Int32
        init(_ i: Int32, _ j: Int32) {
            if i <= j {
                a = i
                b = j
            } else {
                a = j
                b = i
            }
        }
    }

    /// Compute topology stats from flat or `[F,3]` face indices and vertex count.
    public static func stats(vertexCount: Int, faces: [Int32]) -> MeshTopologyStats {
        precondition(faces.count % 3 == 0, "faces length must be multiple of 3")
        let faceCount = faces.count / 3
        var edgeUse: [EdgeKey: Int] = [:]
        edgeUse.reserveCapacity(faceCount * 2)
        var invalid = 0

        for f in 0..<faceCount {
            let i0 = faces[f * 3]
            let i1 = faces[f * 3 + 1]
            let i2 = faces[f * 3 + 2]
            let ok =
                i0 >= 0 && i1 >= 0 && i2 >= 0
                && Int(i0) < vertexCount && Int(i1) < vertexCount && Int(i2) < vertexCount
                && i0 != i1 && i1 != i2 && i0 != i2
            if !ok {
                invalid += 1
                continue
            }
            edgeUse[EdgeKey(i0, i1), default: 0] += 1
            edgeUse[EdgeKey(i1, i2), default: 0] += 1
            edgeUse[EdgeKey(i2, i0), default: 0] += 1
        }

        var boundary = 0
        var nonManifold = 0
        for (_, count) in edgeUse {
            if count == 1 { boundary += 1 }
            else if count >= 3 { nonManifold += 1 }
        }

        return MeshTopologyStats(
            vertexCount: vertexCount,
            faceCount: faceCount,
            boundaryEdgeCount: boundary,
            nonManifoldEdgeCount: nonManifold,
            invalidFaceCount: invalid
        )
    }

    /// Compare two meshes extracted from the same SDF (counts + boundary).
    public static func compare(
        labelA: String,
        statsA: MeshTopologyStats,
        labelB: String,
        statsB: MeshTopologyStats
    ) -> String {
        var lines: [String] = []
        lines.append("=== Mesh A/B topology ===")
        lines.append("\(labelA): \(statsA.summaryLine)")
        lines.append("\(labelB): \(statsB.summaryLine)")
        let dV = statsB.vertexCount - statsA.vertexCount
        let dF = statsB.faceCount - statsA.faceCount
        let dB = statsB.boundaryEdgeCount - statsA.boundaryEdgeCount
        lines.append("delta (\(labelB)-\(labelA)): ΔV=\(dV) ΔF=\(dF) Δboundary=\(dB)")
        if statsA.boundaryEdgeCount == 0 && statsB.boundaryEdgeCount > 0 {
            lines.append(
                "VERDICT: \(labelB) opens boundaries that \(labelA) does not → likely \(labelB) extract/weld bug"
            )
        } else if statsB.boundaryEdgeCount == 0 && statsA.boundaryEdgeCount > 0 {
            lines.append(
                "VERDICT: \(labelA) opens boundaries that \(labelB) does not → likely \(labelA) extract bug"
            )
        } else if statsA.boundaryEdgeCount > 0 && statsB.boundaryEdgeCount > 0 {
            lines.append(
                "VERDICT: BOTH open (boundary>0) → likely SDF/model quality, not Metal-only"
            )
        } else {
            lines.append("VERDICT: BOTH closed (no boundary edges) → holes may be color/shading or visual melts")
        }
        return lines.joined(separator: "\n")
    }
}
