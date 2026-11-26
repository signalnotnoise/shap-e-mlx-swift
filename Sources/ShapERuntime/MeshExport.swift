import Foundation

public enum MeshExportError: Error, LocalizedError {
    case emptyMesh
    case faceVertexMismatch

    public var errorDescription: String? {
        switch self {
        case .emptyMesh: return "Cannot export empty mesh."
        case .faceVertexMismatch: return "Face index out of range for vertex list."
        }
    }
}

public enum MeshExport {
    /// Write a Wavefront OBJ. Faces are 1-indexed.
    /// - Parameters:
    ///   - colors: optional RGB floats length `vertexCount * 3` (writes `v x y z r g b`)
    public static func writeOBJ(
        path: URL,
        vertices: [Float],
        faces: [Int32],
        colors: [Float]? = nil,
        comment: String? = nil
    ) throws {
        precondition(vertices.count % 3 == 0)
        precondition(faces.count % 3 == 0)
        let vCount = vertices.count / 3
        if vCount == 0 || faces.isEmpty {
            throw MeshExportError.emptyMesh
        }
        if let colors {
            precondition(colors.count == vCount * 3, "colors must be V*3")
        }

        var text = ""
        text.reserveCapacity(vertices.count * 12 + faces.count * 8)
        if let comment {
            for line in comment.split(separator: "\n") {
                text += "# \(line)\n"
            }
        }
        text += "# vertices \(vCount) faces \(faces.count / 3)\n"

        for i in 0..<vCount {
            let x = vertices[i * 3]
            let y = vertices[i * 3 + 1]
            let z = vertices[i * 3 + 2]
            if let colors {
                let r = colors[i * 3]
                let g = colors[i * 3 + 1]
                let b = colors[i * 3 + 2]
                text += "v \(x) \(y) \(z) \(r) \(g) \(b)\n"
            } else {
                text += "v \(x) \(y) \(z)\n"
            }
        }
        for f in 0..<(faces.count / 3) {
            let i0 = Int(faces[f * 3])
            let i1 = Int(faces[f * 3 + 1])
            let i2 = Int(faces[f * 3 + 2])
            guard i0 >= 0, i1 >= 0, i2 >= 0, i0 < vCount, i1 < vCount, i2 < vCount else {
                throw MeshExportError.faceVertexMismatch
            }
            // OBJ is 1-based
            text += "f \(i0 + 1) \(i1 + 1) \(i2 + 1)\n"
        }

        try text.write(to: path, atomically: true, encoding: .utf8)
    }

    /// Convenience from `RuntimeTensor` mesh fields.
    public static func writeOBJ(path: URL, mesh: MeshResult, comment: String? = nil) throws {
        let verts = try mesh.vertices.float32Values()
        let faces = try mesh.faces.int32Values()
        var colors: [Float]? = nil
        if let rgba = mesh.vertexChannels?["RGBA"] {
            let vals = try rgba.float32Values()
            // RGBA → RGB for OBJ
            let vCount = verts.count / 3
            var rgb = [Float](repeating: 0, count: vCount * 3)
            for i in 0..<vCount {
                rgb[i * 3] = vals[i * 4]
                rgb[i * 3 + 1] = vals[i * 4 + 1]
                rgb[i * 3 + 2] = vals[i * 4 + 2]
            }
            colors = rgb
        }
        try writeOBJ(path: path, vertices: verts, faces: faces, colors: colors, comment: comment)
    }
}
