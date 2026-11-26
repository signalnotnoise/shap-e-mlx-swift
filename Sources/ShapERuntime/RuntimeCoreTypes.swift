import Foundation

public struct ArtifactVersionInfo: Equatable, Sendable, Codable {
    public let artifactVersion: Int
    public let manifestPath: String
    public let familyVersions: [String: Int]

    public init(artifactVersion: Int, manifestPath: String, familyVersions: [String: Int] = [:]) {
        self.artifactVersion = artifactVersion
        self.manifestPath = manifestPath
        self.familyVersions = familyVersions
    }
}

public struct BoundingBox: Equatable, Sendable, Codable {
    public let minCorner: [Float]
    public let maxCorner: [Float]

    public init(minCorner: [Float], maxCorner: [Float]) {
        self.minCorner = minCorner
        self.maxCorner = maxCorner
    }
}

public struct LatentResult: Sendable {
    public var latent: RuntimeTensor
    public var predXStart: RuntimeTensor
    public var sigmas: RuntimeTensor
    public var seed: Int
    public var guidanceScale: Double
    public var steps: Int
    public var timings: [String: Double]
    public var artifactInfo: ArtifactVersionInfo

    public init(
        latent: RuntimeTensor,
        predXStart: RuntimeTensor,
        sigmas: RuntimeTensor,
        seed: Int,
        guidanceScale: Double,
        steps: Int,
        timings: [String: Double],
        artifactInfo: ArtifactVersionInfo
    ) {
        self.latent = latent
        self.predXStart = predXStart
        self.sigmas = sigmas
        self.seed = seed
        self.guidanceScale = guidanceScale
        self.steps = steps
        self.timings = timings
        self.artifactInfo = artifactInfo
    }
}

public struct MeshResult: Sendable, Codable {
    public var vertices: RuntimeTensor
    public var faces: RuntimeTensor
    public var vertexChannels: [String: RuntimeTensor]?
    public var bbox: BoundingBox
    public var timings: [String: Double]
    public var artifactInfo: ArtifactVersionInfo

    public init(
        vertices: RuntimeTensor,
        faces: RuntimeTensor,
        vertexChannels: [String: RuntimeTensor]?,
        bbox: BoundingBox,
        timings: [String: Double],
        artifactInfo: ArtifactVersionInfo
    ) {
        self.vertices = vertices
        self.faces = faces
        self.vertexChannels = vertexChannels
        self.bbox = bbox
        self.timings = timings
        self.artifactInfo = artifactInfo
    }
}

public struct MeshBatchResult: Sendable {
    public var meshes: [MeshResult]
    public var bbox: BoundingBox
    public var timings: [String: Double]
    public var artifactInfo: ArtifactVersionInfo

    public init(
        meshes: [MeshResult],
        bbox: BoundingBox,
        timings: [String: Double],
        artifactInfo: ArtifactVersionInfo
    ) {
        self.meshes = meshes
        self.bbox = bbox
        self.timings = timings
        self.artifactInfo = artifactInfo
    }
}
