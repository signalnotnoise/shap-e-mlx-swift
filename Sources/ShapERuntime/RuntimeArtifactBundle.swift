import Foundation

public enum RuntimeArtifactError: Error, Equatable, LocalizedError {
    case manifestNotFound(URL)
    case familyNotFound(name: String, manifestURL: URL, available: [String])
    case missingMetadata(family: String, key: String)
    case invalidMetadata(family: String, key: String, expected: String)
    case missingTensor(family: String, tensorName: String)
    case tensorShapeMismatch(name: String, expected: [Int], actual: [Int])
    case missingTensorBlob(family: String, tensorName: String, url: URL)
    case unsupportedTensorDataType(tensorName: String, dtype: String)
    case invalidTensorShape(name: String, shape: [Int])
    case invalidTensorByteCount(name: String, expected: Int, actual: Int)
    case tensorTypeMismatch(name: String, expected: TensorDataType, actual: TensorDataType)

    public var errorDescription: String? {
        switch self {
        case let .manifestNotFound(url):
            return "Runtime artifact manifest not found: \(url.path). Prepare artifacts offline before constructing the production runtime."
        case let .familyNotFound(name, manifestURL, available):
            let availableNames = available.sorted().joined(separator: ", ")
            let rendered = availableNames.isEmpty ? "<none>" : availableNames
            return "Runtime artifact family '\(name)' not found in \(manifestURL.path). Available families: \(rendered)."
        case let .missingMetadata(family, key):
            return "Runtime artifact family '\(family)' is missing required metadata '\(key)'."
        case let .invalidMetadata(family, key, expected):
            return "Runtime artifact family '\(family)' has invalid metadata '\(key)'; expected \(expected)."
        case let .missingTensor(family, tensorName):
            return "Runtime artifact family '\(family)' is missing tensor '\(tensorName)' in the manifest."
        case let .tensorShapeMismatch(name, expected, actual):
            return "Tensor '\(name)' has shape \(actual) but expected \(expected)."
        case let .missingTensorBlob(family, tensorName, url):
            return "Tensor blob missing for family '\(family)' tensor '\(tensorName)': \(url.path). Re-export the runtime artifacts offline."
        case let .unsupportedTensorDataType(tensorName, dtype):
            return "Unsupported tensor dtype '\(dtype)' for tensor '\(tensorName)'."
        case let .invalidTensorShape(name, shape):
            return "Invalid tensor shape for '\(name)': \(shape)."
        case let .invalidTensorByteCount(name, expected, actual):
            return "Tensor '\(name)' has \(actual) bytes but expected \(expected)."
        case let .tensorTypeMismatch(name, expected, actual):
            return "Tensor '\(name)' expected \(expected.rawValue) values but stores \(actual.rawValue)."
        }
    }
}

public struct RuntimeArtifactBundle: Sendable {
    public static let manifestFileName = "manifest.json"

    public let bundleURL: URL
    public let manifest: RuntimeArtifactManifest

    public init(bundlePath: String) throws {
        try self.init(bundleURL: URL(fileURLWithPath: bundlePath))
    }

    public init(bundleURL: URL) throws {
        let resolvedBundleURL = bundleURL.standardizedFileURL
        let manifestURL = Self.manifestURL(for: resolvedBundleURL)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw RuntimeArtifactError.manifestNotFound(manifestURL)
        }

        let data = try Data(contentsOf: manifestURL)
        let decoder = JSONDecoder()
        self.bundleURL = resolvedBundleURL
        self.manifest = try decoder.decode(RuntimeArtifactManifest.self, from: data)
    }

    public var manifestURL: URL {
        Self.manifestURL(for: bundleURL)
    }

    public var artifactInfo: ArtifactVersionInfo {
        ArtifactVersionInfo(
            artifactVersion: manifest.artifactVersion,
            manifestPath: manifestURL.path,
            familyVersions: manifest.families.mapValues(\.version)
        )
    }

    public static func manifestURL(for bundleURL: URL) -> URL {
        bundleURL.standardizedFileURL.appendingPathComponent(manifestFileName, isDirectory: false)
    }

    public func familyVersion(_ family: String) throws -> Int {
        try requireFamily(family).version
    }

    public func requireFamily(_ family: String) throws -> ArtifactFamilyManifest {
        guard let familyManifest = manifest.families[family] else {
            throw RuntimeArtifactError.familyNotFound(
                name: family,
                manifestURL: manifestURL,
                available: Array(manifest.families.keys)
            )
        }
        return familyManifest
    }

    public func loadFamilyTensors(_ family: String) throws -> [String: RuntimeTensor] {
        let familyManifest = try requireFamily(family)
        var tensors: [String: RuntimeTensor] = [:]
        tensors.reserveCapacity(familyManifest.tensors.count)

        for (name, entry) in familyManifest.tensors {
            tensors[name] = try TensorBlobLoader.loadTensor(
                from: bundleURL,
                entry: entry,
                family: family
            )
        }
        return tensors
    }

    public func loadClipTextArtifact() throws -> ClipTextArtifact {
        let family = try requireFamily("clip_text")
        let tensors = try loadFamilyTensors("clip_text")

        let contextLength = try metadataInt(family, familyName: "clip_text", key: "context_length")
        let transformerWidth = try metadataInt(family, familyName: "clip_text", key: "transformer_width")
        let transformerHeads = try metadataInt(family, familyName: "clip_text", key: "transformer_heads")
        let transformerLayers = try metadataInt(family, familyName: "clip_text", key: "transformer_layers")

        return ClipTextArtifact(
            contextLength: contextLength,
            transformerWidth: transformerWidth,
            transformerHeads: transformerHeads,
            transformerLayers: transformerLayers,
            state: tensors
        )
    }

    public func loadText300MArtifact() throws -> Text300MArtifact {
        let family = try requireFamily("text300m")
        let tensors = try loadFamilyTensors("text300m")
        return Text300MArtifact(
            nCtx: try metadataInt(family, familyName: "text300m", key: "n_ctx"),
            dLatent: try metadataInt(family, familyName: "text300m", key: "d_latent"),
            state: tensors
        )
    }

    public func loadImage300MArtifact() throws -> Image300MArtifact {
        let family = try requireFamily("image300m")
        let tensors = try loadFamilyTensors("image300m")
        return Image300MArtifact(
            nCtx: try metadataInt(family, familyName: "image300m", key: "n_ctx"),
            dLatent: try metadataInt(family, familyName: "image300m", key: "d_latent"),
            state: tensors
        )
    }

    public func loadClipVisionArtifact() throws -> ClipVisionArtifact {
        let familyName = "clip_vision"
        let family = try requireFamily(familyName)
        let tensors = try loadFamilyTensors(familyName)

        return ClipVisionArtifact(
            inputResolution: try metadataInt(family, familyName: familyName, key: "input_resolution"),
            patchSize: try metadataInt(family, familyName: familyName, key: "patch_size"),
            transformerWidth: try metadataInt(family, familyName: familyName, key: "transformer_width"),
            transformerHeads: try metadataInt(family, familyName: familyName, key: "transformer_heads"),
            transformerLayers: try metadataInt(family, familyName: familyName, key: "transformer_layers"),
            outputDim: try metadataInt(family, familyName: familyName, key: "output_dim"),
            state: tensors
        )
    }

    public func loadDiffusionConfig() throws -> RuntimeDiffusionConfig {
        let familyName = "diffusion"
        let family = try requireFamily(familyName)
        let tensors = try loadFamilyTensors(familyName)

        return RuntimeDiffusionConfig(
            alphasCumprod: try requireTensor(tensors, family: familyName, name: "alphas_cumprod"),
            betas: try requireTensor(tensors, family: familyName, name: "betas"),
            posteriorLogVarianceClipped: try requireTensor(
                tensors,
                family: familyName,
                name: "posterior_log_variance_clipped"
            ),
            sqrtRecipAlphasCumprod: try requireTensor(
                tensors,
                family: familyName,
                name: "sqrt_recip_alphas_cumprod"
            ),
            sqrtRecipm1AlphasCumprod: try requireTensor(
                tensors,
                family: familyName,
                name: "sqrt_recipm1_alphas_cumprod"
            ),
            posteriorMeanCoef1: try requireTensor(tensors, family: familyName, name: "posterior_mean_coef1"),
            posteriorMeanCoef2: try requireTensor(tensors, family: familyName, name: "posterior_mean_coef2"),
            numTimesteps: try metadataInt(family, familyName: familyName, key: "num_timesteps"),
            modelMeanType: try metadataString(
                family,
                familyName: familyName,
                key: "model_mean_type",
                defaultValue: "epsilon"
            ),
            channelBiases: tensors["channel_biases"],
            channelScales: tensors["channel_scales"]
        )
    }

    public func loadDecoderParamsArtifact() throws -> DecoderParamsArtifact {
        let familyName = "decoder_params"
        let family = try requireFamily(familyName)
        let tensors = try loadFamilyTensors(familyName)
        let paramShapes = try metadataParamShapes(family, familyName: familyName, key: "param_shapes")

        var projections: [String: DecoderProjectionWeights] = [:]
        projections.reserveCapacity(paramShapes.count)

        for param in paramShapes {
            let safeName = sanitize(param.name)
            projections[param.name] = DecoderProjectionWeights(
                weightVCD: try requireTensor(tensors, family: familyName, name: "proj_\(safeName)_weight_vcd"),
                biasVC: try requireTensor(tensors, family: familyName, name: "proj_\(safeName)_bias_vc"),
                lnWeightC: try requireTensor(tensors, family: familyName, name: "proj_\(safeName)_ln_weight_c"),
                lnBiasC: try requireTensor(tensors, family: familyName, name: "proj_\(safeName)_ln_bias_c")
            )
        }

        return DecoderParamsArtifact(
            paramShapes: paramShapes,
            latentCtx: try metadataInt(family, familyName: familyName, key: "latent_ctx"),
            latentWarpName: try metadataString(family, familyName: familyName, key: "latent_warp_name"),
            latentWarpCoeff1: try metadataDouble(family, familyName: familyName, key: "latent_warp_coeff1"),
            projections: projections,
            lnEps: try metadataDouble(family, familyName: familyName, key: "ln_eps", defaultValue: 1e-5)
        )
    }

    public func loadRendererArtifact() throws -> RendererArtifact {
        let familyName = "renderer"
        let family = try requireFamily(familyName)
        let tensors = try loadFamilyTensors(familyName)

        var staticBiases: [Int: RuntimeTensor] = [:]
        for layer in 0...6 {
            staticBiases[layer] = try requireTensor(tensors, family: familyName, name: "mlp_\(layer)_bias")
        }

        var staticWeights: [Int: RuntimeTensor] = [:]
        for layer in 4...6 {
            staticWeights[layer] = try requireTensor(tensors, family: familyName, name: "mlp_\(layer)_weight")
        }

        return RendererArtifact(
            gridSize: try metadataInt(family, familyName: familyName, key: "grid_size"),
            bboxMin: try requireTensor(tensors, family: familyName, name: "bbox_min"),
            bboxMax: try requireTensor(tensors, family: familyName, name: "bbox_max"),
            staticWeights: staticWeights,
            staticBiases: staticBiases,
            directionDependentShape: try metadataBool(
                family,
                familyName: familyName,
                key: "direction_dependent_shape",
                defaultValue: true
            ),
            separateNerfChannels: try metadataBool(
                family,
                familyName: familyName,
                key: "separate_nerf_channels",
                defaultValue: true
            ),
            separateCoarseChannels: try metadataBool(
                family,
                familyName: familyName,
                key: "separate_coarse_channels",
                defaultValue: true
            ),
            channelActivation: try metadataString(
                family,
                familyName: familyName,
                key: "channel_activation",
                defaultValue: "sigmoid"
            ),
            sdfActivation: try metadataString(
                family,
                familyName: familyName,
                key: "sdf_activation",
                defaultValue: "tanh"
            ),
            textureChannels: try metadataStringArray(
                family,
                familyName: familyName,
                key: "texture_channels",
                defaultValue: ["R", "G", "B"]
            )
        )
    }

    private func requireTensor(
        _ tensors: [String: RuntimeTensor],
        family: String,
        name: String
    ) throws -> RuntimeTensor {
        guard let tensor = tensors[name] else {
            throw RuntimeArtifactError.missingTensor(family: family, tensorName: name)
        }
        return tensor
    }

    private func metadataInt(
        _ family: ArtifactFamilyManifest,
        familyName: String,
        key: String
    ) throws -> Int {
        guard let value = family.metadata[key] else {
            throw RuntimeArtifactError.missingMetadata(family: familyName, key: key)
        }
        guard let intValue = value.intValue else {
            throw RuntimeArtifactError.invalidMetadata(family: familyName, key: key, expected: "an integer")
        }
        return intValue
    }

    private func metadataDouble(
        _ family: ArtifactFamilyManifest,
        familyName: String,
        key: String,
        defaultValue: Double? = nil
    ) throws -> Double {
        guard let value = family.metadata[key] else {
            guard let defaultValue else {
                throw RuntimeArtifactError.missingMetadata(family: familyName, key: key)
            }
            return defaultValue
        }
        guard let doubleValue = value.doubleValue else {
            throw RuntimeArtifactError.invalidMetadata(family: familyName, key: key, expected: "a number")
        }
        return doubleValue
    }

    private func metadataBool(
        _ family: ArtifactFamilyManifest,
        familyName: String,
        key: String,
        defaultValue: Bool? = nil
    ) throws -> Bool {
        guard let value = family.metadata[key] else {
            guard let defaultValue else {
                throw RuntimeArtifactError.missingMetadata(family: familyName, key: key)
            }
            return defaultValue
        }
        guard let boolValue = value.boolValue else {
            throw RuntimeArtifactError.invalidMetadata(family: familyName, key: key, expected: "a boolean")
        }
        return boolValue
    }

    private func metadataString(
        _ family: ArtifactFamilyManifest,
        familyName: String,
        key: String,
        defaultValue: String? = nil
    ) throws -> String {
        guard let value = family.metadata[key] else {
            guard let defaultValue else {
                throw RuntimeArtifactError.missingMetadata(family: familyName, key: key)
            }
            return defaultValue
        }
        guard let stringValue = value.stringValue else {
            throw RuntimeArtifactError.invalidMetadata(family: familyName, key: key, expected: "a string")
        }
        return stringValue
    }

    private func metadataStringArray(
        _ family: ArtifactFamilyManifest,
        familyName: String,
        key: String,
        defaultValue: [String]? = nil
    ) throws -> [String] {
        guard let value = family.metadata[key] else {
            guard let defaultValue else {
                throw RuntimeArtifactError.missingMetadata(family: familyName, key: key)
            }
            return defaultValue
        }
        guard let arrayValue = value.arrayValue else {
            throw RuntimeArtifactError.invalidMetadata(family: familyName, key: key, expected: "an array of strings")
        }
        let strings = arrayValue.compactMap(\.stringValue)
        guard strings.count == arrayValue.count else {
            throw RuntimeArtifactError.invalidMetadata(family: familyName, key: key, expected: "an array of strings")
        }
        return strings
    }

    private func metadataParamShapes(
        _ family: ArtifactFamilyManifest,
        familyName: String,
        key: String
    ) throws -> [DecoderParamShape] {
        guard let value = family.metadata[key] else {
            throw RuntimeArtifactError.missingMetadata(family: familyName, key: key)
        }
        guard let entries = value.arrayValue else {
            throw RuntimeArtifactError.invalidMetadata(
                family: familyName,
                key: key,
                expected: "an array of [name, shape] pairs"
            )
        }

        return try entries.map { entry in
            guard let pair = entry.arrayValue, pair.count == 2 else {
                throw RuntimeArtifactError.invalidMetadata(
                    family: familyName,
                    key: key,
                    expected: "an array of [name, shape] pairs"
                )
            }
            guard let name = pair[0].stringValue else {
                throw RuntimeArtifactError.invalidMetadata(
                    family: familyName,
                    key: key,
                    expected: "parameter names as strings"
                )
            }
            guard let shapeValues = pair[1].arrayValue else {
                throw RuntimeArtifactError.invalidMetadata(
                    family: familyName,
                    key: key,
                    expected: "shape arrays of integers"
                )
            }
            let shape = shapeValues.compactMap(\.intValue)
            guard shape.count == shapeValues.count else {
                throw RuntimeArtifactError.invalidMetadata(
                    family: familyName,
                    key: key,
                    expected: "shape arrays of integers"
                )
            }
            return DecoderParamShape(name: name, shape: shape)
        }
    }

    private func sanitize(_ name: String) -> String {
        name.replacingOccurrences(of: ".", with: "__")
    }
}
