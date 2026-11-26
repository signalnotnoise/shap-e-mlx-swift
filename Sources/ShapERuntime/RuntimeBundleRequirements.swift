import Foundation

/// Files a runtime bundle must contain before it can be used.
///
/// The set is `manifest.json`, the CLIP BPE file, and the tensor paths the
/// manifest lists. `textOnly` keeps text-family tensors and skips image
/// families. The checksum list is not consulted here. `fetch_weights.py`
/// applies the same set and also requires each path to be checksummed.
public enum RuntimeBundleRequirements {
    public static let manifestFileName = RuntimeArtifactBundle.manifestFileName
    public static let tokenizerFileName = ClipBPELocation.fileName
    public static let textFamilies: Set<String> = [
        "clip_text",
        "text300m",
        "diffusion",
        "decoder_params",
        "renderer",
    ]
    public static let imageFamilies: Set<String> = [
        "clip_vision",
        "image300m",
    ]

    public static func requiredRelativePaths(
        manifest: RuntimeArtifactManifest,
        textOnly: Bool
    ) -> [String] {
        var paths = [manifestFileName, tokenizerFileName]
        for (family, familyManifest) in manifest.families {
            if textOnly && !textFamilies.contains(family) {
                continue
            }
            for entry in familyManifest.tensors.values where !paths.contains(entry.path) {
                paths.append(entry.path)
            }
        }
        return paths
    }

    /// Throws when a required bundle file is absent.
    public static func validate(bundle: RuntimeArtifactBundle, textOnly: Bool) throws {
        for relative in requiredRelativePaths(manifest: bundle.manifest, textOnly: textOnly) {
            let url = bundle.bundleURL.appendingPathComponent(relative)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw RuntimeArtifactError.missingRequiredFile(name: relative, url: url)
            }
        }
    }
}
