import Foundation

/// Where ShapECLI finds the CLIP BPE vocabulary.
public enum ClipBPELocation {
    /// File name stored beside `manifest.json` in the Hugging Face runtime bundle.
    public static let fileName = "bpe_simple_vocab_16e6.txt.gz"

    /// `<bundle>/bpe_simple_vocab_16e6.txt.gz`.
    public static func fileURL(inBundle bundleURL: URL) -> URL {
        bundleURL.appendingPathComponent(fileName, isDirectory: false)
    }

    /// Resolves the vocabulary path ShapECLI should open.
    ///
    /// An explicit `--bpe` path is used when that file exists. When the flag is
    /// omitted, or when that path is missing, this returns
    /// `<bundle>/bpe_simple_vocab_16e6.txt.gz` if the file is there.
    /// Returns nil when neither file exists. `bundleURL` may be nil when the
    /// bundle has not been resolved yet.
    public static func resolve(cliPath: String?, bundleURL: URL?) -> URL? {
        if let cliPath {
            let trimmed = cliPath.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                let explicit = URL(fileURLWithPath: trimmed)
                if FileManager.default.fileExists(atPath: explicit.path) {
                    return explicit
                }
            }
        }
        guard let bundleURL else { return nil }
        let bundled = fileURL(inBundle: bundleURL)
        guard FileManager.default.fileExists(atPath: bundled.path) else { return nil }
        return bundled
    }
}
