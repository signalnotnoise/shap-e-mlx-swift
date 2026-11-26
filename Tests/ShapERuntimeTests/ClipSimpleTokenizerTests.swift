import XCTest
@testable import ShapERuntime

final class ClipSimpleTokenizerTests: XCTestCase {
    /// BPE file shipped at the root of the Hugging Face runtime bundle.
    /// The unit suite does not download that bundle. These parity tests skip
    /// unless a local copy is already on disk.
    private var vocabularyURL: URL? {
        let environment = ProcessInfo.processInfo.environment
        var candidates: [URL] = []
        if let explicit = environment["SHAP_E_BPE"], !explicit.isEmpty {
            candidates.append(URL(fileURLWithPath: explicit, isDirectory: false))
        }
        if let bundle = environment["SHAP_E_BUNDLE"], !bundle.isEmpty {
            candidates.append(
                URL(fileURLWithPath: bundle, isDirectory: true)
                    .appendingPathComponent("bpe_simple_vocab_16e6.txt.gz", isDirectory: false)
            )
        }
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        candidates.append(
            packageRoot.appendingPathComponent(
                "runtime_bundle/bpe_simple_vocab_16e6.txt.gz",
                isDirectory: false
            )
        )
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    private func makeTokenizer() throws -> ClipSimpleTokenizer {
        guard let vocabularyURL else {
            throw XCTSkip(
                "CLIP BPE vocabulary is not present. Set SHAP_E_BPE, set SHAP_E_BUNDLE, "
                    + "or place bpe_simple_vocab_16e6.txt.gz in runtime_bundle/."
            )
        }
        return try ClipSimpleTokenizer(vocabularyURL: vocabularyURL)
    }

    func testEncodeParityForRepresentativePrompts() throws {
        let tokenizer = try makeTokenizer()

        XCTAssertEqual(try tokenizer.encode("a red chair"), [320, 736, 4269])
        XCTAssertEqual(try tokenizer.encode("A fancy castle!"), [320, 6733, 3540, 256])
        XCTAssertEqual(try tokenizer.encode("hello   world"), [3306, 1002])
        XCTAssertEqual(try tokenizer.encode("cafe\u{0301}"), [15304])
    }

    func testTokenizeParityAndPadding() throws {
        let tokenizer = try makeTokenizer()

        XCTAssertEqual(
            try tokenizer.tokenize(["a red chair"], contextLength: 8)[0],
            [49406, 320, 736, 4269, 49407, 0, 0, 0]
        )
        XCTAssertEqual(
            try tokenizer.tokenize(["A fancy castle!"], contextLength: 8)[0],
            [49406, 320, 6733, 3540, 256, 49407, 0, 0]
        )
        XCTAssertEqual(
            try tokenizer.tokenize(["hello   world"], contextLength: 8)[0],
            [49406, 3306, 1002, 49407, 0, 0, 0, 0]
        )
    }

    func testTokenizeTruncationKeepsEndOfText() throws {
        let tokenizer = try makeTokenizer()
        let prompt = "this is a deliberately longer prompt to exercise truncation in the clip tokenizer path"

        XCTAssertEqual(
            try tokenizer.tokenize([prompt], contextLength: 10)[0],
            [49406, 589, 533, 320, 35163, 5349, 20198, 531, 7016, 49407]
        )
    }

    func testLoadsClipTextArtifactFromRuntimeBundle() throws {
        let bundleURL = RuntimeBundleFixture.bundleURL
        let bundle = try RuntimeArtifactBundle(bundleURL: bundleURL)
        let artifact = try bundle.loadClipTextArtifact()

        XCTAssertEqual(artifact.contextLength, 77)
        XCTAssertEqual(artifact.transformerWidth, 8)
        XCTAssertEqual(artifact.transformerHeads, 2)
        XCTAssertEqual(artifact.transformerLayers, 3)
        XCTAssertEqual(artifact.state["text_projection"]?.shape, [2, 2])
        XCTAssertEqual(artifact.state["token_embedding.weight"]?.shape, [4, 2])
    }
}
