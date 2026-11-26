import Foundation
import XCTest
@testable import ShapERuntime

final class ClipBPELocationTests: XCTestCase {
    private func makeBundle(includeBPE: Bool) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("shap-e-bpe-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        if includeBPE {
            try Data("bpe-stub".utf8).write(to: ClipBPELocation.fileURL(inBundle: root))
        }
        return root
    }

    func testOmittingCLIPathUsesBPEFileBesideTheBundle() throws {
        let bundle = try makeBundle(includeBPE: true)
        defer { try? FileManager.default.removeItem(at: bundle) }

        let resolved = ClipBPELocation.resolve(cliPath: nil, bundleURL: bundle)

        XCTAssertEqual(
            resolved?.standardizedFileURL.path,
            ClipBPELocation.fileURL(inBundle: bundle).standardizedFileURL.path
        )
    }

    func testExplicitCLIPathWinsOverTheBundleFile() throws {
        let bundle = try makeBundle(includeBPE: true)
        defer { try? FileManager.default.removeItem(at: bundle) }
        let other = bundle.appendingPathComponent("other-vocab.txt.gz")
        try Data("other".utf8).write(to: other)

        let resolved = ClipBPELocation.resolve(cliPath: other.path, bundleURL: bundle)

        XCTAssertEqual(resolved?.standardizedFileURL.path, other.standardizedFileURL.path)
    }

    func testMissingExplicitPathFallsBackToTheBundleFile() throws {
        let bundle = try makeBundle(includeBPE: true)
        defer { try? FileManager.default.removeItem(at: bundle) }

        let resolved = ClipBPELocation.resolve(
            cliPath: bundle.appendingPathComponent("missing.txt.gz").path,
            bundleURL: bundle
        )

        XCTAssertEqual(
            resolved?.standardizedFileURL.path,
            ClipBPELocation.fileURL(inBundle: bundle).standardizedFileURL.path
        )
    }

    func testOmittingCLIPathReturnsNilWhenTheBundleHasNoBPEFile() throws {
        let bundle = try makeBundle(includeBPE: false)
        defer { try? FileManager.default.removeItem(at: bundle) }

        XCTAssertNil(ClipBPELocation.resolve(cliPath: nil, bundleURL: bundle))
        XCTAssertNil(ClipBPELocation.resolve(cliPath: "   ", bundleURL: bundle))
    }
}
