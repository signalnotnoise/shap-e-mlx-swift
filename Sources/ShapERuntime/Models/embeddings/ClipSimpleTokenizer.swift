import Foundation
import zlib

public enum ClipTokenizerError: Error, Equatable, LocalizedError {
    case invalidVocabulary(URL)
    case unsupportedToken(String)
    case gzipInitFailed(Int32)
    case gzipInflateFailed(Int32)

    public var errorDescription: String? {
        switch self {
        case let .invalidVocabulary(url):
            return "Invalid CLIP tokenizer vocabulary at \(url.path)."
        case let .unsupportedToken(token):
            return "Tokenizer produced a token not present in the CLIP vocabulary: \(token)"
        case let .gzipInitFailed(code):
            return "Failed to initialize gzip decompression (zlib code \(code))."
        case let .gzipInflateFailed(code):
            return "Failed to decompress gzip vocabulary (zlib code \(code))."
        }
    }
}

private struct StringPair: Hashable {
    let first: String
    let second: String
}

public final class ClipSimpleTokenizer: @unchecked Sendable {
    public static let startOfText = "<|startoftext|>"
    public static let endOfText = "<|endoftext|>"

    public let vocabularyURL: URL
    public let encoder: [String: Int]
    public let startOfTextToken: Int
    public let endOfTextToken: Int

    private let byteEncoder: [UInt8: String]
    private let byteEncoderValues: [String]
    private let bpeRanks: [StringPair: Int]
    private let pattern: NSRegularExpression
    private var cache: [String: [String]]

    public init(vocabularyURL: URL) throws {
        self.vocabularyURL = vocabularyURL.standardizedFileURL
        let byteEncoding = Self.buildByteEncoder()
        self.byteEncoder = byteEncoding.mapping
        self.byteEncoderValues = byteEncoding.values

        let merges = try Self.loadMerges(from: self.vocabularyURL)
        var vocabulary = self.byteEncoderValues
        vocabulary.append(contentsOf: vocabulary.map { "\($0)</w>" })
        vocabulary.append(contentsOf: merges.map { $0.first + $0.second })
        vocabulary.append(Self.startOfText)
        vocabulary.append(Self.endOfText)

        var encoder: [String: Int] = [:]
        encoder.reserveCapacity(vocabulary.count)
        for (index, token) in vocabulary.enumerated() {
            encoder[token] = index
        }
        self.encoder = encoder
        self.startOfTextToken = encoder[Self.startOfText] ?? 49406
        self.endOfTextToken = encoder[Self.endOfText] ?? 49407
        self.bpeRanks = Dictionary(uniqueKeysWithValues: merges.enumerated().map { ($0.element, $0.offset) })
        self.pattern = try NSRegularExpression(
            pattern: #"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#,
            options: [.caseInsensitive]
        )
        self.cache = [
            Self.startOfText: [Self.startOfText],
            Self.endOfText: [Self.endOfText],
        ]
    }

    public func encode(_ text: String) throws -> [Int] {
        let cleaned = Self.whitespaceClean(Self.basicClean(text)).lowercased()
        let nsRange = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
        let matches = pattern.matches(in: cleaned, options: [], range: nsRange)
        var encoded: [Int] = []

        for match in matches {
            guard let range = Range(match.range, in: cleaned) else {
                continue
            }
            let token = String(cleaned[range])
            let transformed = token.utf8.map { byteEncoder[$0] ?? "" }.joined()
            for bpeToken in bpe(transformed) {
                guard let tokenID = encoder[bpeToken] else {
                    throw ClipTokenizerError.unsupportedToken(bpeToken)
                }
                encoded.append(tokenID)
            }
        }

        return encoded
    }

    public func tokenize(_ texts: [String], contextLength: Int = 77) throws -> [[Int32]] {
        precondition(contextLength > 0, "contextLength must be positive.")

        return try texts.map { text in
            var row = [Int32](repeating: 0, count: contextLength)
            var ids = try [startOfTextToken] + encode(text) + [endOfTextToken]
            if ids.count > contextLength {
                ids = Array(ids.prefix(contextLength))
                ids[contextLength - 1] = endOfTextToken
            }
            for (index, value) in ids.enumerated() {
                row[index] = Int32(value)
            }
            return row
        }
    }

    private func bpe(_ token: String) -> [String] {
        if let cached = cache[token] {
            return cached
        }

        guard !token.isEmpty else {
            return []
        }

        var word = token.dropLast().map(String.init)
        word.append(String(token.suffix(1)) + "</w>")
        var pairs = Self.getPairs(word)

        if pairs.isEmpty {
            let single = [token + "</w>"]
            cache[token] = single
            return single
        }

        while true {
            guard let bigram = pairs.min(by: { rank(for: $0) < rank(for: $1) }) else {
                break
            }
            guard bpeRanks[bigram] != nil else {
                break
            }

            var newWord: [String] = []
            var index = 0
            while index < word.count {
                guard let firstIndex = word[index...].firstIndex(of: bigram.first) else {
                    newWord.append(contentsOf: word[index...])
                    break
                }

                newWord.append(contentsOf: word[index..<firstIndex])
                index = firstIndex

                if index < word.count - 1, word[index] == bigram.first, word[index + 1] == bigram.second {
                    newWord.append(bigram.first + bigram.second)
                    index += 2
                } else {
                    newWord.append(word[index])
                    index += 1
                }
            }

            word = newWord
            if word.count == 1 {
                break
            }
            pairs = Self.getPairs(word)
        }

        cache[token] = word
        return word
    }

    private func rank(for pair: StringPair) -> Int {
        bpeRanks[pair] ?? .max
    }

    private static func loadMerges(from vocabularyURL: URL) throws -> [StringPair] {
        let rawData = try Data(contentsOf: vocabularyURL)
        let textData: Data
        if vocabularyURL.pathExtension == "gz" {
            textData = try gunzip(rawData)
        } else {
            textData = rawData
        }
        guard let text = String(data: textData, encoding: .utf8) else {
            throw ClipTokenizerError.invalidVocabulary(vocabularyURL)
        }

        let mergeCount = 49_152 - 256 - 2
        return text
            .split(separator: "\n", omittingEmptySubsequences: false)
            .dropFirst()
            .prefix(mergeCount)
            .compactMap { line in
                let pieces = line.split(separator: " ", omittingEmptySubsequences: true)
                guard pieces.count == 2 else {
                    return nil
                }
                return StringPair(first: String(pieces[0]), second: String(pieces[1]))
            }
    }

    private static func basicClean(_ text: String) -> String {
        htmlUnescape(htmlUnescape(text))
            .precomposedStringWithCanonicalMapping
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func whitespaceClean(_ text: String) -> String {
        text.replacingOccurrences(
            of: #"\s+"#,
            with: " ",
            options: .regularExpression
        ).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func htmlUnescape(_ text: String) -> String {
        var result = ""
        var index = text.startIndex

        while index < text.endIndex {
            if text[index] == "&", let semicolon = text[index...].firstIndex(of: ";") {
                let entityStart = text.index(after: index)
                let entity = String(text[entityStart..<semicolon])
                if let decoded = decodeHTMLEntity(entity) {
                    result.append(decoded)
                    index = text.index(after: semicolon)
                    continue
                }
            }
            result.append(text[index])
            index = text.index(after: index)
        }

        return result
    }

    private static func decodeHTMLEntity(_ entity: String) -> String? {
        switch entity {
        case "amp":
            return "&"
        case "lt":
            return "<"
        case "gt":
            return ">"
        case "quot":
            return "\""
        case "apos":
            return "'"
        case "nbsp":
            return " "
        default:
            break
        }

        if entity.hasPrefix("#x") || entity.hasPrefix("#X") {
            let hex = entity.dropFirst(2)
            if let scalar = UInt32(hex, radix: 16).flatMap(UnicodeScalar.init) {
                return String(scalar)
            }
        } else if entity.hasPrefix("#") {
            let decimal = entity.dropFirst()
            if let scalar = UInt32(decimal).flatMap(UnicodeScalar.init) {
                return String(scalar)
            }
        }

        return nil
    }

    private static func buildByteEncoder() -> (mapping: [UInt8: String], values: [String]) {
        var bytes = Array(UInt8(ascii: "!")...UInt8(ascii: "~"))
        bytes.append(contentsOf: UInt8(0xA1)...UInt8(0xAC))
        bytes.append(contentsOf: UInt8(0xAE)...UInt8(0xFF))

        var codePoints = bytes.map(Int.init)
        var next = 0
        for byte in UInt8.min...UInt8.max where !bytes.contains(byte) {
            bytes.append(byte)
            codePoints.append(256 + next)
            next += 1
        }

        var mapping: [UInt8: String] = [:]
        var values: [String] = []
        mapping.reserveCapacity(bytes.count)
        values.reserveCapacity(bytes.count)
        for (byte, codePoint) in zip(bytes, codePoints) {
            let scalar = String(UnicodeScalar(codePoint)!)
            mapping[byte] = scalar
            values.append(scalar)
        }
        return (mapping, values)
    }

    private static func getPairs(_ word: [String]) -> Set<StringPair> {
        guard word.count >= 2 else {
            return []
        }

        var pairs: Set<StringPair> = []
        pairs.reserveCapacity(word.count - 1)
        for index in 0..<(word.count - 1) {
            pairs.insert(StringPair(first: word[index], second: word[index + 1]))
        }
        return pairs
    }

    private static func gunzip(_ data: Data) throws -> Data {
        guard !data.isEmpty else {
            return Data()
        }

        var stream = z_stream()
        var output = Data()
        let chunkSize = 64 * 1024

        let status = data.withUnsafeBytes { rawBuffer -> Int32 in
            guard let source = rawBuffer.baseAddress?.assumingMemoryBound(to: Bytef.self) else {
                return Z_DATA_ERROR
            }

            stream.next_in = UnsafeMutablePointer(mutating: source)
            stream.avail_in = uInt(rawBuffer.count)
            return inflateInit2_(&stream, 15 + 32, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size))
        }

        guard status == Z_OK else {
            throw ClipTokenizerError.gzipInitFailed(status)
        }

        defer {
            inflateEnd(&stream)
        }

        var inflateStatus: Int32 = Z_OK
        repeat {
            var buffer = [UInt8](repeating: 0, count: chunkSize)
            inflateStatus = buffer.withUnsafeMutableBufferPointer { pointer in
                stream.next_out = pointer.baseAddress
                stream.avail_out = uInt(pointer.count)
                return inflate(&stream, Z_NO_FLUSH)
            }

            let produced = chunkSize - Int(stream.avail_out)
            if produced > 0 {
                output.append(buffer, count: produced)
            }
        } while inflateStatus == Z_OK

        guard inflateStatus == Z_STREAM_END else {
            throw ClipTokenizerError.gzipInflateFailed(inflateStatus)
        }

        return output
    }
}
