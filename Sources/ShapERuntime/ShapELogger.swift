//
// ShapELogger.swift
// ShapERuntime
//

import Foundation

/// A simple logger that hides output behind a debug flag.
public struct ShapELogger {
    /// Set this to `true` to enable print statements.
    nonisolated(unsafe) public static var isDebugEnabled: Bool = false

    /// Logs the provided items if `isDebugEnabled` is `true`.
    public static func debug(_ items: Any..., separator: String = " ", terminator: String = "\n") {
        if isDebugEnabled {
            let output = items.map { "\($0)" }.joined(separator: separator)
            print(output, terminator: terminator)
        }
    }
}
