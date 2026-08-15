@preconcurrency import CoreVideo
import Foundation

/// A validated, positive tensor shape in row-major dimension order.
@frozen
public struct TensorShape: Sendable, Hashable, Codable {
    /// Positive sizes from the outermost to innermost dimension.
    public let dimensions: [Int]
    /// The product of all dimensions.
    public let elementCount: Int

    /// Creates a nonempty shape with positive dimensions and a representable size.
    ///
    /// - Throws: ``ML5Error/invalidTensorShape(_:)`` for empty, nonpositive, or
    ///   overflowing dimensions.
    public init(_ dimensions: [Int]) throws {
        var count = 1
        for dimension in dimensions {
            let product = count.multipliedReportingOverflow(by: dimension)
            guard dimension > 0, !product.overflow else {
                throw ML5Error.invalidTensorShape(dimensions)
            }
            count = product.partialValue
        }
        guard !dimensions.isEmpty else {
            throw ML5Error.invalidTensorShape(dimensions)
        }
        self.dimensions = dimensions
        self.elementCount = count
    }

    /// Decodes dimensions and revalidates the shape invariants.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(container.decode([Int].self))
    }

    /// Encodes the ordered dimensions as a single array.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(dimensions)
    }
}

/// A dense row-major tensor of finite double-precision values.
@frozen
public struct Tensor: Sendable, Hashable, Codable {
    /// The validated tensor dimensions.
    public let shape: TensorShape
    /// Finite elements in row-major order.
    public let values: [Double]

    /// Creates a tensor whose value count exactly matches its shape.
    ///
    /// - Throws: ``ML5Error/invalidTensorElementCount(expected:actual:)`` or
    ///   ``ML5Error/invalidNumericValue(field:)``.
    public init(shape: TensorShape, values: [Double]) throws {
        guard values.count == shape.elementCount else {
            throw ML5Error.invalidTensorElementCount(
                expected: shape.elementCount,
                actual: values.count
            )
        }
        guard values.allSatisfy(\.isFinite) else {
            throw ML5Error.invalidNumericValue(field: "tensor")
        }
        self.shape = shape
        self.values = values
    }

    /// Decodes storage and revalidates its shape and numeric invariants.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            shape: container.decode(TensorShape.self, forKey: .shape),
            values: container.decode([Double].self, forKey: .values)
        )
    }

    /// Encodes the shape and row-major values.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(shape, forKey: .shape)
        try container.encode(values, forKey: .values)
    }

    private enum CodingKeys: String, CodingKey {
        case shape
        case values
    }
}

/// Byte layout for a value-semantic image feature.
@frozen
public enum ML5ImagePixelFormat: String, CaseIterable, Sendable, Hashable, Codable {
    /// One unsigned luminance byte per pixel.
    case grayscale8
    /// Red, green, blue, and alpha bytes per pixel.
    case rgba8
    /// Blue, green, red, and alpha bytes per pixel.
    case bgra8

    /// Number of stored bytes for one pixel.
    public var bytesPerPixel: Int {
        switch self {
        case .grayscale8:
            1
        case .rgba8, .bgra8:
            4
        }
    }
}

/// Immutable image pixels safe to serialize and move across actors.
@frozen
public struct ML5Image: Sendable, Hashable, Codable {
    /// Pixel width.
    public let width: Int
    /// Pixel height.
    public let height: Int
    /// Stored bytes between the start of adjacent rows.
    public let bytesPerRow: Int
    /// Channel byte layout.
    public let pixelFormat: ML5ImagePixelFormat
    /// Exactly `bytesPerRow * height` bytes, including any row padding.
    public let data: Data

    /// Creates a validated image from owned pixel bytes.
    ///
    /// `bytesPerRow` defaults to the tightly packed row size.
    ///
    /// - Throws: ``ML5Error/invalidImage(reason:)`` for invalid dimensions,
    ///   row stride, overflow, or byte count.
    public init(
        width: Int,
        height: Int,
        bytesPerRow: Int? = nil,
        pixelFormat: ML5ImagePixelFormat,
        data: Data
    ) throws {
        guard width > 0, height > 0 else {
            throw ML5Error.invalidImage(reason: "Width and height must be positive.")
        }
        let minimumRowResult = width.multipliedReportingOverflow(
            by: pixelFormat.bytesPerPixel
        )
        guard !minimumRowResult.overflow else {
            throw ML5Error.invalidImage(reason: "The packed row size overflowed Int.")
        }
        let resolvedBytesPerRow = bytesPerRow ?? minimumRowResult.partialValue
        guard resolvedBytesPerRow >= minimumRowResult.partialValue else {
            throw ML5Error.invalidImage(reason: "The row stride is smaller than one pixel row.")
        }
        let byteCountResult = resolvedBytesPerRow.multipliedReportingOverflow(by: height)
        guard !byteCountResult.overflow, data.count == byteCountResult.partialValue else {
            throw ML5Error.invalidImage(
                reason:
                    "Pixel data must contain exactly bytesPerRow multiplied by height bytes."
            )
        }
        self.width = width
        self.height = height
        self.bytesPerRow = resolvedBytesPerRow
        self.pixelFormat = pixelFormat
        self.data = data
    }

    /// Decodes pixels and revalidates their dimensions and storage size.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            width: container.decode(Int.self, forKey: .width),
            height: container.decode(Int.self, forKey: .height),
            bytesPerRow: container.decode(Int.self, forKey: .bytesPerRow),
            pixelFormat: container.decode(ML5ImagePixelFormat.self, forKey: .pixelFormat),
            data: container.decode(Data.self, forKey: .data)
        )
    }

    /// Encodes dimensions, channel layout, stride, and owned bytes.
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(width, forKey: .width)
        try container.encode(height, forKey: .height)
        try container.encode(bytesPerRow, forKey: .bytesPerRow)
        try container.encode(pixelFormat, forKey: .pixelFormat)
        try container.encode(data, forKey: .data)
    }

    /// Copies a Core Video pixel buffer into immutable, transport-safe storage.
    ///
    /// Supported buffers use 8-bit grayscale, RGBA, or BGRA pixels.
    ///
    /// - Throws: ``ML5Error/invalidImage(reason:)`` when the buffer format or
    ///   accessible storage is unsupported.
    public init(pixelBuffer: CVPixelBuffer) throws {
        try self.init(pixelBuffer: pixelBuffer, operations: .system)
    }

    init(
        pixelBuffer: CVPixelBuffer,
        operations: CoreVideoPixelBufferOperations
    ) throws {
        guard
            let pixelFormat = ML5ImagePixelFormat(
                coreVideoPixelFormat: operations.pixelFormat(pixelBuffer)
            )
        else {
            throw ML5Error.invalidImage(reason: "The Core Video pixel format is unsupported.")
        }
        operations.lock(pixelBuffer, .readOnly)
        defer { operations.unlock(pixelBuffer, .readOnly) }
        guard let baseAddress = operations.baseAddress(pixelBuffer) else {
            throw ML5Error.invalidImage(reason: "The Core Video pixel storage is inaccessible.")
        }
        let width = operations.width(pixelBuffer)
        let height = operations.height(pixelBuffer)
        let bytesPerRow = operations.bytesPerRow(pixelBuffer)
        let byteCount = bytesPerRow.multipliedReportingOverflow(by: height)
        guard !byteCount.overflow else {
            throw ML5Error.invalidImage(reason: "The Core Video byte count overflowed Int.")
        }
        try self.init(
            width: width,
            height: height,
            bytesPerRow: bytesPerRow,
            pixelFormat: pixelFormat,
            data: Data(bytes: baseAddress, count: byteCount.partialValue)
        )
    }

    /// Copies the image into a mutable Core Video pixel buffer for Core ML.
    ///
    /// RGBA storage is channel-swizzled into Core Video's widely supported BGRA
    /// layout; grayscale and BGRA storage retain their channel order.
    ///
    /// - Returns: A newly allocated buffer owned by the caller.
    /// - Throws: ``ML5Error/invalidImage(reason:)`` when Core Video cannot allocate
    ///   or expose the requested storage.
    public func makePixelBuffer() throws -> CVPixelBuffer {
        try makePixelBuffer(operations: .system)
    }

    func makePixelBuffer(
        operations: CoreVideoPixelBufferOperations
    ) throws -> CVPixelBuffer {
        var pixelBuffer: CVPixelBuffer?
        let status = operations.create(
            width,
            height,
            pixelFormat.coreVideoPixelFormat,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else {
            throw ML5Error.invalidImage(reason: "Core Video could not allocate pixel storage.")
        }
        operations.lock(pixelBuffer, [])
        defer { operations.unlock(pixelBuffer, []) }
        guard let destination = operations.baseAddress(pixelBuffer) else {
            throw ML5Error.invalidImage(reason: "The allocated Core Video storage is inaccessible.")
        }

        let destinationBytesPerRow = operations.bytesPerRow(pixelBuffer)
        let copiedBytesPerRow = min(bytesPerRow, destinationBytesPerRow)
        data.withUnsafeBytes { source in
            for row in 0..<height {
                let sourceOffset = row * bytesPerRow
                let destinationRow = destination.advanced(by: row * destinationBytesPerRow)
                if pixelFormat == .rgba8 {
                    let destinationBytes = destinationRow.assumingMemoryBound(to: UInt8.self)
                    for pixel in 0..<width {
                        let destinationOffset = pixel * 4
                        let sourcePixel = sourceOffset + destinationOffset
                        destinationBytes[destinationOffset] = source[sourcePixel + 2]
                        destinationBytes[destinationOffset + 1] = source[sourcePixel + 1]
                        destinationBytes[destinationOffset + 2] = source[sourcePixel]
                        destinationBytes[destinationOffset + 3] = source[sourcePixel + 3]
                    }
                } else {
                    source.copyBytes(
                        to: UnsafeMutableRawBufferPointer(
                            start: destinationRow,
                            count: copiedBytesPerRow
                        ),
                        from: sourceOffset..<(sourceOffset + copiedBytesPerRow)
                    )
                }
            }
        }
        return pixelBuffer
    }

    private enum CodingKeys: String, CodingKey {
        case width
        case height
        case bytesPerRow
        case pixelFormat
        case data
    }
}

struct CoreVideoPixelBufferOperations: @unchecked Sendable {
    var create: (Int, Int, OSType, inout CVPixelBuffer?) -> CVReturn
    var pixelFormat: (CVPixelBuffer) -> OSType
    var width: (CVPixelBuffer) -> Int
    var height: (CVPixelBuffer) -> Int
    var bytesPerRow: (CVPixelBuffer) -> Int
    var baseAddress: (CVPixelBuffer) -> UnsafeMutableRawPointer?
    var lock: (CVPixelBuffer, CVPixelBufferLockFlags) -> Void
    var unlock: (CVPixelBuffer, CVPixelBufferLockFlags) -> Void

    static let system = Self(
        create: { width, height, format, output in
            CVPixelBufferCreate(
                kCFAllocatorDefault,
                width,
                height,
                format,
                nil,
                &output
            )
        },
        pixelFormat: CVPixelBufferGetPixelFormatType,
        width: CVPixelBufferGetWidth,
        height: CVPixelBufferGetHeight,
        bytesPerRow: CVPixelBufferGetBytesPerRow,
        baseAddress: CVPixelBufferGetBaseAddress,
        lock: { buffer, flags in
            CVPixelBufferLockBaseAddress(buffer, flags)
        },
        unlock: { buffer, flags in
            CVPixelBufferUnlockBaseAddress(buffer, flags)
        }
    )
}

/// A homogeneous Core ML-compatible sequence.
@frozen
public enum FeatureSequence: Sendable, Hashable, Codable {
    /// An ordered sequence of Unicode strings.
    case strings([String])
    /// An ordered sequence of signed 64-bit integers.
    case integers([Int64])
}

extension ML5ImagePixelFormat {
    fileprivate init?(coreVideoPixelFormat: OSType) {
        switch coreVideoPixelFormat {
        case kCVPixelFormatType_OneComponent8:
            self = .grayscale8
        case kCVPixelFormatType_32RGBA:
            self = .rgba8
        case kCVPixelFormatType_32BGRA:
            self = .bgra8
        default:
            return nil
        }
    }

    fileprivate var coreVideoPixelFormat: OSType {
        switch self {
        case .grayscale8:
            kCVPixelFormatType_OneComponent8
        case .rgba8, .bgra8:
            kCVPixelFormatType_32BGRA
        }
    }
}
