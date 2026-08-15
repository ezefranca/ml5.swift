@preconcurrency import CoreVideo
import Foundation
import Testing

@testable import ML5

@Suite("ML5 structured values and schemas")
struct ML5StructuredValueTests {
    @Test("Tensor shapes and storage enforce dense finite invariants")
    func tensors() throws {
        let shape = try TensorShape([2, 3])
        let tensor = try Tensor(shape: shape, values: [0, 1, 2, 3, 4, 5])

        #expect(shape.dimensions == [2, 3])
        #expect(shape.elementCount == 6)
        #expect(tensor.values.last == 5)
        #expect(
            try JSONDecoder().decode(Tensor.self, from: JSONEncoder().encode(tensor)) == tensor
        )

        for dimensions in [[], [2, 0], [-1], [Int.max, 2]] {
            #expect(throws: ML5Error.invalidTensorShape(dimensions)) {
                _ = try TensorShape(dimensions)
            }
        }
        #expect(throws: ML5Error.invalidTensorElementCount(expected: 6, actual: 1)) {
            _ = try Tensor(shape: shape, values: [1])
        }
        #expect(throws: ML5Error.invalidNumericValue(field: "tensor")) {
            _ = try Tensor(shape: shape, values: [0, 1, 2, 3, 4, .infinity])
        }
    }

    @Test(
        "Images validate storage and round-trip through Core Video",
        arguments: ML5ImagePixelFormat.allCases)
    func images(_ format: ML5ImagePixelFormat) throws {
        let bytes = Array(0..<(2 * format.bytesPerPixel)).map(UInt8.init)
        let image = try ML5Image(
            width: 2,
            height: 1,
            pixelFormat: format,
            data: Data(bytes)
        )

        #expect(image.width == 2)
        #expect(image.height == 1)
        #expect(image.bytesPerRow == bytes.count)
        #expect(image.pixelFormat == format)
        #expect(image.data == Data(bytes))
        #expect(
            try JSONDecoder().decode(ML5Image.self, from: JSONEncoder().encode(image)) == image
        )

        let pixelBuffer = try image.makePixelBuffer()
        let decoded = try ML5Image(pixelBuffer: pixelBuffer)
        #expect(decoded.width == image.width)
        #expect(decoded.height == image.height)
        if format == .rgba8 {
            #expect(decoded.pixelFormat == .bgra8)
            #expect(decoded.data.prefix(bytes.count) == Data([2, 1, 0, 3, 6, 5, 4, 7]))
        } else {
            #expect(decoded.pixelFormat == image.pixelFormat)
            #expect(decoded.data.prefix(bytes.count) == Data(bytes))
        }
    }

    @Test("Images reject invalid dimensions, stride, overflow, and byte counts")
    func invalidImages() {
        #expect(throws: ML5Error.self) {
            _ = try ML5Image(width: 0, height: 1, pixelFormat: .grayscale8, data: Data())
        }
        #expect(throws: ML5Error.self) {
            _ = try ML5Image(
                width: Int.max,
                height: 1,
                pixelFormat: .rgba8,
                data: Data()
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try ML5Image(
                width: 2,
                height: 1,
                bytesPerRow: 1,
                pixelFormat: .grayscale8,
                data: Data()
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try ML5Image(
                width: 1,
                height: Int.max,
                bytesPerRow: 2,
                pixelFormat: .grayscale8,
                data: Data()
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try ML5Image(
                width: 2,
                height: 2,
                pixelFormat: .grayscale8,
                data: Data([0, 1])
            )
        }
    }

    @Test("Unsupported Core Video pixel formats are rejected")
    func unsupportedPixelFormat() throws {
        let pixelBuffer = try makePixelBuffer(format: kCVPixelFormatType_64RGBAHalf)

        #expect(throws: ML5Error.self) {
            _ = try ML5Image(pixelBuffer: pixelBuffer)
        }
    }

    @Test("Core Video failures and externally supplied RGBA layouts are explicit")
    func coreVideoFailures() throws {
        let image = try ML5Image(
            width: 1,
            height: 1,
            pixelFormat: .grayscale8,
            data: Data([42])
        )
        let pixelBuffer = try makePixelBuffer(format: kCVPixelFormatType_OneComponent8)

        var inaccessibleInput = CoreVideoPixelBufferOperations.system
        inaccessibleInput.baseAddress = { _ in nil }
        #expect(throws: ML5Error.self) {
            _ = try ML5Image(pixelBuffer: pixelBuffer, operations: inaccessibleInput)
        }

        var overflowingInput = CoreVideoPixelBufferOperations.system
        overflowingInput.height = { _ in Int.max }
        overflowingInput.bytesPerRow = { _ in 2 }
        #expect(throws: ML5Error.self) {
            _ = try ML5Image(pixelBuffer: pixelBuffer, operations: overflowingInput)
        }

        var rgbaInput = CoreVideoPixelBufferOperations.system
        rgbaInput.pixelFormat = { _ in kCVPixelFormatType_32RGBA }
        #expect(
            try ML5Image(pixelBuffer: pixelBuffer, operations: rgbaInput).pixelFormat == .rgba8
        )

        var failedCreation = CoreVideoPixelBufferOperations.system
        failedCreation.create = { _, _, _, _ in kCVReturnInvalidArgument }
        #expect(throws: ML5Error.self) {
            _ = try image.makePixelBuffer(operations: failedCreation)
        }

        var missingCreation = CoreVideoPixelBufferOperations.system
        missingCreation.create = { _, _, _, _ in kCVReturnSuccess }
        #expect(throws: ML5Error.self) {
            _ = try image.makePixelBuffer(operations: missingCreation)
        }

        var inaccessibleOutput = CoreVideoPixelBufferOperations.system
        inaccessibleOutput.baseAddress = { _ in nil }
        #expect(throws: ML5Error.self) {
            _ = try image.makePixelBuffer(operations: inaccessibleOutput)
        }
    }

    @Test("Every feature kind validates, projects, hashes, and serializes")
    func featureKinds() throws {
        let tensor = try Tensor(shape: TensorShape([2]), values: [1, 2])
        let image = try ML5Image(
            width: 1,
            height: 1,
            pixelFormat: .rgba8,
            data: Data([1, 2, 3, 4])
        )
        let values: [FeatureValue] = [
            .number(1.5),
            .integer(2),
            .string("three"),
            .boolean(true),
            .array([4, 5]),
            .dictionary(["six": 6]),
            .tensor(tensor),
            .sequence(.strings(["seven"])),
            .sequence(.integers([8])),
            .image(image),
        ]

        #expect(
            values.map(\.kind) == [
                .number, .integer, .string, .boolean, .array, .dictionary, .tensor,
                .sequence, .sequence, .image,
            ]
        )
        #expect(values[0].numericValue == 1.5)
        #expect(values[1].numericValue == 2)
        for value in values.dropFirst(2) {
            #expect(value.numericValue == nil)
        }

        let vector = try FeatureVector(
            Dictionary(
                uniqueKeysWithValues: values.enumerated().map {
                    (FeatureName(rawValue: "value\($0.offset)"), $0.element)
                })
        )
        #expect(
            try JSONDecoder().decode(FeatureVector.self, from: JSONEncoder().encode(vector))
                == vector
        )
        let output = try ModelOutput(["value": .tensor(tensor)])
        #expect(
            try JSONDecoder().decode(ModelOutput.self, from: JSONEncoder().encode(output))
                == output
        )
    }

    @Test("Structured feature validation rejects malformed collections")
    func invalidCollections() throws {
        #expect(throws: ML5Error.emptyCollection(field: "value")) {
            _ = try FeatureVector(["value": .array([])])
        }
        #expect(throws: ML5Error.invalidNumericValue(field: "value")) {
            _ = try FeatureVector(["value": .array([.nan])])
        }
        #expect(throws: ML5Error.emptyCollection(field: "value")) {
            _ = try FeatureVector(["value": .dictionary([:])])
        }
        for key in ["", " untrimmed"] {
            #expect(throws: ML5Error.invalidDictionaryKey(field: "value")) {
                _ = try FeatureVector(["value": .dictionary([key: 1])])
            }
        }
        #expect(throws: ML5Error.invalidNumericValue(field: "value")) {
            _ = try FeatureVector(["value": .dictionary(["key": .infinity])])
        }

        let valid = try Tensor(shape: TensorShape([1]), values: [1])
        #expect(try FeatureVector(["value": .tensor(valid)])["value"] == .tensor(valid))
    }

    @Test("Decoding cannot bypass public value invariants")
    func invalidDecoding() throws {
        let decoder = JSONDecoder()
        #expect(throws: ML5Error.self) {
            _ = try decoder.decode(FeatureName.self, from: Data(#"" bad ""#.utf8))
        }
        #expect(throws: ML5Error.self) {
            _ = try decoder.decode(OutputName.self, from: Data(#"" bad ""#.utf8))
        }
        #expect(throws: ML5Error.self) {
            _ = try decoder.decode(TensorShape.self, from: Data("[0]".utf8))
        }
        #expect(throws: ML5Error.self) {
            _ = try decoder.decode(
                Tensor.self,
                from: Data(#"{"shape":[2],"values":[1]}"#.utf8)
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try decoder.decode(
                ML5Image.self,
                from: Data(
                    #"{"width":1,"height":1,"bytesPerRow":1,"pixelFormat":"grayscale8","data":""}"#
                        .utf8
                )
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try decoder.decode(
                FeatureValue.self,
                from: Data(#"{"kind":"array","array":[]}"#.utf8)
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try decoder.decode(
                FeatureVector.self,
                from: Data(#"{" bad ":{"kind":"number","number":1}}"#.utf8)
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try decoder.decode(
                ModelOutput.self,
                from: Data(#"{" bad ":{"kind":"number","number":1}}"#.utf8)
            )
        }
        #expect(throws: ML5Error.self) {
            _ = try decoder.decode(
                FeatureField.self,
                from: Data(#"{"name":"x","kind":"tensor","isRequired":true}"#.utf8)
            )
        }

        let field = try FeatureField(name: "x", kind: .number)
        let duplicateData = try JSONEncoder().encode([field, field])
        #expect(throws: ML5Error.duplicateFeatureName("x")) {
            _ = try decoder.decode(FeatureSchema.self, from: duplicateData)
        }
    }

    @Test("Feature fields reject contradictory kinds, shapes, and defaults")
    func invalidFields() throws {
        let shape = try TensorShape([2])
        let otherShape = try TensorShape([1])
        let otherTensor = try Tensor(shape: otherShape, values: [1])

        #expect(throws: ML5Error.self) {
            _ = try FeatureField(name: "value", kind: .tensor)
        }
        #expect(throws: ML5Error.self) {
            _ = try FeatureField(name: "value", kind: .number, tensorShape: shape)
        }
        #expect(
            throws: ML5Error.featureKindMismatch(
                name: "value", expected: .number, actual: .string
            )
        ) {
            _ = try FeatureField(name: "value", kind: .number, defaultValue: .string("x"))
        }
        #expect(throws: ML5Error.emptyCollection(field: "value")) {
            _ = try FeatureField(name: "value", kind: .array, defaultValue: .array([]))
        }
        #expect(
            throws: ML5Error.tensorShapeMismatch(
                name: "value", expected: [2], actual: [1]
            )
        ) {
            _ = try FeatureField(
                name: "value",
                kind: .tensor,
                tensorShape: shape,
                defaultValue: .tensor(otherTensor)
            )
        }
    }

    @Test("Ordered schemas apply strict, default, optional, and unknown policies")
    func schemas() throws {
        let shape = try TensorShape([2])
        let tensor = try Tensor(shape: shape, values: [1, 2])
        let fields = [
            try FeatureField(name: "input", kind: .tensor, tensorShape: shape),
            try FeatureField(name: "bias", kind: .number, defaultValue: .number(0.5)),
            try FeatureField(name: "note", kind: .string, isRequired: false),
        ]
        let schema = try FeatureSchema(fields)

        #expect(schema.fields == fields)
        #expect(schema.names == ["input", "bias", "note"])
        #expect(
            try JSONDecoder().decode(FeatureSchema.self, from: JSONEncoder().encode(schema))
                == schema
        )

        let complete = try FeatureVector([
            "input": .tensor(tensor),
            "bias": .number(1),
            "note": .string("ready"),
        ])
        #expect(try schema.resolve(complete) == complete)
        #expect(
            try schema.orderedValues(in: complete) == [
                .tensor(tensor), .number(1), .string("ready"),
            ]
        )

        let minimal = try FeatureVector(["input": .tensor(tensor)])
        let defaulted = try schema.resolve(minimal, missing: .useDefaults)
        #expect(defaulted["bias"] == .number(0.5))
        #expect(defaulted["note"] == nil)

        let unknown = try FeatureVector([
            "input": .tensor(tensor),
            "aaa": .integer(1),
            "extra": .boolean(true),
        ])
        let preserved = try schema.resolve(
            unknown,
            missing: .useDefaults,
            unknown: .preserve
        )
        #expect(preserved["extra"] == .boolean(true))
        #expect(preserved["aaa"] == .integer(1))

        #expect(throws: ML5Error.unexpectedFeature("aaa")) {
            try schema.resolve(unknown, missing: .useDefaults)
        }
        #expect(throws: ML5Error.missingFeature("bias")) {
            try schema.resolve(minimal)
        }
        #expect(throws: ML5Error.missingFeature("input")) {
            try schema.resolve(
                FeatureVector(["bias": .number(1)]),
                missing: .useDefaults
            )
        }
        #expect(
            throws: ML5Error.featureKindMismatch(
                name: "input", expected: .tensor, actual: .array
            )
        ) {
            try schema.resolve(
                FeatureVector([
                    "input": .array([1, 2]),
                    "bias": .number(1),
                    "note": .string("ready"),
                ])
            )
        }

        let wrongTensor = try Tensor(shape: TensorShape([1, 2]), values: [1, 2])
        #expect(
            throws: ML5Error.tensorShapeMismatch(
                name: "input", expected: [2], actual: [1, 2]
            )
        ) {
            try schema.resolve(
                FeatureVector([
                    "input": .tensor(wrongTensor),
                    "bias": .number(1),
                    "note": .string("ready"),
                ])
            )
        }
    }

    @Test("Schemas require at least one uniquely named field")
    func invalidSchemas() throws {
        #expect(throws: ML5Error.self) {
            _ = try FeatureSchema([])
        }
        let field = try FeatureField(name: "value", kind: .number)
        #expect(throws: ML5Error.duplicateFeatureName("value")) {
            _ = try FeatureSchema([field, field])
        }
    }
}

private func makePixelBuffer(format: OSType) throws -> CVPixelBuffer {
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        1,
        1,
        format,
        nil,
        &buffer
    )
    #expect(status == kCVReturnSuccess)
    return try #require(buffer)
}
