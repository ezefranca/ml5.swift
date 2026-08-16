// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "ml5.swift",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ML5", targets: ["ML5"]),
        .executable(name: "ML5SmokeSample", targets: ["ML5SmokeSample"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/swiftlang/swift-testing",
            revision: "swift-6.2.3-RELEASE"
        )
    ],
    targets: [
        .target(
            name: "ML5",
            resources: [.process("Resources/PrivacyInfo.xcprivacy")]
        ),
        .testTarget(
            name: "ML5Tests",
            dependencies: [
                "ML5",
                .product(name: "Testing", package: "swift-testing"),
            ],
            resources: [.copy("Resources/BundledModel.model-fixture")]
        ),
        .executableTarget(name: "ML5SmokeSample", dependencies: ["ML5"]),
    ],
    swiftLanguageModes: [.v6]
)
