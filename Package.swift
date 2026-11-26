// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "ShapERuntime",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "ShapERuntime",
            targets: ["ShapERuntime"]
        ),
        .executable(
            name: "ShapECLI",
            targets: ["ShapECLI"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.1")
    ],
    targets: [
        .target(
            name: "ShapERuntime",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ],
            resources: [.process("MarchingCubes.metal")]
        ),
        .executableTarget(
            name: "ShapECLI",
            dependencies: [
                "ShapERuntime",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "ShapERuntimeTests",
            dependencies: ["ShapERuntime"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
