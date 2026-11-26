# ShapERuntime

A Swift package that runs the Shap-E text-to-3D and image-to-3D pipeline on Apple platforms with [MLX](https://github.com/ml-explore/mlx-swift).

The library product is `ShapERuntime`. The executable product is `ShapECLI`. Model weights are not in this repository.

## Install

This repository has no version tag yet, so the snippet tracks `main`.

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/signalnotnoise/shap-e-mlx-swift.git", branch: "main"),
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: [
            .product(name: "ShapERuntime", package: "shap-e-mlx-swift"),
        ]
    ),
]
```

`import ShapERuntime` in the client target. Platforms declared by this package are macOS 14, iOS 17, and visionOS 1.

## Usage

`bundleURL` is the downloaded runtime bundle directory (`manifest.json` at its root). `bpeURL` is `bpe_simple_vocab_16e6.txt.gz` in that same directory.

```swift
import ShapERuntime

let runtime = try ShapERuntime(
    modelBundleURL: bundleURL,
    bpeVocabularyURL: bpeURL
)

let mesh = try await runtime.generate(prompt: "a red chair")
let vertices = try mesh.verticesArray() // [Float], length V * 3
let faces = try mesh.facesArray()       // [Int32], length F * 3
```

`generate(prompt:)` returns one `ShapEMeshResult`. The `batchSize` overload returns `[ShapEMeshResult]`. `generate(image:)` takes a `CGImage` or an `MLXArray` and returns `[ShapEMeshResult]`. Text weights load in `init`. Image weights (`clip_vision` and `image300m`) load on the first image call.

`ShapECLI` is a separate executable for local parity dumps and a CPU-versus-Metal mesh comparison. Pass the bundle and the BPE file explicitly:

```bash
ShapECLI --ab-mesh \
  --bundle runtime_bundle \
  --bpe runtime_bundle/bpe_simple_vocab_16e6.txt.gz \
  --prompt "a red chair"
```

`run_tests.sh` builds with `xcodebuild` because a plain `swift test` or `swift run` can miss MLX's `default.metallib`. This repository's CI uses that same `xcodebuild` path. The CLI was not run in CI.

## Weights

Weights come from [signalNotNoise/shap-e-mlx](https://huggingface.co/signalNotNoise/shap-e-mlx), pinned to revision `d71f70a09b9c03535188d006a719edda5c19a6eb`. That revision currently lists 1,104 files. Upstream release notes record the snapshot as 5,143,254,750 bytes. This package does not re-measure the payload, and CI does not download it.

`scripts/fetch_weights.py` downloads that revision and checks every line in `checksums.sha256` (64 hex characters, two spaces, relative path). A mismatch or a missing checksummed file is a failure.

```bash
python3 -m pip install huggingface_hub
python3 scripts/fetch_weights.py --output runtime_bundle
```

The directory is gitignored. To check a directory you already have, without contacting Hugging Face:

```bash
python3 scripts/fetch_weights.py --verify-only --output runtime_bundle
```

`--text-only` downloads `clip_text`, `text300m`, `diffusion`, `decoder_params`, `renderer`, and the root files (`manifest.json`, the BPE vocabulary, checksums, provenance, licenses, README). That set is what `generate(prompt:)` loads. `generate(image:)` also needs `clip_vision` and `image300m`, which `--text-only` omits. Text-only verification skips checksum entries for files that were not downloaded. Files that are present still have to match.

## Platforms and what is tested

| Platform | Declared | Tested in this repository's CI |
| --- | --- | --- |
| macOS 14+ | yes | yes, GitHub-hosted `macos-26`, Xcode 26.6, `xcodebuild test -scheme ShapERuntime-Package -destination platform=macOS` |
| iOS 17+ | yes | no |
| visionOS 1+ | yes | no |

The macOS job does not download weights. Unit tests use `Tests/ShapERuntimeTests/Fixtures/runtime_bundle_stub/`, a small random-weight fixture with no trained parameters. Three CLIP tokenizer parity tests throw `XCTSkip` unless `bpe_simple_vocab_16e6.txt.gz` is available via `SHAP_E_BPE`, `SHAP_E_BUNDLE`, or `runtime_bundle/bpe_simple_vocab_16e6.txt.gz`. No test in this package loads the full weight bundle. Linux cannot build the MLX Metal shaders, so there is no Linux job.

## Limitations

- Mesh generation quality, Torch parity, and CPU-versus-Metal mesh checks need the downloaded weights. They are not part of the unit tests or CI. `ShapECLI --parity` and `--image-parity` also expect reference files (for example `case.json` and `python_*.npy`) that are not in this repository.
- The stub fixture has no `clip_vision` or `image300m` tensors, so it cannot exercise image generation.
- iOS and visionOS are declared and are not built here. `.spi.yml` asks the Swift Package Index to build those Apple platforms and to generate documentation for `ShapERuntime`. This CI workflow does not generate documentation, and there is no checked-in DocC catalog.
- There is no semver tag. Swift Package Index listing also requires a public repository. Visibility of this repository is left as it is.

## Attribution

This package is MIT licensed. See [LICENSE](LICENSE).

The Swift sources are a port of [Shap-E](https://github.com/openai/shap-e) (MIT, Copyright (c) 2023 OpenAI) and depend on the CLIP BPE vocabulary from [CLIP](https://github.com/openai/CLIP) (MIT, Copyright (c) 2021 OpenAI). Those license texts are [licenses/Shap-E.txt](licenses/Shap-E.txt) and [licenses/CLIP.txt](licenses/CLIP.txt). MLX and mlx-swift are MIT licensed dependencies from Apple and are not vendored. Full notices are in [NOTICE](NOTICE).

## Provenance

Copied from [`shap_e/swift`](https://github.com/signalnotnoise/apple-mlx-foundations/tree/v0.1.0/shap_e/swift) in [apple-mlx-foundations](https://github.com/signalnotnoise/apple-mlx-foundations) tag `v0.1.0`, commit `018215715177943fe72cea8e540bc7b80dcb593f`. That directory matches `main` of the source repository. Its public history is one snapshot commit (author Alex Otero, 2026-11-17), so this package is a clean copy rather than a `git filter-repo` extraction. Two test paths were updated for the new layout: the non-SwiftPM fixture fallback, and the CLIP vocabulary lookup, which now skips when the weight bundle is absent instead of searching a monorepo virtualenv that does not exist here.
