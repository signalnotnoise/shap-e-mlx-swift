# Engineering standards

Invariants for this package. Generic Swift style is not restated here.

## Tools and the mlx-swift pin

`Package.swift:1` sets `swift-tools-version: 6.2`.

The dependency is `https://github.com/ml-explore/mlx-swift` `"0.31.3"..<"0.31.5"` (`Package.swift:22`), the same range as triposr-mlx-swift. mlx-swift 0.31.5 and later require swift-tools-version 6.3, and this manifest is 6.2, so do not raise the cap without moving the tools version. The floor is 0.31.3, the version `Package.resolved` pins and CI tests.

`Package.resolved:4-11` pins mlx-swift `0.31.3` at revision `61b9e011e09a62b489f6bd647958f1555bdf2896`. `README.md:98` says the recorded M3 Max CLI run used that pin. Do not change the `Package.swift` range without updating that lockfile pin.

## Tensor layout

Shapes below are the ones named in source comments and checks.

- CLIP image input after `ImageProcessor.processCLIPImage`: float32 `[1, 224, 224, 3]`, shorter side scaled to 224, then center-cropped, CLIP mean/std applied (`ImageProcessor.swift:21-26`). The bitmap context is not flipped (`ImageProcessor.swift:61-63`).
- CLIP text tokens are integer rows. `embedText` returns float `[batch, textProjectionDim]` after L2 normalization (`ClipTextEmbedder.swift:237-249`).
- Text300M latent is channel-then-token: `[B, inputChannels, T]`. Output is `[B, outputChannels, T]`, documented as predicted `[eps, var]` (`Text300MTransformer.swift:356-362`).
- `ImageParity.swift:193-194` says Karras latent storage is token-major `[B, tokens*channels]`, while the shared component input is `[B, channels, tokens]`.
- Decoder `project` requires rank-2 latent `[B, D]` (`DecoderParamsProjector.swift:174-185`). `D` must be divisible by `latentCtx` from the artifact (`DecoderParamsProjector.swift:195-201`).
- SDF grids are `[X, Y, Z]` float32, already border-padded. Isosurface level is 0. Output verts are `[V, 3]`, faces `[F, 3]` int32 (`MarchingCubes.swift:18-23`).
- `MeshResult.vertices` and `faces` are `RuntimeTensor` values (`RuntimeCoreTypes.swift:56-58`). OBJ export treats vertex floats as triples and face indexes as triples (`MeshExport.swift:26-27`).

`n_ctx` and `d_latent` are integers in the `text300m` family metadata (`RuntimeArtifactBundle.swift:140-141`). Do not hardcode a production latent length in new call sites. The stub comparison table is `Tests/ShapERuntimeTests/Fixtures/README.md:28-35`.

## Weights and integrity

No trained weights are committed. `.gitignore` ignores `runtime_bundle/`. The checked-in tensors are the random stub under `Tests/ShapERuntimeTests/Fixtures/runtime_bundle_stub/`.

The download pin in `scripts/fetch_weights.py:31-32` is repo `signalNotNoise/shap-e-mlx`, revision `d71f70a09b9c03535188d006a719edda5c19a6eb`. `README.md:96` says Hub tag `v0.1.0` resolves to that commit and that one local run verified 1,102 checksummed files. This file does not restate a byte size beyond the fixture README's "5.143 GB" label (`Fixtures/README.md:30`).

Checksum lines are 64 hex characters, two spaces, then a relative path (`fetch_weights.py:64-76`). `verify_checksums` (`fetch_weights.py:180-218`):

- Full verification fails if any listed file is missing or the digest mismatches.
- `--text-only` skips a missing file only when `is_text_scope_path` is false (`fetch_weights.py:202`).
- Text scope is `ROOT_FILES`, `licenses/*`, and `*.bin` under `clip_text`, `text300m`, `diffusion`, `decoder_params`, and `renderer` (`fetch_weights.py:34-61`).
- Required files are `manifest.json`, `bpe_simple_vocab_16e6.txt.gz`, and the tensor paths the manifest lists for the scope (`fetch_weights.py:140-177`). Text-only omits image-family paths. A checksum list that only names `README.md` fails.
- A directory whose only matching payload is `README.md` must exit nonzero. `test_text_only_fails_when_only_readme_is_present` (`scripts/test_fetch_weights.py:48`) covers that case. `test_truncated_checksum_list_fails_against_manifest` (`scripts/test_fetch_weights.py:106`) covers a checksum list that omits a manifest tensor path.

`RuntimeArtifactBundle` requires `manifest.json` and loads each family listed there (`RuntimeArtifactBundle.swift:49-116`). Blob bytes are raw tensors, not safetensors. Canonical names are the manifest tensor names plus the on-disk file each entry points at. This package does not duplicate a per-tensor byte size table.

BPE search order when `--bpe` is omitted or that path is missing:

1. `ClipBPELocation.resolve` uses an existing `--bpe` path, else `<bundle>/bpe_simple_vocab_16e6.txt.gz` (`ClipBPELocation.swift:20-34`).
2. `resolveBPE` then checks `.mlx_venv` CLIP paths and `./bpe_simple_vocab_16e6.txt.gz` (`main.swift:212-233`).

Bundle discovery looks for `manifest.json` (`main.swift:193-210`).

## Parity

`ShapECLI --parity` and `--image-parity` write `.npy` dumps (`main.swift:255-402`, `ImageParity.swift:89`). `README.md:115` says those modes expect reference files such as `case.json` and `python_*.npy`, and those files are not in this repository. Do not claim a numeric correlation from a run that is not recorded here.

The M3 Max mesh counts in `README.md:109` are a recorded local A/B, not a unit-test assertion.

## Memory and Metal

`ShapECLI` sets `Memory.cacheLimit` to `512 * 1024 * 1024` for the parity dump (`main.swift:266`) and to `768 * 1024 * 1024` for `--ab-mesh` (`main.swift:441`) and image parity (`ImageParity.swift:97`).

Volume queries default to `queryBatchSize` 65536 (`ShapERuntime.swift:256`, `STFMeshRenderer.swift:244`). `generate(prompt:)` without `batchSize` uses CPU marching cubes (`ShapERuntime.swift:229`). The batch overload and the CLI can pass `useMetalMarchingCubes: true`.

Metal kernels are `countTriangles` and `extractMesh` (`MetalMarchingCubes.swift:49-51`). The library is `Bundle.module` (`MetalMarchingCubes.swift:46`). Both backends read `LookupTable.shared`, which is built from `MC_TABLE` (`MarchingCubes.swift:243`, `MetalMarchingCubes.swift:58`).

Tests that touch MLX pin the default stream to CPU in `setUp` (for example `Text300MTransformerTests.swift:10-14`). That does not remove the need for `default.metallib` at process start (`run_tests.sh:3-7`).
