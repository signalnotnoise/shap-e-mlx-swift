# Memory and performance

## Cache limit

`ShapECLI` sets MLX `Memory.cacheLimit` in three places:

| Mode | Bytes | Source |
| --- | --- | --- |
| `--parity` | `512 * 1024 * 1024` | `Sources/ShapECLI/main.swift:266` |
| `--ab-mesh` | `768 * 1024 * 1024` | `Sources/ShapECLI/main.swift:441` |
| `--image-parity` | `768 * 1024 * 1024` | `Sources/ShapECLI/ImageParity.swift:97` |

`ShapERuntime` does not set `cacheLimit`. Callers that are not `ShapECLI` keep MLX's own default. This repository does not document that default.

## Batching

`generate` defaults `queryBatchSize` to 65536 and `gridSize` to 128 (`ShapERuntime.swift:255-256`). `STFMeshRenderer.extractMeshes` uses the same query-batch default (`STFMeshRenderer.swift:244-253`) and rejects a grid smaller than 2.

The single-prompt `generate` always uses CPU marching cubes (`ShapERuntime.swift:229`). Metal is available on the batch overload (`ShapERuntime.swift:258`) and on `ShapECLI --ab-mesh`.

Partial previews inside the batch sampler also force CPU marching cubes. The comment is `ShapERuntime.swift:304`.

## M3 Max red chair

These numbers are copied from `README.md:90-111`. They are one local measurement, not a CI assertion.

- Machine: Apple M3 Max, 14 CPU cores (10 performance, 4 efficiency), 30 GPU cores, 36 GB unified memory. macOS 27.3 (26C5047e), Xcode 27.0 (27A5252f), Swift 6.4.
- Command: Release `ShapECLI --ab-mesh` with `--seed 7 --steps 64 --grid 64`, prompt `a red chair`, guidance left at the CLI default 15. Bundle path in the README is `~/.cache/shap-e-mlx`. mlx-swift 0.31.3.
- `/usr/bin/time -l`: 39.42 s real, 1.29 s user, 2.19 s sys. Peak memory footprint 5,489,169,280 bytes. Maximum resident set size 5,856,296,960 bytes.
- CPU and Metal meshes matched: 9,882 vertices, 19,684 faces, closed, 0 boundary edges.
- Metal marching cubes 0.052 s. CPU marching cubes 0.816 s.
- The README's shaded preview is `docs/red-chair.png`, 768 by 768, neutral gray, -90 degrees about X because the mesh is Z-up. `powermetrics` was not run. The README says it needs root, so there is no GPU-utilization percentage.

`README.md:94` also records `xcodebuild test -scheme ShapERuntime-Package` on that machine: 40 tests, 0 failures, three CLIP tokenizer tests skipped until the vocabulary file was supplied, then those three passed.

Nothing in `Sources/` hardcodes 9,882 vertices or 39.42 seconds.
