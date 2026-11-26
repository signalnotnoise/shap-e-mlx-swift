# Test Fixtures

This directory contains checked-in data fixtures used by the `ShapERuntimeTests` unit test suite.

---

## `runtime_bundle_stub/`

A **minimal, randomly-valued stub** of the `ShapERuntime` model bundle format.

### Purpose

The full production bundle contains 5.143 GB (4.790 GiB) of real trained
weights across seven families and is excluded from version control via
`.gitignore` (`runtime_bundle/`). To allow the unit tests to exercise bundle
loading, parsing, manifest validation, and tensor deserialization without
requiring the real weights, this stub provides the same **manifest JSON schema**
and a five-family subset of the directory layout (`clip_text`, `text300m`,
`diffusion`, `decoder_params`, `renderer`), with **tiny random tensors** in
place of actual model parameters. It does not contain `clip_vision` or
`image300m`. The layout is the format-v1 runtime bundle this package loads.
The original specification lives in `docs/FILE-STRUCTURE.md` in
[apple-mlx-foundations](https://github.com/signalnotnoise/apple-mlx-foundations)
and is not part of this repository.

### Properties

| Property | Real Bundle | This Stub |
|---|---|---|
| Weights | Real trained model (1,088 tensors, 5.143 GB total Hub bundle) | 28 tiny random tensors (few KB) |
| Transformer width | 1024 | 8 |
| Transformer layers | 24 | 3 |
| Attention heads | 16 | 2 |
| Diffusion timesteps | 1024 | 2 |
| Grid size | 128+ | 32 |

### What it tests

- `RuntimeArtifactBundle` manifest loading and JSON decoding
- `TensorBlobLoader` raw binary tensor deserialization
- `ClipTextArtifact`, `Text300MArtifact`, `DecoderParamsArtifact`, `RendererArtifact` construction
- Shape and metadata validation error paths

### What it does NOT test

- Numerical correctness of model outputs (use the Python golden tests for that)
- End-to-end mesh generation (requires the real bundle)

### How to update

There is currently no checked-in stub generator or `--stub` export flag. If
the bundle schema changes (for example, new tensor families or metadata keys
are added), edit `runtime_bundle_stub/manifest.json` and its tensor blobs
deliberately, then update this README and the affected tests in the same
change. Do not point the production exporter at this fixture: it writes full
trained tensors.
