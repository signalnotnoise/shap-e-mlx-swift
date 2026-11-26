# Changelog

## Unreleased

- Package copyright line is Copyright (c) 2026 Ismael Alex Otero. signalnotnoise is credited as the project maintainer in the README.
- Initial standalone package. `ShapERuntime` and `ShapECLI` are copied from
  `shap_e/swift` in
  [apple-mlx-foundations](https://github.com/signalnotnoise/apple-mlx-foundations)
  tag `v0.1.0` (commit `018215715177943fe72cea8e540bc7b80dcb593f`).
  That directory is unchanged between the tag and `main`. The public
  history of the directory is one snapshot commit, so this repository
  starts from a clean copy. See NOTICE.
- Package manifest is at the repository root. Platforms remain macOS 14,
  iOS 17, and visionOS 1. The mlx-swift dependency is
  `"0.31.3"..<"0.31.5"`, matching triposr-mlx-swift. mlx-swift 0.31.5 and
  later require swift-tools-version 6.3, and this manifest stays on 6.2.
  `Package.resolved` pins 0.31.3.
- Tokenizer parity tests skip when the CLIP BPE file from the weight
  bundle is absent. The checked-in stub fixture does not require the
  full bundle.
- `ShapECLI` uses `<bundle>/bpe_simple_vocab_16e6.txt.gz` when `--bpe`
  is omitted. An explicit `--bpe` path is used when that file exists.
- Added a macOS GitHub Actions workflow, `.spi.yml`, and
  `scripts/fetch_weights.py` for the pinned Hugging Face snapshot.
- A checksum list that only names optional files such as README.md fails.
  Required files are manifest.json, the BPE tokenizer, and the manifest
  tensor paths for the chosen scope.
- Added `AGENTS.md`, `docs/engineering_standards.md`, the knowledge graph,
  and `docs/wiki/` pages. `scripts/verify_docs.py` checks their line anchors
  against the working tree and does not require `verified_commit` to be an
  ancestor of `HEAD`.
