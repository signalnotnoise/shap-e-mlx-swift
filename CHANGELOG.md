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
  iOS 17, and visionOS 1. The mlx-swift dependency remains
  `from: "0.29.1"`, with `Package.resolved` still pinning 0.31.3.
- Tokenizer parity tests skip when the CLIP BPE file from the weight
  bundle is absent. The checked-in stub fixture does not require the
  full bundle.
- `ShapECLI` uses `<bundle>/bpe_simple_vocab_16e6.txt.gz` when `--bpe`
  is omitted. An explicit `--bpe` path is used when that file exists.
- Added a macOS GitHub Actions workflow, `.spi.yml`, and
  `scripts/fetch_weights.py` for the pinned Hugging Face snapshot.
- `--text-only` verification fails when a checksummed file inside the
  text scope is missing. Missing files outside that scope are skipped.
