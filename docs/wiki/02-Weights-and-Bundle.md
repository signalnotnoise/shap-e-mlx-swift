# Weights and bundle

## What is pinned

`scripts/fetch_weights.py:31-32` pins `signalNotNoise/shap-e-mlx` at revision `d71f70a09b9c03535188d006a719edda5c19a6eb`.

`README.md:96` says tag `v0.1.0` resolves to that commit and that one download verified 1,102 checksummed files. The stub fixture README calls the real Hub bundle 1,088 tensors and 5.143 GB (`Tests/ShapERuntimeTests/Fixtures/README.md:30`). This package does not re-measure that payload in CI.

Trained weights are not in git. `runtime_bundle/` is gitignored. The only committed tensors are the random stub.

## Manifest and blobs

`RuntimeArtifactBundle` reads `manifest.json` from the bundle directory (`RuntimeArtifactBundle.swift:49-116`). Init throws `manifestNotFound` when that file is absent (`RuntimeArtifactBundle.swift:62-63`). `loadText300MArtifact` reads metadata keys `n_ctx` and `d_latent` (`RuntimeArtifactBundle.swift:140-141`). `loadClipTextArtifact` reads `context_length`, `transformer_width`, `transformer_heads`, and `transformer_layers` (`RuntimeArtifactBundle.swift:118-133`). `RuntimeBundleRequirements.validate` (`RuntimeBundleRequirements.swift:41-46`) throws `missingRequiredFile` when the manifest, the BPE file, or a scoped tensor path is missing. It does not read `checksums.sha256`.

`TensorBlobLoader` loads the raw file named by each manifest entry (`TensorBlobLoader.swift:130-156`). These are not safetensors. A full per-file size table is not checked into this repo. Integrity for a download is `checksums.sha256`.

## fetch_weights.py

Checksum lines: 64 hex digits, two spaces, relative path (`fetch_weights.py:64-76`).

`verify_checksums` (`fetch_weights.py:180-218`) hashes listed files, then `ensure_required_files_are_checksummed` (`fetch_weights.py:160-177`) requires `manifest.json`, `bpe_simple_vocab_16e6.txt.gz`, and the manifest tensor paths for the scope. Those paths come from `required_bundle_paths` (`fetch_weights.py:140-157`), not from the checksum list. A list that only names `README.md` fails. A missing listed file still fails, except under `--text-only` when `is_text_scope_path` is false (`fetch_weights.py:202`).

Text scope (`fetch_weights.py:34-61`):

- Root files: `manifest.json`, `bpe_simple_vocab_16e6.txt.gz`, `checksums.sha256`, `provenance.json`, `LICENSE`, `README.md`.
- `licenses/*`.
- `*.bin` in `clip_text`, `text300m`, `diffusion`, `decoder_params`, `renderer`.

`clip_vision` and `image300m` are outside that scope (`fetch_weights.py:41-44`). A text-only tree that lacks them can still verify. A tree that only has a matching `README.md`, and is missing the manifest, the BPE file, or a text-family tensor, must exit nonzero. `test_text_only_fails_when_only_readme_is_present` (`scripts/test_fetch_weights.py:48`) covers that case. `test_truncated_checksum_list_fails_against_manifest` (`scripts/test_fetch_weights.py:106`) covers a checksum list that omits a manifest tensor path while the blob itself is on disk.

## BPE fallback

`ClipBPELocation.resolve` (`ClipBPELocation.swift:20-34`):

- Non-empty `--bpe` wins when that file exists.
- Otherwise `<bundle>/bpe_simple_vocab_16e6.txt.gz` when it exists.

`ShapECLI.resolveBPE` calls that first, then `.mlx_venv` site-packages paths and `./bpe_simple_vocab_16e6.txt.gz` (`main.swift:212-233`).

`ShapERuntime.init` does not search. The caller passes `bpeVocabularyURL` (`ShapERuntime.swift:122`).

Tests look for `SHAP_E_BPE`, then `SHAP_E_BUNDLE/bpe_simple_vocab_16e6.txt.gz`, then `runtime_bundle/bpe_simple_vocab_16e6.txt.gz` (`ClipSimpleTokenizerTests.swift:11-29`). Without one of those, three tokenizer tests `XCTSkip`.
