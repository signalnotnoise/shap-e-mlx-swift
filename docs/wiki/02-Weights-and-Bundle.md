# Weights and bundle

## What is pinned

`scripts/fetch_weights.py:23-24` pins `signalNotNoise/shap-e-mlx` at revision `d71f70a09b9c03535188d006a719edda5c19a6eb`.

`README.md:96` says tag `v0.1.0` resolves to that commit and that one download verified 1,102 checksummed files. The stub fixture README calls the real Hub bundle 1,088 tensors and 5.143 GB (`Tests/ShapERuntimeTests/Fixtures/README.md:30`). This package does not re-measure that payload in CI.

Trained weights are not in git. `runtime_bundle/` is gitignored. The only committed tensors are the random stub.

## Manifest and blobs

`RuntimeArtifactBundle` reads `manifest.json` from the bundle directory (`RuntimeArtifactBundle.swift:45-113`). `loadText300MArtifact` reads metadata keys `n_ctx` and `d_latent` (`RuntimeArtifactBundle.swift:133-138`). `loadClipTextArtifact` reads `context_length`, `transformer_width`, `transformer_heads`, and `transformer_layers` (`RuntimeArtifactBundle.swift:115-130`).

`TensorBlobLoader` loads the raw file named by each manifest entry (`TensorBlobLoader.swift:130-156`). These are not safetensors. A full per-file size table is not checked into this repo. Integrity for a download is `checksums.sha256`.

## fetch_weights.py

Checksum lines: 64 hex digits, two spaces, relative path (`fetch_weights.py:60-72`).

`verify_checksums` (`fetch_weights.py:98-131`) hashes every present listed file. A digest mismatch fails. A missing file fails, except under `--text-only` when `is_text_scope_path` is false (`fetch_weights.py:116-118`).

Text scope (`fetch_weights.py:26-57`):

- Root files: `manifest.json`, `bpe_simple_vocab_16e6.txt.gz`, `checksums.sha256`, `provenance.json`, `LICENSE`, `README.md`.
- `licenses/*`.
- `*.bin` in `clip_text`, `text300m`, `diffusion`, `decoder_params`, `renderer`.

`clip_vision` and `image300m` are outside that scope (`fetch_weights.py:33-36`). A text-only tree that lacks them can still verify. A tree that only has a matching `README.md`, and is missing the manifest, the BPE file, or a text-family tensor, must exit nonzero. The test is `scripts/test_fetch_weights.py:32`.

## BPE fallback

`ClipBPELocation.resolve` (`ClipBPELocation.swift:20-34`):

- Non-empty `--bpe` wins when that file exists.
- Otherwise `<bundle>/bpe_simple_vocab_16e6.txt.gz` when it exists.

`ShapECLI.resolveBPE` calls that first, then `.mlx_venv` site-packages paths and `./bpe_simple_vocab_16e6.txt.gz` (`main.swift:212-233`).

`ShapERuntime.init` does not search. The caller passes `bpeVocabularyURL` (`ShapERuntime.swift:104`).

Tests look for `SHAP_E_BPE`, then `SHAP_E_BUNDLE/bpe_simple_vocab_16e6.txt.gz`, then `runtime_bundle/bpe_simple_vocab_16e6.txt.gz` (`ClipSimpleTokenizerTests.swift:11-29`). Without one of those, three tokenizer tests `XCTSkip`.
