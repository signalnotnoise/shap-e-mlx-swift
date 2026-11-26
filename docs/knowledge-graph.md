# Knowledge graph

Verified source commit: `9d042666fbfbd40b07417cd23dfe34d0731d673f` on `cursor/shap-e-runtime-package-ea5b`.

That SHA is the commit recorded when the graph was measured. `scripts/verify_docs.py` does not ask git whether this commit is an ancestor of `HEAD`. It checks the cited lines in the working tree.

Machine-readable index: [knowledge-graph.json](knowledge-graph.json).

## Dataflow

```mermaid
flowchart TD
  prompt[Text prompt] --> tok[ClipSimpleTokenizer]
  tok --> textEnc[ClipTextEmbedder]
  textEnc --> samp[KarrasSampler plus Text300MTransformer]
  samp --> latent["latent batch by dLatent"]
  latent --> proj[DecoderParamsProjector]
  proj --> stf[STFMeshRenderer]
  stf --> cpu[MarchingCubes.extract]
  stf --> metal[MetalMarchingCubes]
  cpu --> mesh[MeshResult]
  metal --> mesh
  mesh --> obj[MeshExport.writeOBJ]
  image[CGImage] --> prep[ImageProcessor]
  prep --> vision[ClipImageEmbedder]
  vision --> imageTx[Image300MTransformer]
  imageTx --> samp
```

`dLatent` and `nCtx` are bundle metadata (`RuntimeArtifactBundle.swift:140-141`), not literals in the sampler.

## Where to find things

| Task | Location |
| --- | --- |
| Tokenizer | `Sources/ShapERuntime/Models/embeddings/ClipSimpleTokenizer.swift:29`, `Sources/ShapERuntime/ClipBPELocation.swift:4-35` |
| Text encoder | `Sources/ShapERuntime/Models/embeddings/ClipTextEmbedder.swift:116-122` |
| Diffusion sampling | `Sources/ShapERuntime/KarrasSampler.swift:316`, `Sources/ShapERuntime/Text300MTransformer.swift:363-424` |
| Latent decoding | `Sources/ShapERuntime/DecoderParamsProjector.swift:175-249`, `Sources/ShapERuntime/STFMeshRenderer.swift:354-406` |
| Marching cubes, CPU and Metal | `Sources/ShapERuntime/MarchingCubes.swift:24-125`, `Sources/ShapERuntime/MetalMarchingCubes.swift:40-72`, `Sources/ShapERuntime/MCTable.swift:2` |
| Mesh export | `Sources/ShapERuntime/MeshExport.swift:19-70`, `Sources/ShapERuntime/MeshTopology.swift:22-112` |
| CLI and A/B mesh | `Sources/ShapECLI/main.swift:418-593`, `Sources/ShapECLI/main.swift:212-234` |
| Weights fetch and verify | `scripts/fetch_weights.py:34-61`, `scripts/fetch_weights.py:180-218` |
| Tests | `Tests/ShapERuntimeTests/RuntimeBundleFixture.swift:3-25`, `run_tests.sh:17-20` |
| CI and manifest | `Package.swift:1-22`, `.github/workflows/ci.yml:36-48`, `Package.resolved:4-11` |

More detail: [wiki/01-Architecture-and-Dataflow.md](wiki/01-Architecture-and-Dataflow.md), [wiki/02-Weights-and-Bundle.md](wiki/02-Weights-and-Bundle.md), [wiki/03-Build-CI-and-Testing.md](wiki/03-Build-CI-and-Testing.md), [wiki/04-Memory-and-Performance.md](wiki/04-Memory-and-Performance.md).
