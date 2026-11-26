# Architecture and dataflow

Library product `ShapERuntime` (`Package.swift:12-15`). The end-to-end type is `ShapERuntime` (`ShapERuntime.swift:68-583`).

## Text path

1. `init(modelBundleURL:bpeVocabularyURL:)` loads the manifest, CLIP tokenizer, CLIP text embedder, Text300M, the Karras denoiser, the decoder projector, and the STF renderer (`ShapERuntime.swift:104-156`). Image families are not loaded here (`ShapERuntime.swift:142`).
2. `ClipSimpleTokenizer` reads `bpe_simple_vocab_16e6.txt.gz` (`ClipSimpleTokenizer.swift:29`). `encode` is `ClipSimpleTokenizer.swift:76`. `tokenize` is `ClipSimpleTokenizer.swift:99`.
3. `ClipTextEmbedder.embedText` tokenizes, runs the causal text transformer, pools at the end-of-text index, projects, and L2-normalizes (`ClipTextEmbedder.swift:116-122` and `ClipTextEmbedder.swift:237-249`). Output shape is `[batch, textProjectionDim]`.
4. `Text300MTransformer.predict` checks latent `[B, inputChannels, T]`, timesteps `[B]`, and CLIP `[B, clipFeatureDim]`, and returns `[B, outputChannels, T]` (`Text300MTransformer.swift:363-423`). The convenience initializer defaults `nHeads` to 16 and infers width, layers, and channel counts from tensors (`Text300MTransformer.swift:180-196`).
5. `KarrasSampler` (`KarrasSampler.swift:316`) drives `GaussianToKarrasDenoiser` (`KarrasSampler.swift:72`). `ShapERuntime` builds the latent shape as `(batchSize, textArtifact.dLatent)` (`ShapERuntime.swift:279` in the batch `generate`). `ImageParity.swift:193-194` describes stored latents as token-major `[B, tokens*channels]` versus component input `[B, channels, tokens]`.
6. `DecoderParamsProjector.project` accepts `[B, D]` (`DecoderParamsProjector.swift:174-185`) and emits one tensor per decoder parameter.
7. `STFMeshRenderer.extractMeshes` queries the field in batches and meshes each item (`STFMeshRenderer.swift:242-270`). `nerstfForward` is `STFMeshRenderer.swift:354-406`.

`generate(prompt:)` returns one mesh and passes `useMetalMarchingCubes: false` (`ShapERuntime.swift:209-230`). The `batchSize` overload can set the Metal flag (`ShapERuntime.swift:248-261`).

## Image path

`generate(image: CGImage)` calls `ImageProcessor.processCLIPImage` (`ShapERuntime.swift:415`). That function returns float32 `[1, 224, 224, 3]` (`ImageProcessor.swift:21-26`). `generate(image: MLXArray)` documents the same layout (`ShapERuntime.swift:436-437`).

`ensureImagePipeline` loads `ClipImageEmbedder` and `Image300MTransformer` on first use (`ShapERuntime.swift:163-187`). The stub fixture has no `clip_vision` or `image300m` (`Tests/ShapERuntimeTests/Fixtures/README.md:20-22`).

## Marching cubes and export

CPU: `MarchingCubes.extract` (`MarchingCubes.swift:24-125`). The comment at `MarchingCubes.swift:18-23` states SDF `[X, Y, Z]`, bbox min `[3]`, bbox size `[3]`, verts `[V, 3]`, faces `[F, 3]`.

The 256-case table is `MC_TABLE` (`MCTable.swift:2`). `LookupTable` fills MLX buffers from it (`MarchingCubes.swift:221-243`).

Metal: `MetalMarchingCubes` loads `countTriangles` and `extractMesh` from `Bundle.module` (`MetalMarchingCubes.swift:46-51`) and uploads `LookupTable.shared` (`MetalMarchingCubes.swift:58-59`). Kernels read `numTrisTable` (`MarchingCubes.metal:65-66`).

`MeshExport.writeOBJ` writes `v x y z` or `v x y z r g b`, and 1-based `f` records (`MeshExport.swift:19-67`). `MeshTopology` reports boundary and non-manifold edges (`MeshTopology.swift:22-112`). The CLI A/B path calls both backends from `runMeshAB` (`main.swift:418-593`).

This package does not call TripoSR. There is no second mesh stack in `Sources/`.
