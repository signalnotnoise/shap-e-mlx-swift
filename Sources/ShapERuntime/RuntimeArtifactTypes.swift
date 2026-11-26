import Foundation

public struct Text300MArtifact: Equatable, Sendable {
    public let nCtx: Int
    public let dLatent: Int
    public let state: [String: RuntimeTensor]

    public init(
        nCtx: Int,
        dLatent: Int,
        state: [String: RuntimeTensor]
    ) {
        self.nCtx = nCtx
        self.dLatent = dLatent
        self.state = state
    }
}

public struct Image300MArtifact: Equatable, Sendable {
    public let nCtx: Int
    public let dLatent: Int
    public let state: [String: RuntimeTensor]

    public init(
        nCtx: Int,
        dLatent: Int,
        state: [String: RuntimeTensor]
    ) {
        self.nCtx = nCtx
        self.dLatent = dLatent
        self.state = state
    }
}

public struct ClipVisionArtifact: Equatable, Sendable {
    public let inputResolution: Int
    public let patchSize: Int
    public let transformerWidth: Int
    public let transformerHeads: Int
    public let transformerLayers: Int
    public let outputDim: Int
    public let state: [String: RuntimeTensor]

    public init(
        inputResolution: Int,
        patchSize: Int,
        transformerWidth: Int,
        transformerHeads: Int,
        transformerLayers: Int,
        outputDim: Int,
        state: [String: RuntimeTensor]
    ) {
        self.inputResolution = inputResolution
        self.patchSize = patchSize
        self.transformerWidth = transformerWidth
        self.transformerHeads = transformerHeads
        self.transformerLayers = transformerLayers
        self.outputDim = outputDim
        self.state = state
    }
}

public struct RuntimeDiffusionConfig: Equatable, Sendable {
    public let alphasCumprod: RuntimeTensor
    public let betas: RuntimeTensor
    public let posteriorLogVarianceClipped: RuntimeTensor
    public let sqrtRecipAlphasCumprod: RuntimeTensor
    public let sqrtRecipm1AlphasCumprod: RuntimeTensor
    public let posteriorMeanCoef1: RuntimeTensor
    public let posteriorMeanCoef2: RuntimeTensor
    public let numTimesteps: Int
    public let modelMeanType: String
    public let channelBiases: RuntimeTensor?
    public let channelScales: RuntimeTensor?

    public init(
        alphasCumprod: RuntimeTensor,
        betas: RuntimeTensor,
        posteriorLogVarianceClipped: RuntimeTensor,
        sqrtRecipAlphasCumprod: RuntimeTensor,
        sqrtRecipm1AlphasCumprod: RuntimeTensor,
        posteriorMeanCoef1: RuntimeTensor,
        posteriorMeanCoef2: RuntimeTensor,
        numTimesteps: Int,
        modelMeanType: String = "epsilon",
        channelBiases: RuntimeTensor? = nil,
        channelScales: RuntimeTensor? = nil
    ) {
        self.alphasCumprod = alphasCumprod
        self.betas = betas
        self.posteriorLogVarianceClipped = posteriorLogVarianceClipped
        self.sqrtRecipAlphasCumprod = sqrtRecipAlphasCumprod
        self.sqrtRecipm1AlphasCumprod = sqrtRecipm1AlphasCumprod
        self.posteriorMeanCoef1 = posteriorMeanCoef1
        self.posteriorMeanCoef2 = posteriorMeanCoef2
        self.numTimesteps = numTimesteps
        self.modelMeanType = modelMeanType
        self.channelBiases = channelBiases
        self.channelScales = channelScales
    }
}

public struct DecoderParamShape: Equatable, Sendable {
    public let name: String
    public let shape: [Int]

    public init(name: String, shape: [Int]) {
        self.name = name
        self.shape = shape
    }
}

public struct DecoderProjectionWeights: Equatable, Sendable {
    public let weightVCD: RuntimeTensor
    public let biasVC: RuntimeTensor
    public let lnWeightC: RuntimeTensor
    public let lnBiasC: RuntimeTensor

    public init(
        weightVCD: RuntimeTensor,
        biasVC: RuntimeTensor,
        lnWeightC: RuntimeTensor,
        lnBiasC: RuntimeTensor
    ) {
        self.weightVCD = weightVCD
        self.biasVC = biasVC
        self.lnWeightC = lnWeightC
        self.lnBiasC = lnBiasC
    }
}

public struct DecoderParamsArtifact: Equatable, Sendable {
    public let paramShapes: [DecoderParamShape]
    public let latentCtx: Int
    public let latentWarpName: String
    public let latentWarpCoeff1: Double
    public let projections: [String: DecoderProjectionWeights]
    public let lnEps: Double

    public init(
        paramShapes: [DecoderParamShape],
        latentCtx: Int,
        latentWarpName: String,
        latentWarpCoeff1: Double,
        projections: [String: DecoderProjectionWeights],
        lnEps: Double = 1e-5
    ) {
        self.paramShapes = paramShapes
        self.latentCtx = latentCtx
        self.latentWarpName = latentWarpName
        self.latentWarpCoeff1 = latentWarpCoeff1
        self.projections = projections
        self.lnEps = lnEps
    }
}

public struct RendererArtifact: Equatable, Sendable {
    public let gridSize: Int
    public let bboxMin: RuntimeTensor
    public let bboxMax: RuntimeTensor
    public let staticWeights: [Int: RuntimeTensor]
    public let staticBiases: [Int: RuntimeTensor]
    public let directionDependentShape: Bool
    public let separateNerfChannels: Bool
    public let separateCoarseChannels: Bool
    public let channelActivation: String
    public let sdfActivation: String
    public let textureChannels: [String]

    public init(
        gridSize: Int,
        bboxMin: RuntimeTensor,
        bboxMax: RuntimeTensor,
        staticWeights: [Int: RuntimeTensor],
        staticBiases: [Int: RuntimeTensor],
        directionDependentShape: Bool = true,
        separateNerfChannels: Bool = true,
        separateCoarseChannels: Bool = true,
        channelActivation: String = "sigmoid",
        sdfActivation: String = "tanh",
        textureChannels: [String] = ["R", "G", "B"]
    ) {
        self.gridSize = gridSize
        self.bboxMin = bboxMin
        self.bboxMax = bboxMax
        self.staticWeights = staticWeights
        self.staticBiases = staticBiases
        self.directionDependentShape = directionDependentShape
        self.separateNerfChannels = separateNerfChannels
        self.separateCoarseChannels = separateCoarseChannels
        self.channelActivation = channelActivation
        self.sdfActivation = sdfActivation
        self.textureChannels = textureChannels
    }
}
