# The learning models
#
# Each of the models provide the following functions:
#
# reset!: resets the internal weight estimate
# feedback!: updates weight estimate in light of feedback
# diffuse!: diffuses weight estimate if in volatile environment
# confidence: pre-feedback decision confidence
# learningrate (only few models): learning rate that would be used in feedback!

include("probit_regression.jl")

abstract type LearningModel end
getw(m::LearningModel) = m.w

# -----------------------------------------------------------------------------
# Oracle model, no learning
# -----------------------------------------------------------------------------

struct OracleModel <: LearningModel
    w::Vector{Float64}
    σμm2::Float64
    function OracleModel(conf::ConfParse, N, α)
        σμ = parse(Float64, retrieve(conf, "task", "sigmu"))
        return new(zeros(N), 1/abs2(σμ))
    end
end
function reset!(m::OracleModel, μw, Σw, w)
    m.w .= w
end
feedback!(m::OracleModel, corrchoice, t, x, w) = m.w
diffuse!(m::OracleModel) = nothing
confidence(m::OracleModel, choice, t, x, w) = ProbitRegression.Φ(
    choice * dot(x, w) / √(t + m.σμm2))


# -----------------------------------------------------------------------------
# Delta rule
# w += α (θcorr - θchosen)/|θcorr| x
# -----------------------------------------------------------------------------

struct DeltaModel <: LearningModel
    α::Float64
    w::Vector{Float64}
    σμm2::Float64
    function DeltaModel(conf::ConfParse, N, α)
        σμ = parse(Float64, retrieve(conf, "task", "sigmu"))
        return new(α, zeros(N), 1/abs2(σμ))
    end
end
function reset!(m::DeltaModel, μw, Σw, w)
    m.w .= μw
end
function feedback!(m::DeltaModel, corrchoice, t, x, w)
    θchosen = dot(x, m.w)
    # update only if incorrect, in which case w += α corrchoice x
    if Int64(sign(θchosen)) != corrchoice
        m.w .+= (m.α * corrchoice) * x
    end
    return m.w
end
diffuse!(m::DeltaModel) = nothing
confidence(m::DeltaModel, choice, t, x, w) = ProbitRegression.Φ(
    choice * dot(x, m.w) / √(t + m.σμm2))
# learning rate is α for incorrect, and 0 for correct choices
function learningrate(m::DeltaModel, corrchoice, t, x, w)
    θchosen = dot(x, m.w)
    return Int64(sign(θchosen)) == corrchoice ? 0.0 : m.α
end


# -----------------------------------------------------------------------------
# Normalized delta rule
# w += α (θcorr - θchosen)/|θcorr| x, subsequent normalization
# -----------------------------------------------------------------------------

struct NormDeltaModel <: LearningModel
    α::Float64
    w::Vector{Float64}
    σμm2::Float64
    function NormDeltaModel(conf::ConfParse, N, α)
        σμ = parse(Float64, retrieve(conf, "task", "sigmu"))
        return new(α, zeros(N))
    end
end
function reset!(m::NormDeltaModel, μw, Σw, w)
    m.w .= μw
end
function feedback!(m::NormDeltaModel, corrchoice, t, x, w)
    θchosen = dot(x, m.w)
    # update only if incorrect, in which case w += α corrchoice x
    if Int64(sign(θchosen)) != corrchoice
        m.w .+= (m.α * corrchoice) * x
    end
    m.w .*= norm(w) / norm(m.w)
    return m.w
end
diffuse!(m::NormDeltaModel) = nothing
confidence(m::NormDeltaModel, choice, t, x, w) = ProbitRegression.Φ(
    choice * dot(x, m.w) / √(t + m.σμm2))
# learning rate is α for incorrect, and 0 for correct choices
function learningrate(m::NormDeltaModel, corrchoice, t, x, w)
    θchosen = dot(x, m.w)
    return Int64(sign(θchosen)) == corrchoice ? 0.0 : m.α
end


# -----------------------------------------------------------------------------
# Gradient ascent on the feedback likelihood
# -----------------------------------------------------------------------------

struct LikelihoodGradientModel <: LearningModel
    α::Float64
    σμm2::Float64
    w::Vector{Float64}
    function LikelihoodGradientModel(conf::ConfParse, N, α)
        σμ = parse(Float64, retrieve(conf, "task", "sigmu"))
        return new(α, 1/abs2(σμ), zeros(N))
    end
end
function reset!(m::LikelihoodGradientModel, μw, Σw, w)
    m.w .= μw
end
function feedback!(m::LikelihoodGradientModel, correctchoice, t, x, w)
    xt = x / √(t + m.σμm2)
    ywx = correctchoice * dot(m.w, xt)
    m.w .+= m.α * correctchoice * ProbitRegression.NoverΦ(ywx) * xt
    return m.w
end
diffuse!(m::LikelihoodGradientModel) = nothing
confidence(m::LikelihoodGradientModel, choice, t, x, w) = ProbitRegression.Φ(
    choice * dot(x, m.w) / √(t + m.σμm2))


# -----------------------------------------------------------------------------
# Generic model for all models based on probit regression
# -----------------------------------------------------------------------------

struct ProbitModel{T <: ProbitRegression.ProbitModel} <: LearningModel
    σμm2::Float64
    probit::T
end
getw(m::ProbitModel) = ProbitRegression.getμw(m.probit)
function feedback!(m::ProbitModel, corrchoice, t, x, w) 
    ProbitRegression.update!(m.probit, x / √(t + m.σμm2), corrchoice)
    return ProbitRegression.getμw(m.probit)
end
diffuse!(m::ProbitModel) = ProbitRegression.diffuse!(m.probit)
confidence(m::ProbitModel, choice, t, x, w) = ProbitRegression.Φ(
    choice * dot(x, ProbitRegression.getμw(m.probit)) / 
    √(t + m.σμm2 + x' * ProbitRegression.getΣw(m.probit) * x))


# -----------------------------------------------------------------------------
# Assumed density filtering (full/diagonal covariance)
# -----------------------------------------------------------------------------

const ADFModel=ProbitModel{ProbitRegression.ADF}
function ADFModel(conf::ConfParse, N, α)
    σμ = parse(Float64, retrieve(conf, "task", "sigmu"))
    diffstats = readdiffstats(conf, N)
    if diffstats == nothing
        return ADFModel(1/abs2(σμ), 
            ProbitRegression.ADF(zeros(N), Matrix{Float64}(I, N, N), 
                Matrix{Float64}(I, N, N), zeros(N), zeros(N, N)))
    else
        return ADFModel(1/abs2(σμ), 
            ProbitRegression.ADF(zeros(N), Matrix{Float64}(I, N, N),
                diffstats...))
    end        
end
function reset!(m::ADFModel, μw, Σw, w)
    m.probit.μw .= μw
    m.probit.Σw.data .= Σw
end
# learning rate is everything excluding y and Σw x
function learningrate(m::ADFModel, corrchoice, t, x, w)
    tnorm = √(t + m.σμm2)
    xt = x / tnorm
    xΣwx = √(1 + (xt' * m.probit.Σw * xt)[1])
    ywx = corrchoice * dot(xt, m.probit.μw) / xΣwx
    return ProbitRegression.NoverΦ(ywx) / (xΣwx * tnorm)
end

const ADFDiagModel=ProbitModel{ProbitRegression.ADFDiag}
function ADFDiagModel(conf::ConfParse, N, α)
    σμ = parse(Float64, retrieve(conf, "task", "sigmu"))
    diffstats = readdiffstats(conf, N)
    if diffstats == nothing
        return ADFDiagModel(1/abs2(σμ), 
            ProbitRegression.ADFDiag(zeros(N), Matrix{Float64}(I, N, N),
                Matrix{Float64}(I, N, N), zeros(N), zeros(N, N)))
    else
        return ADFDiagModel(1/abs2(σμ), 
            ProbitRegression.ADFDiag(zeros(N), Matrix{Float64}(I, N, N),
                diffstats...))
    end
end
function reset!(m::ADFDiagModel, μw, Σw, w)
    m.probit.μw .= μw
    m.probit.Σw .= diag(Σw)
end


# -----------------------------------------------------------------------------
# 2nd order Taylor expansion of likelihood
# -----------------------------------------------------------------------------

const TaylorModel=ProbitModel{ProbitRegression.Taylor}
function TaylorModel(conf::ConfParse, N, α)
    σμ = parse(Float64, retrieve(conf, "task", "sigmu"))
    diffstats = readdiffstats(conf, N)
    if diffstats == nothing
        return TaylorModel(1/abs2(σμ), 
            ProbitRegression.Taylor(zeros(N), Matrix{Float64}(I, N, N),
                Matrix{Float64}(I, N, N), zeros(N), zeros(N, N)))
    else
        return TaylorModel(1/abs2(σμ), 
            ProbitRegression.Taylor(zeros(N), Matrix{Float64}(I, N, N),
                diffstats...))
    end
end
function reset!(m::TaylorModel, μw, Σw, w)
    m.probit.μw .= μw
    m.probit.Σw.data .= Σw
end


# -----------------------------------------------------------------------------
# Gibbs sampling (does not support diffusion)
# -----------------------------------------------------------------------------

const GibbsModel=ProbitModel{ProbitRegression.Gibbs}
function GibbsModel(conf::ConfParse, N, α)
    σμ = parse(Float64, retrieve(conf, "task", "sigmu"))
    samples = parse(Int64, retrieve(conf, "gibbsmodel", "samples"))
    burnin = parse(Int64, retrieve(conf, "gibbsmodel", "burnin"))
    return GibbsModel(1/abs2(σμ),
        ProbitRegression.Gibbs(zeros(N), Matrix{Float64}(I, N, N),
            samples, burnin))
end
function reset!(m::GibbsModel, μw, Σw, w)
    m.probit.Σμ0 .= Σw \ μw
    m.probit.Σw.data .= Σw
    s = sampler(MvNormal(μw, Σw))
    m.probit.wsam .= Vector{Float64}[rand(s) for i in 1:m.probit.N]
    m.probit.xy = Vector{Vector{Float64}}(undef, 0)
end


# -----------------------------------------------------------------------------
# Particle filter (only works with diffusion)
# -----------------------------------------------------------------------------

const ParticleFilterModel=ProbitModel{ProbitRegression.ParticleFilter}
function ParticleFilterModel(conf::ConfParse, N, α)
    σμ = parse(Float64, retrieve(conf, "task", "sigmu"))
    particles = parse(Int64, retrieve(conf, "pfmodel", "particles"))
    diffstats = readdiffstats(conf, N)
    diffstats != nothing || error("Particle filter only support diffusive transitions")
    Aw, bw, Σd = diffstats
    return ParticleFilterModel(1/abs2(σμ),
        ProbitRegression.ParticleFilter(zeros(N), Matrix{Float64}(I, N, N),
            Aw, bw, Σd, particles))
end
function reset!(m::ParticleFilterModel, μw, Σw, w)
    s = sampler(MultivariateNormal(μw, Σw))
    m.probit.wp .= Vector{Float64}[rand(s) for i in 1:length(m.probit.wp)]
end


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

const modeldict = Dict{String,Tuple{DataType,Bool}}(
    "oracle" => (OracleModel, false),
    "delta" => (DeltaModel, true),
    "normdelta" => (NormDeltaModel, true),
    "lhgrad" => (LikelihoodGradientModel, true),
    "adf" => (ADFModel, false),
    "adfdiag" => (ADFDiagModel, false),
    "taylor" => (TaylorModel, false),
    "gibbs" => (GibbsModel, false),
    "pf" => (ParticleFilterModel, false),
)
requiresα(modelname) = modeldict[modelname][2]
function createmodel(conf::ConfParse, modelname, N, α)
    if modelname ∈ keys(modeldict)
        if requiresα(modelname) && α == nothing
            error("Model $modelname requires learning rate parameter")
        else
            return modeldict[modelname][1](conf, N, α)
        end
    else
        error("Unknown model $modelname")
    end
end
