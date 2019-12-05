# Different models to perform logistic regression with weight diffusion
#
# The models assume a likelihood
#
# p(yn | xn, w) = Φ( yn wn^T xn )
#
# where yn in { -1, 1 }, xn is the observable, and w are the model's parameters.
# Inbetween observations, wn is assumed to diffuse according to the linear
# Gaussian model
#
# p( wn+1 | wn ) = N( wn+1 | Aw wn + bn , Σd ) .
#
# The prior before the first observation is
#
# p( w1 ) = N ( w1 | μ0, Σ0 ) .

module ProbitRegression

using Compat.LinearAlgebra
using SpecialFunctions, Distributions

import Compat.LinearAlgebra.Cholesky
import Compat.LinearAlgebra.lowrankdowndate
import Compat.undef

include("tnsampler.jl")

abstract type ProbitModel end

# ----------------------------------------------------------------------------
# Constants and common functions
# ----------------------------------------------------------------------------

const sqrttwo = √(2)

# cumulative Gaussians utility functions
invΦ(p) = sqrttwo * erfinv(2p - 1)
Φ(x) = (1 + erf(x / sqrttwo)) / 2

# computes N(x|0,1) / Φ(x), more stable for asymptotes at small x
NoverΦ(x) = (x > -6 ? sqrt(2 / π) * exp(-0.5abs2(x)) / (1 + erf(x / sqrttwo)) : 
                      -x / (1 - 1/abs2(x) + 3/x^4))

# ----------------------------------------------------------------------------
# Assumed density filtering
# ----------------------------------------------------------------------------

struct ADF <: ProbitModel
    μw::Vector{Float64}
    Σw::Symmetric{Float64,Matrix{Float64}}
    Aw::Matrix{Float64}
    bw::Vector{Float64}
    Σd::Symmetric{Float64,Matrix{Float64}}

    ADF(μ0, Σ0, Aw, bn, Σd) = new(μ0, Symmetric(Σ0), Aw, bn, Symmetric(Σd))
end

@inline getμw(m::ADF) = m.μw
@inline getΣw(m::ADF) = m.Σw

function update!(m::ADF, x::Vector{Float64}, y::Int64)
    xΣwx = √(1 + (x' * m.Σw * x)[1])
    ywx = y * dot(x, m.μw) / xΣwx
    Cw = NoverΦ(ywx)
    Ccov = abs2(Cw) + Cw * ywx
    Σwx = m.Σw * x
    # update
    m.μw .+= (y * Cw / xΣwx) * Σwx
    m.Σw.data .-= (Ccov / abs2(xΣwx)) * Σwx * Σwx'
end
function diffuse!(m::ADF)
    m.μw .= m.Aw * m.μw + m.bw
    m.Σw.data .+= m.Σd
end


# ----------------------------------------------------------------------------
# Assumed density filtering with diagonal covariance
# ----------------------------------------------------------------------------

struct ADFDiag <: ProbitModel
    μw::Vector{Float64}
    Σw::Vector{Float64}
    Aw::Matrix{Float64}
    bw::Vector{Float64}
    Σd::Vector{Float64}

    ADFDiag(μ0, Σ0, Aw, bn, Σd) = new(μ0, diag(Σ0), Aw, bn, diag(Σd))
end

@inline getμw(m::ADFDiag) = m.μw
@inline getΣw(m::ADFDiag) = Diagonal(m.Σw)

function update!(m::ADFDiag, x::Vector{Float64}, y::Int64)
    xΣwx = √(1 + dot(m.Σw, abs2.(x)))    # x' Σw x for diagonal Σw
    ywx = y * dot(x, m.μw) / xΣwx
    Cw = NoverΦ(ywx)
    Ccov = abs2(Cw) + Cw * ywx
    Σwx = m.Σw .* x                      # Σw x for diagonal Σw
    # update
    m.μw .+= (y * Cw / xΣwx) * Σwx
    m.Σw .-= (Ccov / abs2(xΣwx)) * abs2.(Σwx)  # diagonal of Σwx Σwx'
end
function diffuse!(m::ADFDiag)
    m.μw .= m.Aw * m.μw .+ m.bw
    m.Σw .+= m.Σd
end


# ----------------------------------------------------------------------------
# Approximating the log-posterior by a 2nd-order Taylor expansion
# ----------------------------------------------------------------------------

struct Taylor <: ProbitModel
    μw::Vector{Float64}
    Σw::Symmetric{Float64,Matrix{Float64}}
    Aw::Matrix{Float64}
    bw::Vector{Float64}
    Σd::Symmetric{Float64,Matrix{Float64}}

    Taylor(μ0, Σ0, Aw, bn, Σd) = new(μ0, Symmetric(Σ0), Aw, bn, Symmetric(Σd))
end

@inline getμw(m::Taylor) = m.μw
@inline getΣw(m::Taylor) = m.Σw

function update!(m::Taylor, x::Vector{Float64}, y::Int64)
    ywx = y * dot(x, m.μw)
    Cw = NoverΦ(ywx)
    Ccov = abs2(Cw) + Cw * ywx
    Σwx = m.Σw * x
    # update
    m.Σw.data .-= Ccov / (1 + Ccov * dot(x, Σwx)) * Σwx * Σwx'
    m.μw .+= y * Cw * m.Σw * x
end
function diffuse!(m::Taylor)
    m.μw .= m.Aw * m.μw .+ m.bw
    m.Σw.data .+= m.Σd
end


# ----------------------------------------------------------------------------
# Gibbs sampling from posterior - does not support diffusion
# ----------------------------------------------------------------------------

mutable struct Gibbs <: ProbitModel
    Σμ0::Vector{Float64}           # Σ0^-1 μ0
    Σw::Symmetric{Float64,Matrix{Float64}}  # current w sampling covariance
    N::Int64                       # samples to take
    Nbi::Int64                     # burnin samples
    wsam::Vector{Vector{Float64}}  # w samples, N x Dw
    xy::Vector{Vector{Float64}}    # store for yn * xn

    function Gibbs(μ0, Σ0, N, Nbi)
        s = sampler(MvNormal(μ0, Σ0))
        wsam = Vector{Float64}[rand(s) for i in 1:N]
        new(Σ0 \ μ0, Symmetric(Σ0), N, Nbi, wsam, Vector{Vector{Float64}}(undef, 0))
    end
end

@inline getμw(m::Gibbs) = mean(m.wsam)
@inline getΣw(m::Gibbs) = cov(hcat(m.wsam...)')

function _resamplew(m::Gibbs, w, Σwchol)
    Σμsum = copy(m.Σμ0)
    for xyi in m.xy
        μ = dot(w, xyi)
        # faster samples than rand(TruncatedNormal(μ, 1, 0, Inf)
        aux = μ + randnt(-μ, Inf, Φ(μ))  # sample from aux~N(μ, 1) given aux > 0
        Σμsum .+= xyi .* aux
    end
    # draw w ~ N(Σw Σμsum, Σw)
    return m.Σw * Σμsum + Σwchol' * randn(length(Σμsum))
end
function update!(m::Gibbs, x::Vector{Float64}, y::Int64)
    # include observation in current stats
    push!(m.xy, x * y)
    Σwx = m.Σw * x
    m.Σw.data .-= Σwx * Σwx' / (1 + dot(x, Σwx))
    Σwchol = cholesky(m.Σw).U
    # burnin
    w = m.wsam[end]
    for i = 1:m.Nbi
        w .= _resamplew(m, w, Σwchol)
    end
    # 'real' sampling
    m.wsam[1] = w
    for i = 2:m.N
        m.wsam[i] .= _resamplew(m, m.wsam[i-1], Σwchol)
    end
end
diffuse!(m::Gibbs) = nothing


# ----------------------------------------------------------------------------
# Particle filter - only works with diffusion
# ----------------------------------------------------------------------------

# 1) Sample from ADF proposal
#    q(wn | xn, yn, wn-1) = ADF with prior p(wn | A wn-1 + b, Σd)
#
#    yields μprop = A wn-1 + b + Cw yn / √(1 + xn^T Σd xn) Σd xn 
#           Σprop = Σd + Ccov ((Σd^-1 + xn xn^T)^-1 - Σd)
#
#    with Cw = NoverΦ( yn (A wn-1 + b)^T xn / √(1 + xn^T Σd xn) )
#         Ccov = Cw^2 + Cw yn (A wn-1 + b)^T xn / √(1 + xn^T Σd xn)
#
# 2) Weight by likelihood x prior / proposal
#              p(yn | xn, wn) p(wn | wn-1) / q(wn | xn, yn, wn-1)
#    = Φ(yn xn^T wn) N(wn | A wn-1 + b, Σd) / N(wn | μprop, Σprop)
#    = Φ(yn xn^T wn) exp( logdet(Σprop) - logdet(Σd)
#            - 1/2 wres^T Σd^-1 wres^T + 1/2 propres^T Σprod^-1 propres )
#    with wres = wn - A wn-1 - b
#      propres = wn - μprop

struct ParticleFilter <: ProbitModel
    Aw::Matrix{Float64}
    bw::Vector{Float64}
    Σd::Matrix{Float64}
    Σdinv::Matrix{Float64}
    Σdchol::Cholesky{Float64,Array{Float64,2}}
    wp::Vector{Vector{Float64}}
    diffused::Vector{Bool}   # updatable vector within immutable struct

    # μ0 and Σ0 here are pre-diffusion
    function ParticleFilter(μ0, Σ0, Aw, bw, Σd, N)
        @assert isposdef(Σd)
        s = sampler(MvNormal(μ0, Σ0))
        wp = Vector{Float64}[rand(s) for i in 1:N]
        new(Aw, bw, Σd, inv(Σd), cholesky(Σd, Val(false)), wp, [false])
    end
end

# in constrat to the other models, the particle filter's state encoded by
# the particles reflects the estimate _before_ the diffusion rather than after.
# Therefore, getμw(.) and getΣw(.) need to take into account the diffusion
getμw(m::ParticleFilter) = m.diffused[1] ? m.Aw * mean(m.wp) + m.bw : mean(m.wp)
getΣw(m::ParticleFilter) = m.diffused[1] ? m.Aw * cov(hcat(m.wp...)') * m.Aw' + m.Σd : cov(hcat(m.wp...)')

struct _ParticleFilterCache
    x::Vector{Float64}
    y::Int64
    xx::Matrix{Float64}
    xσdx::Float64
    xΣdx::Float64
    μpre::Vector{Float64}
    detpre::Float64
    function _ParticleFilterCache(m::ParticleFilter, x::Vector{Float64}, y::Int64)
        xx = x * x'
        Σdx = m.Σd * x
        xΣdx = (x' * Σdx)[1]
        xσdx = √(1 + xΣdx)
        new(x, y, xx, xσdx, xΣdx, (y / xσdx) * Σdx, xΣdx / abs2(xσdx))
    end
end
function _samplewp(m::ParticleFilter, c::_ParticleFilterCache, w::Vector{Float64})
    w = m.Aw * w + m.bw
    # sample new weights from proposal
    # q(wn | xn, yn, wn-1) = N(wn | μpro, Σpro)
    # To improve speed, Σpro is never computed explicitly
    ywx = c.y * dot(w, c.x) / c.xσdx
    Cw = NoverΦ(ywx)
    @assert Cw >= 0
    Ccov = abs2(Cw) + Cw * ywx
    @assert 0 <= Ccov <= 1
    μpro = w + Cw * c.μpre
    # for the below, the compiler needed some type inference help
    ΣproL::LowerTriangular{Float64,Array{Float64,2}} = lowrankdowndate(m.Σdchol, √(Ccov) * c.μpre).L
    wnew = μpro + ΣproL * randn(length(μpro))
    # compute particle weight λi
    # p(yn | xn, wn) p(wn | wn-1) / q(wn | xn, yn, wn-1)
    wres = wnew - w
    μres = wnew - μpro
    Σproinv = m.Σdinv .+ (Ccov / (1 + (1 - Ccov) * c.xΣdx)) * c.xx
    λ = Φ(c.y * dot(wnew, c.x)) * √(1 - Ccov * c.detpre) * exp(
        0.5 * (dot(μres, Σproinv * μres) - dot(wres, m.Σdinv * wres)))
    return wnew, λ
end
function _systematicresample(m::ParticleFilter, wpnew::Vector{Vector{Float64}}, 
                             λ::Vector{Float64}, λsum::Float64)
    N = length(wpnew)
    Uincr = λsum / N
    U = Uincr * rand()
    λs = 0.0
    wpi = 1
    for i = 1:N
        k = 0
        λs += λ[i]
        while λs > U
            k += 1
            U += Uincr
        end
        # move k replicas of wpnew[i] into wp
        for j = 1:k
            m.wp[wpi] = wpnew[i]
            wpi += 1
        end
    end
end
function update!(m::ParticleFilter, x::Vector{Float64}, y::Int64)
    N = length(m.wp)
    # resample weight vectors
    c = _ParticleFilterCache(m, x, y)
    λ = Vector{Float64}(undef, N)
    wpnew = Vector{Vector{Float64}}(undef, N)
    λsum = 0.0
    for i = 1:N
        wpnew[i], λi = _samplewp(m, c, m.wp[i])
        λsum += λi
        λ[i] = λi
    end
    # re-draw weight vectors according to particle weights
    _systematicresample(m, wpnew, λ, λsum)
    m.diffused[1] = false
end
function diffuse!(m::ParticleFilter)
    m.diffused[1] = true
end

end # module