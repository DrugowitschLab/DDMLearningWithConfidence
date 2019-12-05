# set of commonly shares functions, including
#
# - Definition of task types
# - Definition of input types
# - Standard diffusion model helper functions

using Compat.LinearAlgebra, Compat.Printf
using Optim, ConfParser, Distributions, DiffModels, DataFrames, HDF5

import Compat.occursin, Compat.range, Compat.undef

# discretization when computing expectations over p(μ)
const μDISCR=100
# minimal size of weight for diffusion simulation
const MINDIFFW=1e-30

# -----------------------------------------------------------------------------
# General helper functions
# -----------------------------------------------------------------------------

# performs stabilized Gram Schmidt orthonormalization on the columns of U
function gram_schmidt!(U::Matrix{T}) where T <: Real
    n, m = size(U)
    @assert n ≥ 1
    @inbounds for k = 1:m
        for j = 1:k-1
            dotjk = U[1,j] * U[1,k]
            @simd for i = 2:n
                dotjk += U[i,j] * U[i,k]
            end
            @simd for i = 1:n
                U[i,k] -= dotjk * U[i,j]
            end
        end
        uknorm = abs2(U[1,k])
        @simd for i = 2:n
            uknorm += abs2(U[i,k])
        end
        uknorm = √(uknorm)
        @simd for i = 1:n
            U[i,k] /= uknorm
        end
    end
end
function gram_schmidt(V::Matrix{T}) where T <: Real
    U = copy(V)
    gram_schmidt!(U)
    return U
end

# angle (in radians) between two vectors
angle(a::AbstractVector, b::AbstractVector) = acos(min(1.0, abs(
    dot(a, b) / (norm(a) * norm(b)))))


# -----------------------------------------------------------------------------
# I/O functions, converting DataFrame to HDF5 and back
# -----------------------------------------------------------------------------

const dfsymtoh5str = Dict{Symbol,String}(
    :μ => "mu")
const h5strtodfsym = Dict(dfsymtoh5str[k] => k for k in keys(dfsymtoh5str))

function _writecompressedtable(fullfilename, df, verbose::Bool=true)
    !verbose || println("Writing data to $fullfilename")
    h5open(fullfilename, "w") do file
        for colname in names(df)
            file[get(dfsymtoh5str, colname, String(colname)),
                 "blosc", 9] = df[colname] 
        end
    end
end
writecompressedtable(filename, df, verbose::Bool=true) = _writecompressedtable(
    "$(filename).h5", df, verbose)
writecompressedtable(filename, df, fromrep, verbose::Bool=true) = _writecompressedtable(
    @sprintf("%s_%06d.h5", filename, fromrep), df, verbose)

# read table, potentially from multiple files
function readcompressedtable(filename, verbose::Bool=true)
    df = DataFrame()
    # collect list of target files
    filedir = dirname(filename)
    filebase = basename(filename)
    reunnumbered = Regex("^$(filebase).h5")
    renumbered = Regex("^$(filebase)_(\\d{6}).h5")
    istargetfile(f) = (isfile(joinpath(filedir, f)) &&
        (occursin(reunnumbered, f) || occursin(renumbered, f)))
    targetfiles = filter(istargetfile, readdir(filedir))
    if isempty(targetfiles)
        error("No data file matching $filename found")
    end
    # if only single one exists: load it
    if length(targetfiles) == 1
        f = joinpath(filedir, targetfiles[1])
        !verbose || println("Reading single data file $f")
        local h5dict
        h5open(f, "r") do file
            h5dict = read(file)
        end
        dfdict = Dict(get(h5strtodfsym, n, Symbol(n)) => h5dict[n] for n in keys(h5dict))
        return DataFrame(dfdict)
    end
    # multiple? determine order
    numtargets = filter(f -> occursin(renumbered, f), targetfiles)
    sort!(numtargets,
        by = f->parse(Int64, match(renumbered, f).captures[1]))
    if length(numtargets) < length(targetfiles)
        @assert length(targetfiles) == length(numtargets) + 1
        unnumtarget = filter(f -> occursin(reunnumbered, f), targetfiles)[1]
        pushfirst!(numtargets, unnumtarget)
        !verbose || println("Loading unnumbered before numbered data")
    end
    dfs = DataFrame[]
    for f in numtargets
        ff = joinpath(filedir, f)
        !verbose || println("Reading data from $ff")
        local h5dict
        h5open(ff, "r") do file
            h5dict = read(file)
        end
        dfdict = Dict(get(h5strtodfsym, n, Symbol(n)) => h5dict[n] for n in keys(h5dict))
        push!(dfs, DataFrame(dfdict))
    end
    return vcat(dfs...)
end


# -----------------------------------------------------------------------------
# Diffusion model helper functions
# -----------------------------------------------------------------------------

# prob. correct for bound θ, drift μ
@inline ddmPC(θ, μ) = 1 / (1 + exp(-2θ * abs(μ)))
# decision time for bound θ, drift μ
@inline ddmDT(θ, μ) = iszero(μ) ? abs2(θ) : θ / μ * tanh(θ * μ)

# expected prob. correct / decision time for bound θ, drifts μ ~ N(0, σμ^2)
function ddmavgperf(θ, σμ)
    PC, DT, Zμ = 0.0, 0.0, 0.0
    σμ2 = abs2(σμ)
    for μ in range(-3σμ, stop=3σμ, length=μDISCR)
        pμ = exp(- abs2(μ) / 2σμ2)
        Zμ += pμ
        PC += pμ * ddmPC(θ, μ)
        DT += pμ * ddmDT(θ, μ)
    end
    return PC / Zμ, DT / Zμ
end

# expected reward for bound θ, drifts μ ~ N(0, σμ^2), accum. cost c
function ddmER(θ, σμ, c)
    PC, DT = ddmavgperf(θ, σμ)
    return PC - c * DT
end
# expected reward rate for bound θ, drifts μ ~ N(0, σμ^2), accum. cost c, iti
function ddmERR(θ, σμ, c, iti)
    PC, DT = ddmavgperf(θ, σμ)
    return (PC - c * DT) / (DT + iti)
end

# returns bound that maximizes expected reward
function ddmoptimθ(σμ, c)
    f(θ) = -ddmER(θ[1], σμ, c)
    res = optimize(f, [0.0], [Inf], [0.5], Fminbox(LBFGS()))
    return Optim.minimizer(res)[1]
end

# returns bound that maximizes expected reward rate
function ddmoptimθ(σμ, c, iti)
    f(θ) = -ddmERR(θ[1], σμ, c, iti)
    res = optimize(f, [0.0], [Inf], [0.5], Fminbox(LBFGS()))
    return Optim.minimizer(res)[1]
end

# simulates multivariate diffusion and returns (choice, t, x), where
# dx ~ N( μx dt, Σx dt ), and bounds {-θ,θ} on w^T x. choice ∈ {-1, 1}
function simdiffusion(w, μx, Σx, θ)
    N = length(μx)
    # lower-bound weights to avoid zero drift rates / diffusion variance
    if sum(abs2, w) < N * MINDIFFW
        #println("setting w to >zero")
        w = MINDIFFW * ones(N)
    end
    # sample (z, t) from 1D diffusion model
    μz = dot(w, μx)
    σz = √(w' * Σx * w)
    t, c = rand(sampler(ConstDrift(μz / σz, 1), ConstSymBounds(θ / σz, 1)))
    z = c ? θ : -θ
    # find x(t) for given z(t)
    if N == 1
        # single dimension, such that z = w * x
        x = z ./ w
    else
        # draw x ~ N(μx t, Σx t), subject to w^T x(t) = z(t)
        x = rand(MultivariateNormal(zeros(N), t * Σx))
        x .+= t * μx - Σx * w * (dot(w, x) - z + t * dot(w, μx)) / (w' * Σx * w)
    end
    return c ? 1 : -1, t, x
end


# -----------------------------------------------------------------------------
# Tasks types
# -----------------------------------------------------------------------------

abstract type BaseTask end
drawμ(task::BaseTask) = task.σμ * randn()
getσμ(task::BaseTask) = task.σμ

# expected reward task
struct ERTask <: BaseTask
    σμ::Float64
    c::Float64
    θ::Float64
    # initialize with optimized bound
    ERTask(σμ, c) = new(σμ, c, ddmoptimθ(σμ, c))
end
taskperf(t::ERTask, EPC, EDT) = EPC - t.c * EDT
taskmaxperf(t::ERTask) = taskperf(t, ddmavgperf(t.θ, t.σμ)...)
taskrandperf(t::ERTask) = 0.5

# reward rate task
struct RRTask <: BaseTask
    σμ::Float64
    c::Float64
    iti::Float64
    θ::Float64
    # initialize with optimized bound
    RRTask(σμ, c, iti) = new(σμ, c, iti, ddmoptimθ(σμ, c, iti))
end
taskperf(t::RRTask, EPC, EDT) = (EPC - t.c * EDT) / (EDT + t.iti)
taskmaxperf(t::RRTask) = taskperf(t, ddmavgperf(t.θ, t.σμ)...)
taskrandperf(t::RRTask) = 0.5 / t.iti

# creates task from conf file
function createtask(conf::ConfParse)
    tasktype = uppercase(retrieve(conf, "task", "type"))
    tasktype ∈ ("RR", "ER") || error("Unknown task type $(tasktype) in configuration file")
    σμ = parse(Float64, retrieve(conf, "task", "sigmu"))
    c = parse(Float64, retrieve(conf, "task", "c"))
    if tasktype == "RR"
        iti = parse(Float64, retrieve(conf, "task", "iti"))
        return RRTask(σμ, c, iti)
    else
        return ERTask(σmu, c)
    end
end

# read diffusion statistics, returns nothing if no diffusion
function readdiffstats(conf::ConfParse, N)
    diffspeed = parse(Float64, retrieve(conf, "task", "diffspeed"))
    if diffspeed ≈ 0
        return nothing
    else
        # diffusive weight model is w(n+1) = λ (w(n) - μw) + μw + ηw,
        # with λ = 1-diffspeed, and ηw ~ N(0, σd^2),
        # and σd = σw √(1-λ^2) to match a steady-state SD of σw
        μw = parse(Float64, retrieve(conf, "task", "muw"))
        σw = parse(Float64, retrieve(conf, "task", "sigw"))
        λ = 1 - diffspeed
        Aw = λ * Matrix{Float64}(I, N, N)
        bw = fill((1-λ)*μw, N)
        Σd = Matrix(abs2(σw) * (1 - abs2(λ)) * I, N, N)
        return Aw, bw, Σd
    end
end


# -----------------------------------------------------------------------------
# Input generation
# -----------------------------------------------------------------------------

# generic functions for all inputs
abstract type BaseInputs end
getw(inp::BaseInputs) = inp.w
getN(inp::BaseInputs) = length(inp.w)
getangerr(inp::BaseInputs, w) = angle(inp.w, w) * 180/π
getIloss(inp::BaseInputs, w) = abs2(dot(inp.w, w)) / (
    w' * inp.Σx * w * abs2(sum(abs2, inp.w)))
function avgperf(inp::BaseInputs, task::BaseTask, w)
    σμ = getσμ(task)
    DT, PC, Zμ = 0.0, 0.0, 0.0
    # diffusion SD for given w
    σw = √(w' * inp.Σx * w)
    for μi in range(-3σμ, stop=3σμ, length=μDISCR)
        # diffusion drift for w and μi
        μw = dot(w, getμx(inputs, μi))
        # compute decision time and probability correct for given moments
        pμ = exp(-0.5abs2(μi / σμ))  # add normalization constant after iter
        Zμ += pμ
        # probability "correct" defined relative to sign of μ rather than μw
        PCμ = ddmPC(task.θ / σw, μw / σw)
        PC += pμ * (sign(μw) == sign(μi) ? PCμ : 1 - PCμ)
        DT += pμ * ddmDT(task.θ / σw, μw / σw)
    end
    # finalize performance measures
    PC /= Zμ
    DT /= Zμ
    return PC, DT
end

# linear uncorrelated inputs
struct LinUncorrInputs <: BaseInputs
    w::Vector{Float64}
    μw::Vector{Float64}
    σw::Float64
    Σx::Symmetric{Float64,Matrix{Float64}}
    # initialize according to conf file
    function LinUncorrInputs(conf::ConfParse, task, N)
        σw = parse(Float64, retrieve(conf, "task", "sigw"))
        μw = parse(Float64, retrieve(conf, "task", "muw"))
        new(fill(μw, N), fill(μw, N), σw, Symmetric(Matrix(I / (N*abs2(μw)), N, N)))
    end
end
wmoments(inp::LinUncorrInputs) = inp.μw, Matrix(abs2(inp.σw) * I, length(inp.w), length(inp.w))
getμx(inp::LinUncorrInputs, μ) = (μ / sum(abs2, inp.w)) * inp.w
getμxprime(inp::LinUncorrInputs) = inp.w / sum(abs2, inp.w)
getΣx(inp::LinUncorrInputs) = inp.Σx
function samplew!(inp::LinUncorrInputs)
    N = length(inp.w)
    inp.w .= inp.μw + inp.σw * randn(N)
    inp.Σx.data .= Matrix(I / sum(abs2, inp.w), N, N)
    return inp.w
end
function diffuse!(inp::LinUncorrInputs, Aw, bw, Σd)
    N = length(inp.w)
    inp.w .= Aw * inp.w + bw + rand(MultivariateNormal(zeros(N), Σd))
    inp.Σx.data .= Matrix(I / sum(abs2, inp.w), N, N)
    return inp.w
end

# linear correlated inputs
struct LinCorrInputs <: BaseInputs
    w::Vector{Float64}
    μw::Float64
    σw::Float64
    f0::Float64
    noisemult::Float64
    a::Vector{Float64}
    Σx::Symmetric{Float64,Matrix{Float64}}  # ensure that cov is Symmetric
    # initialize according to conf file
    function LinCorrInputs(conf::ConfParse, task, N)
        μw = parse(Float64, retrieve(conf, "task", "muw"))
        σw = parse(Float64, retrieve(conf, "task", "sigw"))
        f0 = parse(Float64, retrieve(conf, "lincorrinputs", "f0"))
        noisemult = parse(Float64, retrieve(conf, "lincorrinputs", "noisemult"))
        new(fill(μw, N), μw, σw, f0, noisemult, zeros(N), Symmetric(Matrix{Float64}(I, N, N)))
    end
end
wmoments(inp::LinCorrInputs) = fill(inp.μw, length(inp.w)), Matrix(abs2(inp.σw) * I, length(inp.w), length(inp.w))
getμx(inp::LinCorrInputs, μ) = (μ / sum(abs2, inp.w)) * inp.w .+ inp.a
getμxprime(inp::LinCorrInputs) = inp.w ./ sum(abs2, inp.w)
getΣx(inp::LinCorrInputs) = inp.Σx
function samplew!(inp::LinCorrInputs)
    N = length(inp.w)
    inp.w .= inp.μw .+ inp.σw .* randn(N)
    # find a orthogonal to w that minimizes ||a - f0||
    inp.a .= inp.f0 .* (1 .- (sum(inp.w)/sum(abs2, inp.w)).*inp.w)
    # redraw Σx that matches w, starting with eigenvectors
    inp.Σx.data[:,1] .= inp.w
    inp.Σx.data[:,2:end] .= randn(N, N-1)
    gram_schmidt!(inp.Σx.data)
    # generate eigenvalues and Σx
    D = max.([1; inp.noisemult * exp.(-(1:N-1))], 0.001) ./ sum(abs2, inp.w)
    inp.Σx.data .= inp.Σx.data * Diagonal(D) * inp.Σx.data'
    # return new weight vector
    return inp.w
end
function diffuse!(inp::LinCorrInputs, Aw, bw, Σd)
    N = length(inp.w)
    prevnorm = norm(inp.w)
    prevw = copy(inp.w) / prevnorm
    # diffuse weight, update mean input parameter
    inp.w .= Aw * inp.w + bw + rand(MultivariateNormal(zeros(N), Σd))
    inp.a .= inp.f0 .* (1 .- (sum(inp.w)/sum(abs2, inp.w)).*inp.w)
    # find rotation matrix U mapping prevw to current w to update Σx
    newnorm = norm(inp.w)
    neww = inp.w / newnorm
    U::Matrix{Float64} = hcat(prevw, neww, randn(N, N-2))
    gram_schmidt!(U)
    U[:,2] = neww   # prevent orthogonalization of current w
    U .= hcat(neww, 2*dot(prevw,neww)*neww - prevw, U[:,3:end]) / U
    inp.Σx.data .= abs2(prevnorm / newnorm) * U * inp.Σx * U'
    # return new weight vector
    return inp.w
end

# create inputs from conf file
function createinputs(conf::ConfParse, task, N)
    inputtype = lowercase(retrieve(conf, "task", "inputs"))
    if inputtype == "linuncorr"
        return LinUncorrInputs(conf, task, N)
    elseif inputtype == "lincorr"
        return LinCorrInputs(conf, task, N)
    else
        error("Unknown inputs type $(inputtype) in configuration file")
    end
end
