# script to plot sequential choice dependencies
#
# Call the script as
#
# plotSeqDep.jl configfile model N
#
# where conf/configfile.ini is the configuation file, model is the learning model,
# and N is the input dimensionality. To function generates the figure and writes
# it to
#
# figs/seqdep_configfile_model_N.pdf

import Cairo, Fontconfig
using StatsBase, SpecialFunctions, Colors, Gadfly

include("common.jl")

# plot settings
const μbins = 10  # needs to be even


# parse command line arguments
function parseargs()
    length(ARGS) == 3 || error("Expected three parameters")
    conffile = ARGS[1]
    modelname = ARGS[2]
    N = parse(Int64, ARGS[3])
    return conffile, modelname, N
end

function readconf(conffile)
    conf = ConfParse("conf/$(conffile).ini")
    parse_conf!(conf)
    task = createtask(conf)
    trialsavg = parse(Int64, retrieve(conf, "plotseqdep", "trialsavg"))
    return task, trialsavg
end

# fits cumulative gaussian to y = f(x), returning ML parameters
function fitcumgauss(x, y)
    logΦ(x) = log1p(erf(x / √(2))) - log(2)
    function f(θ)
        r = (x .- θ[1]) ./ θ[2]
        return -sum(y .* logΦ.(r) + (1 .- y) .* logΦ.(-r))
    end
    res = optimize(f, [-Inf, 0.0], [Inf, Inf], [0.0, 100.0], Fminbox(LBFGS()))
    return Optim.minimizer(res)
end

# compute sequential dependency statistics
function compseqstats(df, trialsavg)
    # identify μ bin boundaries by μ quantiles
    μb = fill(NaN, μbins+1)
    μb[1] = -Inf
    μci = μbins ÷ 2 + 1
    for i = 1:(μci-1)
        # symmetrized quantiles
        μb[i+1] = 0.5*(quantile(df[:μ], i/μbins)
            - quantile(df[:μ], ((μbins-i)/μbins)))
    end
    μb[μci] = 0.0
    μb[(μci+1):end] = -μb[(μci-1):-1:1]
    # identify μ bin in data (superslow - can be improved?)
    df[:μbin] = sum(((df[:μ] .> μb[1:(end-1)]') .& (df[:μ] .<= μb[2:end]')) 
                           .* (1:μbins)', dims=2)[:,1]

    #df[:μbin] = [findfirst((df[i,:μ] .> μb[1:(end-1)]) .& (df[i,:μ] .<= μb[2:end]))
    #             for i = 1:size(df, 1)]
    μm = [mean(df[df[:μbin] .== i,:μ]) for i=1:μbins]
    # add more trial stats
    df[:prevμ] = [df[1,:μ]; df[1:(end-1),:μ]]
    df[:prevμbin] = [df[1,:μbin]; df[1:(end-1),:μbin]]
    df[:corr] = round.(Int64, sign.(df[:μ])) .== df[:choice]
    df[:prevcorr] = [df[1,:corr]; df[1:(end-1),:corr]]
    # remove all but trialsavg last trials
    maxtrial = maximum(df[:trial])
    df = df[df[:trial] .> maxtrial - trialsavg,:]
    # compute statistics conditional on previous (μbin, corr) and current μbin
    dfseq = by(df, [:prevμbin, :prevcorr, :μbin]) do df
        DataFrame(pr = mean(0.5(df[:choice] .+ 1)))
    end
    # add mean μ
    dfseq[:μm] = μm[dfseq[:μbin]]
    return dfseq
end

# fits psychometric curves to compute repetition bias
function fitrepbias(dfseq)
    Φ(x) = (1 + erf(x / √(2))) / 2
    repbias = Dict{Bool,Vector{Float64}}()
    μm = fill(NaN, μbins ÷ 2)
    for prevcorr ∈ (true, false)
        b = fill(NaN, μbins ÷ 2)
        for i in 1:(μbins ÷ 2)
            # average across neg and pos bins
            μbinin = (dfseq[:prevμbin] .== i) .& (dfseq[:prevcorr] .== prevcorr)
            μn, σn = fitcumgauss(dfseq[μbinin,:μm], dfseq[μbinin,:pr])
            μbinip = (dfseq[:prevμbin] .== (μbins-i+1)) .& (dfseq[:prevcorr] .== prevcorr)
            μp, σp = fitcumgauss(dfseq[μbinip,:μm], dfseq[μbinip,:pr])
            b[i] = 0.5*(Φ(-μp / σp)-0.5 + 0.5-Φ(-μn / σn))

            @printf("Fitting prevμ bin %2d, μ = %6.3f, σ = %5.3f, p(c=1|μ=0)= %5.3f\n", 
                i, μn, σn, Φ(-μn / σn))
            @printf("Fitting prevμ bin %2d, μ = %6.3f, σ = %5.3f, p(c=1|μ=0)= %5.3f\n", 
                μbins-i+1, μp, σp, Φ(-μp / σp))

            if prevcorr
                μm[i] = 0.5*(mean(dfseq[dfseq[:μbin] .== μbins-i+1,:μm])  - 
                    mean(dfseq[dfseq[:μbin] .== i,:μm]))
            end
        end
        repbias[prevcorr] = b
    end
    print("μ magnitudes")
    for i = 1:(μbins ÷ 2)
        @printf(" %5.3f", μm[i])
    end
    println()
    return repbias, μm
end

# generates sequential psychometric function shift plot
function genseqpsychplot(dfseq, prevcorr)
    μbincols = colormap("RdBu", μbins)
    return plot([layer(dfseq[(dfseq[:prevcorr] .== prevcorr) .& (dfseq[:prevμbin] .== i), :],
                       x="μm", y="pr", Geom.line, Theme(default_color=μbincols[i]))
          for i = 1:μbins]...,
        Guide.xlabel("μ"), Guide.ylabel("p(choice = 1)"),
        Coord.cartesian(ymin=0, ymax=1))
end

# generate repetition bias plot
function genrepbiasplot(repbias, μm)
    return plot(layer(x=μm, y=repbias[true], Geom.line, Theme(default_color="green")),
        layer(x=μm, y=repbias[false], Geom.line, Theme(default_color="red")),
        Guide.xlabel("|μ|"), Guide.ylabel("repetition bias"))
end

conffile, modelname, N = parseargs()
# TODO: remove 'task' reading (as not required?)
task, trialsavg = readconf(conffile)
maxperf = taskmaxperf(task)
randperf = taskrandperf(task)

datafile = "data/learning_$(conffile)_$(modelname)_$(N)"
println("Reading data from $datafile")
df = readcompressedtable(datafile)
println("Computing statistics")
dfseq = compseqstats(df, trialsavg)
repbias, μm = fitrepbias(dfseq)
# psychometric curves after correct choices
println("Generating plots")
pcorr = genseqpsychplot(dfseq, true)
pincorr = genseqpsychplot(dfseq, false)
pbias = genrepbiasplot(repbias, μm)
outfile = "figs/seq_$(conffile)_$(modelname)_$(N).pdf"
draw(PDF(outfile, 6inch, 3*4inch), vstack(pcorr, pincorr, pbias))
println("Plot written to $outfile")
