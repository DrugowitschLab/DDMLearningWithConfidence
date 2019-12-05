# script to plot summaries of the the learning performance of different models
#
# Call the script as
#
# plotLearningSummary.jl inptype learntype
#
# where inptype ∈ {"corr", "uncorr"}, and learntype ∈ {"learn", "ss"}.

import Cairo, Fontconfig
using Colors, Gadfly

include("common.jl")

# plot settings
const meantheme = Theme(default_color="black")
const t = Theme()
const uppctl = 0.75
const lopctl = 0.25
const plotcols = Dict{String,RGB}(
        "opt"             => RGB(0.0 , 0.0 , 0.0 ),
        "adf"             => RGB(0.93, 0.42, 0.0 ),
        "adfdiag"         => RGB(0.95, 0.61, 0.33),
        "taylor"          => RGB(0.18, 0.63, 0.73),
        "delta_0.10"      => RGB(0.80, 0.0 , 0.0 ),
        "delta_0.30"      => RGB(0.81, 0.16, 0.16),
        "delta_0.50"      => RGB(0.83, 0.33, 0.33),
        "delta_0.70"      => RGB(0.84, 0.49, 0.49),
        "delta_0.90"      => RGB(0.86, 0.66, 0.66),
        "normdelta_0.10"  => RGB(0.0 , 0.80, 0.0 ),
        "normdelta_0.30"  => RGB(0.16, 0.81, 0.16),
        "normdelta_0.50"  => RGB(0.33, 0.83, 0.33),
        "normdelta_0.70"  => RGB(0.49, 0.84, 0.49),
        "normdelta_0.90"  => RGB(0.66, 0.86, 0.66),
        "lhgrad_0.10"     => RGB(0.0 , 0.0 , 0.80),
        "lhgrad_0.30"     => RGB(0.16, 0.16, 0.81),
        "lhgrad_0.50"     => RGB(0.33, 0.33, 0.83),
        "lhgrad_0.70"     => RGB(0.49, 0.49, 0.84),
        "lhgrad_0.90"     => RGB(0.66, 0.66, 0.86)
    )
# inputs dimensions to plot
const Ns = (2, 5, 10, 50)
const αs = (0.1, 0.3, 0.5, 0.7, 0.9)
const αstrs = (@sprintf("%4.2f", α) for α in αs)

# parse command line arguments
function parseargs()
    length(ARGS) == 2 || error("Expected two parameters")
    inptype = ARGS[1]
    learntype = ARGS[2]
    if inptype ∉ ("corr", "uncorr")
        error("Unkown input type $inptype")
    end
    if learntype ∉ ("learn", "ss")
        error("Unknown learn type $learntype")
    end
    return inptype, learntype
end

function readconf(conffile)
    conf = ConfParse("conf/$(conffile).ini")
    parse_conf!(conf)
    task = createtask(conf)
    return task
end

# function to compute per-trial statistics of the given dataset
function pertrialstats(df, task, maxperf, randperf)
    df[:corr] = df[:choice] .== Int64.(sign.(df[:μ]))
    df[:Ifrac] = df[:modelI] ./ df[:trueI]
    df[:relperf] = (df[:Eperf] .- randperf) / (maxperf-randperf)

    dfstats = by(df, :trial, 
        df -> DataFrame(
            pc = mean(df[:corr]),
        
            t = mean(df[:t]),
            tmed = quantile(df[:t], 0.5),
            tup =  quantile(df[:t], uppctl),
            tlo = quantile(df[:t], lopctl),
        
            perf = (taskperf(task, mean(df[:corr]), mean(df[:t])) .- randperf) / (maxperf-randperf),

            avgpc = mean(df[:EPC]),
            avgpcmed = quantile(df[:EPC], 0.5),
            avgpcup =  quantile(df[:EPC], uppctl),
            avgpclo = quantile(df[:EPC], lopctl),
        
            avgt = mean(df[:EDT]),
            avgtmed = quantile(df[:EDT], 0.5),
            avgtup =  quantile(df[:EDT], uppctl),
            avgtlo = quantile(df[:EDT], lopctl),

            avgperf = mean(df[:relperf]),
            avgperfmed = quantile(df[:relperf], 0.5),
            avgperfup = quantile(df[:relperf], uppctl),
            avgperflo = quantile(df[:relperf], lopctl),

            Ifrac = mean(df[:Ifrac]),
            Ifracmed = quantile(df[:Ifrac], 0.5),
            Ifracup = quantile(df[:Ifrac], uppctl),
            Ifraclo = quantile(df[:Ifrac], lopctl),
        
            angerr = mean(df[:angerr]),
            angerrmed = quantile(df[:angerr], 0.5),
            angerrup = quantile(df[:angerr], uppctl),
            angerrlo = quantile(df[:angerr], lopctl),
        ))
end

# loads data for a single model
function loadmodeldata(modelname, conffile, task, maxperf, randperf, α::Union{Nothing,String}=nothing)
    md = Vector{DataFrame}(undef, length(Ns))
    for n in 1:length(Ns)
        datafile = α == nothing ?
            "data/learning_$(conffile)_$(modelname)_$(Ns[n])" :
            "data/learning_$(conffile)_$(modelname)_$(Ns[n])_$(α)"
        md[n] = pertrialstats(readcompressedtable(datafile), task, maxperf, randperf)
    end
    return md
end

# loads and processes the data
function loaddata(inptype, learntype)
    conffile = "$(learntype)$(inptype)"
    # performance ranges
    task = readconf(conffile)
    maxperf = taskmaxperf(task)
    randperf = taskrandperf(task)
    # load probabilistic models
    d = Dict{String,Vector{DataFrame}}()
    lm(m, α::Union{Nothing,String}=nothing) = loadmodeldata(m, conffile, task, maxperf, randperf, α)
    d["opt"] = lm(learntype == "learn" ? "gibbs" : "pf")
    d["adf"] = lm("adf")
    d["adfdiag"] = lm("adfdiag")
    d["taylor"] = lm("taylor")
    # load models with learning rates
    for α in αstrs
        d["delta_$α"] = lm("delta", α)
        d["normdelta_$α"] = lm("normdelta", α)
        d["lhgrad_$α"] = lm("lhgrad", α)
    end
    return d
end

# generate plot
function genperfplot(d, perfmeasure, perfname, perflims)
    plots = Matrix{Plot}(undef, length(Ns), 4)
    pm(m, n) = layer(d[m][n], x="trial", y=perfmeasure, 
        Geom.line, Theme(default_color=plotcols[m]))
    for n in 1:length(Ns)
        # probabilistic models
        plots[n,1] = plot(
            [pm(m, n) for m ∈ ("opt", "adf", "adfdiag", "taylor")]...,
            Guide.xlabel("trial"), Guide.ylabel(perfname),
            Coord.cartesian(ymin=perflims[1], ymax=perflims[2])
            )
        # delta rule
        plots[n,2] = plot(
            [pm("delta_$αstr", n) for αstr ∈ αstrs]...,
            Guide.xlabel("trial"), Guide.ylabel(perfname),
            Coord.cartesian(ymin=perflims[1], ymax=perflims[2])
            )
        # normalized delta rule
        plots[n,3] = plot(
            [pm("normdelta_$αstr", n) for αstr ∈ αstrs]...,
            Guide.xlabel("trial"), Guide.ylabel(perfname),
            Coord.cartesian(ymin=perflims[1], ymax=perflims[2])
            )
        # likelihood gradient
        plots[n,4] = plot(
            [pm("lhgrad_$αstr", n) for αstr ∈ αstrs]...,
            Guide.xlabel("trial"), Guide.ylabel(perfname),
            Coord.cartesian(ymin=perflims[1], ymax=perflims[2])
            )
    end
    return gridstack(plots)
end

# main
inptype, learntype = parseargs()
d = loaddata(inptype, learntype)

println("Generating reward rate plot")
p = genperfplot(d, "avgperf", "Rel rew rate", (0.0, 1.0))
outfile = "figs/learnsum_$(learntype)$(inptype)_rewrate.pdf"
draw(PDF(outfile, 4*6inch, length(Ns)*4inch), p)
println("Plot written to $outfile")

println("Generating Fisher information plot")
p = genperfplot(d, "Ifrac", "Rel Fisher info", (0.0, 1.0))
outfile = "figs/learnsum_$(learntype)$(inptype)_Ifrac.pdf"
draw(PDF(outfile, 4*6inch, length(Ns)*4inch), p)
println("Plot written to $outfile")

println("Generating angular error plot")
p = genperfplot(d, "angerr", "Angular error", (0.0, 90.0))
outfile = "figs/learnsum_$(learntype)$(inptype)_angerr.pdf"
draw(PDF(outfile, 4*6inch, length(Ns)*4inch), p)
println("Plot written to $outfile")
