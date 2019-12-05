# script to plot summaries of the the steady-state performance of different models
#
# Call the script as
#
# plotSSSummary.jl inptype
#
# where inptype ∈ {"corr", "uncorr"}.

import Cairo, Fontconfig
using DataFrames, Colors, Gadfly

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
const αstrs = Tuple([@sprintf("%4.2f", α) for α in αs])

# parse command line arguments
function parseargs()
    length(ARGS) == 1 || error("Expected one parameter")
    inptype = ARGS[1]
    if inptype ∉ ("corr", "uncorr")
        error("Unkown input type $inptype")
    end
    return inptype
end

function readconf(conffile)
    conf = ConfParse("conf/$(conffile).ini")
    parse_conf!(conf)
    task = createtask(conf)
    trialsavg = parse(Int64, retrieve(conf, "plotsssummary", "trialsavg"))
    return task, trialsavg
end

# function to compute per-trial statistics of the given dataset
function avgstats(df, task, maxperf, randperf)
    df[:corr] = df[:choice] .== Int64.(sign.(df[:μ]))
    df[:Ifrac] = df[:modelI] ./ df[:trueI]
    df[:relperf] = (df[:Eperf] .- randperf) / (maxperf-randperf)

    dfstats = Dict{String,Float64}(
            "avgperf" => mean(df[:relperf]),
            "avgperfmed" => quantile(df[:relperf], 0.5),
            "avgperfup" => quantile(df[:relperf], uppctl),
            "avgperflo" => quantile(df[:relperf], lopctl),
            "avgperfsd" => √(var(df[:relperf])),
            "avgperfsem" => √(var(df[:relperf])/length(df[:relperf])),

            "Ifrac" => mean(df[:Ifrac]),
            "Ifracmed" => quantile(df[:Ifrac], 0.5),
            "Ifracup" => quantile(df[:Ifrac], uppctl),
            "Ifraclo" => quantile(df[:Ifrac], lopctl),
            "Ifracsd" => √(var(df[:Ifrac])),
            "Ifracsem" => √(var(df[:Ifrac])/length(df[:Ifrac])),

            "angerr" => mean(df[:angerr]),
            "angerrmed" => quantile(df[:angerr], 0.5),
            "angerrup" => quantile(df[:angerr], uppctl),
            "angerrlo" => quantile(df[:angerr], lopctl),
            "angerrsd" => √(var(df[:angerr])),
            "angerrsem" => √(var(df[:angerr])/length(df[:angerr]))            
        )
    return dfstats
end

# loads data for a single model
function loadmodeldata(modelname, conffile, task, trialsavg,
    maxperf, randperf, α::Union{Nothing,String}=nothing)
    md = Vector{Dict{String,Float64}}(undef, length(Ns))
    for n in 1:length(Ns)
        datafile = α == nothing ?
            "data/learning_$(conffile)_$(modelname)_$(Ns[n])" :
            "data/learning_$(conffile)_$(modelname)_$(Ns[n])_$(α)"
        println("Processing $datafile")
        df = readcompressedtable(datafile)
        maxtrial = maximum(df[:trial])
        df = df[df[:trial] .> maxtrial - trialsavg,:]
        md[n] = avgstats(readcompressedtable(datafile), task, maxperf, randperf)
    end
    return md
end

# loads and processes the data
function loaddata(inptype)
    conffile = "ss$(inptype)"
    # performance ranges
    task, trialsavg = readconf(conffile)
    maxperf = taskmaxperf(task)
    randperf = taskrandperf(task)
    # load probabilistic models
    d = Dict{String,Vector{Dict{String,Float64}}}()
    lm(m, α::Union{Nothing,String}=nothing) = loadmodeldata(
        m, conffile, task, trialsavg, maxperf, randperf, α)
    d["opt"] = lm("pf")
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

# generates plot for single N (=Ns[Ni]), single perf. measure
function genNperfplot(d, Ni, perfmeasure, perfname, perflims)
    # helper functions for heuristic models with learning rates
    ys(m) = Float64[d["$(m)_$(αstr)"][Ni][perfmeasure] for αstr in αstrs]
    ymins(m) = Float64[
        d["$(m)_$(αstr)"][Ni][perfmeasure] - d["$(m)_$(αstr)"][Ni]["$(perfmeasure)sem"] 
        for αstr in αstrs]
    ymaxs(m) = Float64[
        d["$(m)_$(αstr)"][Ni][perfmeasure] + d["$(m)_$(αstr)"][Ni]["$(perfmeasure)sem"] 
        for αstr in αstrs]
    plot([layer(x=[0.0, 1.0], y=[1,1] * d[m][Ni][perfmeasure], 
            Geom.line, Theme(default_color=plotcols[m]))
          for m ∈ ("opt", "adf", "adfdiag", "taylor")]...,
        [layer(x=collect(αs), y=ys(m), ymin=ymins(m), ymax=ymaxs(m),
            Geom.line, Geom.errorbar, Theme(default_color=plotcols["$(m)_$(αstrs[1])"]))
          for m ∈ ("normdelta", "lhgrad")]...,
        Guide.xlabel("learning rate"), Guide.ylabel("$(perfname), N=$(Ns[Ni])"),
        Coord.cartesian(ymin=perflims[1], ymax=perflims[2])
        )
end

# generate plot
function genperfplot(d)
    plots = Matrix{Plot}(undef, length(Ns), 3)
    for Ni in 1:length(Ns)
        plots[Ni, 1] = genNperfplot(d, Ni, "avgperf", "Rel rew rate", (0.0, 1.0))
        plots[Ni, 2] = genNperfplot(d, Ni, "Ifrac", "Rel Fisher info", (0.0, 1.0))
        plots[Ni, 3] = genNperfplot(d, Ni, "angerr", "Angular error", (0.0, 90.0))
    end
    return gridstack(plots)
end

# main
inptype = parseargs()
d = loaddata(inptype)
p = genperfplot(d)
outfile = "figs/sssum_ss$(inptype).pdf"
draw(PDF(outfile, 3*6inch, length(Ns)*4inch), p)
println("Plot written to $outfile")
