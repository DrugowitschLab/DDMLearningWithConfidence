# Generates figures summarizing performance collected by simLearning.
#
# Call the script as
#
# plotLearning.jl configfile model N [α]
#
# where conf/configfile.ini is the configuation file, model is the learning model,
# and N is the input dimensionality. α is the optional learning rate (only required
# for certain models). To function generates the figure and writes it to
#
# figs/learning_configfile_model_N.pdf

import Cairo, Fontconfig
using StatsBase, Gadfly

include("common.jl")
include("models.jl")

# plot settings
const meantheme = Theme(default_color="black")
const t = Theme()
const uppctl = 0.75
const lopctl = 0.25

# parse command line arguments
function parseargs()
    length(ARGS) ∈ (3, 4) || error("Expected three or four parameters")
    conffile = ARGS[1]
    modelname = ARGS[2]
    N = parse(Int64, ARGS[3])
    α = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : nothing
    return conffile, modelname, N, α
end

function readconf(conffile)
    conf = ConfParse("conf/$(conffile).ini")
    parse_conf!(conf)
    task = createtask(conf)
    return task
end

conffile, modelname, N, α = parseargs()
if requiresα(modelname) && α == nothing
    error("Model $(modelname) requires learning rate parameter")
end
task = readconf(conffile)
maxperf = taskmaxperf(task)
randperf = taskrandperf(task)
const basefilename = requiresα(modelname) ?
    "learning_$(conffile)_$(modelname)_$(N)_$(@sprintf("%.2f", α))" :
    "learning_$(conffile)_$(modelname)_$(N)"

# read and process data
datafile = "data/$(basefilename)"
df = readcompressedtable(datafile)

println("Computing statistics")
df[:corr] = df[:choice] .== Int64.(sign.(df[:μ]))
df[:Ifrac] = df[:modelI] ./ df[:trueI]
df[:relperf] = (df[:Eperf] .- randperf) / (maxperf-randperf)

dftrials = by(df, :trial, 
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

# plot results
println("Generating plots")
ppc = plot(dftrials, x="trial", y="pc", Geom.line,
    Guide.xlabel("trial"), Guide.ylabel("p(corr)"),
    Coord.cartesian(ymin=0.4, ymax=1), meantheme)
pepc = plot(dftrials,
    layer(x="trial", y="avgpc", Geom.line, meantheme),
    layer(x="trial", y="avgpcmed", ymin="avgpclo", ymax="avgpcup",
        Geom.line, Geom.ribbon),
    Guide.xlabel("trial"), Guide.ylabel("&lt;p(corr)&gt;"),
    Coord.cartesian(ymin=0.4, ymax=1), t)
pdt = plot(dftrials, 
    layer(x="trial", y="t", Geom.line, meantheme),
    layer(x="trial", y="tmed", ymin="tlo", ymax="tup", Geom.line, Geom.ribbon),
    Guide.xlabel("trial"), Guide.ylabel("dt"), Coord.cartesian(ymin=0), t)
pedt = plot(dftrials,
    layer(x="trial", y="avgt", Geom.line, meantheme),
    layer(x="trial", y="avgtmed", ymin="avgtlo", ymax="avgtup",
        Geom.line, Geom.ribbon),
    Guide.xlabel("trial"), Guide.ylabel("&lt;dt&gt;"), Coord.cartesian(ymin=0), t)
pperf = plot(dftrials, x="trial", y="perf", Geom.line,
    Guide.xlabel("trial"), Guide.ylabel("frac rew rate"),
    Coord.cartesian(ymin=0.0, ymax=1.1), meantheme)
peperf = plot(dftrials,
    layer(x="trial", y="avgperf", Geom.line, meantheme),
    layer(x="trial", y="avgperfmed", ymin="avgperflo", ymax="avgperfup",
        Geom.line, Geom.ribbon),
    Guide.xlabel("trial"), Guide.ylabel("&lt;frac rew rate&gt;"),
    Coord.cartesian(ymin=0.0, ymax=1.1), t)
pangerr = plot(dftrials, 
    layer(x="trial", y="angerr", Geom.line, meantheme),
    layer(x="trial", y="angerrmed", ymin="angerrlo", ymax="angerrup",
        Geom.line, Geom.ribbon),
    Guide.xlabel("trial"), Guide.ylabel("angular error"), Coord.cartesian(ymin=0.0), t)
pIfrac = plot(dftrials,
    layer(x="trial", y="Ifrac", Geom.line, meantheme),
    layer(x="trial", y="Ifracmed", ymin="Ifraclo", ymax="Ifracup",
        Geom.line, Geom.ribbon),
    Guide.xlabel("trial"), Guide.ylabel("frac Fisher info"),
    Coord.cartesian(ymin=0.0, ymax=1.1), t)

p = gridstack(Union{Plot,Gadfly.Compose.Context}[ppc pepc; pdt pedt; pperf peperf; pangerr pIfrac])
plotfile = "figs/$(basefilename).pdf"
println("Writing plot to $plotfile")
draw(PDF(plotfile, 8.5inch, 11inch), p)
