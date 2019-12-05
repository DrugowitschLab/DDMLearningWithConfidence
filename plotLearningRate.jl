# Generates learning rate over confidence figures.
#
# Call the script as
#
# plotLearningRate.jl configfile model
#
# where conf/configfile.ini is the configuation file, model is the learning model,
# and N is the input dimensionality. α is the optional learning rate (only required
# for certain models).
#
# To function generates the figure and writes it to
#
# figs/learningrate_configfile_model_N[_α].pdf

import Cairo, Fontconfig
using Colors, Gadfly

include("common.jl")
include("models.jl")


# plot settings
const t = Theme(default_color = colorant"red", point_size=1pt, highlight_width=0pt)
const corrcol = RGB(0.0, 0.8, 0.0)
const incorrcol = RGB(0.8, 0.0, 0.0)
const corrαcol = Dict{String,RGB}(
    "0.10" => RGB(0.0 , 0.8 , 0.0 ),
    "0.50" => RGB(0.33, 0.83, 0.33),
    "0.90" => RGB(0.66, 0.86, 0.66))
const incorrαcol = Dict{String,RGB}(
    "0.10" => RGB(0.8 , 0.0 , 0.0 ),
    "0.50" => RGB(0.83, 0.33, 0.33),
    "0.90" => RGB(0.86, 0.66, 0.66))
# imput dimensions / learning rates to plot
const Ns = (2, 5, 10, 50)
const αs = ("0.10", "0.50", "0.90")
# how many points to plot (subsampling; nothing avoids subsampling)
const subsamplesize = 1000

# parse command line arguments
function parseargs()
    length(ARGS) == 2 || error("Expected two parameters")
    conffile = ARGS[1]
    modelname = ARGS[2]
    return conffile, modelname
end

# loads data from file
function loaddata(conffile, modelname)
    subsample(df) = subsamplesize == nothing ? 
        df : df[rand(1:size(df, 1), subsamplesize), :]
    if requiresα(modelname)
        df = Matrix{DataFrame}(undef, length(Ns), length(αs))
        for i = 1:length(Ns), j = 1:length(αs)
            N, α = Ns[i], αs[j]
            datafile = "data/learningrate_$(conffile)_$(modelname)_$(N)_$(α)"
            println("Processing $datafile")
            df[i,j] = subsample(readcompressedtable(datafile))
        end
    else
        df = Vector{DataFrame}(undef, length(Ns))
        for i = 1:length(Ns)
            N = Ns[i]
            datafile = "data/learningrate_$(conffile)_$(modelname)_$(N)"
            println("Processing $datafile")
            df[i] = subsample(readcompressedtable(datafile))
        end
    end
    return df
end

# generates plots
function genlrplots(df)
    plots = Matrix{Plot}(undef, length(Ns), 2)
    hasα = ndims(df) == 2
    for i = 1:length(Ns)
        if hasα
            dfcorr = DataFrame[
                df[i,j][df[i,j][:conf] .>= 0.5, :] for j in 1:length(αs)]
            dfincorr = DataFrame[
                df[i,j][df[i,j][:conf] .< 0.5, :] for j in 1:length(αs)]
            plots[i,1] = plot(
                [layer(dfcorr[j], x="conf", y="lr", Geom.point, 
                    Theme(t, default_color=corrαcol[αs[j]])) for j in 1:length(αs)]...,
                [layer(dfincorr[j], x="conf", y="lr", Geom.point,
                    Theme(t, default_color=incorrαcol[αs[j]])) for j in 1:length(αs)]...,
                Guide.xlabel("confidence"), Guide.ylabel("learning rate, N=$(Ns[i])"),
                Coord.cartesian(xmin=0, xmax=1)
                )
            plots[i,2] = plot(
                [layer(dfcorr[j], x="conf", y="wdiff", Geom.point, 
                    Theme(t, default_color=corrαcol[αs[j]])) for j in 1:length(αs)]...,
                [layer(dfincorr[j], x="conf", y="wdiff", Geom.point,
                    Theme(t, default_color=incorrαcol[αs[j]])) for j in 1:length(αs)]...,
                Guide.xlabel("confidence"), Guide.ylabel("|| w(n+1) - w(n) ||"),
                Coord.cartesian(xmin=0, xmax=1)
                )
        else
            dfi = df[i]
            dfcorr = dfi[dfi[:conf] .>= 0.5, :]
            dfincorr = dfi[dfi[:conf] .< 0.5, :]
            plots[i,1] = plot(
                layer(dfcorr, x="conf", y="lr", Geom.point, Theme(t, default_color=corrcol)),
                layer(dfincorr, x="conf", y="lr", Geom.point, Theme(t, default_color=incorrcol)),
                Guide.xlabel("confidence"), Guide.ylabel("learning rate, N=$(Ns[i])"),
                Coord.cartesian(xmin=0, xmax=1)
                )
            plots[i,2] = plot(
                layer(dfcorr, x="conf", y="wdiff", Geom.point, Theme(t, default_color=corrcol)),
                layer(dfincorr, x="conf", y="wdiff", Geom.point, Theme(t, default_color=incorrcol)),
                Guide.xlabel("confidence"), Guide.ylabel("|| w(n+1) - w(n) ||"),
                Coord.cartesian(xmin=0, xmax=1)
                )
        end
    end
    return gridstack(plots)
end

conffile, modelname = parseargs()
df = loaddata(conffile, modelname)
println("Generating plot")
p = genlrplots(df)
outfile = "figs/learningrate_$(conffile)_$(modelname).pdf"
draw(PDF(outfile, 2*6inch, length(Ns)*4inch), p)
println("Plot written to $outfile")
