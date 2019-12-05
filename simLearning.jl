# Simulates learning from a random initial state across many repetitions
#
# Call the script as
#
# simLearning.jl configfile model N [α] [fromrep-torep]
#
# where conf/configfile.ini is the configuation file, model is the learning model,
# and N is the input dimensionality. α is the optional learning rate (only required
# for certain models). If [fromrep-torep] is provided, only the specified
# repetitions are performed, and the filename contains torep.
#
# The function simulates a certain number of trials across many repetisions, and
# writes the output to
#
# data/learning_configfile_model_N[_α][_fromrep].csv

using ProgressMeter

include("common.jl")
include("models.jl")

# parse command line arguments
function parseargs()
    length(ARGS) ∈ (3, 4, 5) || error("Expected three to five arguments")
    conffile = ARGS[1]
    modelname = ARGS[2]
    N = parse(Int64, ARGS[3])
    α = nothing
    fromrep = nothing
    torep = nothing
    if length(ARGS) == 4
        m = match(r"^(\d+)-(\d+)", ARGS[4])
        if m == nothing
            α = parse(Float64, ARGS[4])
        else
            fromrep = parse(Int64, m.captures[1])
            torep = parse(Int64, m.captures[2])
        end
    elseif length(ARGS) == 5
        α = parse(Float64, ARGS[4])
        m = match(r"^(\d+)-(\d+)", ARGS[5])
        if m == nothing
            error("Expected fifth argument of form fromrep-torep")
        end
        fromrep = parse(Int64, m.captures[1])
        torep = parse(Int64, m.captures[2])
    end
    return conffile, modelname, N, α, fromrep, torep
end

# read configuration file
function readconf(conffile, modelname, N, α)
    conf = ConfParse("conf/$(conffile).ini")
    parse_conf!(conf)
    task = createtask(conf)
    diffstats = readdiffstats(conf, N)
    model = createmodel(conf, modelname, N, α)
    inputs = createinputs(conf, task, N)
    trials = parse(Int64, retrieve(conf, "simlearning", "trials"))
    repetitions = parse(Int64, retrieve(conf, "simlearning", "repetitions"))
    return task, diffstats, model, inputs, trials, repetitions
end

# runs single repetition, i.e. sequence of trials
function runrep(task, diffstats, model, inputs, trials)
    σμ = getσμ(task)
    withdiff = diffstats != nothing
    # resample w and get μ-independent input statistics
    samplew!(inputs)
    μxprime = getμxprime(inputs)
    trueI = dot(μxprime, getΣx(inputs) \ μxprime)
    # reset/initialize model
    reset!(model, wmoments(inputs)..., getw(inputs))

    # run 'trials' trials and collect response statistics
    df = DataFrame(trial=Int64[], μ=Float64[], choice=Int64[], t=Float64[],
        EPC=Float64[], EDT=Float64[], Eperf=Float64[],
        trueI=Float64[], modelI=Float64[], angerr=Float64[])
    for trial = 1:trials
        # draw latent state, simulate model choice, and provide model feedback
        μ = drawμ(task)
        choice, t, x = simdiffusion(getw(model), getμx(inputs, μ), getΣx(inputs), task.θ)
        w = feedback!(model, μ > 0 ? 1 : -1, t, x, getw(inputs))

        # collect statistics
        EPC, EDT = avgperf(inputs, task, w)
        Eperf = taskperf(task, EPC, EDT)
        angerr = getangerr(inputs, w)
        modelI = getIloss(inputs, w)
        push!(df, [trial, μ, choice, t, EPC, EDT, Eperf, trueI, modelI, angerr])

        # diffuse model and state
        if withdiff
            diffuse!(inputs, diffstats...)
            diffuse!(model)
        end
    end
    return df
end

# collect data across multiple repetitions
function runsim(task, diffstats, model, inputs, trials, fromrep, torep, verbose::Bool=false)
    # first run is separate
    if verbose
        p = Progress(torep-fromrep+1, 1, "Simulating...")
    end
    df = runrep(task, diffstats, model, inputs, trials)
    df[:repetition] = 1
    !verbose || next!(p)
    # others are in loop
    for repetition = (fromrep+1):torep
        dfrep = runrep(task, diffstats, model, inputs, trials)
        dfrep[:repetition] = repetition
        append!(df, dfrep)
        !verbose || next!(p)
    end
    return df
end

# main
conffile, modelname, N, α, fromrep, torep = parseargs()
hasreprange = fromrep != nothing
task, diffstats, model, inputs, trials, repetitions = readconf(conffile, modelname, N, α)
if !hasreprange
    fromrep = 1
    torep = repetitions
end
print("Running simulations for conf $(conffile), model $(modelname), N=$(N), rep=$(fromrep)-$(torep)")
!requiresα(modelname) || print(", α=$(@sprintf("%.2f", α))")
println()
#df = runsim(task, model, inputs, trials, 2, true)
#@profile df = runsim(task, diffstats, model, inputs, trials, repetitions, true)
#Profile.print()
df = runsim(task, diffstats, model, inputs, trials, fromrep, torep, true)
fnα = requiresα(modelname) ? @sprintf("_%.2f", α) : ""
datafile = "data/learning_$(conffile)_$(modelname)_$(N)$(fnα)"
if hasreprange
    writecompressedtable(datafile, df, fromrep)
else
    writecompressedtable(datafile, df)
end
