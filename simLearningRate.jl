# Simulates long sequence of trials and stores the learning rate in each trial
#
# Call the script as
#
# simLearningRate.jl configfile model N [α]
#
# where conf/configfile.ini is the configuation file, model is the learning model,
# and N is the input dimensionality. α is the optional learning rate (only required
# for certain models). The function simulates a certain number of
# trials across many repetisions, and writes the output to
#
# data/learningrate_configfile_model_N[_α].jld

using ProgressMeter

include("common.jl")
include("models.jl")

# parse command line arguments
function parseargs()
    length(ARGS) ∈ (3, 4) || error("Expected three or four parameters")
    conffile = ARGS[1]
    modelname = ARGS[2]
    N = parse(Int64, ARGS[3])
    α = length(ARGS) > 3 ? parse(Float64, ARGS[4]) : nothing
    return conffile, modelname, N, α
end

# read configuration file
function readconf(conffile, modelname, N, α)
    conf = ConfParse("conf/$(conffile).ini")
    parse_conf!(conf)
    task = createtask(conf)
    diffstats = readdiffstats(conf, N)
    if diffstats == nothing
        error("Script only supports volatile environments with diffusion")
    end
    model = createmodel(conf, modelname, N, α)
    if isempty(methods(learningrate,
        (typeof(model), Int64, Float64, Vector{Float64}, Vector{Float64})))
        error("Model $modelname does not support computnig the learning rate")
    end
    inputs = createinputs(conf, task, N)
    burnin = parse(Int64, retrieve(conf, "simlearningrate", "burnin"))
    trials = parse(Int64, retrieve(conf, "simlearningrate", "trials"))
    return task, diffstats, model, inputs, burnin, trials
end

# collect data across multiple repetitions
function runsim(task, diffstats, model, inputs, burnin, trials, verbose::Bool=false)
    σμ = getσμ(task)
    # resample w and get μ-independent input statistics
    samplew!(inputs)
    μxprime = getμxprime(inputs)
    trueI = dot(μxprime, getΣx(inputs) \ μxprime)
    # reset/initialize model
    reset!(model, wmoments(inputs)..., getw(inputs))

    # run 'burnin' + 'trials' trials and collect response statistics
    df = DataFrame(trial=Int64[], μ=Float64[], choice=Int64[], t=Float64[],
        EPC=Float64[], EDT=Float64[], Eperf=Float64[],
        trueI=Float64[], modelI=Float64[], angerr=Float64[],
        conf=Float64[], lr=Float64[], wdiff=Float64[])
    if verbose
        p = Progress(trials + burnin, 1, "Simulating...")
    end
    for trial = 1:(burnin + trials)
        # draw latent state, simulate model choice, and provide model feedback
        μ = drawμ(task)
        choice, t, x = simdiffusion(getw(model), getμx(inputs, μ), getΣx(inputs), task.θ)
        corrchoice = μ > 0 ? 1 : -1
        if trial > burnin
            conf = confidence(model, corrchoice, t, x, getw(inputs))
            α = learningrate(model, corrchoice, t, x, getw(inputs))
            wold = copy(getw(model))
        end
        w = feedback!(model, corrchoice, t, x, getw(inputs))

        # collect statistics
        if trial > burnin
            wdiff = norm(w - wold)
            EPC, EDT = avgperf(inputs, task, w)
            Eperf = taskperf(task, EPC, EDT)
            angerr = getangerr(inputs, w)
            modelI = getIloss(inputs, w)
            push!(df, [trial, μ, choice, t, EPC, EDT, Eperf,
                       trueI, modelI, angerr, conf, α, wdiff])
        end

        # diffuse model and state
        diffuse!(inputs, diffstats...)
        diffuse!(model)

        # update progress meter
        !verbose || next!(p)
    end
    return df
end

# main
conffile, modelname, N, α = parseargs()
task, diffstats, model, inputs, burnin, trials = readconf(conffile, modelname, N, α)
print("Running simulations for conf $(conffile), model $(modelname), N=$(N)")
!requiresα(modelname) || print(", α=$(@sprintf("%.2f", α))")
println()
#df = runsim(task, model, inputs, trials, 2, true)
#@profile df = runsim(task, diffstats, model, inputs, trials, repetitions, true)
#Profile.print()
df = runsim(task, diffstats, model, inputs, burnin, trials, true)
if requiresα(modelname)
    datafile = "data/learningrate_$(conffile)_$(modelname)_$(N)_$(@sprintf("%.2f", α))"
else
    datafile = "data/learningrate_$(conffile)_$(modelname)_$(N)"
end
println("Writing data to $datafile")
writecompressedtable(datafile, df)
