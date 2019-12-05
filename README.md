# DDMLearningWithConfidence

This repository contains the scripts that implement the different models described in

Jan Drugowitsch, André G. Mendonça, Zachary F. Mainen, and Alexandre Pouget (2019). [Learning optimal decisions with confidence](https://www.pnas.org/cgi/doi/10.1073/pnas.1906787116). _Proceedings of the National Academy of Sciences_ 116(49), 24872-24880.

The scripts are licensed under the [New BSD License](LICENSE).

## Preliminaries

All scripts have been tested and run under Julia v1.0 on Ubuntu Linux v18.04.2 and MacOS Mojave. It should also work on other operating systems or later Julia versions.

Please note that this repository only contains the scripts to assess learning performance, generate the respective data files, and plot this data. It does not contain pre-computed performance data files. They need to be computed using the provided scripts before any performance data can be plotted. Generating these data files can take a considerable amount of time. They are not provided, as they take up ~67GB of space, and can be reconstructed with the provided scripts.

## Installation

All scripts are written in [Julia](https://julialang.org). To run the scripts, first [download a copy of Julia](https://julialang.org/downloads/) and install it. Then get a [copy of the scripts](https://github.com/DrugowitschLab/DDMLearningWithConfidence/archive/master.zip), and extract them to a folder of your choice.

To install the required Julia libraries, navigate to this folder, and use the following commands at the Julia command-line REPL:
```
]activate .
]instantiate
```
This should download and build the required libraries. An alternative approach to achieve the same is to run
```
julia --project=. -e "using Pkg; Pkg.instantiate()"
```
in a terminal in the folder that contains the scripts.

For further information about how to enter and use the Julia REPL, please consult the [Julia documentation](https://docs.julialang.org) (see standard library `REPL` section). The `]` symbol preceding the above commands initiates the REPL package manager mode. Please consult the Julia documentation (standard library `Pkg` section) for how to enter and leave the package manager mode.

## Usage

All of the provided commands need to be run in a terminal in the folder that contains the model scripts. The resulting learning performance data is written to the `data` folder. The plotting scripts read data from that folder, and write the plots to the `figs` folder. The `conf` folder contains the different model/simulation configurations.

### Generating the learning performance data

The main script generating learning performance data is [simLearning.jl](simLearning.jl). This script takes a few command line arguments, simulates the respective model, and writes the model performance data to a file in the `data` folder. The script is called by
```
julia --project=. simLearning.jl [configfile] [model] [N] [[α]]
```
Upon completion, it writes the collected performance data to
```
data/learning_[configfile]_[model]_[N][[_α]].h5
```

The sript takes the following arguments. `[configfile]` specifies the model/simulation configuration. The `[configfile]` argument needs to correspond to one of the files in the `conf` folder. The following configurations are provided:
```
learncorr          Learning curves with correlated inputs (Fig. 2A)
sscorr             Steady-state performance with correlated inputs (Fig. 2B)
seqdep             Configuration to evaluate sequential choice dependencies (Fig. 4A/B)
learnuncorr        Learning curves with uncorrelated inputs (not used in paper)
ssuncorr           Steady-state performance with uncorrelated inputs (no used in paper)
```

`[model]` specifies the learning model. The following models are available:
```
* probabilistic (do not require α)
adf                Assumed Density Filtering
adfdiag            Assumed Density Filtering, diagonal covariance
taylor             Using Taylor-series expansion of log-likelihood
gibbs              Gibbs sampling (use only with learn* configurations)
pf                 Particle filtering (use only with ss* configurations)
* heuristic (require α)
delta              Standard delta rule
normdelta          Normalized delta rule
lhgrad             Gradient ascent on log-likelihood
```

`[N]` specifies the number of inputs. For the paper, only `N=2` and `N=50` have been used. `[[α]]` specifies the learning rate for the heuristic models, and ignored by the probabilistic models.

As an example, to compute the steady-state performance (correlated inputs) of the delta rule for 50 inputs with learning rate 0.1, run
```
julia --project=. simLearning.jl sscorr delta 50 0.1
```
at the terminal.

For a complete dataset that replicates Fig. 2 in the paper, `simLearning.jl` needs to be run for all models, the `learncorr` and `sscorr` configurations, and `N=2` and `N=50` inputs. For heuristic models, this needs to be done for learning rates `α=0.1`,`0.3`,`0.5`,`0.7`, and `0.9`.

### Generating learning rate data

Figure 1D in the paper shows the learning rate of the ADF model. The associated data is generated by the [simLearningRate.jl](simLearningRate.jl) script that has the following arguments:
```
julia --project=. simLearningRate.jl [configfile] [model] [N] [[α]]
```
The meaning of the arguments is the same as for `simLearning.jl` (see above). The script evaluates the learning rate and writes the data to
```
data/learningrate_[configfile]_[model]_[N][[_α]].h5
```

### Plotting learning performance of individual models

To plot the learning performance of individual models, use
```
julia --project=. plotLearning.jl [configfile] [model] [N] [[α]]
```
The arguments are the same as for `simLearning.jl` (see above). The script  requires the learning rate data of the associated model (see `simLearning.jl`), and generates a summary plot of the learning performance of that model which is written to
```
figs/learning_[configfile]_[model]_[N].pdf
```

### Plotting learning curves across models

To plot a summary of learning curves across different models (as in Fig. 2A), use
```
julia --project=. plotLearningSummary.jl [inptype] [learntype]
```
where `[inptype]` is either `corr` or `uncorr` (correlated or uncorrelated inputs), `[learntype]` is either `learn` or `ss` (learning curves or steady-state performance). Either argument combination loads the associated configuration file.

The learning summary plots require pre-computed learning performance data for the associated configuration file, for all of the above models, and for `N=2`, `5`, `10`, and `50` (can be changed in the script). For heuristic models, it requires the data for learning rates `α=0.1`,`0.3`,`0.5`,`0.7`, and `0.9`.

It generates three plots similar to Fig. 2A in the paper, for three different performance measures and writes them to the `figs` folder.

### Plotting steady state performance across models

To plot a summary of steady-state performances across different models (as in Fig. 2B), use
```
julia --project=. plotSSSummary.jl [inptype]
```
where `[inptype]` is either `corr` (corresponding to configuration `sscorr`) or `uncorr` (corresponding to configuration `ssuncorr`).

As `plotLearningSummary.jl`, the steady state summary plots require pre-compted learning performance for the associated configuation files, for all of the above models, and for `N=2`, `5`, `10`, and `50` (can be changed in the script). For heuristic models, it requires the data for learning rates `α=0.1`,`0.3`,`0.5`,`0.7`, and `0.9`.

### Plot learing rate over confidence

To plot the learning rate over decision confidence (as in Fig. 1D), use
```
julia --project=. plotLearningRate.jl [configfile] [model]
```
where `[configfile]` specifies the configuration file, and `[model]` the model for which to plot the learning rate for.

The plot requires pre-computed learning rate data (using `simLearningRate.jl`; see above) for the specified configuration file and given model for `N=2`, `5`, `10`, and `50` (can be changed in the script). For heuristic models, it requires the data for learning rates `α=0.1`,`0.5`, and `0.9`.

Figure 1D was generated from learning rate data for the `sscorr` configuration and `adf` model.

### Plot sequential choice dependencies

To plot the sequential choice dependencies arising from learning (as in Fig. 4A/B), use
```
julia --project=. plotSeqDep.jl [configfile] [model] [N]
```
where `[configfile]` specifies the configuration file, `[model]` the model, and `[N]` the number of inputs.

The plot requires the pre-computed learning performance (using `simLearning.jl`; see above) for the associated configuration file, model, and input dimensionality. It does not allow specifying a learning rate, and thus does not support heuristic models.

Figure 4 used the `seqdep` configuration, the `adf` model, and `N=2`.
