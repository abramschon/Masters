# Ising Model 

We create an Ising ('Ee-zing') model that models the joint probability distribution over a set of binary neurons. The Ising model is the maximum entropy distribution that reproduces the *mean probability of neurons spiking*, and the *pairwise correlations*. 

The challenge is determining the variables `h` (a vector), `J` (a matrix) such that the Ising model produces statistics consistent with the observed statistics.  

We first determine these variables analytically using Mathematica for a small number of neurons. It should become apparent that analytical solutions are computationally unfeasible for large numbers of neurons. 

We then determine the variables computationally.

## Wolfram Language in Jupyter
We use this [package](https://github.com/WolframResearch/WolframLanguageForJupyter) to allow us to use the Wolfram Language in Jupyter. In short:
- Clone the repo

    `git clone https://github.com/WolframResearch/WolframLanguageForJupyter.git`
- Navigate into the repo and run:
    `./configure-jupyter.wls add`

To get help with the Wolfram Language, see the [documentation](https://reference.wolfram.com/language/).
