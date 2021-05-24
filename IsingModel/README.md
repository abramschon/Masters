# Ising Model 

We create an Ising ('Ee-zing') model that models the joint probability distribution over a set of binary neurons. The Ising model is the maximum entropy distribution that reproduces the mean probability of neurons spiking $\langle \sigma_i \rangle$, and the pairwise correlations $\langle \sigma_i\sigma_j \rangle$. 

The challenge is determining the variables $h$, $J$ such that the Ising model produces statistics consistent with the observed statistics.  

We first determine these variables analytically using Mathematica for a small number of neurons. It should become apparent that analytical solutions are computationally unfeasible for large numbers of neurons. 

We then determine the variables computationally.