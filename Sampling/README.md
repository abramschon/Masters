# Importance Sampling

We need to determine certain expectations. When the number of states in the distribution is exponential in the system size, as is the case with $N$ binary variables, determining these expectations involves performing summations over exponentially many states which can be unfeasible for large systems. This is what motivates importance sampling - drawing a representative sample from the distribution over which we can use to estimate expectations. 

Following the OG Clever Machine [blogposts](https://theclevermachine.wordpress.com/2012/09/24/a-brief-introduction-to-markov-chains/) on sampling methods, we implement a version of Gibbs sampling.

In Gibbs sampling for a multivariate distribution $p(\sigma_1,...,\sigma_N)$, we need analytic expressions for the conditional probabilities of each variable $p(\sigma_i|,\sigma_j, j\neq i)$. In the case of binary variables described by the un-normalised likelihood function $p^*$, using Bayes theorem we have:
$$
 p(\sigma_i|,\sigma_j, j\neq i) = \frac{p^*(\bm{\sigma})}{\sum_{\sigma_i}p^*(\bm{\sigma})}
$$
where we marginalise over the variable $\sigma_i$ in the denominator. We then perform component wise sampling in the same vein as the Metropolis-Hastings algorithm. In pseudo-code:

    sample initial state s=[s[1],...,s[D]] from the prior dist.
    for t in [1,M]:
        for each dimension i:
            draw s[i] from p(s[i]| s[1],..., s[i-1],s[i+1],...,s[D])
    

Look into python [Numba](https://numba.readthedocs.io/en/stable/user/5minguide.html) to speed up code. 