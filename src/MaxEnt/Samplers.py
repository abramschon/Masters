import numpy as np 
from numba import njit, prange

def gibbs_sampling(p,init_states,M,avg_every=1,burn_in=0):
    """
    Returns M x chains samples based on the given unnormalised probability mass function p
    Assumes 0,1 notation
    Args:
        p - unnormalized probability of state s
        init_states - CxN matrix of initial states for each Markov chain
        M - number of samples per chain
        chains - number of chains
        avg_every - how many transitions before we take the next sample
        burn_in - how many samples we initially discard
    """
    if init_states.ndim == 1:
        chains = 1
        N = init_states.shape[0]
    else: #assumes ndim == 2
        chains = init_states.shape[0]
        N = init_states.shape[1]
    
    its = M*avg_every+burn_in # number of iterations per chain
    samples = np.zeros((chains,its,N))
    for c in range(chains):
        samples[c,0,:] = init_states[c] #set the initial state 

    samples = fast_gibbs_sampling(p,N,its,chains,samples)
    samples = samples[:,burn_in:] # discard burn in
    samples = samples.reshape(avg_every,-1,N)[0] # take the first sample of every avg_every samples
    return samples

@njit(parallel=True) #not sure how much of a difference parallel makes here
def fast_gibbs_sampling(p,N,its,chains,samples):
    """
    Runs gibbs sampling and fills out samples
    Burn-in is not included, and we add states each time a dimension is updated, hence states are very correlated.
    """
    for c in prange(chains):
        for t in range(1,its): 
            samples[c,t] = samples[c,t-1] #copy previous state
            i = t % N #which dimension to work on

            state_off = np.copy(samples[c,t])
            state_on = np.copy(samples[c,t])

            state_off[i] =  0 #state with neuron i set to off
            state_on[i] = 1 #state with neuron i set to on

            p_off = p(state_off)
            p_cond_off = p_off / (p_off + p(state_on) ) #calc cond prob that spin i is on given other spin vals

            if np.random.binomial(1,p_cond_off): #draw number from unif distribution to determine whether we update i
                samples[c,t]= state_off
                continue
            samples[c,t]=state_on

    return samples