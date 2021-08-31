import numpy as np
from math import comb
from numba import njit
from .utils import get_state_space
from .Samplers import gibbs_sampling

class Independent:
    """
    Represents an independent or factorised model
    If sampling is True, all estimates will be based off of samples and the state space will not be set up
    Variables:
        N - no. spins
        avgs - vector of expectations for each spin
        spin_vals - the values each spin takes on, typically [0,1] or [-1,1]
        states - matrix of all possible states
        h - vector of the local magnetic field
        Z - the partition function
    """
    def __init__(self, N, avgs, spin_vals=[0,1], sampling=False):
        # set user input
        self.sampling = sampling 
        self.N = N
        self.avgs = avgs
        self.spin_vals = spin_vals
        # initialise h and Z
        self.h = - np.log(avgs/(1-avgs))
        self.Z = np.prod(1+np.exp(-self.h))
        #possibly init state space 
        self.states = None
        if not sampling: # determine all states
            self.states = get_state_space(N, spin_vals, dtype=np.byte)
        
    
    # Methods for calculating probabilities and expectations over entire distribution
    def expectation(self, f):
        """
        Returns the sum over all states of the function f, weighted by the probability distirbution produced by the Ising model. 
        Args:
            f - a function of all the states, must return either a column vector (2^N x 1) or a matrix (2^N x N)
        """
        return f(self.states).T @ self.p(self.states) 

    def averages(self):
        """
        Returns a vector of the expected values <s_i>
        """
        return self.expectation(lambda states: states)

    def p(self, s):
        """
        Returns the normalized probability of the state s given the model parameters 
        Args:
            s - np.array of the state, e.g. np.array([0,0,1]), here the third neuron fires
        """
        return np.exp(-self.H(s)) / self.Z

    def H(self, s):
        """
        Return the hamiltonian H(s) of the state s 
        or the hamiltonian over the states s 
        Args:
            s - np.array of the state/states
        """
        return s@self.h 
    
    def get_samples(self, M=100000, chains = 4):
        h = self.h
        @njit
        def p(s):
            return np.exp(-h.dot(s))
        init_states = np.array([np.random.binomial(1,self.avgs) for c in range(chains)])
        return gibbs_sampling(p,init_states,M,avg_every=self.N,burn_in=0)
        

class PopCount:
    """
    Represents the population count model (reproduces the probability that K neurons fire)
    Variables:
        N - no. spins
        p_K - numpy array of probabilities that K neurons fire
        spin_vals - the values each spin takes on, typically [0,1] or [-1,1]
        states - matrix of all possible states
        h - vector of the local magnetic field
        Z - the partition function
    """
    def __init__(self, N, p_K, spin_vals=[0,1], sampling=False):
        # set user input
        self.N = N
        self.p_K = p_K
        self.spin_vals = spin_vals
        # get all combinations of N choose K
        self.combs = np.array([comb(N,K) for K in np.arange(N+1)])
        #possibly init state space 
        self.states = None
        if not sampling: # determine all states
            self.states = get_state_space(N, spin_vals, dtype=np.byte)
    
    # Methods for calculating probabilities and expectations over entire distribution
    def expectation(self, f):
        """
        Returns the sum over all states of the function f, weighted by the probability distirbution produced by the Ising model. 
        Args:
            f - a function of all the states, must return either a column vector (2^N x 1) or a matrix (2^N x N)
        """
        return f(self.states).T @ self.p(self.states) 

    def prob_k(self):
            """
            Returns a vector of the probability of observing K neurons fire
            """
            count = np.sum(self.states, axis=1) #how many neurons fire in each states
            p = self.p(self.states) #probability of each state
            return np.array([ np.sum(p[ count == i ])  for i in range(self.N + 1)])

    def p(self, s):
        """
        Returns the normalized probability of the state s given the model parameters 
        Args:
            s - np.array of the state, e.g. np.array([0,0,1]), here the third neuron fires
        """
        if s.ndim == 2:  #assumes 0, 1 notation
            K = np.sum(s,axis=1)
        elif s.ndim == 1:
            K = np.sum(s) 
        else:
            return -1
            
        return self.p_K[K] / self.combs[K]

    def get_samples(self, M=100000, chains = 4):
        p_K = np.copy(self.p_K)
        combs = np.copy(self.combs)
        @njit
        def p(s,p_K=p_K):
            K = int(np.sum(s))
            return p_K[K] / combs[K] 
        init_states = np.array([np.random.binomial(1,0.5*np.ones(self.N)) for c in range(chains)])
        return gibbs_sampling(p,init_states,M,avg_every=self.N,burn_in=self.N*10)
    