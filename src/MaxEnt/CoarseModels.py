import numpy as np
from math import comb
from .utils import get_state_space

class Independent:
    """
    Represents an independent or factorised model
    Variables:
        N - no. spins
        avgs - vector of expectations for each spin
        spin_vals - the values each spin takes on, typically [0,1] or [-1,1]
        states - matrix of all possible states
        h - vector of the local magnetic field
        Z - the partition function
    """
    def __init__(self, N, avgs, spin_vals=[0,1]):
        # set user input
        self.N = N
        self.avgs = avgs
        self.spin_vals = spin_vals
        # determine all states
        self.states = get_state_space(N, spin_vals, dtype=np.byte)
        # randomly initialise h and J
        self.h = - np.log(avgs/(1-avgs))
        self.Z = np.prod(1+np.exp(-self.h))
    
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
    
    def correlations(self):
        """
        Returns a matrix of the expected values <s_i s_j> where i != j
        """
        return np.triu( self.states.T@np.diag(self.p(self.states))@self.states, 1)

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
    def __init__(self, N, p_K, spin_vals=[0,1]):
        # set user input
        self.N = N
        self.p_K = p_K
        self.spin_vals = spin_vals
        # determine all states
        self.states = get_state_space(N, spin_vals, dtype=np.byte)
        # get all combinations of N choose K
        self.combs = np.array([comb(N,K) for K in np.arange(N+1)])
    
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
    
    def correlations(self):
        """
        Returns a matrix of the expected values <s_i s_j> where i != j
        """
        return np.triu( self.states.T@np.diag(self.p(self.states))@self.states, 1)

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
    