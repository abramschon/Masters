import numpy as np
from numba import njit

class Ising:
    """
    Represents an Ising model.
    Variables:
        N - no. spins
        avgs - vector of expectations for each spin
        corrs - matrix of pairwise correlations
        lr - learning rate
        spin_vals - the values each spin takes on, typically [0,1] or [-1,1]
        states - matrix of all possible states
        h - vector of the local magnetic fields 
        J - matrix of the pairwise couplings 
        Z - the current value of the partition function
    """
    def __init__(self, N, avgs, corrs, lr=0.1, spin_vals=[0,1]):
        # set user input
        self.N = N
        self.avgs = avgs
        self.corrs = corrs
        self.lr = lr
        self.spin_vals = spin_vals
        # determine all states
        self.states = np.array([self.to_binary(n) for n in range(2**N)]) 
        # randomly initialise h and J
        self.h = np.random.random_sample((N))
        self.J = np.triu( np.random.random_sample((N,N)), 1)
        # work out the partition function Z
        self.Z = self.calc_Z()
    
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

    def p_unnormalized(self, s):
        """
        Returns the unnormalized probability (not divided Z) of the state s given the model parameters 
        Args:
            s - np.array of the state
        """
        return np.exp(-self.H(s))

    def H(self, s):
        """
        Return the hamiltonian H(s) of the state s if s.ndim == 1, 
        or the hamiltonian over the states s if s.ndim == 2
        Args:
            s - np.array of the state/states
        """
        if s.ndim==1:
            return s@self.h + s@self.J@s 
        
        return s@self.h + np.sum(s@self.J*s, axis=1)
            
    def calc_Z(self):
        """ 
        Calculates the partition function Z based on the current h and J.
        """
        return np.sum( self.p_unnormalized(self.states) )
    
    def to_binary(self, n):
        """
        Returns a binary rep of the int n as an array of size N, e.g. Assuming N = 5, 3 -> np.array([0,0,0,1,1]) 
        Not particularly efficient, but since it is only used once at the start, this is alright
        """
        b = np.zeros(self.N)
        for i in range(self.N):
            if n % 2 == 1: b[self.N-1-i]=1 # index N-1-i otherwise numbers are reversed
            n//=2
            if n==0: break
        return b

    # Methods for gradient ascent
    def gradient_ascent(self):
        """
        Performs gradient ascent on the log-likelihood and updates h and J
        """
        self.h, self.J, self.Z = self.fast_gradient_ascent(self.states,self.h,self.J,self.Z,self.avgs,self.corrs,self.lr)
        return True
        
    @staticmethod
    @njit
    def fast_gradient_ascent(states,h,J,Z,avgs,corrs,lr):
        """
        Performs gradient ascent but makes use of Numba to hopefully speed this up
        Unfortunately, Numba does not play nicely with class functions, hence the static method
        and having to rewrite a bunch of stuff. 
        """
        def H(states,h,J):
            return states@h + np.sum(states@J*states, axis=1)

        steps = 500
        for _ in range(steps): #update this condition to check accuracy
        
            # current prob of states
            p_states = np.exp( - H(states, h, J) ) / Z

            # work out corrections to h
            mod_avgs = states.T @ p_states #model averages
            h_new = h + lr*( mod_avgs - avgs )

            # work out corrections to J
            mod_corrs = np.triu( states.T@np.diag(p_states)@states, 1)
            J_new = J + lr*( mod_corrs - corrs )

            # perform the update 
            h = h_new 
            J = J_new
            Z = np.sum( np.exp( -H(states, h, J)) ) 

        return h, J, Z