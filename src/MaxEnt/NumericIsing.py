import numpy as np
from numba import njit
from .utils import get_state_space

class Ising:
    """
    Represents an Ising model. Only works for N < 10ish.
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
        self.states = get_state_space(N, spin_vals, dtype=np.byte)
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
    
    def pert_init(self):
        """
        Initialise weights based on estimates from the perturbative results
        Div by 0 issue if any average is 0
        """
        self.h = np.log( (1/self.avgs) - 1)
        prod_avgs = np.outer(self.avgs,self.avgs)
        self.J = -np.log( (self.corrs / prod_avgs) + np.tril( np.ones((self.N,self.N)))  ) 
        return True

    # Methods for gradient ascent
    def gradient_ascent(self, max_it = 500):
        """
        Performs gradient ascent on the log-likelihood and updates h and J
        """
        for _ in range(max_it):
            self.update_params()
        return True
        
    def update_params(self):
        """
        Works out the gradients and updates the parameters
        """
        # current prob of states
        p_states = self.p(self.states)

        # work out corrections to h
        mod_avgs = self.states.T @ p_states #model averages
        h_new = self.h + self.lr*( mod_avgs - self.avgs )

        # work out corrections to J
        mod_corrs = np.triu( self.states.T@np.diag(p_states)@self.states, 1)
        J_new = self.J + self.lr*( mod_corrs - self.corrs )

        # perform the update 
        self.h = h_new 
        self.J = J_new
        self.Z = self.calc_Z()

        return True
