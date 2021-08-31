import numpy as np
from numba import njit
from .utils import get_state_space

class KPair:
    """
    Represents an K-pairwise model. Only works for N < 10ish.
    Variables:
        N - no. spins
        avgs - vector of expectations for each spin
        corrs - matrix of pairwise correlations
        p_k - vector of probabilities of observing K neurons fire.
        lr - learning rate
        spin_vals - the values each spin takes on, typically [0,1] or [-1,1]
        states - matrix of all possible states
        h - vector of the local magnetic fields 
        J - matrix of the pairwise couplings 
        V - vector related to population spike distribution
        Z - the current value of the partition function
    """
    def __init__(self, N, avgs, corrs, p_k, lr=0.1, spin_vals=[0,1]):
        # set user input
        self.N = N
        self.avgs = avgs
        self.corrs = corrs
        self.p_k = p_k
        self.lr = lr
        self.spin_vals = spin_vals
        # determine all states
        self.states = get_state_space(N, spin_vals, dtype=np.byte)
        # randomly initialise h and J
        self.h = np.random.random_sample((N))
        self.J = np.triu( np.random.random_sample((N,N)), 1)
        self.V = np.random.random_sample((N+1))
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
            return s@self.h + s@self.J@s + self.V[np.sum(s)] # assumes 0,1 representation
        
        return s@self.h + np.sum(s@self.J*s, axis=1) + self.V[np.sum(s,axis=1)]
            
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
    def gradient_ascent(self):
        """
        Performs gradient ascent on the log-likelihood and updates h and J
        """
        self.h, self.J, self.V, self.Z = self.fast_gradient_ascent(self.states,self.h,self.J,self.V,self.Z,self.avgs,self.corrs,self.p_k,self.lr)
        return True
        
    @staticmethod
    # @njit #seems to make little difference here
    def fast_gradient_ascent(states,h,J,V,Z,avgs,corrs,p_k,lr):
        """
        Performs gradient ascent but makes use of Numba to hopefully speed this up
        Unfortunately, Numba does not play nicely with class functions, hence the static method
        and having to rewrite a bunch of stuff. 
        """
        def H(states,h,J,V):
            return states@h + np.sum(states@J*states, axis=1) + V[np.sum(states,axis=1)]

        count = np.sum(states, axis=1) #how many neurons fire in each state
        N = states.shape[1]
        steps = 500
        for _ in range(steps): #update this condition to check accuracy
        
            # current prob of states
            p_states = np.exp( - H(states, h, J, V) ) / Z

            # work out corrections to h
            mod_avgs = states.T @ p_states #model averages
            h_new = h + lr*( mod_avgs - avgs )

            # work out corrections to J
            mod_corrs = np.triu( states.T@np.diag(p_states)@states, 1)
            J_new = J + lr*( mod_corrs - corrs )

            # work out corrections to V
            mod_p_k = np.array([ np.sum(p_states[ count == i ])  for i in range(N + 1)])
            V_new = V + lr*( mod_p_k - p_k )

            # perform the update 
            h = h_new 
            J = J_new
            V = V_new
            Z = np.sum( np.exp( -H(states, h, J, V)) ) 

        return h, J, V, Z
