import numpy as np

def main():
    N = 4
    h = np.random.random_sample((N))
    J = np.triu( np.random.random_sample((N,N)), 1)
    K = np.zeros((N,N,N))

    alpha = 1
    for i in range(N-2):
        for j in range(i+1,N-1):
            for k in range(j+1, N):
                K[i,j,k] = alpha 
                alpha /= 10
    ex = ThreeWise(N, h, J, K)
    
    #calc 3 wise correlations
    for i in range(N-2):
        for j in range(i+1,N-1):
            for k in range(j+1, N):
                print(f"Correlation between {i},{j},{k}\n", ex.expectation(lambda s: s[:,i]*s[:,j]*s[:,k]) )

class ThreeWise:
    """
    Represents a Boltzman distribution with 3-wise interactions.
    Variables:
        N - no. spins
        h - vector of the local magnetic fields 
        J - matrix of the pairwise interactions 
        K - tensor of the 3-wise interactions
        spin_vals - the values each spin takes on, typically [0,1] or [-1,1]
        states - matrix of all possible states
        Z - the current value of the partition function
    """
    def __init__(self, N, h, J, K, spin_vals=[0,1]):
        # set user input
        self.N = N
        self.h = h
        self.J = J
        self.K = K
        self.spin_vals = spin_vals
        # determine all states
        self.states = np.array([self.to_binary(n) for n in range(2**N)]) 
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
            return -self.h@s - self.J@s@s - self.K@s@s@s

        return -s@self.h - np.sum(s@self.J*s, axis=1) - np.einsum("ij,ik,il,jkl->i", s,s,s,self.K)
            
    def calc_Z(self):
        """ 
        Calculates the partition function Z based on the current h and J.
        """
        return np.sum( self.p_unnormalized(self.states) )
    
    def to_binary(self, n):
        """
        Returns a binary rep of the int n as an array of size N, e.g. Assuming N = 5, 3 -> np.array([0,0,0,1,1]) 
        """
        b = np.zeros(self.N)
        for i in range(self.N):
            if n % 2 == 1: b[self.N-1-i]=1 # index N-1-i otherwise numbers are reversed
            n//=2
            if n==0: break
        return b
    
if __name__ == "__main__":
    main()
