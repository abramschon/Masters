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
                print(f"Correlation between {i},{j},{k}\n",ex.expectation(lambda s: s[0]*s[1]*s[2], [i,j,k], ex.p))

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
    
    # Methods for calculating probabilities and expectations
    def expectation(self, f, ind, p):
        """
        Returns the sum over all states of the function f, weighted by p. 
        Args:
            f - a function of a subset of the spins
            ind - indices of the spins which are involved in f
            p - a function of the state, for instance the probability p of observing the state
        """
        exp = 0
        for s in self.states:
            exp += f( [s[i] for i in ind] ) * p(s)
        
        return exp

    def averages(self):
        """
        Returns a vector of the expected values <s_i>
        """
        averages = np.zeros(self.N)
        for i in range(self.N):
            averages[i] = self.expectation(lambda s: s[0], [i], self.p)
        return averages
    
    def correlations(self):
        """
        Returns a matrix of the expected values <s_i s_j> where i != j
        """
        correlations = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(i+1,self.N):
                correlations[i,j] = self.expectation(lambda s: s[0]*s[1], [i,j], self.p)
        return correlations

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
        Return the hamiltonian H(s) of the state s
        Args:
            s - np.array of the state
        """
        return -self.h@s - self.J@s@s - self.K@s@s@s
            
    def calc_Z(self):
        """ 
        Calculates the partition function Z based on the current h and J.
        """
        # (the lambda function just returns 1 since this is just a sum of p over all states) 
        Z = self.expectation(lambda args: 1, [], self.p_unnormalized) 
        return Z 
    
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
