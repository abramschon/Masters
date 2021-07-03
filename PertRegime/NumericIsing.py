import numpy as np

def main():
    N = 3
    avgs = 0.5*np.ones(N) # prob of every neuron firing in a window is 0.5
    corrs = 0.2*np.triu(np.ones((N,N)),1) # prob of 2 neurons firing in the same window is 0.2 
    ising = Ising(N, avgs, corrs, lr=0.5) 
    ising.gradient_ascent() # 100 steps 
    pred_avgs = ising.averages()
    pred_corrs = ising.correlations()
    print("Predicted averages:", pred_avgs, "Predicted correlations:", pred_corrs,sep="\n")
    

class Ising:
    """
    Represents an Ising model.
    Variables:
        N - no. spins
        av_s - vector of expectations for each spin
        av_ss - matrix of pairwise correlations
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
        Returns a vector of the predicted expected values
        """
        averages = np.zeros(self.N)
        for i in range(self.N):
            averages[i] = self.expectation(lambda s: s[0], [i], self.p)
        return averages
    
    def correlations(self):
        """
        Returns a matrix of the predicted correlations values.
        We also predict the auto-correlations, though this model is not trained on them.
        """
        correlations = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(i,self.N):
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
        return -self.h.dot(s) - s@self.J@s 
            
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

    # Methods for gradient ascent
    def gradient_ascent(self):
        """
        Performs gradient ascent on the log-likelihood and updates h and J
        """
        steps = 100
        for _ in range(steps): #update this condition to check accuracy
            # work out corrections to h
            h_new = self.h
            for i in range(self.N):
                dLdhi = self.avgs[i] - self.expectation( lambda s: s[0], [i], self.p )
                h_new[i] += self.lr * dLdhi 

            # work out corrections to J
            J_new = self.J
            for i in range(self.N-1):
                for j in range(i+1,self.N):
                    dLdJij = self.corrs[i,j] - self.expectation( lambda s: s[0]*s[1], [i,j], self.p )
                    J_new[i,j] += self.lr * dLdJij 

            # perform the update
            self.h = h_new
            self.J = J_new
            self.Z = self.calc_Z()
    
if __name__ == "__main__":
    main()
