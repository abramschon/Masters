import numpy as np
import matplotlib.pyplot as plt
import time

def main():
    N = 5
    avgs = 0.5*np.ones(N) # prob of every neuron firing in a window is 0.5
    corrs = 0.2*np.triu(np.ones((N,N)),1) # prob of 2 neurons firing in the same window is 0.2 
    print("Init model")
    ising = Ising(N, avgs, corrs, lr=0.5) 
    print("Starting grad ascent")
    ising.gradient_ascent() # 100 steps 
    print("Stop grad ascent")
    pred_avgs = ising.averages()
    pred_corrs = ising.correlations()
    print("Predicted averages:", pred_avgs, "Predicted correlations:", pred_corrs,sep="\n")
    print(f"P({ising.states[0]})={ising.p(ising.states[0])}")

    # Calculate average times
    reps = 10
    Ns = np.arange(1,13)
    times = np.zeros( (reps,len(Ns)) )
    for i in range(reps):
        for N in Ns:
            print(N)
            avgs = 0.5*np.ones(N) # prob of every neuron firing in a window is 0.5
            corrs = 0.2*np.triu(np.ones((N,N)),1) # prob of 2 neurons firing in the same window is 0.2 
            ising = Ising(N, avgs, corrs, lr=0.5) 
            start = time.time()
            ising.gradient_ascent() # 100 steps 
            stop = time.time()
            times[i,N-1]=stop-start
    
    av_times = np.mean(times,0)
    std_times = np.std(times,0)

    plt.plot(Ns, av_times, "k.")
    plt.plot(Ns, av_times+2*std_times/np.sqrt(reps), "r_")
    plt.plot(Ns, av_times-2*std_times/np.sqrt(reps), "r_")
    plt.semilogy()
    plt.title("Time for 100 steps of grad. ascent vs. system size")
    plt.xlabel("System size")
    plt.ylabel("Log Time (seconds)")
    plt.show()

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
    
    # Methods for calculating probabilities and expectations over entire distribution
    def expectation(self, f):
        """
        Returns the sum over all states of the function f, weighted by the probability distirbution produced by the Ising model. 
        Args:
            f - a function of all the states
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
            return -s@self.h - s@self.J@s 
        
        return -s@self.h - np.sum(s@self.J*s, axis=1)
            
    def calc_Z(self):
        """ 
        Calculates the partition function Z based on the current h and J.
        """
        # (the lambda function just returns 1 since this is just a sum of p over all states) 
        Z = np.sum( self.p_unnormalized(self.states) )
        return Z 
    
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
        steps = 100
        for _ in range(steps): #update this condition to check accuracy
            # work out corrections to h
            h_new = self.h + self.lr*(self.avgs-self.averages())

            # work out corrections to J
            J_new = self.J + self.lr*(self.corrs-self.correlations())

            # perform the update
            self.h = h_new
            self.J = J_new
            self.Z = self.calc_Z()
    
if __name__ == "__main__":
    main()
