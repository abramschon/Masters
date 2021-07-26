import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange

def main():
    av_time_grad_ascent()

def test_samples():
    N = 10
    avgs = 0.5*np.ones(N) # prob of every neuron firing in a window is 0.5
    corrs = 0.2*np.triu(np.ones((N,N)),1) # prob of 2 neurons firing in the same window is 0.2 
    
    ising = Ising(N, avgs, corrs, lr=0.5) 
    
    M = 500000
    chains = 10
    start = time.time()
    samples = ising.gibbs_sampling(M,chains)
    stop = time.time()
    print(f"Time to generate {chains}x{M} samples", stop-start)

    samples = samples.reshape(-1,N) #collapse different trials 

    means_true = ising.averages()
    means_sample = np.mean(samples, axis=0)
    
    dp = 5 #decimal places
    print("True means", np.round(means_true,dp), "Sample means", np.round(means_sample,dp), sep="\n")
    print("Difference in means", means_true - means_sample, sep="\n")

    corrs_true = ising.correlations()
    corrs_sample = np.triu((samples.T@samples) / samples.shape[0],k=1)
    print("True correlations", np.round(corrs_true,dp), "Sample correlations", np.round(corrs_sample,dp), sep="\n")
    print("Difference in correlations", corrs_true - corrs_sample, sep="\n")

def fit_example():
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

def av_time_grad_ascent():
    # Calculate average times
    reps = 50
    Ns = np.arange(1,11)
    times = np.zeros( (reps,len(Ns)) )
    for i in range(reps):
        if not (i+1)%10:
            print("Repetitions: ", i+1)
        for N in Ns:
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
    plt.title("Time for 100 steps of grad. ascent vs. system size")
    plt.xlabel("System size")
    plt.ylabel("Time (seconds)")
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
    def __init__(self, N, avgs, corrs, lr=0.1, spin_vals=np.array([0,1])):
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
            return states.dot(h) + np.sum(states.dot(J)*states, axis=1)

        steps = 500
        for _ in range(steps): #update this condition to check accuracy
        
            # current prob of states
            p_states = np.exp( - H(states, h, J) ) / Z

            # work out corrections to h
            mod_avgs = states.T.dot(p_states) #model averages
            h_new = h + lr*( mod_avgs - avgs )

            # work out corrections to J
            mod_corrs = np.triu( states.T.dot(np.diag(p_states)).dot(states), 1)
            J_new = J + lr*( mod_corrs - corrs )

            # perform the update 
            h = h_new 
            J = J_new
            Z = np.sum( np.exp( -H(states, h, J)) ) 

        return h, J, Z

    def gibbs_sampling(self,M,chains=1):
        p_ind = self.avgs*(self.spin_vals[1]-self.spin_vals[0]) + self.spin_vals[0]
        samples = np.zeros((chains, M, self.N))
        for c in range(chains):
            samples[c,0,:] = np.random.binomial(1,p_ind)*(self.spin_vals[1]-self.spin_vals[0]) + self.spin_vals[0] #set the initial state 

        return self.fast_gibbs_sampling(self.h, self.J, self.spin_vals, samples)

    @staticmethod
    @njit #not sure how much of a difference parallel makes here
    def fast_gibbs_sampling(h, J, spin_vals, samples):
        """
        Given a probability distribution p, returns M samples using gibbs sampling
        Burn-in is not included, and we add states everytime a dimension is updated, hence states are very correlated.
        """
        for c in prange(samples.shape[0]):
            for t in range(1,samples.shape[1]): 
                samples[c,t] = samples[c,t-1] #copy previous state
                i = t % samples.shape[2] #which dimension to work on
                
                state_off = np.copy(samples[c,t])
                state_on = np.copy(samples[c,t])

                state_off[i] =  spin_vals[0] #state with neuron i set to off
                state_on[i] = spin_vals[1] #state with neuron i
                
                p_on = np.exp( - state_on.dot(h) - state_on.dot(J).dot(state_on) )
                p_on = p_on / (p_on + np.exp( - state_off.dot(h) - state_off.dot(J).dot(state_off) ) ) #calc cond prob that spin i is on given other spin vals
                
                if np.random.binomial(1,p_on): #draw number from unif distribution to determine whether we update i
                    samples[c,t]=state_on
                    continue
                samples[c,t]=state_off
        
        return samples

if __name__ == "__main__":
    main()