import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange

def main():
    N=50
    
    avgs = 0.5*np.ones(N) # prob of every neuron firing in a window is 0.5
    corrs = 0.2*np.triu(np.ones((N,N)),1) # prob of 2 neurons firing in the same window is 0.2 
    print("Init model")
    ising = NumIsing(N, avgs, corrs, lr=0.5) 
    
    print("State space ", 2**N)
    N_samples=5000; chains=4; N_sets=40; updates_per_set=100; M = N_samples*chains
    print(f"{N_sets} sets of {M} samples will be generated, for a total of {N_sets*M} states")
    
    print("Starting gradient ascent with sampling")
    start = time.time()
    ising.num_gradient_ascent(N_samples, chains, N_sets, updates_per_set) 
    stop = time.time()
    print("Stop grad ascent, time: ",stop-start,"s")
    
    pred_avgs = ising.averages()
    print("Predicted averages:", np.round(pred_avgs,4))
    # pred_corrs = ising.correlations()
    # print("Predicted correlations:", np.round(pred_corrs,4))


def test_correlations():
    N = 5
    avgs = 0.5*np.ones(N) # prob of every neuron firing in a window is 0.5
    corrs = 0.2*np.triu(np.ones((N,N)),1) # prob of 2 neurons firing in the same window is 0.2 
    
    print("Init model")
    ising = NumIsing(N, avgs, corrs, lr=0.5,analytic=True) 
    
    print("Starting grad ascent")
    ising.num_gradient_ascent() #
    print("Stop grad ascent")
    
    pred_avgs = ising.mod_avgs
    pred_corrs = ising.mod_corrs
    print("Predicted averages:", np.round(pred_avgs,4), "Predicted correlations:", np.round(pred_corrs,4),sep="\n")

    #test averages
    print("Default", ising.correlations())
    print("Analytic", ising.correlations(analytic=True))
    print("Compute",ising.correlations(compute=True))
    ising.save_samples()
    print("After more samples",ising.correlations(compute=True))

def test_samples(): #fix this
    N = 10
    print("Init model")
    ising = NumIsing(N, 0.5*np.ones(N), 0.2*np.triu(np.ones((N,N)),1), lr=0.5,analytic=True) 

    M = 500000
    chains = 10
    start = time.time()
    samples = ising.gibbs_sampling(M,chains)
    stop = time.time()
    print(f"Time to generate {chains}x{M} samples", stop-start)

    means_true = ising.averages(analytic=True)
    means_sample = np.mean(samples, axis=0)
    
    dp = 5 #decimal places
    print("True means", np.round(means_true,dp), "Sample means", np.round(means_sample,dp), sep="\n")
    print("Difference in means", means_true - means_sample, sep="\n")

    corrs_true = ising.correlations(analytic=True)
    corrs_sample = np.triu((samples.T@samples) / samples.shape[0],k=1)
    print("True correlations", np.round(corrs_true,dp), "Sample correlations", np.round(corrs_sample,dp), sep="\n")
    print("Difference in correlations", corrs_true - corrs_sample, sep="\n")


class NumIsing:
    """
    Represents an Ising model for large number of neurons where sampling is necesssary
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
    def __init__(self, N, avgs, corrs, lr=0.1, spin_vals=np.array([0,1]), analytic=False):
        # set user input
        self.N = N
        self.avgs = avgs
        self.corrs = corrs
        self.lr = lr
        self.spin_vals = spin_vals
        
        # randomly initialise h and J
        self.h = np.random.random_sample((N))
        self.J = np.triu( np.random.random_sample((N,N)), 1)
        
        self.Z = -1 # arbitrarily set Z to -1, we can determine this via sampling if necessary

        # to later save estimated model avgs, corrs and samples from gradient ascent
        self.mod_avgs = None
        self.mod_corrs = None
        self.samples = None

        # optionally, set up state space to compute quantities analytically
        self.states = None
        if analytic:
            self.set_state_space()

            
    # Calculating probabilities and expectations over entire distribution analytically
    def to_binary(self, n):
        """
        Returns a binary rep of the int n as an array of size N, e.g. Assuming N = 5, 3 -> np.array([0,0,0,1,1]) 
        Not particularly efficient, but since it is only used once at the start for small N, this is okay
        """
        b = np.zeros(self.N)
        for i in range(self.N):
            if n % 2 == 1: b[self.N-1-i]=1 # index N-1-i otherwise numbers are reversed
            n//=2
            if n==0: break
        return b

    def set_state_space(self):
        """
        Sets up the state space, but only if analytic expectations are needed
        """
        self.states = np.array([self.to_binary(n) for n in range(2**self.N)]) 
        return True

    @staticmethod
    @njit
    def H(s, h, J):
        """
        Return the hamiltonian H(s) of the state s if s.ndim == 1, otherwise returns the hamiltonian over the states s 
        """
        if s.ndim==1:
            return s.dot(h) + s.dot(J).dot(s) 
        return s.dot(h) + np.sum(s.dot(J)*s, axis=1)

    @staticmethod
    @njit
    def p_unnormalized(s, h, J):
        """
        Returns the unnormalized probability (not divided by Z) of the state/states s 
        """
        if s.ndim==1:
            return np.exp( - s.dot(h) - s.dot(J).dot(s) )
        return np.exp(-s.dot(h) - np.sum(s.dot(J)*s, axis=1))

    def check_analytic(self):
        """
        Checks whether user really wants analytic expressions and if so, sets up neccessary 2^N quantites
        If all goes well, returns True
        """
        if type(self.states) is not np.ndarray: #check if state space has been initialised
            proceed = input("Warning: model was not initialised for analytic expectations.\nProcced? (y/n)")
            if proceed != "y":
                return False
            self.set_state_space()
        if self.Z == -1: #check if partition function is up to date
            self.Z = np.sum( self.p_unnormalized(self.states, self.h, self.J) )
        return True

    def averages(self, analytic=False, compute=False):
        """
        Returns a vector of the expected values <s_i>
        By default, returns the average computed during model fitting.
        However, you can also elect to explicitly compute the averages based on the currently saved samples or analytically compute the averages
        If there are currently no saved samples, returns False
        """
        if analytic:
            if self.check_analytic():
                return self.states.T.dot(self.p_unnormalized(self.states, self.h, self.J)) / self.Z
            return -1 #something went wrong

        # otherwise, return sample expectations
        if type(self.samples) is not np.ndarray: #check whether there are currently samples 
            print("No samples currently saved, run:\nsave_samples()")
            return False
        if compute: #explicitly compute averages based on current samples
            self.mod_avgs = np.mean(self.samples,axis=0)
            return self.mod_avgs  
        if type(self.mod_avgs) is np.ndarray: #return saved samples
            return self.mod_avgs 
        print("No averages currenly saved, rerun setting compute=True")
        return False
    
    def correlations(self, analytic=False, compute=False):
        """
        Returns a matrix of the expected values <s_i s_j> where i != j
        By default, returns the correlations computed during model fitting.
        However, you can also elect to explicitly compute the correlations based on the currently saved samples or analytically compute the correlations.
        If there are currently no saved samples, returns False
        """
        if analytic:
            if self.check_analytic():
                return  np.triu( self.states.T@np.diag(self.p_unnormalized(self.states, self.h, self.J))@self.states, 1) / self.Z 
            return -1 #something went wrong

        # otherwise, return sample expectations
        if type(self.samples) is not np.ndarray: #check whether there are currently samples 
            print("No samples currently saved, run:\nsave_samples()")
            return False
        if compute: #explicitly compute averages based on current samples
            self.mod_corrs = np.triu((self.samples.T@self.samples) / self.samples.shape[0] ,k=1) #update saved correlations
            return self.mod_corrs
        if type(self.mod_corrs) is np.ndarray: #return saved samples
            return self.mod_corrs 
        print("No correlations currenly saved, rerun setting compute=True")
        return False

    # Methods for gradient ascent
    def num_gradient_ascent(self, N_samples=1000, chains=4, N_sets=10, updates_per_set=100, burn_in = 0, lr=0.1):
        """
        Performs gradient ascent, but uses gibbs sampling to work out expectations
        Args:
            N_samples: how many samples we generate via gibbs sampling
            N_sets: how many times we generate a set of samples
            updates_per_set: for each set of samples, how many gradient ascent updates we perform
        """
        burn_in = 0
        M = (N_samples - burn_in)*chains #number of samples
        h = self.h
        J = self.J

        for _ in range(N_sets):
            samples = self.gibbs_sampling(N_samples,chains) #generate set of samples with current h and J
            
            for i in range(updates_per_set): #gradient ascent based on the samples
                if i == 0:
                    Ps = np.ones(M) 
                else:
                    Ps = self.p_unnormalized(samples, h-self.h, J-self.J) # like the likelihood of each state, but based on samples 
                denom = M * np.mean(Ps) 

                mod_avgs = samples.T.dot(Ps) / denom
                mod_corrs = self.sample_corrs(samples,Ps) / denom
                
                #update h and J
                h = h + lr*(mod_avgs - self.avgs)
                J = J + lr*(mod_corrs - self.corrs)
            
            #update h and J that generate samples
            self.h = h
            self.J = J
        
        # save the estimated expectations and samples for use later
        self.mod_avgs = mod_avgs 
        self.mod_corrs = mod_corrs
        self.samples = samples
        self.Z = -1 # normalisation constant is now out of date
        return True

    @staticmethod
    @njit
    def sample_corrs(X,Y):
        """
        A convenience method used above
        X is an M x N matrix of states
        Y is an M vector of values for each state
        """
        N = X.shape[1]
        corrs = np.zeros((N,N))
        for i in range(N-1):
            for j in range(i+1,N):
                corrs[i,j] = np.sum( Y[ X[:,i]*X[:,j] == 1 ] )
        return corrs
    
    def save_samples(self,M=10000,chains=4,burn_in=0):
        """
        Saves states sampled based on the current model, which can be used to compute expectations
        """
        add_samples = self.gibbs_sampling(M,chains,burn_in)
        if type(self.samples) is np.ndarray: #we have existing samples
            self.samples = np.concatenate( (self.samples, add_samples), axis=0 )
            return True
        self.samples = add_samples #we don't have existing samples
        return True


    def gibbs_sampling(self,M,chains=1,burn_in=0):
        """
        Returns (M - burn_in)*chains samples based on the current Ising model
        """
        p_ind = self.avgs*(self.spin_vals[1]-self.spin_vals[0]) + self.spin_vals[0] 
        
        samples = np.zeros((chains, M, self.N))
        for c in range(chains):
            samples[c,0,:] = np.random.binomial(1,p_ind)*(self.spin_vals[1]-self.spin_vals[0]) + self.spin_vals[0] #set the initial state 

        samples = self.fast_gibbs_sampling(self.h, self.J, self.spin_vals, samples)
        samples = samples[:,burn_in:].reshape(-1,self.N)
        return samples

    @staticmethod
    @njit(parallel=True) #not sure how much of a difference parallel makes here
    def fast_gibbs_sampling(h, J, spin_vals, samples):
        """
        Given the weights h and J for an Ising model, and initial states specified by the initial entries of samples, 
        samples states from the Ising model and returns them.
        Burn-in is not included, and we add states each time a dimension is updated, hence states are very correlated.
        """
        N = samples.shape[2] 
        for c in prange(samples.shape[0]):
            for t in range(1,samples.shape[1]): 
                samples[c,t] = samples[c,t-1] #copy previous state
                i = t % N #which dimension to work on
                
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