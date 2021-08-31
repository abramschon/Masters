import numpy as np

def to_binary(n, width, spin_vals=[0,1], dtype=np.byte):
    """
    Returns a binary rep of the int n as an array of size width, e.g. Assuming N = 5, 3 -> np.array([0,0,0,1,1]) 
    Not particularly efficient, but since it is only used once at the start for small N, this is okay
    """
    b = np.ones(width,dtype=dtype)*spin_vals[0] 
    for i in range(width):
        if n % 2 == 1: 
            b[width-1-i]=spin_vals[1] # index N-1-i otherwise numbers are reversed
        else:
            b[width-1-i]=spin_vals[0]
        n//=2
        if n==0: break
    return b

def get_state_space(width, spin_vals=[0,1], dtype=None):
    """
    Sets up the state space, but only if analytic expectations are needed
    """
    return np.array([to_binary(n, width, spin_vals, dtype) for n in range(2**width)],dtype=dtype) 