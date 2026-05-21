####
# Code scaffold for Homework 2/3 in Assignment 6
####

import numpy as np

# Import the input file
X0 = np.loadtxt('Input.txt', ndmin=2)  # initial state
In = np.loadtxt('Input2.txt')          # seed and number of simulations

# Set the seed and the number of simulations from the input file
np.random.seed(seed=int(In[0]))
NrSimulations = int(In[1])

# ----- Fixed Quantities -----
# Stoichiometric matrix
#     r1   r2   r3 
S = np.array([
    [-1,  0,  0],  # X1
    [-1,  1,  1],  # X2
    [ 0,  0, -1]   # X3
])

# Reaction parameters
k = [0.01, 0.1, 0.01]
t_final = 10  # final time

# ----- Reaction Rate Functions -----
def propensities(X, k, t):
    R = np.zeros((3, 1))
    R[0] = k[0] * X[0] * X[1]
    R[1] = k[1] * 0.5 * (np.sin(t * 180) + 2)
    R[2] = k[2] * X[1] * X[2]
    return R

def Time_To_Next_Reaction(lam):
    """
    Samples from an exponential distribution with rate "lam". 
    """
    r = np.random.rand()
    while r == 0:
        r = np.random.rand()
    return (1.0 / lam) * np.log(1.0 / r)

def Find_Reaction_Index(a, B):
    """ 
    Takes in the reaction rate vector and returns the index of the reaction.
    """
    r = np.random.rand()
    while r == 0:
        r = np.random.rand()
    return np.sum(np.cumsum(a) < r * B)

def SSA(Stochiometry, X0, t_final, k):
    """
    The Stochastic Simulation Algorithm.
    """
    # For storage
    X_store = []
    T_store = []
    
    # Initialize
    t = 0.0
    x = X0.copy()
    X_store.append(x[1, 0])
    T_store.append(t)

    while t < t_final:
        # Compute reaction rate functions
        a = propensities(x, k, t)
        B = np.sum(a)
        
        # 1. When? Compute first Jump Time
        if B == 0:
            return np.array(X_store), np.array(T_store)
            
        tau = Time_To_Next_Reaction(B)
        
        # Stopping criterion
        if (t + tau > t_final):
            return np.array(X_store), np.array(T_store)
        else:
            # 2. What? Find and execute the reaction
            t = t + tau
            j = Find_Reaction_Index(a, B)
            
            if j < len(a):
                x = x + Stochiometry[:, [j]]
            
            # Update our Storage
            X_store.append(x[1, 0])
            T_store.append(t)

# Run a number of simulations and save the respective trajectories
for i in range(NrSimulations):
    states, times = SSA(S, X0, t_final, k)
    # Save trajectory
    output = np.concatenate((np.array(times, ndmin=2), np.array(states, ndmin=2)), axis=0)
    np.savetxt(f'Task2Traj{i+1}.txt', output, delimiter=',', fmt='%1.3f')