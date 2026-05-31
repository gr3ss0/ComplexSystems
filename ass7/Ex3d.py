####
# Code scaffold for Homework 2/3 in Assignment 6
####

import numpy as np
import matplotlib.pyplot as plt

#import the input file
X0 = np.array([[100], [0]]) #initial state
In = np.loadtxt('Input.txt') #seed and number of simulations
#set the seed and the number of simulations from the input file
np.random.seed(seed=int(In[0]))
NrSimulations = int(In[1])

#-----Fixed Quantities-----
# Stoichiometric matrix

# <------- fill in the model specifics ---->

#               r1   r2  r3 
S = np.array([  [-1, 0],#X1
				[1, -1],#X2
				])

#reaction parameters
k = [0.5, 0.3]

t_final =  24 #final time

# <------------------------------>

# <------- fill in the reaction rate functions ---->
#reaction propensities
def propensities(X,k):
		R = np.zeros((3,1))
		R[0] = k[0]*X[0]
		R[1] = k[1]*X[1]
		return R
# <------------------------------>

def Time_To_Next_Reaction(lam):
	"""
	@brief The function samples from an exponential distribution with rate "lam". 
	@param lam : real value positive.
	"""

	# small hack as the numpy uniform random number includes 0
	r = np.random.rand()
	while r == 0:
		r = np.random.rand()

	return (1.0/lam)*np.log(1.0/r)

def Find_Reaction_Index(a):
	"""	
	@brief The function takes in the reaction rate vector and returns
	the index of the reaction to be fired of a possible reaction candidate.
	@param a : Array (num_reaction,1) 

	"""
	# small hack as the numpy uniform random number includes 0
	r = np.random.rand()
	while r == 0:
		r = np.random.rand()

	return np.sum(np.cumsum(a) < r*np.sum(a))

def SSA(Stochiometry,X0,t_final,k, timed=False):
	"""
	@brief  The Stochastic Simulation Algorithm. Given the stochiometry,
	propensities and the initial state; the algorithm
	gives a stochastic trajectory of the Kurtz process until $t_final.$
	
	@param Stochiometry : Numpy Array (Num_species,Num_reaction).
	@param X_0: Numpy Array (Num_species, 1).
	@param t_final : positive number.
	@param k1,k2,k3,k4: positive numbers  (reaction rate parameters)

	"""

	#for storage
	X_store = []
	T_store = []
	#initialize
	t = 0.0
	x = X0
	X_store.append(x)
	T_store.append(t)

	while t < t_final:
		#compute reaction rate functions
		a = propensities(x,k)
		# 1. When? Compute first Jump Time
		tau = Time_To_Next_Reaction(np.sum(a))
		""" Stopping criterium: Test if we have jumped too far and if
		yes, return the stored variables (states, times)
		"""


		if timed and not np.isinf(tau):
			while len(T_store) < int(t + tau) +1 and len(T_store) < int(t_final) + 1: # BUG, somehow causes error. check if we have jumped over the next integer time point
				X_store.append(x) # Store pre-reaction state for this interval
				T_store.append(float(len(T_store)))
				
		if (t + tau > t_final) or (np.sum(a) == 0):
			while len(T_store)<int(t_final)+1:
				#print(len(T_store))
				# Update our Storage
				X_store.append(x)
				T_store.append(len(T_store))
			return np.array(X_store),np.array(T_store)
		
		else:
			# Since we have not, we need to find the next reaction
			t = t + tau #update time
			#2. What? find reaction to execute and execute the reaction
			j = Find_Reaction_Index(a)
			x = x + Stochiometry[:,[j]]
			if not timed:
				X_store.append(x)
				T_store.append(t)
		


def compute_mean_std():
	sample_means = []
	sim_sizes = []
	for n in [10,40,160]:
		x1_trace = []
		for i in range(n):
			# get a single realisation
			states, times = SSA(S,X0,5,k, timed=True)
			x1_trace.append(states[5,1,0])
		x1_mean = np.sum(np.array(x1_trace) / n)
		sample_means.append(x1_mean)
		sim_sizes.append(n)
	return sim_sizes, sample_means


def analytical_solution(t, k):
	ka = k[0]
	ke = k[1]
	res = ke * X0[0, 0] / (ke - ka) * (np.exp(ka * t) - np.exp(-ke * t))
	return res

# def compute_error(sample_means, expectation):
# 	errors = []
# 	for mean in sample_means:
# 		error = np.abs(mean - expectation)
# 		errors.append(error)
# 	return errors

expectation = analytical_solution(5, k)

experimets = []
N = []
# 10 repetition of the same size
for i in range(10):

	for n in [10,40,160]:
		# for one size
		x1_trace = []
		for i in range(n):
			# get a single realisation
			states, times = SSA(S,X0,5,k, timed=True)
			x1_trace.append(states[5,1,0])
		# compute means
		x1_mean = np.sum(np.array(x1_trace) / n)
		N.append(n)
		error = np.abs(x1_mean - expectation)
		experimets.append(error)

print(experimets)
print(N)
#experimets = np.array(experimets)
# np.savetxt('Errors.txt', experimets, delimiter=',', fmt='%1.2f')

# x = 10 * np.array([10, 40, 160])


#plt.errorbar(N, y=experimets, yerr=np.array(experimets)/np.sqrt(np.array(N)), fmt='o', label='Error with Std Dev')
plt.scatter(N, y=experimets, label='Mean Error', color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Simulations')
plt.ylabel('Error')
plt.legend()
plt.savefig('Error_Plot.png')