'''
Example power analysis (i.e., a priori sample size calculation)
for 0D data, one-sample design.
'''


import numpy as np
from scipy import stats


def power_1sample(n, effect, alpha=0.05):
	delta  = effect * n ** 0.5   # non-centrality parameter
	u      = stats.t.isf( alpha , n-1 )
	return stats.nct.sf( u , n-1 , delta )


def sample_size_1sample(effect, alpha=0.05, target_power=0.8, n_range=(5,50)):
	'''
	Adjust n_range to a broader sample size range if necessary
	
	Note that stats.nct.sf does not handle n<5 well
	'''
	n   = np.arange( *n_range )
	p   = np.array([power_1sample(nn, effect, alpha) for nn in n])
	ind = np.argwhere(p > target_power).flatten()[0]
	return n[ind]



#(0) Set parameters:
alpha   = 0.05
power   = 0.8   # desired power
effect  = 0.5   # effect size ( mean / sd )



#(1) Calculate sample size analytically:
n       = sample_size_1sample(effect, alpha, power)
print( f'Analytically calculated sample size: {n}' )




