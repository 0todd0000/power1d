'''
Example power analysis (i.e., a priori sample size calculation)
for 0D data, two-sample design.
'''


import numpy as np
from scipy import stats


def power_2sample(nA, nB, effect, alpha=0.05):
	delta  = effect * (( nA * nB ) / ( nA + nB )) ** 0.5  # noncentrality parameter
	v      = nA + nB - 2  # degrees of freedom
	u      = stats.t.isf( alpha , v )
	return stats.nct.sf( u , v , delta )



def sample_size_2sample(effect, alpha=0.05, target_power=0.8, n_range=(5,50)):
	'''
	Adjust n_range to a broader sample size range if necessary
	'''
	n   = np.arange( *n_range )
	p   = np.array([power_2sample(nn, nn, effect, alpha) for nn in n])
	ind = np.argwhere(p > target_power).flatten()[0]
	return n[ind]



#(0) Set parameters:
alpha   = 0.05
power   = 0.8   # desired power
effect  = 0.6   # effect size ( mean / sd )



#(1) Calculate sample size analytically:
n       = sample_size_2sample(effect, alpha, power)
print( f'Analytically calculated sample size: {n}' )




