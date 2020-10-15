'''
Example "post hoc" power calculation for 0D data, one-sample design.
'''


import numpy as np
from scipy import stats
import power1d




def power_1sample(n, effect, alpha=0.05):
	delta  = effect * n ** 0.5   # noncentrality parameter
	u      = stats.t.isf( alpha , n-1 )
	return stats.nct.sf( u , n-1 , delta )



#(0) Set parameters:
alpha   = 0.05
J       = 12    # sample size
effect  = 0.7   # effect size ( mean / sd )



#(1) Calculate power analytically:
power    = power_1sample(J, effect, alpha)
print( 'Analytically calculated power: %.3f' %power )



#(2) Calculate power numerically using power1d
np.random.seed(0)
Q        = 2  # a short continuum
baseline = power1d.geom.Null( Q )
signal0  = power1d.geom.Null( Q )
signal1  = power1d.geom.Constant( Q , amp = effect )
noise    = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 1 )
model0   = power1d.models.DataSample( baseline , signal0 , noise , J = J)
model1   = power1d.models.DataSample( baseline , signal1 , noise , J = J)
teststat = power1d.stats.t_1sample_fn( J )
emodel0  = power1d.models.Experiment( model0 , teststat )
emodel1  = power1d.models.Experiment( model1 , teststat )
# simulate experiments:
sim      = power1d.ExperimentSimulator( emodel0 , emodel1 )
results  = sim.simulate( 10000 , progress_bar = True )
# create ROI
roi      = np.array( [ True , False ] )
results.set_roi( roi )
print( 'Numerically estimated power: %.3f' %results.p_reject1 )



