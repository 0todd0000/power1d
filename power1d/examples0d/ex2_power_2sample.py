'''
Example "post hoc" power calculation for 0D data, two-sample design.
'''


import numpy as np
from scipy import stats
import power1d




def power_2sample(nA, nB, effect, alpha=0.05):
	delta  = effect * (( nA * nB ) / ( nA + nB )) ** 0.5  # noncentrality parameter
	v      = nA + nB - 2  # degrees of freedom
	u      = stats.t.isf( alpha , v )
	return stats.nct.sf( u , v , delta )



#(0) Set parameters:
alpha   = 0.05
JA      = 10    # sample size (group A)
JB      = 14    # sample size (group B)
effect  = 0.8   # effect size ( mean / sd )



#(1) Calculate power analytically:
power    = power_2sample(JA, JB, effect, alpha)
print( 'Analytically calculated power: %.3f' %power )



#(2) Calculate power numerically using power1d
np.random.seed(0)
Q        = 2  # a short continuum
baseline = power1d.geom.Null( Q )
signal0  = power1d.geom.Null( Q )
signal1  = power1d.geom.Constant( Q , amp = effect )
noiseA   = power1d.noise.Gaussian( JA , Q , mu = 0 , sigma = 1 )
noiseB   = power1d.noise.Gaussian( JB , Q , mu = 0 , sigma = 1 )
modelA0  = power1d.models.DataSample( baseline , signal0 , noiseA , J = JA )
modelA1  = power1d.models.DataSample( baseline , signal0 , noiseA , J = JA )
modelB0  = power1d.models.DataSample( baseline , signal0 , noiseB , J = JB )
modelB1  = power1d.models.DataSample( baseline , signal1 , noiseB , J = JB )
teststat = power1d.stats.t_2sample_fn( JA, JB )
emodel0  = power1d.models.Experiment( [modelA0,modelB0] , teststat )
emodel1  = power1d.models.Experiment( [modelA1,modelB1] , teststat )
# simulate experiments:
sim       = power1d.ExperimentSimulator(emodel0, emodel1)
results   = sim.simulate(10000, progress_bar=True)
# create ROI
roi      = np.array( [ True , False ] )
results.set_roi( roi )
print( 'Numerically estimated power: %.3f' %results.p_reject1 )



