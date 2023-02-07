
'''
Validate the noncentral t approach to continuum-level
power calculation.
'''

import numpy as np
import power1d






#(0) Set parameters:
J      = 8     # sample size
Q      = 201   # continuum size
W      = 40.0  # smoothness
### derived parameters:
df     = J - 1 # degrees of freedom
### power parameters:
alpha  = 0.05  # Type I error rate
effect = 0.8   # effect size
delta  = effect * J ** 0.5 # non-centrality parameter





#(1) Theoretical power using on-central t method:
zstar  = power1d.prob.t_isf(alpha, df, Q, W)          #critical threshold (under the null)
power0 = power1d.prob.nct_sf(zstar, df, Q, W, delta)  #power
print('Theoretical power:  %.05f' %power0)




#(2) Validate numerically using power1d:
np.random.seed(0)
baseline = power1d.geom.Null( Q )
signal0  = power1d.geom.Null( Q )
signal1  = power1d.geom.Constant( Q , amp = effect )
noise    = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 1 , fwhm = W )
model0   = power1d.models.DataSample( baseline , signal0 , noise , J = J)
model1   = power1d.models.DataSample( baseline , signal1 , noise , J = J)
teststat = power1d.stats.t_1sample_fn( J )
emodel0  = power1d.models.Experiment( model0 , teststat )
emodel1  = power1d.models.Experiment( model1 , teststat )
# simulate experiments:
sim      = power1d.ExperimentSimulator( emodel0 , emodel1 )
results  = sim.simulate( 1000 , progress_bar = True )
print( 'Simulated power: %.3f' %results.p_reject1 )


