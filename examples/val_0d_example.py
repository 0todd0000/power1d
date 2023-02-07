
import numpy as np
from scipy import stats
import power1d






#(0) Set parameters:
alpha  = 0.05
J      = 12    #sample size
effect = 0.8   #effect size ( mean * sd )
### derived parameters:
df     = J - 1   #degrees of freedom
delta  = effect * J ** 0.5   #non-centrality parameter



#(1) Analytical power:
u      = stats.t.isf( alpha , df )
p0     = stats.nct.sf( u , df , delta )
print( 'Analytical power: %.3f' %p0 )



#(2) Numerically estimated power:
np.random.seed(0)
Q        = 2  #a short continuum
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
results  = sim.simulate( 1000 , progress_bar = True )
# create ROI
roi      = np.array( [ True , False ] )
results.set_roi( roi )
print( 'Simulated power: %.3f' %results.p_reject1 )










