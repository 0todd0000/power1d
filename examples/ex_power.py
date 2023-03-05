
import numpy as np
import matplotlib.pyplot as plt
import power1d

import time


#(0) Create geometry:
np.random.seed(10)
J = 5   # sample size
Q = 101 # continuum size
baseline = power1d.geom.Null( Q )
signal0  = power1d.geom.Null( Q )
signal1  = power1d.geom.GaussianPulse( Q , q = 60 , fwhm = 30, amp = 2.0 )
noise    = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 1 )
# noise    = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 1 , fwhm = 20 )

model0   = power1d.models.DataSample( baseline , signal0 , noise , J = J )
model1   = power1d.models.DataSample( baseline , signal1 , noise , J = J )

teststat = power1d.stats.t_1sample_fn( J )

emodel0  = power1d.models.Experiment( model0 , teststat )
emodel1  = power1d.models.Experiment( model1 , teststat )

# simulate experiments:
sim       = power1d.ExperimentSimulator(emodel0, emodel1)
results   = sim.simulate(10000, progress_bar=True)



#(2) Plot:
plt.close('all')
plt.figure(figsize=(8,6))
results.plot()
plt.show()



