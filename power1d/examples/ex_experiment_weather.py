
import os
import numpy as np
from matplotlib import pyplot
import power1d






#(0) Load weather data:
data     = power1d.data.weather()
y        = data['Continental']



#(1) Create DataSample model:
np.random.seed(0)
J        = 8  # sample size
Q        = 365  # continuum size
baseline = power1d.geom.Continuum1D( y.mean(axis=0) )
signal0  = power1d.geom.Null( Q )
signal1  = power1d.geom.GaussianPulse( Q , q = 200 , amp = 6 , fwhm = 100 )

noise0   = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 0.3 )
noise1   = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 5 , fwhm = 70 )
noise    = power1d.noise.Additive( noise0 , noise1 )

model0   = power1d.models.DataSample( baseline , signal0 , noise , J = J)
model1   = power1d.models.DataSample( baseline , signal1 , noise , J = J)


teststat = power1d.stats.t_1sample_fn( J )
emodel0  = power1d.models.Experiment( model0 , teststat )
emodel1  = power1d.models.Experiment( model1 , teststat )

# simulate experiments:
sim       = power1d.ExperimentSimulator(emodel0, emodel1)
results   = sim.simulate(200, progress_bar=True)

results.set_coi_radius(50)

roi       = np.array( [False]*Q )
roi[150:250] = True
# roi[:] = True

results.set_roi( roi )




#(2) Plot:
pyplot.close('all')
pyplot.figure(figsize=(8,6))
### create axes:
results.plot()
pyplot.show()




