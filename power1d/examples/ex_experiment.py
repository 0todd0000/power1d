
import numpy as np
from matplotlib import pyplot
import power1d




#(0) Create geometry:
np.random.seed(2)
J = 8   # sample size
Q = 101 # continuum size
baseline = power1d.geom.Null( Q )
signal   = power1d.geom.GaussianPulse( Q , q = 60 , fwhm = 30, amp = 3 )
noise    = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 1 , fwhm = 20 )

model    = power1d.models.DataSample( baseline , signal , noise , J = J )

teststat = power1d.stats.t_1sample
emodel   = power1d.models.Experiment( model , teststat )
emodel.simulate( 50 )




#(2) Plot:
pyplot.close('all')
### create figure:
pyplot.figure(figsize=(4,3))
### create axes:
ax       = pyplot.axes([0.14,0.18,0.84,0.8])
ax.plot( emodel.Z.T, color='k', lw=0.5 )
ax.set_xlabel('Continuum position', size=12)
ax.set_ylabel('Test statistic value', size=12)
pyplot.show()

# pyplot.savefig('/Users/todd/Documents/Projects/projects/power1d/figsnew/experiment.pdf')

