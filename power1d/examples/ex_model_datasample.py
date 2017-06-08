
import numpy as np
from matplotlib import pyplot
import power1d




#(0) Create geometry:
np.random.seed(10)
J = 8   # sample size
Q = 101 # continuum size
baseline = power1d.geom.Null( Q )
signal   = power1d.geom.GaussianPulse( Q , q = 60 , fwhm = 30, amp = 8 )
noise    = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 1 , fwhm = 20 )
model    = power1d.models.DataSample( baseline , signal , noise)


#(2) Plot:
pyplot.close('all')
pyplot.figure(figsize=(8,3))
### create axes:
ax0      = pyplot.axes([0.09,0.18,0.41,0.8])
ax1      = pyplot.axes([0.57,0.18,0.41,0.8])
AX       = [ax0,ax1]
### plot:
model.plot( ax = ax0, lw=5 )
model.random( )
model.plot( ax = ax1, lw=5 )
pyplot.setp(AX, ylim=(-2.5, 9.5))
[ax.set_xlabel('Continuum position', size=12)  for ax in AX]
ax0.set_ylabel('Continuum value', size=12)
ax0.legend( [ax0.lines[0], ax0.lines[-1]], ['Noise', 'Mean'] )
labels   = '(a)', '(b)'
[ax.text(0.03, 0.91, s, size=12, transform=ax.transAxes)   for ax,s in zip(AX,labels)]
pyplot.show()

# pyplot.savefig('/Users/todd/Documents/Projects/projects/power1d/figsnew/data_sample.pdf')

