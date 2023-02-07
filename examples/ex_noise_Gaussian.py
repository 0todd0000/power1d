
import numpy as np
from matplotlib import pyplot
import power1d




#(0) Create geometry:
np.random.seed(0)
J = 8   # sample size
Q = 101 # continuum size
noise0 = power1d.noise.Gaussian( J, Q , mu = 0 , sigma = 1 )
noise1 = power1d.noise.SmoothGaussian( J, Q , mu = 0 , sigma = 1 , fwhm = 20 , pad = True )


#(2) Plot:
pyplot.close('all')
pyplot.figure(figsize=(8,3))
### create axes:
ax0      = pyplot.axes([0.09,0.18,0.41,0.8])
ax1      = pyplot.axes([0.57,0.18,0.41,0.8])
AX       = [ax0,ax1]
### plot:
noise0.plot(ax=ax0, color='k', lw=0.5)
noise1.plot(ax=ax1, color='k', lw=0.5)
pyplot.setp(AX, ylim=(-3.5, 3.5))
[ax.set_xlabel('Continuum position', size=12)  for ax in AX]
ax0.set_ylabel('Continuum value', size=12)
labels   = '(a)', '(b)'
[ax.text(0.03, 0.91, s, size=12, transform=ax.transAxes)   for ax,s in zip(AX,labels)]
pyplot.show()



