
import os
import numpy as np
import matplotlib.pyplot as plt
import power1d






#(0) Load weather data:
data     = power1d.data.weather()
y        = data['Continental']



#(1) Create DataSample model:
np.random.seed(5)
J        = 8  # sample size
Q        = 365  # continuum size
baseline = power1d.geom.Continuum1D( y.mean(axis=0) )
signal   = power1d.geom.Null( Q )
noise0   = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 0.3 )
noise1   = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 3 , fwhm = 70 )
noise    = power1d.noise.Additive( noise0 , noise1 )
model    = power1d.models.DataSample( baseline , signal , noise , J = J)


#(2) Plot:
plt.close('all')
plt.figure(figsize=(8,3))
### create axes:
ax0      = plt.axes([0.09,0.18,0.41,0.8])
ax1      = plt.axes([0.57,0.18,0.41,0.8])
AX       = [ax0,ax1]
### plot:
model.plot( ax = ax0, lw=5 )
model.random( )
model.plot( ax = ax1, lw=5 )
# plt.setp(AX, ylim=(-2.5, 9.5))
[ax.set_xlabel('Continuum position', size=12)  for ax in AX]
ax0.set_ylabel('Continuum value', size=12)
ax0.legend( [ax0.lines[0], ax0.lines[-1]], ['Noise', 'Mean'], loc='upper right' )
labels   = '(a)', '(b)'
[ax.text(0.03, 0.91, s, size=12, transform=ax.transAxes)   for ax,s in zip(AX,labels)]
plt.show()



