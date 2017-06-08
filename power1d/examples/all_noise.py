
import numpy as np
from matplotlib import pyplot
import power1d





#(0) Construct primitives:
np.random.seed(1)
J,Q       = 10,101





noise     = []
### basic types:
noise.append(   power1d.noise.ConstantUniform( J, Q, x0=-2, x1=2 )   )
noise.append(   power1d.noise.ConstantGaussian( J, Q, mu=0, sigma=1 )   )
noise.append(   power1d.noise.Uniform( J, Q, x0=-2, x1=2 )   )
noise.append(   power1d.noise.Gaussian( J, Q, mu=0, sigma=1 )   )
noise.append(   power1d.noise.Skewed( J, Q, mu=0, sigma=1, alpha=3 )   )
noise.append(   power1d.noise.SmoothGaussian( J, Q, mu=0, sigma=1, fwhm=20, pad=False )   )
noise.append(   power1d.noise.SmoothSkewed( J, Q, mu=0, sigma=1, alpha=3, fwhm=20, pad=False )   )
### compound type (Additive):
noise0    = power1d.noise.Gaussian( J, Q, mu=0, sigma=0.1 )
noise1    = power1d.noise.SmoothGaussian( J, Q, mu=0, sigma=1.5, fwhm=40, pad=False )
noise.append(   power1d.noise.Additive( noise0, noise1 )    )
### compound type (Mixture):
noise0    = power1d.noise.Gaussian( 2, Q, mu=0, sigma=0.5 )
noise1    = power1d.noise.SmoothGaussian( 40, Q, mu=2.0, sigma=0.5, fwhm=20, pad=False )
noise.append(   power1d.noise.Mixture( noise0, noise1 )    )
### compound type (Scaled):
noise0    = power1d.noise.Gaussian( J, Q, mu=0, sigma=1 )
scale     = np.linspace(0, 2, Q)
noise.append(   power1d.noise.Scaled( noise0, scale )   )
### compound type (SignalDependent):
noise0    = power1d.noise.Gaussian(J=J, Q=101, sigma=0.2)
signal    = power1d.geom.GaussianPulse(Q=101, q=65, amp=2.5, fwhm=20)
fn        = lambda nvalue,svalue: nvalue + nvalue*svalue**2
noise.append(   power1d.noise.SignalDependent(noise0, signal, fn=fn)   )






# #(1) Plot:
# pyplot.close('all')
# pyplot.figure(figsize=(12,6))
# # for i,x in enumerate(noise):
# 	ax = pyplot.subplot(3, 5, i+1)
# 	x.plot(ax, color='b', lw=0.5)
# 	ax.text(0.05, 0.9, '%s' %x.__class__.__name__, transform=ax.transAxes, bbox=dict(facecolor='w', alpha=0.5))
# pyplot.suptitle('All power1d.noise models', size=20)
# pyplot.show()



#(1) Plot:
pyplot.close('all')
pyplot.figure(figsize=(12,6))
axx    = np.linspace(0.05, 0.82, 5)
axy    = np.linspace(0.71, 0.09, 3)
axw    = 0.17
axh    = 0.26
ind    = -1
i      = -1
AX     = []
IND    = [0,1,2,3,4, 5,6,  10,11,12,13]
for axyy in axy:
	for axxx in axx:
		ind += 1
		if ind not in IND:
			continue
		i   += 1
		ax = pyplot.axes( [ axxx, axyy, axw, axh ] )
		c  = noise[i]
		c.plot(ax=ax, color='k', lw=0.5)
		ax.text(0.05, 0.9, '%s' %c.__class__.__name__, size=13, transform=ax.transAxes, bbox=dict(facecolor='w', alpha=0.5))
		AX.append(ax)
pyplot.setp(AX, ylim=(-3,5))
for ax in AX[7:]:
	ax.set_xlabel('Continuum position', size=12)
for ax in [AX[0], AX[5], AX[7]]:
	ax.set_ylabel('Continuum value', size=12)
pyplot.show()


# pyplot.savefig('/Users/todd/Documents/Projects/projects/power1d/figsnew/all_noise.pdf')


