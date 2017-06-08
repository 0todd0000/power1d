
import numpy as np
from matplotlib import pyplot
import power1d




#(0) Create models:
np.random.seed(1)
J,Q,fwhm  = 8, 101, 25
### create baseline and noise for all models:
baseline  = power1d.geom.Null(Q)
noise     = power1d.noise.SmoothGaussian(J=J, Q=Q, sigma=1, fwhm=fwhm, pad=True)
### model 0:  inflated variance:
sIV,wIV   = 1.5, 10
signal0   = power1d.noise.SmoothGaussian(J=1, Q=Q, sigma=sIV, fwhm=wIV, pad=False)
### model 1:  NC-RFT
signal1   = power1d.geom.Constant(Q=Q, amp=1.5)
### model 2:  power1
signal2   = power1d.geom.GaussianPulse(Q=Q, q=75, fwhm=50, amp=2.0)
noise20   = power1d.noise.SmoothGaussian(J=J, Q=Q, sigma=1, fwhm=fwhm, pad=True)
noise21   = power1d.noise.Gaussian(J=J, Q=Q, sigma=0.1)
noise2    = power1d.noise.Additive(noise20, noise21)
signal2n  = power1d.geom.Exponential(Q=Q, x0=0.2, x1=2.0, rate=3)
noise2    = power1d.noise.Scaled(noise2, signal2n)
noise2.random()
noise2.random()
noise2.random()



# (1) Plot:
pyplot.close('all')
fontname = u'DejaVu Sans'
fig = pyplot.figure(figsize=(8,2.5))
### create axes:
axx = np.linspace(0.06,0.7,3)
axy = 0.19
axw = 0.29
axh = 0.78
AX  = [pyplot.axes([x, axy, axw, axh])  for x in axx]
ax0,ax1,ax2  = AX
[ax.set_yticklabels([])  for ax in [ax1,ax2]]
[pyplot.setp(ax.get_xticklabels() + ax.get_yticklabels(), size=8)  for ax in AX]
### colors and line widths:
cb,cn,cs = 'k', '0.4', (0.4,0.8,0.4)
lws      = 4
### plot model 0 (inflated variance):
baseline.plot(ax=ax0, color=cb)
noise.plot(ax=ax0, color=cn, lw=0.5)
signal0.plot(ax=ax0, color=cs, lw=lws)
### plot model 1 (ncrft):
baseline.plot(ax=ax1, color=cb)
noise.random()
noise.plot(ax=ax1, color=cn, lw=0.5)
signal1.plot(ax=ax1, color=cs, lw=lws)
### plot model 2 (power1d):
baseline.plot(ax=ax2, color=cb)
noise2.plot(ax=ax2, color=cn, lw=0.5)
signal2.plot(ax=ax2, color=cs, lw=lws)
### adjust axes:
pyplot.setp(AX, ylim=(-4,4))
[ax.set_xlabel('Continuum position (%)', size=11, name=fontname)  for ax in AX]
ax0.set_ylabel('Dependent variable', size=11, name=fontname)
### legend:
ax0.legend([ax0.lines[1], ax0.lines[-1]], ['Noise','Signal'])
### panel labels:
labels  = 'Inflated variance', 'Non-central RFT', 'Numerical'
[ax.text(0.05,0.9, '(%s)  %s'%(chr(97+i),s), name=fontname, transform=ax.transAxes, size=11)  for i,(ax,s) in enumerate(zip(AX,labels))]
pyplot.show()


# pyplot.savefig('/Users/todd/Documents/Projects/projects/power1d/figsnew/existing_methods.pdf')
