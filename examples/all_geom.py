
import numpy as np
from matplotlib import pyplot
import power1d





#(0) Construct primitives:
np.random.seed(0)
Q         = 101
continua  = []
continua.append(   power1d.geom.Continuum1D( np.random.randn(Q) )   )
continua.append(   power1d.geom.Constant(Q, 1.3)   )
continua.append(   power1d.geom.Exponential(Q, x0=0.2, x1=2.3, rate=5)   )
continua.append(   power1d.geom.ExponentialSaw(Q, x0=0, x1=30.5, rate=10, cutoff=75)   )
continua.append(   power1d.geom.GaussianPulse(Q, q=60, sigma=None, fwhm=20, amp=3.2)   )
continua.append(   power1d.geom.Linear(Q, x0=0, x1=3.5, slope=None)   )
continua.append(   power1d.geom.Null(Q)   )
continua.append(   power1d.geom.SawPulse(Q, q0=50, q1=80, x0=0, x1=2.5)   )
continua.append(   power1d.geom.SawTooth(Q, q0=3, q1=13, x0=0, x1=2.5, dq=3)   )
continua.append(   power1d.geom.Sigmoid(Q, q0=40, q1=80, x0=-1, x1=2.5)   )
continua.append(   power1d.geom.Sinusoid(Q, q0=0, amp=1, hz=2)   )
continua.append(   power1d.geom.SquarePulse(Q, q0=40, q1=60, x0=-0.5, x1=2.2)   )
continua.append(   power1d.geom.SquareTooth(Q, q0=5, q1=18, x0=-1.2, x1=2.7, dq=8)   )
continua.append(   power1d.geom.TrianglePulse(Q, q0=60, q1=85, x0=-1, x1=3)   )
continua.append(   power1d.geom.TriangleTooth(Q, q0=20, q1=35, x0=-1, x1=2.5, dq=10)   )



#(1) Plot:
pyplot.close('all')
pyplot.figure(figsize=(12,6))
axx    = np.linspace(0.05, 0.82, 5)
axy    = np.linspace(0.71, 0.09, 3)
axw    = 0.17
axh    = 0.26
i      = 0
AX     = []
for axyy in axy:
	for axxx in axx:
		ax = pyplot.axes( [ axxx, axyy, axw, axh ] )
		c  = continua[i]
		c.plot(ax=ax)
		ax.text(0.05, 0.9, '%s' %c.__class__.__name__, size=13, transform=ax.transAxes, bbox=dict(facecolor='w', alpha=0.5))
		AX.append(ax)
		i += 1
pyplot.setp(AX, ylim=(-2.5,4))
for ax in AX[10:]:
	ax.set_xlabel('Continuum position', size=12)
for ax in [AX[0], AX[5], AX[10]]:
	ax.set_ylabel('Continuum value', size=12)
pyplot.show()




