
from matplotlib import pyplot
import power1d




#(0) Create geometry:
Q = 101 # continuum size
y = power1d.geom.GaussianPulse( Q , q = 60 , fwhm = 20, amp = 3.2 )


#(2) Plot:
pyplot.close('all')
pyplot.figure(figsize=(4,3))
ax = pyplot.axes([0.14,0.18,0.85,0.8])
y.plot()
ax.set_xlabel('Continuum position', size=12)
ax.set_ylabel('Continuum value', size=12)
pyplot.show()

# pyplot.savefig('/Users/todd/Documents/Projects/projects/power1d/figsnew/gaussian_pulse.pdf')

