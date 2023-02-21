
import matplotlib.pyplot as plt
import power1d




#(0) Create geometry:
Q = 101 # continuum size
y = power1d.geom.GaussianPulse( Q , q = 60 , fwhm = 20, amp = 3.2 )


#(2) Plot:
plt.close('all')
plt.figure(figsize=(4,3))
ax = plt.axes([0.14,0.18,0.85,0.8])
y.plot()
ax.set_xlabel('Continuum position', size=12)
ax.set_ylabel('Continuum value', size=12)
plt.show()



