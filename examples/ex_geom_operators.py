
import matplotlib.pyplot as plt
import power1d




#(0) Create geometry:
Q = 101 # continuum size
y0 = power1d.geom.GaussianPulse( Q , q = 40 , fwhm = 60, amp = 1.0 )
y1 = power1d.geom.Sinusoid( Q , amp = 1 , hz = 2 )
# y1 = power1d.geom.TriangleTooth( Q , q0 = 0, q1 = 20, x0 = 1, x1 = 2, dq = 0 )
# y1 = power1d.geom.TrianglePulse( Q , q0 = 0, q1 = 100, x0 = 1, x1 = 2 )
# y1 = power1d.geom.Sigmoid( Q , q0 = 10, q1 = 80, x0 = 0.5, x1 = 2 )
# y1 = power1d.geom.Linear( Q , x0 = 0.1, x1 = 5 )
yA = y0 + y1
yB = y0 * y1
yC = y0 ** y1



#(1) Plot:
plt.close('all')
plt.figure(figsize=(8,3))
ax0      = plt.axes([0.09,0.18,0.41,0.8])
ax1      = plt.axes([0.57,0.18,0.41,0.8])
AX       = [ax0,ax1]
y0.plot(ax=ax0, color='b', ls='-', label='y0')
y1.plot(ax=ax0, color='g', ls='--', label='y1')
ax0.legend()
### plot derived geometries:
colors   = 'k', 'r', 'orange'
yA.plot(ax=ax1, label='yA = y0 + y1', color=colors[0], ls='-')
yB.plot(ax=ax1, label='yB = y0 * y1', color=colors[1], ls='--')
yC.plot(ax=ax1, label='yC = y0 ** y1', color=colors[2], ls=':')
### legend and labels:
ax1.legend(loc=(0.15,0.7))
[ax.set_xlabel('Continuum position', size=12) for ax in AX]
ax0.set_ylabel('Continuum value', size=12)
labels   = '(a)', '(b)'
[ax.text(0.03, 0.91, s, size=12, transform=ax.transAxes)   for ax,s in zip(AX,labels)]
plt.show()



