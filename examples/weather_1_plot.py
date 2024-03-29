'''
Canadian temperature data (Ramsay and Silverman, 2005)
'''


import os
import numpy as np
import matplotlib.pyplot as plt
import power1d


#(0) Load weather data:
data     = power1d.data.weather()
y0       = data['Arctic']
y1       = data['Atlantic']
y2       = data['Continental']
y3       = data['Pacific']
Y        = [y0, y1, y2, y3]




#(1) Plot:
plt.close('all')
### create figure:
plt.figure(figsize=(8,3))
### create axes:
ax0      = plt.axes([0.09,0.18,0.41,0.8])
ax1      = plt.axes([0.57,0.18,0.41,0.8])
AX       = [ax0,ax1]
### plot all data:
labels   = 'Arctic', 'Atlantic', 'Continental', 'Pacific'
colors   = 'b', 'orange', 'g', 'r'
ls       = '-', '--', ':', '-.'
x        = np.arange(365)
for y,label,c,lss in zip(Y,labels,colors,ls):
	h    = ax0.plot(y.T, color=c, lw=0.5, ls=lss)
	plt.setp(h[0], label=label)
	### plot mean and SD continua:
	m,s  = y.mean(axis=0), y.std(ddof=1, axis=0)
	ax1.plot(m, color=c, lw=3, ls=lss, label=label)
	ax1.fill_between(x, m-s, m+s, alpha=0.5)
plt.setp(AX, ylim=(-35,25))
### legend:
ax1.legend(fontsize=10, loc=(0.35,0.03))
[ax.set_xlabel('Day', size=14) for ax in AX]
ax0.set_ylabel('Temperature  (deg C)', size=14)
### panel labels:
labels   = '(a)', '(b)'
[ax.text(0.03, 0.91, s, size=12, transform=ax.transAxes)   for ax,s in zip(AX,labels)]
plt.show()



