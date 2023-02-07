
'''
Validate the inflated variance approach to continuum-level
power calculation.

Same as val_power_iv.py, but uses power1d's high-level
interface for numerical simulation.
'''

import os
import numpy as np
from matplotlib import pyplot
import power1d




#(0) Set parameters:



### power parameters:
alpha  = 0.05  # Type I error rate
effects = np.arange( 0.1 , 0.71, 0.1 ) # effect size


#(0) Set parameters:
JJ     = [5, 10, 25]     # sample sizes
Q      = 201   # continuum size
WW0    = [10.0, 20.0, 50.0]  # smoothness values (under the null)
### power parameters:
alpha  = 0.05  #Type I error rate
W1     = 30.0  #continuum smoothness under the alternative hypothesis
sigmas = np.arange(0.5, 2.01, 0.1)   #effect size (as variance under the alternative hypothesis)



# #(1) Theoretical power using on-central t method:
# np.random.seed(0)
# PP0,PP1 = [],[]
# for W0 in WW0:
# 	P0,P1 = [],[]
# 	for J in JJ:
# 		p0,p1 = [],[]
# 		for sigma in sigmas:
# 			df     = J - 1 # degrees of freedom
#
# 			### analytical power:
# 			u      = power1d.prob.t_isf(alpha, df, Q, W0)    #critical threshold (under the null)
# 			f      = float(W1) / W0                          #ratio of signal-to-noise smoothness
# 			Wstar  = W0 * ( (1+sigma**2) / (1+(sigma**2)/(1+f**2)) )**0.5  #smoothness for the alternative
# 			ustar  = u * ( 1 + sigma**2 )**(-0.5)            #threshold for the alternative hypothesis
# 			pp0    = power1d.prob.t_sf(ustar, df, Q, Wstar)  #theoretical power
#
# 			#(2) Validate numerically using power1d:
# 			baseline  = power1d.geom.Null(Q=Q)
# 			signal    = power1d.geom.Null(Q=Q)
# 			noise0    = power1d.noise.SmoothGaussian(Q=Q, sigma=1.0, fwhm=W0, J=J)
# 			noise1    = power1d.noise.SmoothGaussian(Q=Q, sigma=1.0, fwhm=Wstar, J=J)
# 			### data sample models:
# 			model0    = power1d.models.DataSample(baseline, signal, noise0, J=J)
# 			model1    = power1d.models.DataSample(baseline, signal, noise1, J=J)
# 			### experiment models:
# 			teststat  = power1d.stats.t_1sample_fn(J)
# 			expmodel0 = power1d.Experiment(model0, teststat)
# 			expmodel1 = power1d.Experiment(model1, teststat)
# 			### simulate:
# 			sim       = power1d.ExperimentSimulator(expmodel0, expmodel1)
# 			results   = sim.simulate(10000, progress_bar=True)
# 			pp1       = results.sf(ustar)
#
# 			print('Theoretical power:  %.05f' %pp0)
# 			print( 'Simulated power: %.3f' %pp1 )
# 			print
# 			p0.append( pp0 )
# 			p1.append( pp1 )
# 		P0.append( p0 )
# 		P1.append( p1 )
# 	PP0.append( P0 )
# 	PP1.append( P1 )
# PP0,PP1 = np.array(PP0), np.array(PP1)
# dir0    = os.path.dirname( __file__ )
# fname   = os.path.join(dir0, 'val_1d_iv.npz')
# np.savez(fname, PP0=PP0, PP1=PP1)



dir0    = os.path.dirname( __file__ )
fname   = os.path.join(dir0, 'val_1d_iv.npz')
with np.load(fname) as Z:
	PP0 = Z['PP0']
	PP1 = Z['PP1']

#(2) Plot:
pyplot.close('all')
### create figure:
pyplot.figure(figsize=(12,3))
### create axes:
axx    = np.linspace(0.05, 0.71, 3)
axy    = 0.16
axw    = 0.28
axh    = 0.83
AX     = [pyplot.axes([x,axy,axw,axh])  for x in axx]
ax0,ax1,ax2 = AX
colors   = 'b', 'g', 'r'
ls       = '-', '--', ':'
for ax,W,P0,P1 in zip(AX,WW0,PP0,PP1):
	for p0,p1,c,lss,J in zip( P0, P1 , colors, ls, JJ ):
		ax.plot( sigmas , p0 , color=c, ls=lss, label='J = %d' %J )
		ax.plot( sigmas , p1 , 'o', color=c )
		ax.set_xlabel(r'Effect size  ($\sigma$)', size=12)
ax0.set_ylabel('Power', size=12)
ax0.legend( loc=(0.01,0.5) )
pyplot.setp(AX, ylim=(0, 0.75))
### panel labels:
[ax.text(0.05, 0.9, '(%s)  FWHM = %d'%(chr(97+i), W), transform=ax.transAxes, size=12)  for i,(ax,W) in enumerate( zip(AX,WW0) ) ]
pyplot.show()







