
'''
Validate the noncentral t approach to continuum-level
power calculation.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import power1d






#(0) Set parameters:
JJ     = [5, 10, 25]     # sample sizes
Q      = 201   # continuum size
WW     = [10.0, 20.0, 50.0]  # smoothness values
### power parameters:
alpha  = 0.05  # Type I error rate
effects = np.arange( 0.1 , 0.71, 0.1 ) # effect size




# #(1) Theoretical power using on-central t method:
# np.random.seed(0)
# PP0,PP1 = [],[]
# for W in WW:
# 	P0,P1 = [],[]
# 	for J in JJ:
# 		p0,p1 = [],[]
# 		for effect in effects:
# 			df     = J - 1 # degrees of freedom
# 			delta  = effect * J ** 0.5 # non-centrality parameter
#
# 			# analyical:
# 			zstar  = power1d.prob.t_isf(alpha, df, Q, W)          #critical threshold (under the null)
# 			pp0    = power1d.prob.nct_sf(zstar, df, Q, W, delta)  #power
#
# 			#(2) Validate numerically using power1d:
# 			baseline = power1d.geom.Null( Q )
# 			signal0  = power1d.geom.Null( Q )
# 			signal1  = power1d.geom.Constant( Q , amp = effect )
# 			noise    = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 1 , fwhm = W )
# 			model0   = power1d.models.DataSample( baseline , signal0 , noise , J = J)
# 			model1   = power1d.models.DataSample( baseline , signal1 , noise , J = J)
# 			teststat = power1d.stats.t_1sample_fn( J )
# 			emodel0  = power1d.models.Experiment( model0 , teststat )
# 			emodel1  = power1d.models.Experiment( model1 , teststat )
# 			# simulate experiments:
# 			sim      = power1d.ExperimentSimulator( emodel0 , emodel1 )
# 			results  = sim.simulate( 10000 , progress_bar = True )
# 			pp1      = results.p_reject1
# 			print('Theoretical power:  %.05f' %pp0)
# 			print( 'Simulated power: %.3f' %pp1 )
# 			print
# 			p0.append( pp0 )
# 			p1.append( pp1 )
# 		P0.append( p0 )
# 		P1.append( p1 )
# 	PP0.append( P0 )
# 	PP1.append( P1 )
#
# PP0,PP1 = np.array(PP0), np.array(PP1)
# dir0    = os.path.dirname( __file__ )
# fname   = os.path.join(dir0, 'data', 'val_1d_nct.npz')
# np.savez(fname, PP0=PP0, PP1=PP1)



dir0    = os.path.dirname( __file__ )
fname   = os.path.join(dir0, 'data', 'val_1d_nct.npz')
with np.load(fname) as Z:
	PP0 = Z['PP0']
	PP1 = Z['PP1']

#(2) Plot:
plt.close('all')
### create figure:
plt.figure(figsize=(12,3))
### create axes:
axx    = np.linspace(0.05, 0.71, 3)
axy    = 0.16
axw    = 0.28
axh    = 0.83
AX     = [plt.axes([x,axy,axw,axh])  for x in axx]
ax0,ax1,ax2 = AX
colors   = 'b', 'g', 'r'
ls       = '-', '--', ':'
for ax,W,P0,P1 in zip(AX,WW,PP0,PP1):
	for p0,p1,c,lss,J in zip( P0, P1 , colors, ls, JJ ):
		ax.plot( effects , p0 , color=c, ls=lss, label='J = %d' %J )
		ax.plot( effects , p1 , 'o', color=c )
		ax.set_xlabel('Effect size', size=12)
ax0.set_ylabel('Power', size=12)
ax0.legend( loc=(0.01,0.5) )
plt.setp(AX, ylim=(0.1, 1.1))
### panel labels:
[ax.text(0.05, 0.9, '(%s)  FWHM = %d'%(chr(97+i), W), transform=ax.transAxes, size=12)  for i,(ax,W) in enumerate( zip(AX,WW) ) ]
plt.show()



