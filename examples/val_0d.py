
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import power1d






#(0) Set parameters:
alpha   = 0.05
JJ      = [5, 10, 25]    #sample size
effects = np.arange( 0.2 , 1.01, 0.1 )
### derived parameters:



# #(2) Numerically estimated power:
# np.random.seed(0)
# Q        = 2  #a short continuum
# baseline = power1d.geom.Null( Q )
# signal0  = power1d.geom.Null( Q )
# PP0,PP1  = [],[]
# for J in JJ:
# 	noise    = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 1 )
# 	model0   = power1d.models.DataSample( baseline , signal0 , noise , J = J)
# 	teststat = power1d.stats.t_1sample_fn( J )
# 	emodel0  = power1d.models.Experiment( model0 , teststat )
# 	P0,P1    = [] , []
# 	for effect in effects:
# 		signal1  = power1d.geom.Constant( Q , amp = effect )
# 		model1   = power1d.models.DataSample( baseline , signal1 , noise , J = J)
# 		emodel1  = power1d.models.Experiment( model1 , teststat )
# 		# simulate experiments:
# 		sim      = power1d.ExperimentSimulator( emodel0 , emodel1 )
# 		results  = sim.simulate( 1000 , progress_bar = True )
# 		# create ROI
# 		roi      = np.array( [ True , False ] )
# 		results.set_roi( roi )
# 		P1.append( results.p_reject1 )
#
#
# 		#(1) Analytical power:
# 		df     = J - 1   #degrees of freedom
# 		u      = stats.t.isf( alpha , df )
# 		delta  = effect * J ** 0.5   #non-centrality parameter
# 		p0     = stats.nct.sf( u , df , delta )
# 		P0.append( p0 )
#
# 		print( 'Analytical power: %.3f' %p0 )
# 		print( 'Simulated power: %.3f' %results.p_reject1 )
# 		print
# 	PP0.append(P0)
# 	PP1.append(P1)
# PP0,PP1 = np.array(PP0), np.array(PP1)
# dir0    = os.path.dirname( __file__ )
# fname   = os.path.join(dir0, 'data', 'val_0d.npz')
# np.savez(fname, PP0=PP0, PP1=PP1)


dir0    = os.path.dirname( __file__ )
fname   = os.path.join(dir0, 'data', 'val_0d.npz')
with np.load(fname) as Z:
	PP0 = Z['PP0']
	PP1 = Z['PP1']

#(2) Plot:
plt.close('all')
### create figure:
plt.figure(figsize=(4,3))
### create axes:
ax       = plt.axes([0.14,0.18,0.84,0.8])
colors   = 'b', 'g', 'r'
ls       = '-', '--', ':'
for c,lss,P0,P1,J in zip( colors, ls, PP0, PP1 , JJ ):
	ax.plot( effects , P0 , color=c , ls=lss, label='J = %d' %J )
	ax.plot( effects , P1 , 'o', color=c )
	ax.set_xlabel('Effect size', size=12)
	ax.set_ylabel('Power', size=12)
ax.legend()
plt.show()






