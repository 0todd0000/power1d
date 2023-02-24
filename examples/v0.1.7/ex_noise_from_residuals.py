
'''
This script demonstrates how to create a noise object
from experimental residuals.

The experimental data must be saved as a (J,Q) array where:
	J = number of observations
	Q = number of 1D domain nodes
'''


import os
import numpy as np
from matplotlib import pyplot as plt
import power1d


# load experimental data:
fpath    = os.path.join( os.path.dirname( __file__ ), 'data', 'Neptune1999kneeflex.csv')
y        = np.loadtxt( fpath, delimiter=',')

# calculate mean and residuals
m        = y.mean(axis=0)  # mean
r        = y - m           # residuals

np.random.seed(0)
noise    = power1d.noise.from_residuals( r )


plt.close('all')
fig,axs = plt.subplots(2, 2, figsize=(8,6))
ax0,ax1,ax2,ax3 = axs.ravel()
ax0.plot( r.T )
noise.plot( ax=ax1 )
ax2.plot( y.T )
ax3.plot( (m + noise.value).T )
plt.setp(axs[1], xlabel='Time (%)')
ax0.set_ylabel('Residual (deg)')
ax2.set_ylabel('Knee flexion angle (deg)')
titles = 'Experimental residuals', 'Noise model', 'Original data', 'Noise value + mean'
[ax.set_title(s) for ax,s in zip(axs.ravel(), titles)]
plt.tight_layout()
plt.show()


