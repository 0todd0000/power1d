
'''
This script demonstrates how data sample models can be created
directly from an array containing experimental data.

The array shape should be (J,Q) array where:
	J = number of observations
	Q = number of 1D domain nodes
'''

import os
import numpy as np
from matplotlib import pyplot as plt
import power1d


# load an experimental dataset
dir_data = os.path.join( os.path.dirname( __file__ ), 'data' )
fpath    = os.path.join( dir_data, 'Neptune1999kneeflex.csv' )
y        = np.loadtxt( fpath, delimiter=',')
model    = power1d.models.datasample_from_array( y )



plt.close('all')
fig,axs = plt.subplots(1, 3, figsize=(11,3))
ax0,ax1,ax2 = axs
ax0.plot( y.T, lw=0.5 )
ax0.plot( y.mean(axis=0), 'k', lw=3, label='Mean')
ax0.legend()
model.plot( ax=ax1 )
model.random()
model.plot( ax=ax2 )
plt.setp(axs, xlabel='Time (%)')
ax0.set_ylabel('Knee flexion angle (deg)')
titles = 'Experimental dataset', 'DataSample model (random)', 'DataSample model (random)'
[ax.set_title(s) for ax,s in zip(axs,titles)]
plt.tight_layout()
plt.show()


