
import os
import numpy as np
from matplotlib import pyplot as plt
import power1d


fpath    = os.path.join( os.path.dirname( __file__ ), 'data', 'Neptune1999kneeflex.csv')
y        = np.loadtxt( fpath, delimiter=',')
model    = power1d.models.datasample_from_array( y )



plt.close('all')
fig,axs = plt.subplots(1, 3, figsize=(11,3))
ax0,ax1,ax2 = axs
ax0.plot( y.T, lw=0.5 )
ax0.plot( y.mean(axis=0), 'k', lw=3)
model.plot( ax=ax1 )
model.random()
model.plot( ax=ax2 )
plt.tight_layout()
plt.show()


