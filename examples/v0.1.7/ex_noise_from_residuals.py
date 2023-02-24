
import os
import numpy as np
from matplotlib import pyplot as plt
import power1d


fpath    = os.path.join( os.path.dirname( __file__ ), 'data', 'Neptune1999kneeflex.csv')
y        = np.loadtxt( fpath, delimiter=',')
m        = y.mean(axis=0)
r        = y - m


np.random.seed(0)
noise    = power1d.noise.from_residuals( r )


plt.close('all')
fig,axs = plt.subplots(2, 2, figsize=(8,6))
ax0,ax1,ax2,ax3 = axs.ravel()
ax0.plot( r.T )
noise.plot( ax=ax1 )
ax2.plot( y.T )
ax3.plot( (m + noise.value).T )
plt.tight_layout()
plt.show()


