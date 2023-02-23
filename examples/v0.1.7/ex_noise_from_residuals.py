
import os
import numpy as np
from matplotlib import pyplot as plt
import power1d


dir_data  = os.path.join( os.path.dirname(__file__), 'data')
fpath     = os.path.join( dir_data, 'array_multi.csv.gz')
r         = np.loadtxt( fpath, delimiter=',')


# r        = y - y.mean(axis=0)  # residuals
noise    = power1d.noise.from_residuals( r )


plt.close('all')
fig,axs = plt.subplots(1, 2, figsize=(8,3))
ax0,ax1 = axs
ax0.plot( r.T )
noise.plot( ax=ax1 )
plt.tight_layout()
plt.show()


