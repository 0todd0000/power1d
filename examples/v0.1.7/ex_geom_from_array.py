
import numpy as np
from matplotlib import pyplot as plt
import power1d



y        = np.sin( np.linspace(0, 4*np.pi, 101) ) * np.linspace(0.1, 1, 101)**3
g0       = power1d.geom.from_array( y )
g1       = power1d.geom.Continuum1D( y )  # same as power1d.geom.from_array


plt.close('all')
fig,axs = plt.subplots(1, 2, figsize=(8,3))
ax0,ax1 = axs
ax0.plot( y )
g0.plot( ax=ax1, lw=3 )
g1.plot( ax=ax1, marker='o', ms=2 )
plt.tight_layout()
plt.show()


