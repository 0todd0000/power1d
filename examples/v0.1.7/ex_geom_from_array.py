
'''
This script demonstrates how to create geometry objects
from an arbitrary 1D array.

Note that "from_array" and "Continuum1D" are identical.

The "from_array" function is provided simply as an
alternative, clearer way to create geometry objects
from arrays.
'''


import numpy as np
from matplotlib import pyplot as plt
import power1d


# create arbitrary geometry
y        = np.sin( np.linspace(0, 4*np.pi, 101) ) * np.linspace(0.1, 1, 101)**3

# create a geometry object using "from_array"
g0       = power1d.geom.from_array( y )

# create a geometry object using "Continuum1D"
g1       = power1d.geom.Continuum1D( y )  # same as power1d.geom.from_array


plt.close('all')
fig,axs = plt.subplots(1, 2, figsize=(8,3))
ax0,ax1 = axs
ax0.plot( y )
g0.plot( ax=ax1, lw=3, label='...using "from_array"' )
g1.plot( ax=ax1, marker='o', ms=2, label='...using "Continuum1D"' )
ax0.legend()
ax1.legend()
plt.setp(axs, xlabel='Domain position')
ax0.set_ylabel('Dependent variable value')
ax0.set_title('Arbitrary 1D array')
ax1.set_title('Geometry objects...')
plt.tight_layout()
plt.show()


