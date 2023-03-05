
'''
This script demonstrates how data sample models can be created
directly from a file containing an experimental dataset.

The 1D observations should be arranged as a (J,Q) array where:
	J = number of observations
	Q = number of 1D domain nodes

See "ex_model_from_array.py" for more details regarding
creating data sample models from experimental data.
'''

import os
from matplotlib import pyplot as plt
import power1d


dir_data   = os.path.join( os.path.dirname(__file__), 'data')
fpathCSV   = os.path.join( dir_data, 'Atlantic.csv' )
fpathNPYgz = os.path.join( dir_data, 'Atlantic.npy.gz' )
fpathMAT   = os.path.join( dir_data, 'Atlantic.mat' )


model      = power1d.io.file2datasamplemodel( fpathCSV )
model      = power1d.io.file2datasamplemodel( fpathNPYgz )
model      = power1d.io.file2datasamplemodel( fpathMAT )


plt.close('all')
plt.figure()
ax = plt.axes()
model.plot( ax=ax )
plt.setp(ax, xlabel='Day', ylabel='Atlantic temperature')
plt.tight_layout()
plt.show()

