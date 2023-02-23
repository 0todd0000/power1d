
import os
from matplotlib import pyplot as plt
import power1d


dir_data   = os.path.join( os.path.dirname(__file__), 'data')
fpath      = os.path.join( dir_data, 'Atlantic.csv' )
fpath      = os.path.join( dir_data, 'Atlantic.npy.gz' )
fpath      = os.path.join( dir_data, 'Atlantic.mat' )


model      = power1d.io.file2datasamplemodel( fpath )
model      = power1d.io.file2datasamplemodel( fpath )
model      = power1d.io.file2datasamplemodel( fpath )


plt.close('all')
plt.figure( figsize=(8,6) )
ax = plt.axes()
model.plot( ax=ax )
plt.tight_layout()
plt.show()

