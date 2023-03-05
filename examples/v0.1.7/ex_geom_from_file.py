
'''
This script demonstrates how to create geometry objects
directly from CSV files.

The data may be saved in .csv or .csv.gz format

A single 1D array can be saved in the CSV file as either
a single row or a single column.

Multiple 1D arrays must be saved as a (J,Q) array where:
	J = number of observations
	Q = number of 1D domain nodes
'''


import os
from matplotlib import pyplot as plt
import power1d


dir_data  = os.path.join( os.path.dirname(__file__), 'data')

# convert CSV.GZ files to geometry objects:
fpath0g  = os.path.join( dir_data, 'array_single_row.csv.gz')
fpath1g  = os.path.join( dir_data, 'array_single_col.csv.gz')
fpath2g  = os.path.join( dir_data, 'array_multi.csv.gz')
g0       = power1d.io.file2geom( fpath0g )
g1       = power1d.io.file2geom( fpath1g )
g2       = power1d.io.file2geom( fpath2g )



plt.close('all')
fig,axs = plt.subplots(1, 2, figsize=(8,3))
ax0,ax1 = axs
g0.plot( ax=ax0, lw=3, label='Single CSV row' )
g1.plot( ax=ax0, marker='o', ms=2, label='Single CSV column' )
ax0.legend()
ax0.set_title('Single 1D arrays')
[gg.plot( ax=ax1 )  for gg in g2]
ax1.set_title('Multiple 1D arrays as CSV rows')
plt.tight_layout()
plt.show()


