
import os
import numpy as np
from matplotlib import pyplot as plt
import power1d


dir_data  = os.path.join( os.path.dirname(__file__), 'data')

# # convert CSV files to geometry objects:
# fpath0   = os.path.join( dir_data, 'array_single_row.csv')
# fpath1   = os.path.join( dir_data, 'array_single_col.csv')
# fpath2   = os.path.join( dir_data, 'array_multi.csv')
# g0       = power1d.io.file2geom( fpath0 )
# g1       = power1d.io.file2geom( fpath1 )
# g2       = power1d.io.file2geom( fpath2 )


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
g0.plot( ax=ax0, lw=3 )
g1.plot( ax=ax0, marker='o', ms=2 )
[gg.plot( ax=ax1 )  for gg in g2]
plt.tight_layout()
plt.show()


