'''
Datasets.

Currently only one dataset is available:

"weather"
'''

# Copyright (C) 2023  Todd Pataky


import os
import numpy as np


def weather():
	'''
	| This dataset was made available by Prof. James O. Ramsay
	  of McGill University. The dataset was download from:
	| http://www.psych.mcgill.ca/misc/fda/downloads/FDAfuns/Matlab
	| on 28 March 2017 (see the ./examples/weather directory).

	The data have been converted from the original form in
	the "daily.mat" file to NumPy format and saved in
	./power1d/data/weather/daily.npz
	
	This is a convenience function for loading those data.
	
	Arguments:
	
	(None)
	
	Outputs:
	
	A dictionary containing the following keys:
	
	- "Atlantic"
	- "Pacific"
	- "Continental"
	- "Arctic"
	
	Each item is a (J x 365) array where J is the number of
	stations per region.
	
	Example:
	
	.. plot::
		:include-source:

		from matplotlib import pyplot
		import power1d

		data = power1d.data.weather()   #load data dictionary
		y    = data['Continental']   #extract one region
		pyplot.plot(y.T, color="k")
	'''
	dir0     = os.path.dirname(__file__)
	fname    = os.path.join(dir0, 'data', 'weather', 'daily.npz')
	with np.load(fname) as D:
		z    = dict( **D )
	return z

