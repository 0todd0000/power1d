'''
IO functions.

This module contains convenience functions for high-level
creation of geometry and DataSample models from data files.
'''

# Copyright (C) 2023  Todd Pataky

import os
import numpy as np


def file2geom( fpath ):
	'''
	Create Continuum1D geometry object(s) from a CSV file.
	
	Only CSV (and CSV.GZ) files are currently supported.
	
	1D arrays can be saved as either a single row or a
	single column in a CSV file.
	
	2D arrays must be saved with shape (nrow,ncol) where
	nrow is the number of 1D arrays and each row will be
	converted to a Continuum1D object
	
	Arguments:

	*fpath* ---- full path to a CSV file


	Outputs:
	
	*obj* ---- a Continuum1D object
	'''
	import os,pathlib
	from . geom import from_array
	assert os.path.exists( fpath ), f'File "{fpath}" not found.'
	ext   = ''.join(  pathlib.Path( fpath ).suffixes ).lower()
	assert ext in ['.csv', '.csv.gz'], f'File extension must be ".csv" or ".csv.gz"'
	a = np.loadtxt( fpath, delimiter=',' )
	return from_array( a )


def file2datasamplemodel( fpath ):
	'''
	Create a DataSample model from a CSV file.
	
	The following formats are currently supported:
	
	- .csv
	- .csv.gz
	- .mat
	- .npy
	- .npy.gz
	
	
	In all cases, the data file must contain a 2D array
	with shape (J,Q);  J=observations, Q=domain nodes
	
	Arguments:

	*fpath* ---- full path to a data file
	
	Outputs:
	
	*model* ---- a DataSample model
	'''
	
	import pathlib
	from . models import datasample_from_array
	ext   = ''.join(  pathlib.Path( fpath ).suffixes ).lower()
	if ext in ['.csv', '.csv.gz']:
		y = np.loadtxt( fpath, delimiter=',')
	elif ext.lower() == '.mat':
		from scipy.io import loadmat
		m   = loadmat( fpath )
		key = [k for k in m.keys() if not k.startswith('__')][0]
		y   = m[ key ]
	elif ext.lower() == '.npy':
		y = np.load( fpath )
	elif ext.lower() == '.npy.gz':
		y = load_npy_gz( fpath )
	return datasample_from_array( y )


def load(fpath):
	import pickle
	with open(fpath, 'rb') as f:
		obj = pickle.load( f )
	return obj


def load_npy_gz(fpath):
	from gzip import GzipFile
	with GzipFile( fpath, 'r') as f:
		a = np.load( file=f )
	return a


def save_npy_gz(fpath, a):
	from gzip import GzipFile
	with GzipFile( fpath, 'w') as f:
		np.save(file=f, arr=a)


