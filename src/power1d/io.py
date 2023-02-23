'''
IO functions
'''

# Copyright (C) 2023  Todd Pataky

import os
import numpy as np


def file2datasamplemodel( fpath ):
	import pathlib
	from . models import datasample_from_array
	ext   = ''.join(  pathlib.Path( fpath ).suffixes ).lower()
	if ext == '.csv':
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


