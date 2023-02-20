'''
IO functions
'''

# Copyright (C) 2023  Todd Pataky




def load(fpath):
	import pickle
	with open(fpath, 'rb') as f:
		obj = pickle.load( f )
	return obj


def load_npy_gz(fpath):
	from gzip import GzipFile
	import numpy as np
	with GzipFile( fpath, 'r') as f:
		a = np.load( file=f )
	return a


def save_npy_gz(fpath, a):
	from gzip import GzipFile
	import numpy as np
	with GzipFile( fpath, 'w') as f:
		np.save(file=f, arr=a)


