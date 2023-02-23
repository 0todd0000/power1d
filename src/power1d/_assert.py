'''
Common assert functions.

Copyright (C) 2023  Todd Pataky
'''



def _assert_spm1d():
	try:
		import spm1d
	except ModuleNotFoundError:
		import inspect
		currframe   = inspect.currentframe()
		callframe   = inspect.getouterframes(currframe, 2)
		callername  = callframe[1][3]
		raise ModuleNotFoundError( f'Using "{callername}" requires spm1d')
		
