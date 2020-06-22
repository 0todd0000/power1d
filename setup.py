
from setuptools import setup



long_description = '''
**power1d** is a Python package for numerically estimating statistical
power in experiments involving one-dimensional continua.
'''

setup(
	name             = 'power1d',
	version          = '0.1.1',
	description      = 'Numerical Power Estimates for One-Dimensional Continua',
	author           = 'Todd Pataky',
	author_email     = 'spm1d.mail@gmail.com',
	packages         = ['power1d'],
	package_data     = {'power1d' : ['examples/*.*', 'data/*.*'] },
	include_package_data = True,
	long_description = long_description,
	keywords         = ['statistics', 'random field theory', 'Gaussian random fields', 'continuum analysis', 'null hypothesis testing', 'time series analysis'],
	classifiers      = [],
	install_requires = ["numpy", "scipy", "matplotlib"]
)