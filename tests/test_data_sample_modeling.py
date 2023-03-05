
'''
This test script follows the online documentation:
https://spm1d.org/power1d/Examples/DataSample.html
'''

import os
import pytest
import numpy as np
import power1d



dir0         = os.path.dirname( __file__ )
dir_expected = os.path.join( dir0, '_expected', 'data_sample_modeling')



def test_weather_mean():
	data     = power1d.data.weather()  # load data
	y        = data['Continental']     # extract one region
	m        = y.mean( axis=0 )        # mean continuum
	m0       = power1d.io.load_npy_gz( os.path.join(dir_expected, 'mean.npy.gz') )
	assert m == pytest.approx(m0, abs=1e-3)
	

def test_geometries():
	Q         = 365  # continuum size
	g0        = power1d.geom.GaussianPulse( Q , q=200 , fwhm=190 , amp=40 )
	g1        = power1d.geom.Constant( Q , amp=23 )
	baseline  = g0 - g1  # subtract the geometries
	signal    = power1d.geom.GaussianPulse( Q , q=200 , fwhm=100 , amp=5 )
	baseline0 = power1d.io.load( os.path.join(dir_expected, 'baseline.pkl') )
	signal0   = power1d.io.load( os.path.join(dir_expected, 'signal.pkl') )
	assert baseline == baseline0
	assert signal == signal0
	
	
def test_noise():
	np.random.seed(0)
	J        = 8    # sample size
	Q        = 365  # continuum size
	n0       = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 0.3 )
	n1       = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 3 , fwhm = 70 )
	noise    = power1d.noise.Additive( n0 , n1 )
	noise0   = power1d.io.load( os.path.join(dir_expected, 'noise.pkl') )
	assert noise == noise0
	

def test_model():
	np.random.seed(0)
	J        = 8    # sample size
	Q        = 365  # continuum size
	# construct baseline geometry:
	g0       = power1d.geom.GaussianPulse( Q , q=200 , fwhm=190 , amp=40 )
	g1       = power1d.geom.Constant( Q , amp=23 )
	baseline = g0 - g1  # subtract the geometries
	# construct signal geometry:
	signal   = power1d.geom.GaussianPulse( Q , q=200 , fwhm=100 , amp=5 )
	# construct noise model:
	noise0   = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 0.3 )
	noise1   = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 3 , fwhm = 70 )
	noise    = power1d.noise.Additive( noise0 , noise1 )
	# create data sample model:
	model    = power1d.models.DataSample( baseline, signal, noise, J=J )
	model0   = power1d.io.load( os.path.join(dir_expected, 'model.pkl') )
	assert model == model0

	

# test_weather_mean()
# test_geometries()
# test_noise()
# test_model()
