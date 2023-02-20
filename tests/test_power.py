
'''
This test script follows the online documentation:
https://spm1d.org/power1d/Examples/DataSample.html
'''

import os
import pytest
import numpy as np
import power1d



dir0         = os.path.dirname( __file__ )
dir_expected = os.path.join( dir0, '_expected', 'power')



def test_one_sample():
	np.random.seed(0)
	J        = 8    # sample size
	Q        = 365  # continuum size
	# construct baseline geometry:
	g0       = power1d.geom.GaussianPulse( Q , q=200 , fwhm=190 , amp=40 )
	g1       = power1d.geom.Constant( Q , amp=23 )
	baseline = g0 - g1  # subtract the geometries
	# construct signal geometry:
	signal0  = power1d.geom.Null( Q )
	signal1  = power1d.geom.GaussianPulse( Q , q=200 , fwhm=100 , amp=5 )
	# construct noise model:
	noise0   = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 0.3 )
	noise1   = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 3 , fwhm = 70 )
	noise    = power1d.noise.Additive( noise0 , noise1 )
	# create data sample models:
	model0   = power1d.models.DataSample( baseline, signal0, noise, J=J )
	model1   = power1d.models.DataSample( baseline, signal1, noise, J=J )
	# create experiment models:
	teststat = power1d.stats.t_1sample
	emodel0  = power1d.models.Experiment( model0 , teststat )    # null hypothesis
	emodel1  = power1d.models.Experiment( model1 , teststat )    # alternative hypothesis
	# simulate the experiments:
	sim      = power1d.ExperimentSimulator( emodel0 , emodel1 )
	results  = sim.simulate( 200, progress_bar=True )
	# check results:
	results0 = power1d.io.load( os.path.join(dir_expected, 'onesample-results.pkl') )
	assert results == results0
	

def test_two_sample():
	np.random.seed(0)
	JA,JB,Q   = 5, 7, 101
	baseline  = power1d.geom.Null(Q=Q)
	signal0   = power1d.geom.Null(Q=Q)
	signal1   = power1d.geom.GaussianPulse(Q=101, q=65, amp=2.5, sigma=10)
	noise     = power1d.noise.Gaussian(J=5, Q=101, sigma=1)

	#(1) Create data sample models:
	modelA0   = power1d.models.DataSample(baseline, signal0, noise, J=JA) #null A
	modelB0   = power1d.models.DataSample(baseline, signal0, noise, J=JB) #null N
	modelA1   = power1d.models.DataSample(baseline, signal0, noise, J=JA) #alternative A
	modelB1   = power1d.models.DataSample(baseline, signal1, noise, J=JB) #alternative B

	#(2) Create experiment models:
	# teststat  = power1d.stats.t_2sample_fn(JA, JB)
	teststat  = power1d.stats.t_2sample
	expmodel0 = power1d.models.Experiment([modelA0, modelB0], teststat) #null
	expmodel1 = power1d.models.Experiment([modelA1, modelB1], teststat) #alternative

	#(3) Simulate experiments:

	sim       = power1d.ExperimentSimulator(expmodel0, expmodel1)
	results   = sim.simulate(1000, progress_bar=True)
	# check results:
	results0 = power1d.io.load( os.path.join(dir_expected, 'twosample-results.pkl') )
	assert results == results0
	

# test_one_sample()
# test_two_sample()
