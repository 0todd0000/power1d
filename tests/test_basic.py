
'''
Test basic functionality
'''


import numpy as np
import matplotlib.pyplot as plt
import power1d


def test_hashable():
	'''
	Test basic hashing
	'''
	Q    = 365
	g0   = power1d.geom.GaussianPulse( Q , q=200 , fwhm=100 , amp=5 )
	g1   = power1d.geom.GaussianPulse( Q , q=200 , fwhm=100 , amp=5 )
	assert g0 == g1


def test_basic():
	'''
	Test basic syntax to ensure that no errors are generated
	'''
	# create geometry:
	np.random.seed(10)
	J = 5   # sample size
	Q = 101 # continuum size
	baseline = power1d.geom.Null( Q )
	signal0  = power1d.geom.Null( Q )
	signal1  = power1d.geom.GaussianPulse( Q , q = 60 , fwhm = 30, amp = 2.0 )
	noise    = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 1 , fwhm = 20 )
	model00  = power1d.models.DataSample( baseline , signal0 , noise , J = J )
	model01  = power1d.models.DataSample( baseline , signal0 , noise , J = J )
	model10  = power1d.models.DataSample( baseline , signal0 , noise , J = J )
	model11  = power1d.models.DataSample( baseline , signal1 , noise , J = J )
	teststat = power1d.stats.t_2sample_fn( J, J )
	emodel0  = power1d.models.Experiment( [model00,model01] , teststat )
	emodel1  = power1d.models.Experiment( [model10,model11] , teststat )
	# simulate experiments:
	sim       = power1d.ExperimentSimulator(emodel0, emodel1)
	results   = sim.simulate(1000, progress_bar=True)


def test_data():
	'''
	Test internal dataset loading
	'''
	data     = power1d.data.weather()  # load data
	y        = data['Continental']     # extract one region
	m        = y.mean( axis=0 )        # mean continuum
	baseline = power1d.geom.Continuum1D( m )
