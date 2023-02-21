
'''
This test script follows the online documentation:
https://spm1d.org/power1d/Examples/DataSample.html
'''

import os
import pytest
import numpy as np
import power1d



dir0         = os.path.dirname( __file__ )
dir_expected = os.path.join( dir0, '_expected', 'samplesize')



def test_manual():
	J          = 5    # sample size
	Q          = 101  # continuum size
	q          = 65   # signal location
	baseline   = power1d.geom.Null(Q=Q)
	signal0    = power1d.geom.Null(Q=Q)
	signal1    = power1d.geom.GaussianPulse(Q=101, q=q, amp=1.3, sigma=10)
	noise      = power1d.noise.Gaussian(J=5, Q=101, sigma=1)
	# create data sample models:
	model0     = power1d.models.DataSample(baseline, signal0, noise, J=J)  #null
	model1     = power1d.models.DataSample(baseline, signal1, noise, J=J)  #alternative
	# iteratively simulate for a range of sample sizes:
	np.random.seed(0)    #seed the random number generator
	JJ         = [5, 6, 7, 8, 9, 10]  #sample sizes
	tstat      = power1d.stats.t_1sample  #test statistic function
	emodel0    = power1d.models.Experiment(model0, tstat) # null
	emodel1    = power1d.models.Experiment(model1, tstat) # alternative
	sim        = power1d.ExperimentSimulator(emodel0, emodel1)
	# loop through the different sample sizes:
	power_omni = []
	power_coi  = []
	coir       = 3
	for J in JJ:
		emodel0.set_sample_size( J )
		emodel1.set_sample_size( J )
		results = sim.simulate( 1000 )
		results.set_coi( ( q , coir ) )  #create a COI at the signal location
		power_omni.append( results.p_reject1 )  #omnibus power
		power_coi.append( results.p_coi1[0] )   #coi power
	# load expected results:
	fpathNPZ  = os.path.join( dir_expected, 'manual.npz' )
	with np.load( fpathNPZ ) as z:
		po    = z['power_omni']
		pc    = z['power_coi']
	assert power_omni == pytest.approx( po, abs=1e-5 )
	assert power_coi  == pytest.approx( pc, abs=1e-5 )


test_manual()
