
'''
Validate the inflated variance approach to continuum-level
power calculation.

Same as val_power_iv.py, but uses power1d's high-level
interface for numerical simulation.
'''

import numpy as np
import power1d






#(0) Set parameters:
J      = 20     #sample size
Q      = 101   #continuum size
W0     = 20.0  #continuum smoothness under the null hypothesis
### derived parameters:
df     = J-1   #degrees of freedom
### power parameters:
alpha  = 0.05  #Type I error rate
W1     = 10.0  #continuum smoothness under the alternative hypothesis
sigma  = 2.0   #effect size (as variance under the alternative hypothesis)






#(1) Theoretical power using inflated varince method:
'''
The probability of rejecting the null hypothesis when the alternative is true
is given as the probability that "sigma"-scaled random fields with smoothness
"Wstar" will exceed the threshold "ustar" where "Wstar" and "ustar" are
defined as indicated below (Friston et al. 1994)

Theoretical power can also be computed using the following convenience function:
>>> power0 = power1d.prob._power_Friston1994(alpha, df, Q, W0, W1, sigma)
'''
u      = power1d.prob.t_isf(alpha, df, Q, W0)    #critical threshold (under the null)
f      = float(W1) / W0                          #ratio of signal-to-noise smoothness
Wstar  = W0 * ( (1+sigma**2) / (1+(sigma**2)/(1+f**2)) )**0.5  #smoothness for the alternative
ustar  = u * ( 1 + sigma**2 )**(-0.5)            #threshold for the alternative hypothesis
power0 = power1d.prob.t_sf(ustar, df, Q, Wstar)  #theoretical power
print('Theoretical power:  %.05f' %power0)




#(2) Validate numerically:
np.random.seed(0)
baseline  = power1d.geom.Null(Q=Q)
signal    = power1d.geom.Null(Q=Q)
noise0    = power1d.noise.SmoothGaussian(Q=Q, sigma=1.0, fwhm=W0, J=J)
noise1    = power1d.noise.SmoothGaussian(Q=Q, sigma=1.0, fwhm=Wstar, J=J)
### data sample models:
model0    = power1d.models.DataSample(baseline, signal, noise0, J=J)
model1    = power1d.models.DataSample(baseline, signal, noise1, J=J)
### experiment models:
teststat  = power1d.stats.t_1sample_fn(J)
expmodel0 = power1d.Experiment(model0, teststat)
expmodel1 = power1d.Experiment(model1, teststat)
### simulate:
sim       = power1d.ExperimentSimulator(expmodel0, expmodel1)
results   = sim.simulate(1000, progress_bar=True)
power     = results.sf(ustar)
print('Simulated power:    %.05f' %power)





