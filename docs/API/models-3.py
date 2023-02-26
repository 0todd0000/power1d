import matplotlib.pyplot as plt
import power1d

J        = 10
Q        = 101
baseline = power1d.geom.Null( Q=Q )
signal0  = power1d.geom.Null( Q=Q )
signal1  = power1d.geom.GaussianPulse( Q=Q, q=40, fwhm=15, amp=3.5 )
noise    = power1d.noise.Gaussian( J=J, Q=Q, mu=0, sigma=1.0 )
model0   = power1d.models.DataSample(baseline, signal0, noise, J=J)
model1   = power1d.models.DataSample(baseline, signal1, noise, J=J)
emodel   = power1d.models.Experiment( [model0, model1], fn=power1d.stats.t_2sample )
plt.close('all')
emodel.plot()