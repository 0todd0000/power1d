'''
This script replicates all figures and text results from the paper:

Pataky TC (in review) Power1D: A Python toolbox for numerical
power estimates in experiments involving one-dimensional
continua. PeerJ Computer Science.


!!! NOTES !!!

1. This script takes approximately 30 s to run.  It will report
   its progress to the terminal using Python's "print" command,
   by printing the figure or script number (from the paper)
   that it is currently excecuting.  It may appear to get hung
   on certain figures (especially Figures 14 & 15) while it
   runs numerical simulations in the background. If the script
   does not proceed to the next figure / script after about
   30 s then there may be problems and it may indeed be hung.


2. If using iPython and Python 3.X to run this script you
   may need to manually close each generated figure before
   the script moves on.


3. This script uses a small number of simulation iterations
   in order to efficiently generate all results. Since these
   results involve random number generation, the results
   generated by this script are not numerically identical
   to the results in the paper, which were generated using
   a larger number of simulation iterations. Nevertheless,
   the results are qualitatively identical so are directly
   comparable with no change in the paper's interpretations.
   To more precisely replicate the paper's results and
   to achieve closer agreement between simulated and
   theoretical results, increase the number of simulation
   iterations specified in the "sim.simulate" commands below.
   For example, in "Script 2" change the command:

   >>> results = sim.simulate( 500 , progress_bar = False )

   to:

   >>> results = sim.simulate( 10000 , progress_bar = True )



4. For convenience the start of this script lists boolean
   flags for each of the fifteen figures. Setting a figure's
   flag to True will cause it to be generated and setting it
   to False will skip all associated commands and simulations.
   Similarly, flags are provided for the paper's scripts
   (i.e., code chunks from the paper that do not produce
   figures) immediately below the figure flags.
'''

import time
time_beginning = time.time()



from math import log
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import power1d
plt.close('all')



# FIGURE and SCRIPT flags

figure1  = True
figure2  = True
figure3  = True
figure4  = True
figure5  = True

figure6  = True
figure7  = True
figure8  = True
figure9  = True
figure10 = True

figure11 = True
figure12 = True
figure13 = True
figure14 = True
figure15 = True

figure16 = True

script1  = True    # ( Page 9 -- noise control )
script2  = True    # ( Page 17 -- noncentral t )
script3  = True    # ( Pages 18-19 -- inflated variance method )
script4  = True    # ( Pages 20-21 -- noncentral t method )








#-----------------------------------------------
# Script 1  ( Noise (power1d.noise) -- noise control )
#-----------------------------------------------

if script1:
	print('Script 1     ( Noise (power1d.noise) -- noise control )')
	J = 10   # sample size
	Q = 101  # continuum size
	
	np.random.seed(0)
	noise = power1d.noise.Gaussian(J, Q, mu=0, sigma=1)
	print( noise.value[0,0:3] )
	
	noise.random()
	print( noise.value[0,0:3] )
	
	np.random.seed(0)
	noise.random()
	print( noise.value[0,0:3] )
	print('\n\n\n')



#-----------------------------------------------
# Script 2  ( Validations 0D -- noncentral t )
#-----------------------------------------------

if script2:
	print('Script 2     ( Validations 0D -- noncentral t )')
	alpha  = 0.05
	J      = 12
	effect = 0.8
	df     = J - 1
	delta  = effect * J**0.5
	# analytical power:
	u      = stats.t.isf(alpha, df)
	p      = stats.nct.sf(u, df, delta)
	print('Theoretical power: %.3f' %p)
	# numerically estimated power using power1d:
	np.random.seed(0)
	Q        = 2  #a short continuum
	baseline = power1d.geom.Null( Q )
	signal0  = power1d.geom.Null( Q )
	signal1  = power1d.geom.Constant( Q , amp = effect )
	noise    = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 1 )
	model0   = power1d.models.DataSample( baseline , signal0 , noise , J = J)
	model1   = power1d.models.DataSample( baseline , signal1 , noise , J = J)
	teststat = power1d.stats.t_1sample_fn( J )
	emodel0  = power1d.models.Experiment( model0 , teststat )
	emodel1  = power1d.models.Experiment( model1 , teststat )
	# simulate experiments:
	sim      = power1d.ExperimentSimulator( emodel0 , emodel1 )
	results  = sim.simulate( 500 , progress_bar = False )
	# create ROI
	roi      = np.array( [ True , False ] )
	results.set_roi( roi )
	print( 'Simulated power: %.3f' %results.p_reject1 )
	print('\n\n\n')
	


#-----------------------------------------------
# Script 3  ( Validations 1D -- inflated variance method )
#-----------------------------------------------

if script3:
	print('Script 3     ( Validations 1D -- inflated variance method )')
	
	#(0) Set parameters:
	J      = 20    #sample size
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
	print('Theoretical power:  %.3f' %power0)

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
	results   = sim.simulate(1000, progress_bar=False)
	power     = results.sf(ustar)
	print('Simulated power:    %.3f' %power)
	print('\n\n\n')







#-----------------------------------------------
# Script 4  ( Validations 1D  -- noncentral t method )
#-----------------------------------------------

if script4:
	print('Script 4     ( Validations 1D  -- noncentral t method )')
	
	#(0) Set parameters:
	J      = 8     # sample size
	Q      = 201   # continuum size
	W      = 40.0  # smoothness
	### derived parameters:
	df     = J - 1 # degrees of freedom
	### power parameters:
	alpha  = 0.05  # Type I error rate
	effect = 0.8   # effect size
	delta  = effect * J ** 0.5 # non-centrality parameter





	#(1) Theoretical power using on-central t method:
	zstar  = power1d.prob.t_isf(alpha, df, Q, W)          #critical threshold (under the null)
	power0 = power1d.prob.nct_sf(zstar, df, Q, W, delta)  #power
	print('Theoretical power:  %.3f' %power0)




	#(2) Validate numerically using power1d:
	np.random.seed(0)
	baseline = power1d.geom.Null( Q )
	signal0  = power1d.geom.Null( Q )
	signal1  = power1d.geom.Constant( Q , amp = effect )
	noise    = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 1 , fwhm = W )
	model0   = power1d.models.DataSample( baseline , signal0 , noise , J = J)
	model1   = power1d.models.DataSample( baseline , signal1 , noise , J = J)
	teststat = power1d.stats.t_1sample_fn( J )
	emodel0  = power1d.models.Experiment( model0 , teststat )
	emodel1  = power1d.models.Experiment( model1 , teststat )
	# simulate experiments:
	sim      = power1d.ExperimentSimulator( emodel0 , emodel1 )
	results  = sim.simulate( 1000 , progress_bar = False )
	print( 'Simulated power: %.3f' %results.p_reject1 )
	print('\n\n\n')











#-----------------------------------------------
# Figure 1
#-----------------------------------------------

if figure1:
	print('Figure 1...')
	
	# Load weather data:
	data     = power1d.data.weather()
	y0       = data['Arctic']
	y1       = data['Atlantic']
	y2       = data['Continental']
	y3       = data['Pacific']
	Y        = [y0, y1, y2, y3]
	plt.figure(figsize=(8,3))
	### create axes:
	ax0      = plt.axes([0.09,0.18,0.41,0.8])
	ax1      = plt.axes([0.57,0.18,0.41,0.8])
	AX       = [ax0,ax1]
	### plot all data:
	labels   = 'Arctic', 'Atlantic', 'Continental', 'Pacific'
	colors   = 'b', 'orange', 'g', 'r'
	ls       = '-', '--', ':', '-.'
	x        = np.arange(365)
	for y,label,c,lss in zip(Y,labels,colors,ls):
		h    = ax0.plot(y.T, color=c, lw=0.5, ls=lss)
		plt.setp(h[0], label=label)
		### plot mean and SD continua:
		m,s  = y.mean(axis=0), y.std(ddof=1, axis=0)
		ax1.plot(m, color=c, lw=3, ls=lss, label=label)
		ax1.fill_between(x, m-s, m+s, alpha=0.5)
	plt.setp(AX, ylim=(-35,25))
	### legend:
	ax1.legend(fontsize=10, loc=(0.35,0.03))
	[ax.set_xlabel('Day', size=14) for ax in AX]
	ax0.set_ylabel('Temperature  (deg C)', size=14)
	### panel labels:
	labels   = '(a)', '(b)'
	[ax.text(0.03, 0.91, s, size=12, transform=ax.transAxes)   for ax,s in zip(AX,labels)]
	plt.show()

	print('\n\n\n')







#-----------------------------------------------
# Figure 2
#-----------------------------------------------

if figure2:
	print('Figure 2...')
	
	def estimate_fwhm(R):
		'''
		Estimate field smoothness (FWHM) from a set of random fields or a set of residuals.
	
		:Parameters:

			*R* --- a set of random fields, or a set of residuals
		
		:Returns:

			*FWHM* --- the estimated FWHM
	
		Reference:

		Kiebel S, Poline J, Friston K, Holmes A, Worsley K (1999).
		Robust Smoothness Estimation in Statistical Parametric Maps Using Standardized
		Residuals from the General Linear Model. NeuroImage, 10(6), 756-766.
		'''
		eps    = np.finfo(float).eps
		ssq    = (R**2).sum(axis=0)
		dy,dx  = np.gradient(R)
		v      = (dx**2).sum(axis=0)
		v     /= (ssq + eps)
		i      = np.isnan(v)
		v      = v[np.logical_not(i)]
		reselsPerNode = np.sqrt(v / (4*log(2)))
		FWHM   = 1 / reselsPerNode.mean()        #global FWHM estimate:
		return FWHM

	#(0) Load weather data:
	data     = power1d.data.weather()
	y0       = data['Arctic']
	y1       = data['Atlantic']
	y2       = data['Continental']
	y3       = data['Pacific']
	Y        = [y0, y1, y2, y3]

	#(1) Conduct inference on two of the four geographic regions:
	yA,yB    = y1,y2   #Atlantic and Continental
	JA,JB    = yA.shape[0], yB.shape[0]                         #sample sizes
	mA,mB    = yA.mean(axis=0), yB.mean(axis=0)                 #sample means
	sA,sB    = yA.std(ddof=1, axis=0), yB.std(ddof=1, axis=0)   #sample SD
	s        = (   (  (JA-1)*sA*sA + (JB-1)*sB*sB  )  /  ( JA+JB-2 )   )**0.5   #pooled SD
	t        = (mA-mB) / s / (1.0/JA + 1.0/JB)**0.5             # t statistic
	### estimate smoothness:
	rA,rB    = yA - mA, yB - mB    #residuals
	fwhm     = estimate_fwhm( np.vstack( [rA, rB] ) )
	### compute critical threshold (assming sphericity):
	alpha    = 0.05           #Type I error rate
	df       = JA + JB - 2    #degrees of freedom
	Q        = yA.shape[1]    #continuum size
	tstar    = power1d.prob.t_isf(alpha, df, Q, fwhm)
	### compute uncorrected and Bonferroni thresholds for comparison:
	tstar_u  = stats.t.isf(alpha, df)   #uncorrected threshold
	pcrit    = 1 - (1-alpha)**(1.0/Q)
	tstar_b  = stats.t.isf(pcrit, df)   #Bonferroni threshold


	#(2) Plot:
	plt.figure('Figure 2', figsize=(4,3))
		### create axes:
	ax       = plt.axes([0.12,0.18,0.86,0.8])
	ax.plot(t, 'k-', label='Test statistic continuum')
	ax.axhline(tstar_b, color='r', ls='--', label=r'Bonferroni threshold ($\alpha$=0.05)')
	ax.axhline(tstar,   color='g', ls='--', label=r'RFT threshold ($\alpha$=0.05)')
	ax.axhline(tstar_u, color='b', ls='--', label=r'Uncorrected threshold ($\alpha$=0.05)')
	### legend:
	ax.legend(fontsize=8, loc=(0.12,0.67))
	ax.set_xlabel('Day', size=14)
	ax.set_ylabel('t value', size=14)
	plt.show()
	
	print('\n\n\n')
	
	


#-----------------------------------------------
# Figure 3
#-----------------------------------------------

if figure3:
	print('Figure 3...')
	
	#(0) Create models:
	np.random.seed(1)
	J,Q,fwhm  = 8, 101, 25
	### create baseline and noise for all models:
	baseline  = power1d.geom.Null(Q)
	noise     = power1d.noise.SmoothGaussian(J=J, Q=Q, sigma=1, fwhm=fwhm, pad=True)
	### model 0:  inflated variance:
	sIV,wIV   = 1.5, 10
	signal0   = power1d.noise.SmoothGaussian(J=1, Q=Q, sigma=sIV, fwhm=wIV, pad=False)
	### model 1:  NC-RFT
	signal1   = power1d.geom.Constant(Q=Q, amp=1.5)
	### model 2:  power1
	signal2   = power1d.geom.GaussianPulse(Q=Q, q=75, fwhm=50, amp=2.0)
	noise20   = power1d.noise.SmoothGaussian(J=J, Q=Q, sigma=1, fwhm=fwhm, pad=True)
	noise21   = power1d.noise.Gaussian(J=J, Q=Q, sigma=0.1)
	noise2    = power1d.noise.Additive(noise20, noise21)
	signal2n  = power1d.geom.Exponential(Q=Q, x0=0.2, x1=2.0, rate=3)
	noise2    = power1d.noise.Scaled(noise2, signal2n)
	noise2.random()
	noise2.random()
	noise2.random()



	# (1) Plot:
	fontname = u'DejaVu Sans'
	fig = plt.figure('Figure 3', figsize=(8,2.5))
	### create axes:
	axx = np.linspace(0.06,0.7,3)
	axy = 0.19
	axw = 0.29
	axh = 0.78
	AX  = [plt.axes([x, axy, axw, axh])  for x in axx]
	ax0,ax1,ax2  = AX
	[ax.set_yticklabels([])  for ax in [ax1,ax2]]
	[plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), size=8)  for ax in AX]
	### colors and line widths:
	cb,cn,cs = 'k', '0.4', (0.4,0.8,0.4)
	lws      = 4
	### plot model 0 (inflated variance):
	baseline.plot(ax=ax0, color=cb)
	noise.plot(ax=ax0, color=cn, lw=0.5)
	signal0.plot(ax=ax0, color=cs, lw=lws)
	### plot model 1 (ncrft):
	baseline.plot(ax=ax1, color=cb)
	noise.random()
	noise.plot(ax=ax1, color=cn, lw=0.5)
	signal1.plot(ax=ax1, color=cs, lw=lws)
	### plot model 2 (power1d):
	baseline.plot(ax=ax2, color=cb)
	noise2.plot(ax=ax2, color=cn, lw=0.5)
	signal2.plot(ax=ax2, color=cs, lw=lws)
	### adjust axes:
	plt.setp(AX, ylim=(-4,4))
	[ax.set_xlabel('Continuum position (%)', size=11, name=fontname)  for ax in AX]
	ax0.set_ylabel('Dependent variable', size=11, name=fontname)
	### legend:
	ax0.legend([ax0.lines[1], ax0.lines[-1]], ['Noise','Signal'])
	### panel labels:
	labels  = 'Inflated variance', 'Non-central RFT', 'Numerical'
	[ax.text(0.05,0.9, '(%s)  %s'%(chr(97+i),s), name=fontname, transform=ax.transAxes, size=11)  for i,(ax,s) in enumerate(zip(AX,labels))]
	plt.show()

	print('\n\n\n')




#-----------------------------------------------
# Figure 4
#-----------------------------------------------

if figure4:
	print('Figure 4...')
	
	# create geometry:
	Q = 101
	y = power1d.geom.GaussianPulse(Q,q=60,fwhm=20,amp=3.2)
	# plot:
	plt.figure('Figure 4', figsize=(4,3))
	ax = plt.axes([0.12,0.18,0.86,0.8])
	y.plot()
	ax.set_xlabel('Continuum position')
	plt.show()

	print('\n\n\n')
	
	
	

#-----------------------------------------------
# Figure 4
#-----------------------------------------------

if figure5:
	print('Figure 5...')
	
	np.random.seed(0)
	Q         = 101
	continua  = []
	continua.append(   power1d.geom.Continuum1D( np.random.randn(Q) )   )
	continua.append(   power1d.geom.Constant(Q, 1.3)   )
	continua.append(   power1d.geom.Exponential(Q, x0=0.2, x1=2.3, rate=5)   )
	continua.append(   power1d.geom.ExponentialSaw(Q, x0=0, x1=30.5, rate=10, cutoff=75)   )
	continua.append(   power1d.geom.GaussianPulse(Q, q=60, sigma=None, fwhm=20, amp=3.2)   )
	continua.append(   power1d.geom.Linear(Q, x0=0, x1=3.5, slope=None)   )
	continua.append(   power1d.geom.Null(Q)   )
	continua.append(   power1d.geom.SawPulse(Q, q0=50, q1=80, x0=0, x1=2.5)   )
	continua.append(   power1d.geom.SawTooth(Q, q0=3, q1=13, x0=0, x1=2.5, dq=3)   )
	continua.append(   power1d.geom.Sigmoid(Q, q0=40, q1=80, x0=-1, x1=2.5)   )
	continua.append(   power1d.geom.Sinusoid(Q, q0=0, amp=1, hz=2)   )
	continua.append(   power1d.geom.SquarePulse(Q, q0=40, q1=60, x0=-0.5, x1=2.2)   )
	continua.append(   power1d.geom.SquareTooth(Q, q0=5, q1=18, x0=-1.2, x1=2.7, dq=8)   )
	continua.append(   power1d.geom.TrianglePulse(Q, q0=60, q1=85, x0=-1, x1=3)   )
	continua.append(   power1d.geom.TriangleTooth(Q, q0=20, q1=35, x0=-1, x1=2.5, dq=10)   )

	#(1) Plot:
	plt.figure('Figure 5', figsize=(12,6))
	axx    = np.linspace(0.03, 0.82, 5)
	axy    = np.linspace(0.71, 0.04, 3)
	axw    = 0.17
	axh    = 0.26
	i      = 0
	AX     = []
	for axyy in axy:
		for axxx in axx:
			ax = plt.axes( [ axxx, axyy, axw, axh ] )
			c  = continua[i]
			c.plot(ax=ax)
			ax.text(0.05, 0.9, '%s' %c.__class__.__name__, size=13, transform=ax.transAxes, bbox=dict(facecolor='w', alpha=0.5))
			AX.append(ax)
			i += 1
	plt.setp(AX, ylim=(-2.5,4))
	plt.show()

	print('\n\n\n')






#-----------------------------------------------
# Figure 6
#-----------------------------------------------

if figure6:
	print('Figure 6...')
	
	# create geometry:
	Q  = 101
	y0 = power1d.geom.GaussianPulse(Q,q=40,fwhm=60,amp=1)
	y1 = power1d.geom.Sinusoid(Q,amp=1,hz=2)
	# use operators to create more complex geometries:
	yA = y0 + y1
	yB = y0 * y1
	yC = y0 ** y1
	# plot:
	plt.figure('Figure 6', figsize=(8,3))
	ax0      = plt.axes([0.09,0.18,0.41,0.8])
	ax1      = plt.axes([0.57,0.18,0.41,0.8])
	AX       = [ax0,ax1]
	y0.plot(ax=ax0, color='b', label='y0')
	y1.plot(ax=ax0, color='g', label='y1')
	ax0.legend()
	### plot derived geometries:
	colors   = 'k', 'r', 'orange'
	yA.plot(ax=ax1, label='yA = y0 + y1', color=colors[0])
	yB.plot(ax=ax1, label='yB = y0 * y1', color=colors[1])
	yC.plot(ax=ax1, label='yC = y0 ** y1', color=colors[2])
	### legend and labels:
	ax1.legend(loc=(0.15,0.7))
	[ax.set_xlabel('Continuum position', size=12) for ax in AX]
	labels   = '(a)', '(b)'
	[ax.text(0.03, 0.91, s, size=12, transform=ax.transAxes)   for ax,s in zip(AX,labels)]
	plt.show()
	
	print('\n\n\n')
	




#-----------------------------------------------
# Figure 7
#-----------------------------------------------

if figure7:
	print('Figure 7...')
	
	np.random.seed(0)
	J = 8   # sample size
	Q = 101 # continuum size
	noise0 = power1d.noise.Gaussian( J, Q , mu = 0 , sigma = 1 )
	noise1 = power1d.noise.SmoothGaussian( J, Q , mu = 0 , sigma = 1 , fwhm = 20 , pad = True )

	# plot:
	plt.figure('Figure 7', figsize=(8,3))
	### create axes:
	ax0      = plt.axes([0.09,0.18,0.41,0.8])
	ax1      = plt.axes([0.57,0.18,0.41,0.8])
	AX       = [ax0,ax1]
	### plot:
	noise0.plot(ax=ax0, color='k', lw=0.5)
	noise1.plot(ax=ax1, color='k', lw=0.5)
	plt.setp(AX, ylim=(-3.5, 3.5))
	[ax.set_xlabel('Continuum position', size=12)  for ax in AX]
	labels   = '(a)', '(b)'
	[ax.text(0.03, 0.91, s, size=12, transform=ax.transAxes)   for ax,s in zip(AX,labels)]
	plt.show()
	
	print('\n\n\n')
	




#-----------------------------------------------
# Figure 8
#-----------------------------------------------

if figure8:
	print('Figure 8...')
	
	np.random.seed(1)
	J,Q       = 10,101   # sample and continuum sizes
	noise     = []
	### basic types:
	noise.append(   power1d.noise.ConstantUniform( J, Q, x0=-2, x1=2 )   )
	noise.append(   power1d.noise.ConstantGaussian( J, Q, mu=0, sigma=1 )   )
	noise.append(   power1d.noise.Uniform( J, Q, x0=-2, x1=2 )   )
	noise.append(   power1d.noise.Gaussian( J, Q, mu=0, sigma=1 )   )
	noise.append(   power1d.noise.Skewed( J, Q, mu=0, sigma=1, alpha=3 )   )
	noise.append(   power1d.noise.SmoothGaussian( J, Q, mu=0, sigma=1, fwhm=20, pad=False )   )
	noise.append(   power1d.noise.SmoothSkewed( J, Q, mu=0, sigma=1, alpha=3, fwhm=20, pad=False )   )
	### compound type (Additive):
	noise0    = power1d.noise.Gaussian( J, Q, mu=0, sigma=0.1 )
	noise1    = power1d.noise.SmoothGaussian( J, Q, mu=0, sigma=1.5, fwhm=40, pad=False )
	noise.append(   power1d.noise.Additive( noise0, noise1 )    )
	### compound type (Mixture):
	noise0    = power1d.noise.Gaussian( 2, Q, mu=0, sigma=0.5 )
	noise1    = power1d.noise.SmoothGaussian( 40, Q, mu=2.0, sigma=0.5, fwhm=20, pad=False )
	noise.append(   power1d.noise.Mixture( noise0, noise1 )    )
	### compound type (Scaled):
	noise0    = power1d.noise.Gaussian( J, Q, mu=0, sigma=1 )
	scale     = np.linspace(0, 2, Q)
	noise.append(   power1d.noise.Scaled( noise0, scale )   )
	### compound type (SignalDependent):
	noise0    = power1d.noise.Gaussian(J=J, Q=101, sigma=0.2)
	signal    = power1d.geom.GaussianPulse(Q=101, q=65, amp=2.5, fwhm=20)
	fn        = lambda nvalue,svalue: nvalue + nvalue*svalue**2
	noise.append(   power1d.noise.SignalDependent(noise0, signal, fn=fn)   )


	#(1) Plot:
	plt.figure('Figure 8', figsize=(12,6))
	axx    = np.linspace(0.03, 0.82, 5)
	axy    = np.linspace(0.71, 0.04, 3)
	axw    = 0.17
	axh    = 0.26
	ind    = -1
	i      = -1
	AX     = []
	IND    = [0,1,2,3,4, 5,6,  10,11,12,13]
	for axyy in axy:
		for axxx in axx:
			ind += 1
			if ind not in IND:
				continue
			i   += 1
			ax = plt.axes( [ axxx, axyy, axw, axh ] )
			c  = noise[i]
			c.plot(ax=ax, color='k', lw=0.5)
			ax.text(0.05, 0.9, '%s' %c.__class__.__name__, size=13, transform=ax.transAxes, bbox=dict(facecolor='w', alpha=0.5))
			AX.append(ax)
	plt.setp(AX, ylim=(-3,5))
	plt.show()
	
	print('\n\n\n')







#-----------------------------------------------
# Figure 9
#-----------------------------------------------

if figure9:
	print('Figure 9...')
	
	np.random.seed(10)
	J = 8   # sample size
	Q = 101 # continuum size
	baseline = power1d.geom.Null( Q )
	signal   = power1d.geom.GaussianPulse( Q , q = 60 , fwhm = 30, amp = 8 )
	noise    = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 1 , fwhm = 20 )
	model    = power1d.models.DataSample( baseline , signal , noise)

	# plot:
	plt.figure('Figure 9', figsize=(8,3))
	### create axes:
	ax0      = plt.axes([0.09,0.18,0.41,0.8])
	ax1      = plt.axes([0.57,0.18,0.41,0.8])
	AX       = [ax0,ax1]
	### plot:
	model.plot( ax = ax0 )
	model.random( )
	model.plot( ax = ax1 )
	plt.setp(AX, ylim=(-2.5, 9.5))
	[ax.set_xlabel('Continuum position', size=12)  for ax in AX]
	labels   = '(a)', '(b)'
	[ax.text(0.03, 0.91, s, size=12, transform=ax.transAxes)   for ax,s in zip(AX,labels)]
	plt.show()
	
	print('\n\n\n')






#-----------------------------------------------
# Figure 10
#-----------------------------------------------

if figure10:
	print('Figure 10...')
	
	data     = power1d.data.weather()
	y        = data['Continental']
	
	np.random.seed(5)
	J        = 8  # sample size
	Q        = 365  # continuum size
	baseline = power1d.geom.Continuum1D( y.mean(axis=0) )
	signal   = power1d.geom.Null( Q )
	noise0   = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 0.3 )
	noise1   = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 3 , fwhm = 70 )
	noise    = power1d.noise.Additive( noise0 , noise1 )
	model    = power1d.models.DataSample( baseline , signal , noise , J = J)

	# plot:
	plt.figure('Figure 10', figsize=(8,3))
	### create axes:
	ax0      = plt.axes([0.09,0.18,0.41,0.8])
	ax1      = plt.axes([0.57,0.18,0.41,0.8])
	AX       = [ax0,ax1]
	### plot:
	model.plot( ax = ax0 )
	model.random( )
	model.plot( ax = ax1 )
	# plt.setp(AX, ylim=(-2.5, 9.5))
	[ax.set_xlabel('Continuum position', size=12)  for ax in AX]
	labels   = '(a)', '(b)'
	[ax.text(0.03, 0.91, s, size=12, transform=ax.transAxes)   for ax,s in zip(AX,labels)]
	plt.show()
	
	print('\n\n\n')





#-----------------------------------------------
# Figure 11
#-----------------------------------------------

if figure11:
	print('Figure 11...')
	
	np.random.seed(2)
	J = 8   # sample size
	Q = 101 # continuum size
	baseline = power1d.geom.Null( Q )
	signal   = power1d.geom.GaussianPulse( Q , q = 60 , fwhm = 30, amp = 3 )
	noise    = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 1 , fwhm = 20 )
	model    = power1d.models.DataSample( baseline , signal , noise , J = J )
	# experiment model:
	teststat = power1d.stats.t_1sample
	emodel   = power1d.models.Experiment( model , teststat )
	emodel.simulate( 50 )

	# plot:
	plt.figure('Figure 11', figsize=(4,3))
	### create axes:
	ax       = plt.axes([0.14,0.18,0.84,0.8])
	ax.plot( emodel.Z.T, color='k', lw=0.5 )
	ax.set_xlabel('Continuum position', size=12)
	ax.set_ylabel('Test statistic value', size=12)
	plt.show()
	
	print('\n\n\n')
	







#-----------------------------------------------
# Figure 12
#-----------------------------------------------

if figure12:
	print('Figure 12...')
	
	np.random.seed(10)
	J = 5   # sample size
	Q = 101 # continuum size
	baseline = power1d.geom.Null( Q )
	signal0  = power1d.geom.Null( Q )
	signal1  = power1d.geom.GaussianPulse( Q , q = 60 , fwhm = 30, amp = 2.0 )
	noise    = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 1 )
	model0   = power1d.models.DataSample( baseline , signal0 , noise , J = J )
	model1   = power1d.models.DataSample( baseline , signal1 , noise , J = J )
	# experiment model:
	teststat = power1d.stats.t_1sample_fn( J )
	emodel0  = power1d.models.Experiment( model0 , teststat )
	emodel1  = power1d.models.Experiment( model1 , teststat )
	# simulate:
	sim      = power1d.ExperimentSimulator(emodel0, emodel1)
	print('Figure 12')
	results   = sim.simulate(1000, progress_bar=True)
	print('\n\n\n')
	# plot:
	plt.figure('Figure 12', figsize=(8,6))
	results.plot()
	plt.show()
	
	print('\n\n\n')





#-----------------------------------------------
# Figure 13
#-----------------------------------------------

if figure13:
	print('Figure 13...')
	
	np.random.seed(0)
	data     = power1d.data.weather()
	y        = data['Continental']
	J        = 8   # sample size
	Q        = 365  # continuum size
	baseline = power1d.geom.Continuum1D( y.mean(axis=0) )
	signal0  = power1d.geom.Null( Q )
	signal1  = power1d.geom.GaussianPulse( Q , q = 200 , amp = 6 , fwhm = 100 )
	noise0   = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 0.3 )
	noise1   = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 5 , fwhm = 70 )
	noise    = power1d.noise.Additive( noise0 , noise1 )
	model0   = power1d.models.DataSample( baseline , signal0 , noise , J = J)
	model1   = power1d.models.DataSample( baseline , signal1 , noise , J = J)
	teststat = power1d.stats.t_1sample_fn( J )
	emodel0  = power1d.models.Experiment( model0 , teststat )
	emodel1  = power1d.models.Experiment( model1 , teststat )
	# simulate experiments:
	sim       = power1d.ExperimentSimulator(emodel0, emodel1)
	results   = sim.simulate(200, progress_bar=True)
	# set COI radius
	results.set_coi_radius(50)
	# set ROI:
	roi       = np.array( [False]*Q )
	roi[150:250] = True
	results.set_roi( roi )
	# plot:
	plt.figure('Figure 13', figsize=(8,6))
	results.plot()
	plt.show()
	
	print('\n\n\n')




#-----------------------------------------------
# Figure 14
#-----------------------------------------------

if figure14:
	print('Figure 14...')
	
	alpha   = 0.05
	JJ      = [5, 10, 25]    #sample size
	effects = np.arange( 0.2 , 1.01, 0.1 )
	
	# Estimate power:
	np.random.seed(0)
	Q        = 2  # a short continuum
	baseline = power1d.geom.Null( Q )
	signal0  = power1d.geom.Null( Q )
	PP0,PP1  = [],[]  #analytical and numerical power results
	for J in JJ:
		noise    = power1d.noise.Gaussian( J , Q , mu = 0 , sigma = 1 )
		model0   = power1d.models.DataSample( baseline , signal0 , noise , J = J)
		teststat = power1d.stats.t_1sample_fn( J )
		emodel0  = power1d.models.Experiment( model0 , teststat )
		P0,P1    = [] , []
		for effect in effects:
			signal1  = power1d.geom.Constant( Q , amp = effect )
			model1   = power1d.models.DataSample( baseline , signal1 , noise , J = J)
			emodel1  = power1d.models.Experiment( model1 , teststat )
			# simulate experiments:
			sim      = power1d.ExperimentSimulator( emodel0 , emodel1 )
			results  = sim.simulate( 200 , progress_bar = False )
			# create ROI
			roi      = np.array( [ True , False ] )
			results.set_roi( roi )
			P1.append( results.p_reject1 )


			#(1) Analytical power:
			df     = J - 1   #degrees of freedom
			u      = stats.t.isf( alpha , df )
			delta  = effect * J ** 0.5   #non-centrality parameter
			p0     = stats.nct.sf( u , df , delta )
			P0.append( p0 )
		PP0.append(P0)
		PP1.append(P1)


	#(2) Plot:
	plt.figure('Figure 14', figsize=(4,3))
	### create axes:
	ax       = plt.axes([0.14,0.18,0.84,0.8])
	colors   = 'b', 'g', 'r'
	for c,P0,P1,J in zip( colors, PP0, PP1 , JJ ):
		ax.plot( effects , P0 , '-', color=c , label='J = %d' %J )
		ax.plot( effects , P1 , 'o', color=c )
		ax.set_xlabel('Effect size', size=12)
		ax.set_ylabel('Power', size=12)
	ax.legend()
	plt.show()
	
	print('\n\n\n')





#-----------------------------------------------
# Figure 15
#-----------------------------------------------

if figure15:
	print('Figure 15...')
	
	#(0) Set parameters:
	alpha   = 0.05  # Type I error rate
	effects = np.arange( 0.1 , 0.71, 0.1 ) # effect size
	JJ      = [5, 10, 25]     # sample sizes
	Q       = 201   # continuum size
	WW0     = [10.0, 20.0, 50.0]  # smoothness values (under the null)
	### power parameters:
	alpha   = 0.05  #Type I error rate
	W1      = 30.0  #continuum smoothness under the alternative hypothesis
	sigmas  = np.arange(0.5, 2.01, 0.1)   #effect size (as variance under the alternative hypothesis)



	#(1) Theoretical and numerically estimated powers:
	np.random.seed(0)
	PP0,PP1 = [],[]
	for W0 in WW0:
		P0,P1 = [],[]
		for J in JJ:
			p0,p1 = [],[]
			for sigma in sigmas:
				df     = J - 1 # degrees of freedom

				### analytical power:
				u      = power1d.prob.t_isf(alpha, df, Q, W0)    #critical threshold (under the null)
				f      = float(W1) / W0                          #ratio of signal-to-noise smoothness
				Wstar  = W0 * ( (1+sigma**2) / (1+(sigma**2)/(1+f**2)) )**0.5  #smoothness for the alternative
				ustar  = u * ( 1 + sigma**2 )**(-0.5)            #threshold for the alternative hypothesis
				pp0    = power1d.prob.t_sf(ustar, df, Q, Wstar)  #theoretical power

				#(2) Validate numerically using power1d:
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
				results   = sim.simulate(50, progress_bar=False)
				pp1       = results.sf(ustar)
				p0.append( pp0 )
				p1.append( pp1 )
			P0.append( p0 )
			P1.append( p1 )
		PP0.append( P0 )
		PP1.append( P1 )
	PP0,PP1 = np.array(PP0), np.array(PP1)

	#(2) Plot:
	plt.figure('Figure 15', figsize=(12,3))
	### create axes:
	axx    = np.linspace(0.05, 0.71, 3)
	axy    = 0.16
	axw    = 0.28
	axh    = 0.83
	AX     = [plt.axes([x,axy,axw,axh])  for x in axx]
	ax0,ax1,ax2 = AX
	colors   = 'b', 'g', 'r'
	for ax,W,P0,P1 in zip(AX,WW0,PP0,PP1):
		for p0,p1,c,J in zip( P0, P1 , colors, JJ ):
			ax.plot( sigmas , p0 , '-', color=c , label='J = %d' %J )
			ax.plot( sigmas , p1 , 'o', color=c )
			ax.set_xlabel(r'Effect size  ($\sigma$)', size=12)
	ax0.set_ylabel('Power', size=12)
	ax0.legend( loc=(0.01,0.5) )
	plt.setp(AX, ylim=(0, 0.75))
	### panel labels:
	[ax.text(0.05, 0.9, '(%s)  FWHM = %d'%(chr(97+i), W), transform=ax.transAxes, size=12)  for i,(ax,W) in enumerate( zip(AX,WW0) ) ]
	plt.show()
	
	print('\n\n\n')





#-----------------------------------------------
# Figure 16
#-----------------------------------------------

if figure16:
	print('Figure 16...')
	
	#(0) Set parameters:
	JJ     = [5, 10, 25]     # sample sizes
	Q      = 201   # continuum size
	WW     = [10.0, 20.0, 50.0]  # smoothness values
	### power parameters:
	alpha  = 0.05  # Type I error rate
	effects = np.arange( 0.1 , 0.71, 0.1 ) # effect size

	#(1) Theoretical and numerically estimated powers:
	np.random.seed(0)
	PP0,PP1 = [],[]
	for W in WW:
		P0,P1 = [],[]
		for J in JJ:
			p0,p1 = [],[]
			for effect in effects:
				df     = J - 1 # degrees of freedom
				delta  = effect * J ** 0.5 # non-centrality parameter

				# analyical:
				zstar  = power1d.prob.t_isf(alpha, df, Q, W)          #critical threshold (under the null)
				pp0    = power1d.prob.nct_sf(zstar, df, Q, W, delta)  #power

				#(2) Validate numerically using power1d:
				baseline = power1d.geom.Null( Q )
				signal0  = power1d.geom.Null( Q )
				signal1  = power1d.geom.Constant( Q , amp = effect )
				noise    = power1d.noise.SmoothGaussian( J , Q , mu = 0 , sigma = 1 , fwhm = W )
				model0   = power1d.models.DataSample( baseline , signal0 , noise , J = J)
				model1   = power1d.models.DataSample( baseline , signal1 , noise , J = J)
				teststat = power1d.stats.t_1sample_fn( J )
				emodel0  = power1d.models.Experiment( model0 , teststat )
				emodel1  = power1d.models.Experiment( model1 , teststat )
				# simulate experiments:
				sim      = power1d.ExperimentSimulator( emodel0 , emodel1 )
				results  = sim.simulate( 50 , progress_bar = False )
				pp1      = results.p_reject1
				p0.append( pp0 )
				p1.append( pp1 )
			P0.append( p0 )
			P1.append( p1 )
		PP0.append( P0 )
		PP1.append( P1 )


	#(2) Plot:
	plt.figure('Figure 16', figsize=(12,3))
	### create axes:
	axx    = np.linspace(0.05, 0.71, 3)
	axy    = 0.16
	axw    = 0.28
	axh    = 0.83
	AX     = [plt.axes([x,axy,axw,axh])  for x in axx]
	ax0,ax1,ax2 = AX
	colors   = 'b', 'g', 'r'
	for ax,W,P0,P1 in zip(AX,WW,PP0,PP1):
		for p0,p1,c,J in zip( P0, P1 , colors, JJ ):
			ax.plot( effects , p0 , '-', color=c , label='J = %d' %J )
			ax.plot( effects , p1 , 'o', color=c )
			ax.set_xlabel('Effect size', size=12)
	ax0.set_ylabel('Power', size=12)
	ax0.legend( loc=(0.01,0.5) )
	plt.setp(AX, ylim=(0.1, 1.1))
	### panel labels:
	[ax.text(0.05, 0.9, '(%s)  FWHM = %d'%(chr(97+i), W), transform=ax.transAxes, size=12)  for i,(ax,W) in enumerate( zip(AX,WW) ) ]
	plt.show()
	
	print('\n\n\n')
	
	
	

print('Total elpased time: %d s' %(time.time() - time_beginning))

