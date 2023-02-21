'''
Canadian temperature data (Ramsay and Silverman, 2005)
'''


from math import log
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import power1d



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
plt.close('all')

### create figure:
plt.figure(figsize=(4,3))
### create axes:
ax       = plt.axes([0.12,0.18,0.86,0.8])
ax.plot(t, 'k-', label='Test statistic continuum')
ax.axhline(tstar_b, color='r', ls='-', label=r'Bonferroni threshold ($\alpha$=0.05)')
ax.axhline(tstar,   color='g', ls='--', label=r'RFT threshold ($\alpha$=0.05)')
ax.axhline(tstar_u, color='b', ls=':', label=r'Uncorrected threshold ($\alpha$=0.05)')


### legend:
ax.legend(fontsize=8, loc=(0.12,0.67))
ax.set_xlabel('Day', size=14)
ax.set_ylabel('t value', size=14)
plt.show()



