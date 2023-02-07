
from scipy import stats
from matplotlib import pyplot






#(0) Set parameters:
alpha  = 0.05  #Type I error rate
power  = 0.80  #target power
y0     = 50.0  #initial value
sd     = 1.0   #standard deviation
pct    = 2.0   #percent increase



#(1) Compute power for a range of sample sizes:
J      = 2     #sample size
p      = 0.0   #computed power
JJ     = []    #saved sample sizes
PP     = []    #saved power values
while p < power:
	### derived parameters:
	df     = J - 1                 #degrees of freedom
	effect = 0.01 * pct * y0 / sd  #effect size ( sd units )
	delta  = effect * J ** 0.5     #non-centrality parameter
	### analytical power:
	u      = stats.t.isf( alpha , df )
	p      = stats.nct.sf( u , df , delta )
	### save results:
	JJ.append( J )
	PP.append( p )
	### update sample size:
	J += 1



#(2) Plot results:
pyplot.close('all')
pyplot.figure( figsize=(6,4) )
ax = pyplot.axes([0.15,0.15,0.8,0.8])
ax.plot(JJ, PP, 'o-', label='Actual power')
ax.axhline(power, color='k', ls='--', label='Target power')
ax.set_xlabel('Sample size', size=12)
ax.set_ylabel('Power', size=12)
ax.legend()
pyplot.show()








