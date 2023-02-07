
from scipy import stats




#(0) Set parameters:
alpha  = 0.05  #Type I error rate
J      = 8     #sample size
y0     = 50.0  #initial value
sd     = 1.0   #standard deviation
pct    = 2.0   #percent increase
### derived parameters:
df     = J - 1                 #degrees of freedom
effect = 0.01 * pct * y0 / sd  #effect size ( sd units )
delta  = effect * J ** 0.5     #non-centrality parameter



#(1) Analytical power:
u      = stats.t.isf( alpha , df )
p0     = stats.nct.sf( u , df , delta )
print( 'Analytical power: %.3f' %p0 )








