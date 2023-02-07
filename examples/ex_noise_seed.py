
import numpy as np
import power1d





#(0) Construct primitives:
np.random.seed(0)
J     = 10
Q     = 101
noise = power1d.noise.Gaussian( J, Q, mu=0, sigma=0.1 )
print ( noise.value[ 0, 0 : 3 ] )

noise.random()
print ( noise.value[ 0, 0 : 3 ] )


np.random.seed(0)
noise.random()
print ( noise.value[ 0, 0 : 3 ] )
