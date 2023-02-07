'''
power1d: numerical power estimates for one-dimensional continua

Pataky TC (2017) Power1D: A Python toolbox for numerical
power estimates in experiments involving one-dimensional
continua. PeerJ Computer Science 3: e125.
DOI: 10.7717/peerj-cs.125
https://peerj.com/articles/cs-125/


Copyright (C) 2023  Todd Pataky
Version: 0.1.5 (2023-02-07)
'''


__version__ = '0.1.6 (2023-02-07)'


__all__ = ['geom', 'models', 'noise', 'roi', 'stats']



from . import data
from . import geom
from . import models
from . import noise
from . import prob
from . import random
from . import roi
from . import stats


DataSample          = models.DataSample
Experiment          = models.Experiment
ExperimentSimulator = models.ExperimentSimulator
RegionOfInterest    = roi.RegionOfInterest





