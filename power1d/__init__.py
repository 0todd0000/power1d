'''
power1d: numerical power estimates for one-dimensional continua

Pataky TC (in review) Power1D: A Python toolbox for numerical
power estimates in experiments involving one-dimensional
continua. PeerJ Computer Science.


Copyright (C) 2020  Todd Pataky
Version: 0.1.1 (2020/06/22)
'''


__version__ = '0.1.1 (2020/06/22)'


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





