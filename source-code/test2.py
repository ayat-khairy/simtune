from scipy import *  # or from NumPy import *

import sys
import numpy as np
import kernels.priors
import numpy as np
import scipy.stats as sps
from operator import add
import weave
#import scipy.stats as sps
import scipy.special as spe
import logging


a = ones((512,512), 'float64')
b = ones((512,512), 'float64')
# ...do some stuff to fill in b...
# now average
a[1:-1,1:-1] =  (b[1:-1,1:-1] + b[2:,1:-1] + b[:-2,1:-1] \
               + b[1:-1,2:] + b[1:-1,:-2]) / 5.

import weave
expr = "a[1:-1,1:-1] =  (b[1:-1,1:-1] + b[2:,1:-1] + b[:-2,1:-1]" \
                      "+ b[1:-1,2:] + b[1:-1,:-2]) / 5."
weave.blitz(expr)
print ("Done!")
