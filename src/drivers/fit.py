
#%%
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import time
import importlib

from fitmodel import *

try:
    importlib.reload(fit_model)
except Exception as exc:
    print('Not reloading')
    print(exc)

init_params = {
    'm' : (2.2, 'free'), 
    'delta_T' : (14.6, 'free'), 
    'tw' : (3, 'free'), 
    'tau0' : (8.370690225587594e-05, 'fixed'), 
    'A' : (200, 'free'), 
    'b' : (0, 'fixed'),
    'tsink' : (922.6071057715765, 'fixed')
}