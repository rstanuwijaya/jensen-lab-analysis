import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.core.numeric import full
# from scipy.optimize import curve_fit
import lmfit
import os
import time
import importlib
import pandas

from fit_model import *


class RunFit:
    def __init__(self, init_params: dict, fpath: bool, fitradius: int = 1024, test_fit: bool = True, debug: bool = True):
        self.init_params = init_params
        self.fpath = fpath
        self.fname = os.path.basename(fpath)
        self.fitradius = fitradius
        self.test_fit = test_fit
        self.debug = debug
        self.iter = 1

        self.read_file()
        self.init_fit()
        self.eval_fit()

    def read_file(self):
        full_xdata = [x for x in range(-n_tb, n_tb+1)]
        full_ydata = np.loadtxt(self.fpath)
        full_data = zip(full_xdata, full_ydata)
        self.xdata = [x for x in range(-self.fitradius, self.fitradius+1)]
        self.ydata = [y for x, y in full_data if x in self.xdata]
    
    def init_fit(self):
        self.fitmodel = FitModel(self.xdata, self.ydata, self.fitradius, self.init_params, self.fname)
        self.model = lmfit.Model(self.fitmodel.model)
        default_hint = {'value': 1, 'min': -math.inf, 'max': math.inf, 'vary': True, 'expr': None}
        for param_name in self.model.param_names:
            self.model.set_param_hint(param_name, 
                value=self.init_params.get(param_name, default_hint).get('value'),
                min=self.init_params.get(param_name, default_hint).get('min'),
                max=self.init_params.get(param_name, default_hint).get('max'),
                vary=self.init_params.get(param_name, default_hint).get('vary'),
                )
        self.params = self.model.make_params()
        # self.model.print_param_hints()
    
    def eval_fit(self):
        result = self.model.eval(self.params, x=1)
        return result
    
fpath = r'C:\Users\stnav\OneDrive - HKUST Connect\Academics\Jensen Lab\Experimental_Data\VarySync\Q=1200_0.4photon_19.7699MHz\analysis\time_coincidence_count\2674541_Frame2_Exp300_iter1.csv'
test = RunFit({}, fpath, fitradius=2)
