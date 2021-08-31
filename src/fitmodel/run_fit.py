#%%
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

consts = {
    'FrameTime': 2e-6,
    'LaserPeriod': 1/20.07e6,
    'TimeBinLength': 54.32e-12
}

var = {
    'CameraPeriod': None
}

config = {
    'FitRadius': 1024,
}

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

    def read_file(self):
        full_xdata = [x for x in range(-n_tb, n_tb+1)]
        full_ydata = np.loadtxt(self.fpath)
        full_data = zip(full_xdata, full_ydata)
        self.xdata = [x for x in range(-self.fitradius, self.fitradius+1)]
        self.ydata = [y for x, y in full_data if x in self.xdata]
        self.fitmodel = FitModel(self.xdata, self.ydata, self.fitradius, self.init_params, self.fname)
    
    def init_fit(self):
        self.model = lmfit.Model(self.fitmodel.model, name=self.fname)
        default_hint = {'value': 1, 'min': -math.inf, 'max': math.inf, 'vary': True, 'expr': None}
        for param_name in self.model.param_names:
            self.model.set_param_hint(param_name, 
                value=self.init_params.get(param_name, default_hint).get('value', default_hint['value']),
                min=self.init_params.get(param_name, default_hint).get('min', default_hint['min']),
                max=self.init_params.get(param_name, default_hint).get('max', default_hint['max']),
                vary=self.init_params.get(param_name, default_hint).get('vary', default_hint['vary']),
                )
        self.params = self.model.make_params()
    
    def eval_fit(self):
        result = self.model.eval(self.params, x=self.xdata)
        return result
    
    def run_fit(self):
        result = self.model.fit(self.ydata, self.params, x=self.xdata)
        print(result.fit_report())
        result.plot_fit()
        plt.show()
        return result

# test = RunFit(init_params, fpath, fitradius=900)
dirpath = '/mnt/e/Data/JensenLab/VarySyncFine/analysis/data_time_coincidence_count'
fpaths = [os.path.join(dirpath, fname) for fname in os.listdir(dirpath)]
fpaths = reversed(sorted(fpaths))
model_result = None
for fpath in fpaths:
    freq = os.path.basename(fpath).split('_')[0].replace('Freq', '')
    # if freq not in ('19940', '19920', '19960'): continue
    var['CameraPeriod'] = 1/(int(freq)*1e3)
    init_params = {
        'N': {
            'value': consts['FrameTime']/var['CameraPeriod'],
            'vary': False,
        },
        'm': {
            'value': 1.3,
            'vary': True,
        },
        'delta_T': {
            'value': (var['CameraPeriod']-consts['LaserPeriod'])/consts['TimeBinLength'],
            'vary' : False,
        },
        'tw': {
            'value': 8,
            'vary': False,
        },
        'tau0': {
            'value': 0,
            'vary': False,
        },
        'A': {
            'value': 169,
            'vary': True,
        },
        'b': {
            'value': 11,
            'vary': True,
        },
        'Z': {
            'value': 1024*var['CameraPeriod']/55e-9,
            'vary': False,
        },
    }
    if model_result:
        init_params['m']['value'] = model_result.params['m'].value
        init_params['A']['value'] = model_result.params['A'].value
        init_params['b']['value'] = model_result.params['b'].value
    
    model = RunFit(init_params, fpath, fitradius=config['FitRadius'])
    model_result = model.run_fit()
# %%
