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
class RunFit:
    def __init__(self, init_params: dict, fpath: bool, fitradius: tuple = (-1024, 1024), test_fit: bool = True, debug: bool = True):
        self.init_params = init_params
        self.fpath = fpath
        self.fname = os.path.basename(fpath)
        self.min_x = fitradius[0]
        self.max_x = fitradius[1]
        self.test_fit = test_fit
        self.debug = debug
        self.iter = 1

        self.read_file()
        self.init_fit()

    def read_file(self):
        full_xdata = [x for x in range(-n_tb, n_tb+1)]
        full_ydata = np.loadtxt(self.fpath)
        full_data = zip(full_xdata, full_ydata)
        self.xdata = [x for x in range(self.min_x, self.max_x+1)]
        self.ydata = [y for x, y in full_data if x in self.xdata]
        self.fitmodel = FitModel(self.xdata, self.ydata, (self.min_x, self.max_x), self.init_params, self.fname)
    
    def init_fit(self):
        self.model = lmfit.Model(self.fitmodel.model, name=self.fname)
        default_hint = {'value': 1, 'min': -math.inf, 'max': math.inf, 'vary': False, 'expr': None}
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
        self.result = self.model.fit(self.ydata, self.params, x=self.xdata)
        print(self.result.fit_report())
        plt.figure(figsize=(20,30))
        self.result.plot_fit(datafmt='o')
        plt.show()
        self.fit_params = self.result.params
        return self.result

    def save_fit(self, savedirpath=''):
        data = np.array([self.ydata, self.result.best_fit]).transpose()
        savepath = os.path.join(savedirpath, self.fname)
        print('savepath', savepath)
        np.savetxt(savepath, data)

        report = self.result.fit_report()
        with open(savepath.replace('.csv', '_report.txt'), 'w') as fh:
            fh.write(report)
