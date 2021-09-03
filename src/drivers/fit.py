
#%%
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import time

print(__name__)
from fitmodel import RunFit


consts = {
    'FrameTime': 2e-6,
    'LaserPeriod': 1/20.07e6,
    'TimeBinLength': 54.32e-12
}

var = {
    'CameraPeriod': None
}

config = {
    'FitRadius': (-1024, 1024),
}
# test = RunFit(init_params, fpath, fitradius=900)
dirpath = '/mnt/e/Data/JensenLab/VarySyncFine/analysis/data_time_coincidence_count'
fpaths = [os.path.join(dirpath, fname) for fname in os.listdir(dirpath)]
fpaths = (sorted(fpaths))
model_result = None
for fpath in fpaths:
    CameraFreq = os.path.basename(fpath).split('_')[0].replace('Freq', '')
    CameraFreq = int(CameraFreq)
    # if CameraFreq % 100 != 0: continue
    if CameraFreq not in (19780, 19880, 19980): continue
    # if CameraFreq != 19960: continue
    var['CameraPeriod'] = 1/(CameraFreq*1e3)
    init_params = {
        'N': {
            'value': consts['FrameTime']/var['CameraPeriod'],
            'vary': False,
        },
        'm': {
            'value': 1.2899,
            'vary': False,
        },
        'delta_T': {
            'value': (var['CameraPeriod']-consts['LaserPeriod'])/consts['TimeBinLength'],
            # 'value': 23.2771,
            'vary' : False,
        },
        'tw': {
            'value': 13.095,
            'vary': False,
        },
        'tau0': {
            'value': 0,
            'vary': False,
        },
        'A': {
            'value': 326.44,
            'vary': False,
        },
        'b': {
            'value': 0,
            'vary': False,
        },
        'Z': {
            'value': 1024*var['CameraPeriod']/55e-9,
            # 'value': 964.70,
            'vary': False,
        },
    }
    if model_result:
        init_params['m']['value'] = model_result.params['m'].value
        init_params['tw']['value'] = model_result.params['tw'].value
        init_params['A']['value'] = model_result.params['A'].value
        init_params['b']['value'] = model_result.params['b'].value
    
    model = RunFit(init_params, fpath, fitradius=config['FitRadius'])
    model_result = model.run_fit()
    # model.save_fit('/mnt/e/Data/JensenLab/VarySyncFine/analysis/data_fittingmodel')
