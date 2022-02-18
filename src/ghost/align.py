import sys

from math import ceil, floor, pi, acos, prod, sqrt
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from io import StringIO
from getkey import getkey

from multisim import GhostSimulator, GhostAnalyser, SLMPattern, AlignmentHelper

# sys.path.append(os.getenv("HEDS_PYTHON_MODULES", ""))
# from holoeye import slmdisplaysdk

# ErrorCode = slmdisplaysdk.SLMDisplay.ErrorCode
# ShowFlags = slmdisplaysdk.SLMDisplay.ShowFlags

# slm = slmdisplaysdk.SLMDisplay()

k = 2 * pi / 0.632
theta = 0.5 * pi / 180
slm_phys_res = (1080, 1920)
slm_res = [32, 32]

slm_shift = [300, 480]
lens_size = 14

while True:
    ah = AlignmentHelper(slm_res)
    filter = ah.ones()
    pg = SLMPattern(lens_size, slm_phys_res, k, theta)

    amp_pattern = np.zeros((pg.xp_size, pg.yp_size))
    ph_pattern = np.zeros((pg.xp_size, pg.yp_size))

    amp_pattern[:slm_res[0], :slm_res[0]] = filter

    pattern = pg.generate_pattern_grad(amp_pattern, ph_pattern)
    pattern = np.roll(pattern, slm_shift, axis=(0, 1))

    key = getkey()
    if key == 'q':
        break
    elif key == 'w':
        slm_shift[0] -= 10
    elif key == 'a':
        slm_shift[1] += 10
    elif key == 's':
        slm_shift[0] += 10
    elif key == 'd':
        slm_shift[1] -= 10
    elif key == 'i':
        slm_res[0] += 1
        slm_res[1] += 1
    elif key == 'j':
        lens_size -= 1
    elif key == 'k':
        slm_res[0] -= 1
        slm_res[1] -= 1
    elif key == 'l':
        lens_size += 1

    print('----------------------------------')
    print('slm_res:', slm_res)
    print('slm_shift:', slm_shift)
    print('lens_size:', lens_size)
    print('----------------------------------')

    # error = slm.showPhasevalues(pattern)
    # assert error == ErrorCode.NoError, slm.errorString(error)
