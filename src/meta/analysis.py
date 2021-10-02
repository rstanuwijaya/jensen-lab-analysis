# %%
from typing import Any, Dict, Tuple
from scipy.special import erf
from math import sqrt, pi, exp, sin, cos, tan, inf
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
from IPython.display import clear_output

from .slotmodel import SlotModel


class MetaAnalyzer:
    def __init__(self, verbose=False, *, path, canvas, min_slot_size, threshold, init_params, scale, ):
        self.path: os.PathLike = path
        self.scale: float = scale  # pixels per nm
        self.canvas_y: int = canvas[0]
        self.canvas_x: int = canvas[1]
        self.min_slot_size: int = min_slot_size
        self.threshold: int = threshold
        self.init_params: Dict[str, Any] = init_params
        self.params: lmfit.Parameters = None
        self.result: lmfit.model.ModelResult = None
        self.verbose = verbose

        self.image: np.ndarray = self.load_image(self.path)
        self.contours, self.contours_image = self.get_contours(
            self.image)

    def load_image(self, path):
        print(f"Loading Image from {path}")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img[:self.canvas_y, :self.canvas_x]
        return img

    def get_contours(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cnt_img = img_gray * 0
        min_size = self.min_slot_size
        ret, im = cv2.threshold(
            img_gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(
            im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if c.shape[0] > min_size]
        for c in contours:
            cv2.drawContours(cnt_img, [c], -1, 255, -1)
        return contours, cnt_img

    def fit_lattice_lmfit(self):
        y, x = np.arange(self.contours_image.shape[0]), np.arange(
            self.contours_image.shape[1])
        X, Y = np.meshgrid(x, y)
        Y, X = Y.ravel(), X.ravel()
        self.xdata = np.vstack((Y, X))
        self.ydata = self.contours_image.flatten()
        self.model = lmfit.Model(SlotModel.gaussian_fit, name="Slot Model")
        default_hint = {'value': 1, 'min': -inf,
                        'max': inf, 'vary': False, 'expr': None, }
        for param_name in self.model.param_names:
            self.model.set_param_hint(param_name,
                                      value=self.init_params.get(param_name, default_hint).get(
                                          'value', default_hint['value']),
                                      min=self.init_params.get(param_name, default_hint).get(
                                          'min', default_hint['min']),
                                      max=self.init_params.get(param_name, default_hint).get(
                                          'max', default_hint['max']),
                                      vary=self.init_params.get(param_name, default_hint).get(
                                          'vary', default_hint['vary']),
                                      )
        self.params = self.model.make_params()

        if self.verbose:
            def iter_cb(params, iter, resid, *args, **kws):
                clear_output(wait=True)
                R = resid.reshape((self.canvas_y, self.canvas_x))
                plt.imshow(R, cmap="gray")
                plt.show()
                print(f"iter:\t{iter}")
                params.pretty_print(colwidth=10, columns=[
                                    'value', 'stderr', 'vary'])
        else:
            def iter_cb(params, iter, resid, *args, **kws):
                pass

        self.result = self.model.fit(
            self.ydata, self.params, xdata=self.xdata, verbose=True, iter_cb=iter_cb)
        clear_output(wait=True)
        self.params = self.result.params
        return self.result
# %%
