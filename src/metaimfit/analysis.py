# %%
from typing import Any, Dict, Tuple
from scipy.special import erf
from math import sqrt, pi, exp, sin, cos, tan, inf
import lmfit
import numpy as np
import cv2
import os
from IPython.display import clear_output

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


from .slotmodel import SlotModel


class MetaAnalyzer:
    def __init__(self, verbose=False, max_fit_iter=None, *, path, canvas, min_slot_size, threshold, init_params, scale, ):
        self.path: os.PathLike = path
        self.name = os.path.basename(self.path)
        self.scale: float = scale  # pixels per nm
        self.canvas_y: int = canvas[0]
        self.canvas_x: int = canvas[1]
        self.min_slot_size: int = min_slot_size
        self.threshold: int = threshold
        self.init_params: Dict[str, Any] = init_params
        self.max_fit_iter = max_fit_iter
        self.params: lmfit.Parameters = None
        self.result: lmfit.model.ModelResult = None
        self.verbose = verbose

        self.image: np.ndarray = self.load_image(self.path)
        self.contours, self.contours_image = self.get_contours(
            self.image)

        self.init_lmfit()

    def load_image(self, path):
        print(f"Loading Image from {path}")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[self.canvas_y[0]:self.canvas_y[1],
                  self.canvas_x[0]:self.canvas_x[1]]
        return img

    def get_contours(self, img):
        img_gray = img
        cnt_img = img_gray * 0
        min_size = self.min_slot_size
        ret, im = cv2.threshold(
            img_gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(
            im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if c.shape[0] > min_size]
        for c in contours:
            cv2.drawContours(cnt_img, [c], -1, 255, -1)
        # plt.imshow(img + cnt_img, cmap='gray')
        # plt.show()
        return contours, cnt_img

    def init_lmfit(self):
        y, x = np.arange(self.contours_image.shape[0]), np.arange(
            self.contours_image.shape[1])
        X, Y = np.meshgrid(x, y)
        Y, X = Y.ravel(), X.ravel()
        self.xdata = np.vstack((Y, X))
        self.ydata = self.contours_image.flatten()
        self.model = lmfit.Model(SlotModel.gaussian_fit, name=self.name)
        default_hint = {'value': 1, 'min': -inf,
                        'max': inf, 'vary': False, 'expr': None}
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

    def eval_lattice_lmfit(self):
        return self.model.eval(self.params, xdata=self.xdata)

    def fit_lattice_lmfit(self):
        if self.verbose:
            def iter_cb(params, iter, resid, *args, **kws):
                clear_output(wait=True)
                R = resid.reshape(
                    (self.canvas_y[1]-self.canvas_y[0], self.canvas_x[1]-self.canvas_x[0]))
                plt.imshow(R, cmap="gray", vmin=-255, vmax=255)
                plt.show()
                print(f"iter:\t{iter}")
                params.pretty_print(colwidth=10, columns=[
                                    'value', 'stderr', 'vary'])
        else:
            def iter_cb(params, iter, resid, *args, **kws):
                pass

        self.result = self.model.fit(
            self.ydata, self.params, xdata=self.xdata, verbose=True, iter_cb=iter_cb, max_nfev=self.max_fit_iter)
        clear_output(wait=True)
        self.params = self.result.params
        return self.result

    def print_physical_params(self, params):
        keys = set(params.keys())
        values = {x: params[x].value for x in keys}
        stderrs = {x: params[x].stderr or inf for x in keys}
        angle_keys = {'t1', 't2', 'phi'}
        length_keys = keys - angle_keys

        print("---------------------------------------------")
        for key in length_keys:
            print(
                f"{key} \t= {values[key]*self.scale} \t+- {stderrs[key]*self.scale} nm")
        for key in angle_keys:
            print(
                f"{key} \t= {values[key]*180/pi} \t+- {stderrs[key]*180/pi} deg")
        # print(f"l = {values['l']*self.scale} +- {stderrs['l']*self.scale} nm")
        # print(f"w = {values['w']*self.scale} +- {stderrs['w']*self.scale} nm")
        # print(f"px = {values['px']*self.scale} +- {stderrs['px']*self.scale} nm")
        # print(f"py = {values['py']*self.scale} +- {stderrs['py']*self.scale} nm")
        # print(f"dx = {values['dx']*self.scale} +- {stderrs['dx']*self.scale} nm")
        # print(f"dy = {values['dy']*self.scale} +- {stderrs['dy']*self.scale} nm")
        # print(f"t1 = {values['t1']*180/pi} +- {stderrs['t1']*180/pi} deg")
        # print(f"t2 = {values['t2']*180/pi} +- {stderrs['t2']*180/pi} deg")
        print("---------------------------------------------")

    def plot_matplotlib(self):
        Z = self.model.eval(self.params, xdata=self.xdata)
        R = Z - self.contours_image.ravel()
        R = R.reshape(self.contours_image.shape)
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(18.5, 10.5)
        axs[0].imshow(self.image, cmap="gray", vmin=0, vmax=255)
        axs[1].imshow(R, cmap="gray", vmin=-255, vmax=255)
        plt.show()

    def plot_plotly(self):
        print("Please wait, generating plotly plots")
        Z = self.model.eval(self.params, xdata=self.xdata)
        R = Z - self.contours_image.ravel()
        R = R.reshape(self.contours_image.shape)
        fig = make_subplots(
            rows=1, cols=2, shared_xaxes=True, shared_yaxes=True)
        fig.add_trace(go.Heatmap(z=self.image, colorscale='Greys_r',
                      zmin=0, zmax=255), row=1, col=1)
        fig.add_trace(go.Heatmap(z=R, colorscale='Greys_r',
                      zmin=-255, zmax=255), row=1, col=2)
        fig.update_traces(showscale=False)
        fig.update_layout(title_text=f"Analysis for {self.name}")
        fig.show()

# %%
