#%%
%matplotlib ipympl
import os
import cv2
from lmfit.lineshapes import gaussian
from lmfit.minimizer import minimize
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from scipy.optimize import curve_fit    
import lmfit
import math
from math import sqrt, pi, exp, erf, sin, cos, tan
from scipy.special import erf
class SlotModel:
    CANVAS_X = 1500
    CANVAS_Y = 1000
    MIN_SLOT_SIZE = 100
    THRESHOLD = 40

    @staticmethod
    def gaussian_flat(x, mu, w, s):
        return -1. / 2. * (erf((2*mu - w - 2*x)/(2*np.sqrt(2)*s)) - erf((2*mu + w - 2*x)/(2*np.sqrt(2)*s)))

    @staticmethod
    def gaussian_slot(y, x, cy, cx, t, l, w, r):
        # x, y = np.array([[cos(t), sin(t)],[-sin(t), cos(t)]]) @ x, y
        return SlotModel.gaussian_flat(y, cy, l, r) * SlotModel.gaussian_flat(x, cx, w, r)

    @staticmethod
    def gaussian_cell(y, x, my, mx, py, px, dy, dx, l, w, r):
        cy1, cx1 = my - dy, mx - dx
        cy2, cx2 = my + dy, mx + dx
        t1, t2 = 0, 0
        return SlotModel.gaussian_slot(y, x, cy1, cx1, t1, l, w, r) + SlotModel.gaussian_slot(y, x, cy2, cx2, t2, l, w, r)
    
    @staticmethod
    def gaussian_lattice(y, x, y0, x0, py, px, dy, dx, l, w, r):
        y_ = np.mod(y - y0, py)
        x_ = np.mod(x - x0, px)
        return SlotModel.gaussian_cell(y_, x_, py/2, px/2, py, px, dy, dx, l, w, r)

    @staticmethod
    def gaussian_fit(xdata, y0, x0, py, px, dy, dx, l, w, r):
        y, x = xdata
        img =  255 * SlotModel.gaussian_lattice(y, x, y0, x0, py, px, dy, dx, l, w, r)
        return img

class MetaAnalyzer:
    def __init__(self, path, angle_left, angle_right, tolerance, scale):
        self.__path = path
        self.scale = scale # pixels per nm
        if angle_left is not None:
            angle_left = angle_left % 180
            self.bound_min_left = (angle_left - tolerance/2) % 180
            self.bound_max_left = (angle_left + tolerance/2) % 180
        if angle_right is not None:
            angle_right = angle_right % 180
            self.bound_min_right = (angle_right - tolerance/2) % 180
            self.bound_max_right = (angle_right + tolerance/2) % 180
        self.__image = self.load_image(self.__path)
        self.__contours, self.__contours_image = self.get_contours(self.__image)
        # self.img_rectangles, self.rectangles = self.fit_rectangles()
        # self.img_ellipses, self.ellipses = self.fit_ellipses(self.__contours_image)
        # self.param_ellipses = self.analyze_ellipse()
        # self.plot_report(savefig=True)
        self.fit_lattice_lmfit()

    @staticmethod
    def load_image(path):
        print(f"Loading Image from {path}")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img[:SlotModel.CANVAS_Y, :SlotModel.CANVAS_X]
        return img

    @staticmethod
    def get_contours(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res_img = img_gray * 0
        min_size = SlotModel.MIN_SLOT_SIZE
        ret, im = cv2.threshold(img_gray, SlotModel.THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if c.shape[0] > min_size ]
        for c in contours:
            # print(c.shape[0])
            cv2.drawContours(res_img, [c], -1, 255, -1)
        return contours, res_img 

    def fit_lattice_lmfit(self):
        img_gray = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        # img_gray = cv2.GaussianBlur(img_gray, (101, 101), 0)
        self.ydata = img_gray.flatten()
        y, x = np.arange(img_gray.shape[0]), np.arange(img_gray.shape[1])
        X, Y = np.meshgrid(x, y)
        Y, X = Y.ravel(), X.ravel()
        xdata = np.vstack((Y, X))
        self.model = lmfit.Model(SlotModel.gaussian_fit, name="test model")
        default_hint = {'value': 1, 'min': -math.inf, 'max': math.inf, 'vary': False, 'expr': None,}
        self.init_params = {
            'y0': {
                'value': 0,
                'vary': True,
            },
            'x0': {
                'value': 0,
                'vary': True,
            },
            'py': {
                'value': 177,
                'vary': True,
                'min': 0
            },
            'px': {
                'value': 177,
                'vary': True,
                'min': 0
            },
            'dy': {
                'value': 177//4,
                'vary': True,
                'min': 0
            },
            'dx': {
                'value': 177//4,
                'vary': True,
                'min': 0
            },
            'l': {
                'value': 177/2,
                'vary': True,
                'min': 0,
                'max': 177
            },
            'w': {
                'value': 20,
                'vary': True,
                'min': 0,
                'max': 177/4
            },
            'r': {
                'value': 2,
                'vary': True,
                'min': 0
            },
        }
        for param_name in self.model.param_names:
            self.model.set_param_hint(param_name, 
                value=self.init_params.get(param_name, default_hint).get('value', default_hint['value']),
                min=self.init_params.get(param_name, default_hint).get('min', default_hint['min']),
                max=self.init_params.get(param_name, default_hint).get('max', default_hint['max']),
                vary=self.init_params.get(param_name, default_hint).get('vary', default_hint['vary']),
                )
        self.params = self.model.make_params()
        def callback(params, iter, resid, *args, **kws):
            print(iter)

        # self.result = self.model.fit(self.ydata, self.params, xdata=xdata,verbose=True, iter_cb=callback)
        # self.params = self.result.params
        # print(self.result.fit_report())
        Z = self.model.eval(self.params, xdata=xdata)
        R = self.ydata - Z
        Z = Z.reshape(img_gray.shape)
        R = R.reshape(img_gray.shape)
        plt.imshow(Z, interpolation='None', cmap="gray")
        plt.show()
        plt.imshow(R, interpolation='None', cmap="gray")
        plt.show()
    
    def fit_ellipses(self, img=None):
        if img is None:
            img = self.__image
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        contours = self.__contours
        res_img = img
        ellipses = []
        for cnt in contours:
            elp = cv2.fitEllipse(cnt)
            ellipses.append(elp)
            cv2.ellipse(res_img,elp,(255,0,0),2)
        return res_img, ellipses
    
    def analyze_ellipse(self):
        size_l = np.array([e[1][1] for e in self.ellipses])*2*self.scale
        size_w = np.array([e[1][0] for e in self.ellipses])*2*self.scale
        angle = np.array([e[2] for e in self.ellipses])
        return angle, size_l, size_w

    
    def plot_report(self, savefig=False):
        plt.figure(figsize=(20, 20))
        ax1 = plt.subplot(3, 2, 1)
        ax1.imshow(self.img_rectangles)
        ax2 = plt.subplot(3, 2, 2)
        ax2.imshow(self.img_ellipses)
        ax3 = plt.subplot(3, 2, (3, 4))
        ax4 = plt.subplot(3, 2, (5, 6))
        ax3.set_xlim(0, 180)
        ax4.set_xlim(0, 180)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        angle, size_l, size_w = self.param_ellipses


        if hasattr(self, 'bound_min_left'):
            bound_min_left, bound_max_left = self.bound_min_left, self.bound_max_left
            if bound_min_left < bound_max_left:
                zone_left = np.logical_and(angle > bound_min_left, angle < bound_max_left)
            else:
                zone_left = np.logical_or(angle > bound_min_left, angle < bound_max_left)
            angle_left = angle[zone_left]
            size_l_left = size_l[zone_left]
            size_w_left = size_w[zone_left]
        else:
            bound_min_left, bound_max_left = None, None
            angle_left, size_l_left, size_w_left = None, None, None

        if hasattr(self, 'bound_min_right'):
            bound_min_right, bound_max_right = self.bound_min_right, self.bound_max_right
            if bound_min_right < bound_max_right:
                zone_right = np.logical_and(angle > bound_min_right, angle < bound_max_right)
            else:
                zone_right = np.logical_or(angle > bound_min_right, angle < bound_max_right)
            angle_right = angle[zone_right]
            size_l_right = size_l[zone_right]
            size_w_right = size_w[zone_right]
        else:
            bound_min_right, bound_max_right = None, None
            angle_right, size_l_right, size_w_right = None, None, None

        is_right = False

        for (angle, size_l, size_w) in [(angle_left, size_l_left, size_w_left), (angle_right, size_l_right, size_w_right)]:
            if angle is not None: 
                ax3.scatter(angle, size_l)
                bound_min, bound_max = (bound_min_left, bound_max_left) if not is_right else (bound_min_right, bound_max_right)
                report_l = \
                f"""L vs. Angle from t={bound_min}:{bound_max}
N = {angle.shape[0]}
Average T: {np.average(angle)}
Std T: {np.std(angle)}
Average L: {np.average(size_l)}nm
Std L: {np.std(size_l)}nm"""        
                pos_x, pos_y, ha = 0.02 if not is_right else 0.98, 0.95, 'right' if is_right else 'left'
                ax3.text(pos_x, pos_y, report_l, transform=ax3.transAxes, fontsize=14,
                        verticalalignment='top', horizontalalignment=ha, bbox=props)        

                ax4.scatter(angle, size_w)
                report_w = \
                f"""W vs. Angle from t={bound_min}:{bound_max}
N = {angle.shape[0]}
Average T: {np.average(angle)}
Std T: {np.std(angle)}
Average W: {np.average(size_w)}nm
Std W: {np.std(size_w)}nm"""        
                ax4.text(pos_x, pos_y, report_w, transform=ax4.transAxes, fontsize=14,
                        verticalalignment='top', horizontalalignment=ha, bbox=props)     

            is_right = True

        if not savefig:
            plt.show()
        else:
            fname = self.__path.replace(".tif", "analysis.png")
            plt.savefig(fname)

    def fit_rectangles(self):
        img = self.__image
        contours = self.__contours
        res_img = img
        rectangles = []
        for c in contours:
            min_y = min([p[0][0] for p in c])
            max_y = max([p[0][0] for p in c])
            min_x = min([p[0][1] for p in c])
            max_x = max([p[0][1] for p in c])
            rect = cv2.rectangle(res_img, (min_y, min_x), (max_y, max_x), (0, 255, 0), 2)
            rectangles.append(rect)
        return res_img, rectangles


def main():
    dirpath = os.path.abspath("/mnt/e/Data/JensenLab/FIBImage/Set1_FIB")
    for name in os.listdir(dirpath):
        if not name.endswith(".tif"): continue
        path = os.path.join(dirpath, name)
        name = name.replace("roa", "").replace(".tif", "")
        args = name.split("_")
        
        angle_left = -int(args[0])
        angle_right = -int(args[1])
        tolerance = 30
        scale = 500 / 277
        # angle_left, angle_right, tolerance = 90, None, 180
        analysis = MetaAnalyzer(path, angle_left, angle_right, tolerance, scale)
        break

# def main():
#     #testing slot main
#     fig = plt.figure()
#     ax = fig.gca()
#     x = np.linspace(-10, 10, 1000)
#     y = np.linspace(-10, 10, 1000)
#     z = SlotModel.gaussian_flat(x, 0, 4, 0.2)
#     ax.plot(x, z)

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     Y, X = np.meshgrid(y, x)
#     Z = SlotModel.gaussian_slot(Y, X, 0, 0, 4, 2, 0.2)
#     fig.suptitle("Gaussian Slot")
#     ax.plot_surface(X, Y, Z)

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     Z = SlotModel.gaussian_cell(Y, X, 0, 0, 10, 10, 2.5, 2.5, 4, 2, 0.2)
#     fig.suptitle("Slot Cell")
#     ax.plot_surface(X, Y, Z)

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     x = np.linspace(-10, 10, 2000)
#     y = np.linspace(-10, 10, 3000)
#     Y, X = np.meshgrid(y, x)
#     X, Y = np.meshgrid(x, y)
#     Z = SlotModel.gaussian_lattice(Y, X, 0, 0, 10, 10, 2.5, 2.5, 4, 2, 0.2)
#     fig.suptitle("Slot lattice")
#     # ax.plot_surface(X, Y, Z)
#     ax.imshow(Z)

#     plt.show()

if __name__ == '__main__':
    main()

# %%
