#%%
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
from math import sqrt, pi, exp
class SlotModel:
    CANVAS_X = 3000
    CANVAS_Y = 2000

    @staticmethod
    def gaussian_2d(y, x, mu_y, mu_x, s_y, s_x):
        return 1. / np.sqrt(2*np.pi*s_x*s_y) * np.exp(-((x-mu_x)**2/(2*s_x**2)+(y-mu_y)**2/(2*s_y**2)))

    @staticmethod
    def gaussian_slot(y, x, cy, cx, l, w, r):
        img = np.zeros(x.shape)
        bl = cx - w/2
        br = cx + w/2
        bt = cy - l/2
        bb = cy + l/2
        img += SlotModel.gaussian_2d(y, x, bl, bt, r, r)
        img += SlotModel.gaussian_2d(y, x, br, bt, r, r)
        img += SlotModel.gaussian_2d(y, x, bl, bb, r, r)
        img += SlotModel.gaussian_2d(y, x, br, bb, r, r)
        bt, bb = int(bt), int(bb)
        bl, br = int(bl), int(br)
        img[bt:bb, :] = img[:, None][bt, :]
        img[:, bl:br] = img[:, bl][:, None]
        return img

    @staticmethod
    def gaussian_cell(y, x, my, mx, py, px, dx, dy, l, w, r):
        print(mx, my)
        img = np.zeros(x.shape)
        cy1, cx1 = my - dy, mx - dx
        cy2, cx2 = my + dy, mx + dx
        img += SlotModel.gaussian_slot(y, x, cy1, cx1, l, w, r)
        img += SlotModel.gaussian_slot(y, x, cy2, cx2, l, w, r)
        return img
    
    @staticmethod
    def gaussian_lattice(y, x, y0, x0, py, px, dy, dx, l, w, r):
        img = np.zeros(x.shape)
        reps_y = y.shape[0] // py + 1
        reps_x = x.shape[1] // px + 1
        for j in range(reps_y):
            for i in range(reps_x):
                img += SlotModel.gaussian_cell(y, x, j*py - y0, i*px - x0, py, px, dx, dy, l, w, r)
        return img

    @staticmethod
    def single_slot(cell, y, x, l, w, r):
        # rounded rectangular slot oriented vertically
        bl = x - w//2
        br = x + w//2
        bt = y - l//2
        bb = y + l//2
        cv2.circle(cell, (bl + r, bt + r), r, 255, -1)
        cv2.circle(cell, (br - r, bt + r), r, 255, -1)
        cv2.circle(cell, (bl + r, bb - r), r, 255, -1)
        cv2.circle(cell, (br - r, bb - r), r, 255, -1)
        cv2.rectangle(cell, (bl + r, bt), (br - r, bb), 255, -1)
        cv2.rectangle(cell, (bl, bt + r), (br, bb - r), 255, -1)
        return cell

    @staticmethod
    def slot_cell(py, px, dx, dy, l, w, r):
        # unit cell containing a pair of single slots located
        py, px = int(py), int(px)
        img = np.zeros((py, px), np.uint8)
        img = SlotModel.single_slot(img, py//2-dy, px//2+dx, l, w, r)
        img = SlotModel.single_slot(img, py//2+dy, px//2-dx, l, w, r)
        return img

    @staticmethod
    def slot_lattice(xdata, y0, x0, py, px, dy, dx, l, w, r):
        # lattice of the slots on a given canvas
        print(y0, x0, py, px, dy, dx, l, w, r,)
        y0, x0, py, px, dy, dx, l, w, r, = int(y0), int(x0), int(py), int(px), int(dy), int(dx), int(l), int(w), int(r),
        canvas_y, canvas_x = SlotModel.CANVAS_Y, SlotModel.CANVAS_X
        # assert canvas_y%py == 0 and canvas_x%px == 0, "Slot cell dimension is not an integer"
        reps_y, reps_x = canvas_y//py + 1, canvas_x//px + 1
        cell = SlotModel.slot_cell(py, px, dy, dx, l, w, r)
        img = np.tile(cell, (reps_y, reps_x))
        img =  img[y0:y0+canvas_y, x0:x0+canvas_x]
        img = cv2.GaussianBlur(img, (101, 101), 0)
        return img.flatten()

class MetaAnalyzer:
    def __init__(self, path):
        self.__path = path
        self.__image = self.load_image(self.__path)
        self.__contours, self.__contours_image = self.get_contours(self.__image)
        self.__rectangles = self.draw_rectangles()
        # self.test_slot_cell()
        # self.test_slot_array()
        # self.fit_lattice()
        # self.fit_lattice_lmfit()
        # self.fit_lattice_leastsq()

        # self.test_gaussian_2d()
        # self.test_gaussian_slot()
        # self.test_gaussian_cell()
        # self.test_gaussian_lattice()

        self.fit_ellipse()
        print("done")

    @staticmethod
    def load_image(path):
        print(f"Loading Image from {path}")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img[:SlotModel.CANVAS_Y, :SlotModel.CANVAS_X]
        plt.imshow(img)
        plt.show()
        return img

    @staticmethod
    def get_contours(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res_img = img_gray * 0
        min_size = 300
        ret, im = cv2.threshold(img_gray, 70, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if c.shape[0] > min_size ]
        print(f"Using contour detection algorithm for size > {min_size}")
        for c in contours:
            cv2.drawContours(res_img, [c], -1, 255, -1)
        plt.imshow(res_img)
        plt.show()
        return contours, res_img 

    def test_gaussian_2d(self):
        print("testing gaussian")
        y = np.linspace(-100, 100, 200)
        x = np.linspace(-100, 100, 200)
        y, x = np.meshgrid(y, x)
        img = SlotModel.gaussian_2d(y, x, 50, 50, 20, 20)
        plt.imshow(img)
        plt.show()     

    def test_gaussian_slot(self):
        print("testing slot")
        y = np.linspace(0, 500, 400)
        x = np.linspace(0, 500, 400)
        y, x = np.meshgrid(y, x)
        img = SlotModel.gaussian_slot(y, x, 250, 250, 100, 20, 5)
        plt.imshow(img)
        plt.show()     

    def test_gaussian_cell(self):
        print("testing gaussian")
        y = np.linspace(0, 500, 400)
        x = np.linspace(0, 500, 400)
        y, x = np.meshgrid(y, x)
        img = SlotModel.gaussian_cell(y, x, 250, 250, 200, 200, 50, 50, 100, 20, 5)
        plt.imshow(img)
        plt.show()    

    def test_gaussian_lattice(self):
        print("testing gaussian")
        y = np.linspace(0, 3000, 200)
        x = np.linspace(0, 3000, 200)
        y, x = np.meshgrid(y, x)
        img = SlotModel.gaussian_lattice(y, x, 0, 0, 500, 500, 250, 250, 100, 20, 5)
        plt.imshow(img)
        plt.show()     

    def test_slot_cell(self):
        img = SlotModel(self.__image).slot_cell(500, 500, 125, 125, 200, 100, 50)
        plt.imshow(img)
        plt.show()

    def test_slot_array(self):
        img_gray = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        img = SlotModel.slot_lattice(img_gray, 200, 100, 500, 500, 125, 125, 200, 100, 50)

    def fit_lattice(self):
        img_gray = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        func = SlotModel.slot_lattice
        #       y0,     x0,     py,     px,     dy,     dx,     l,      w,      r,
        p0 = (  50,     200,    500,    500,    500/4,    500/4,    200,    40,    20)

        ydata = img_gray.flatten()
        xdata = np.arange(ydata.shape[0])
        popt, pcov = curve_fit(func, xdata, ydata, p0=p0)
        z_fit = func(xdata, *popt)
        Z_fit = z_fit.reshape(img_gray.shape)
        plt.imshow(Z_fit)
        plt.plot()
        print("popt:")
        print(popt)
        print("pcov:")
        print(pcov)

    def fit_lattice_lmfit(self):
        img_gray = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (101, 101), 0)
        self.ydata = img_gray.flatten()
        self.model = lmfit.Model(SlotModel.slot_lattice, name="test model")
        default_hint = {'value': 1, 'min': -math.inf, 'max': math.inf, 'vary': False, 'expr': None,}
        self.init_params = {
            'y0': {
                'value': 50,
                'vary': False,
            },
            'x0': {
                'value': 200,
                'vary': False,
            },
            'py': {
                'value': 500,
                'vary': True,
            },
            'px': {
                'value': 500,
                'vary': True,
            },
            'dy': {
                'value': 500//4,
                'vary': False,
            },
            'dx': {
                'value': 500//4,
                'vary': False,
            },
            'l': {
                'value': 200,
                'vary': True,
            },
            'w': {
                'value': 50,
                'vary': True,
            },
            'r': {
                'value': 10,
                'vary': True,
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
        self.result = self.model.fit(self.ydata, self.params, xdata=None)
        print(self.result.fit_report())
        
    
    def fit_lattice_leastsq():
        img_gray = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (101, 101), 0)

    def fit_ellipse(self):
        img = self.__image
        contours = self.__contours
        res_img = img * 0
        fit_result = []
        for cnt in contours:
            ellipse = cv2.fitEllipse(cnt)
            fit_result.append(ellipse)
            cv2.ellipse(res_img,ellipse,(0,randint(100,255),randint(100,255)),4)
        print("Fitting ellipse")
        plt.imshow(res_img)
        plt.show()            
        size_w = [e[1][0] for e in fit_result]
        size_h = [e[1][1] for e in fit_result]
        angle = [e[2] for e in fit_result]

        print("avg_w: ", sum(size_w)/len(size_w))
        print("avg_h: ", sum(size_h)/len(size_h))
        print("avg_w: ", sum(angle)/len(angle))

    def draw_rectangles(self):
        img = self.__image
        contours = self.__contours
        res_img = img
        for c in contours:
            min_y = min([p[0][0] for p in c])
            max_y = max([p[0][0] for p in c])
            min_x = min([p[0][1] for p in c])
            max_x = max([p[0][1] for p in c])
            cv2.rectangle(res_img, (min_y, min_x), (max_y, max_x), (0, 255, 0), 2)
        plt.imshow(res_img)
        plt.show()            


def main():
    path = os.path.abspath("/home/stnav/jensen-lab/sample/Align_P300L135_003.tif")
    analysis = MetaAnalyzer(path)

if __name__ == '__main__':
    main()

# %%
