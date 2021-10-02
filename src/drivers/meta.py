#%%
import os
import time
import numpy as np
from math import pi
import matplotlib.pyplot as plt

from meta import MetaAnalyzer
from meta import SlotModel

def main():
    analysis_list = []
    path = os.path.abspath("/home/stnav/jensen-lab/sample/Align_P300L135_003.tif")
    canvas_y, canvas_x = 1000, 1500
    min_slot_size = 100
    threshold = 40
    scale = 500/277
    init_params = {
        'y0': {
            'value': 0,
            'vary': True,
        },
        'x0': {
            'value': 0,
            'vary': True,
        },
        'py': {
            'value': 180,
            'vary': True,
            'min': 0
        },
        'px': {
            'value': 183,
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
            'value': 200/2,
            'vary': True,
            'min': 0,
            'max': 177
        },
        'w': {
            'value': 30,
            'vary': True,
            'min': 0,
            'max': 177/4
        },
        'r': {
            'value': 2,
            'vary': True,
            'min': 0
        },
        't1': {
            'value': 0,
            'vary': True,
            'min': -pi/2,
            'max': pi/2
        },
        't2': {
            'value': 0,
            'vary': True,
            'min': -pi/2,
            'max': pi/2
        },
        'phi': {
            'value': -pi/180*2.5,
            'vary': True,
            'min': -pi/2,
            'max': pi/2
        }
    }
    dirpath = os.path.abspath("/mnt/e/Data/JensenLab/FIBImage/Set1_FIB")
    for name in os.listdir(dirpath):
        if not name.endswith(".tif"):
            continue
        path = os.path.join(dirpath, name)
        name = name.replace("roa", "").replace(".tif", "")
        args = name.split("_")

        analysis = MetaAnalyzer(path=path, canvas=(canvas_y, canvas_x), min_slot_size=min_slot_size, threshold=threshold, init_params=init_params, scale=scale, verbose=True)
        result = analysis.fit_lattice_lmfit()
        analysis_list.append(analysis)
        print(result.fit_report())
        Z = analysis.model.eval(analysis.params, xdata=analysis.xdata)
        R = Z - analysis.contours_image.ravel()
        R = R.reshape(analysis.contours_image.shape)
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(18.5, 10.5)
        axs[0].imshow(analysis.image, cmap="gray")
        axs[1].imshow(R, cmap="gray")
        plt.show()
        break

def test_main():
    # # testing slot main
    fig = plt.figure()
    ax = fig.gca()
    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    z = SlotModel.gaussian_flat(x, mu=0, w=4, s=0.2)
    ax.plot(x, z)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    X, Y = np.meshgrid(x, y)
    Z = SlotModel.gaussian_slot(Y, X, cy=0, cx=0, t=0, l=4, w=2, r=0.2)
    fig.suptitle("Gaussian Slot")
    # ax.plot_surface(X, Y, Z)
    ax.imshow(Z)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    X, Y = np.meshgrid(x, y)
    Z = SlotModel.gaussian_cell(Y, X, my=0, mx=0, py=10, px=10, t1=0, t2=0, dy=2.5, dx=2.5, l=4, w=2, r=0.2)
    fig.suptitle("Slot Cell")
    # ax.plot_surface(X, Y, Z)
    ax.imshow(Z)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(-10, 10, 3000)
    y = np.linspace(-10, 10, 2000)
    X, Y = np.meshgrid(x, y)
    Z = SlotModel.gaussian_lattice(Y, X, y0=0, x0=0, phi=pi/6/4, py=10, px=10, t1=0, t2=0, dy=2.5, dx=2.5, l=4, w=2, r=0.2)
    fig.suptitle("Slot lattice")
    # ax.plot_surface(X, Y, Z)
    ax.imshow(Z)

    plt.show()