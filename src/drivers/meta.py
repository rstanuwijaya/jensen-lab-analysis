#%%
import os
import time
from IPython.core.display import clear_output
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import lmfit

from meta import MetaAnalyzer
from meta import SlotModel

def main():
    analysis_list = []
    canvas_y, canvas_x = (0, 1000), (0, 1500)
    min_slot_size = 50
    threshold = 40
    scale = 500/277
    run_fit = True
    show_iter_output = True
    max_fit_iter = None
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
            'min': 0,
        },
        'px': {
            'value': 183,
            'vary': True,
            'min': 0,
        },
        'dy': {
            'value': 177//4,
            'vary': True,
            'min': 0,
        },
        'dx': {
            'value': 177//4,
            'vary': True,
            'min': 0,
        },
        'l': {
            'value': 200/2,
            'vary': True,
            'min': 0,
            'max': 177,
        },
        'w': {
            'value': 30,
            'vary': True,
            'min': 0,
            'max': 177/4,
        },
        'r': {
            'value': 2,
            'vary': True,
            'min': 0,
        },
        't1': {
            'value': 0,
            'vary': True,
            'min': -pi/2,
            'max': pi/2,
        },
        't2': {
            'value': 0,
            'vary': True,
            'min': -pi/2,
            'max': pi/2,
        },
        'phi': {
            'value': -pi/180*2.5,
            'vary': True,
            'min': -pi/2,
            'max': pi/2,
        }
    }
    dirpath = os.path.abspath("/mnt/e/Data/JensenLab/FIBImage/Set1_FIB_ellipse")
    images = []
    for file in os.listdir(dirpath):
        if file.endswith('.tif'):
            images.append(file)
    for i, image in enumerate(images):
        print(f"{i} {image}")
    # selected_index = 0
    selected_index = int(input("Enter the index of the file to be processed"))
    clear_output()
    for i, name in enumerate(images):
        if i != selected_index: continue
        path = os.path.join(dirpath, name)
        name = name.replace("roa", "").replace(".tif", "")
        args = name.split("_")

        analysis = MetaAnalyzer(path=path, canvas=(canvas_y, canvas_x), min_slot_size=min_slot_size, threshold=threshold, init_params=init_params, scale=scale, verbose=show_iter_output, max_fit_iter=max_fit_iter)
        if run_fit:
            result = analysis.fit_lattice_lmfit()
            analysis_list.append(analysis)
            print(result.fit_report())
            analysis.print_physical_params(result.params)
        analysis.plot_matplotlib()
        # analysis.plot_plotly()

def test_main():
    # # testing slot main
    fig = plt.figure()
    ax = fig.gca()
    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    mu, w, s = 0, 4, 0.8
    z = SlotModel.gaussian_flat(x, mu=mu, w=w, s=s)
    np.savetxt(f'out/meta_gaussianFlat_x_mu={mu}-w={w}-s={s}.csv', x)
    np.savetxt(f'out/meta_gaussianFlat_z_mu={mu}-w={w}-s={s}.csv', z)
    ax.plot(x, z)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    X, Y = np.meshgrid(x, y)
    l, w, r = 4, 2, 0.2
    Z = SlotModel.gaussian_slot(Y, X, cy=0, cx=0, t=0, l=l, w=w, r=r)
    np.savetxt(f'out/meta_gaussianSlot_z_l={mu}-w={w}-r={r}.csv', Z)
    fig.suptitle("Gaussian Slot")
    # ax.plot_surface(X, Y, Z)
    ax.imshow(Z)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 10, 1000)
    y = np.linspace(0, 10, 1000)
    X, Y = np.meshgrid(x, y)
    dx, dy = 2.5, 2.5,
    t1, t2 = 0, pi/6 
    Z = SlotModel.gaussian_cell(Y, X, my=5, mx=5, py=10, px=10, t1=t1, t2=t2, dy=dy, dx=dx, l=4, w=2, r=0.2)
    np.savetxt(f'out/meta_gaussianCell_z_t1={t1}-t2={t2}dx={dx}-dy={dy}.csv', Z)
    fig.suptitle("Slot Cell")
    # ax.plot_surface(X, Y, Z)
    ax.imshow(Z)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 60, 1500)
    y = np.linspace(0, 40, 1000)
    X, Y = np.meshgrid(x, y)
    dx, dy = 2.5, 2.5,
    t1, t2 = 0, pi/6 
    Z = SlotModel.gaussian_lattice(Y, X, y0=0, x0=0, phi=pi/6/4, py=10, px=10, t1=t1, t2=t2, dy=dy, dx=dx, l=4, w=2, r=0.2)
    np.savetxt(f'out/meta_gaussianLattice_z_t1={t1}-t2={t2}dx={dx}-dy={dy}.csv', Z)
    fig.suptitle("Slot lattice")
    # ax.plot_surface(X, Y, Z)
    ax.imshow(Z)

    plt.show()