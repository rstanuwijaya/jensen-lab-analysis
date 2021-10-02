import numpy as np
from math import pi, exp, sin, cos, tan
from scipy.special import erf


class SlotModel:
    @staticmethod
    def gaussian_flat(x, *, mu, w, s):
        return -1. / 2. * (erf((2*mu - w - 2*x)/(2*np.sqrt(2)*s)) - erf((2*mu + w - 2*x)/(2*np.sqrt(2)*s)))

    @staticmethod
    def gaussian_slot(y, x, *, cy, cx, t, l, w, r):
        size = y.shape
        x, y = x.ravel(), y.ravel()
        x, y = np.array([[cos(t), -sin(t)], [sin(t), cos(t)]]
                        ) @ np.vstack((x-cx, y-cy)) + np.vstack((cx, cy))
        x, y = x.reshape(size), y.reshape(size)
        return SlotModel.gaussian_flat(y, mu=cy, w=l, s=r) * SlotModel.gaussian_flat(x, mu=cx, w=w, s=r)

    @staticmethod
    def gaussian_cell(y, x, *, my, mx, py, px, t1, t2, dy, dx, l, w, r):
        cy1, cx1 = my - dy, mx - dx
        cy2, cx2 = my + dy, mx + dx
        return SlotModel.gaussian_slot(y, x, cy=cy1, cx=cx1, t=t1, l=l, w=w, r=r) + SlotModel.gaussian_slot(y, x, cy=cy2, cx=cx2, t=t2, l=l, w=w, r=r)

    @staticmethod
    def gaussian_lattice(y, x, *, y0, x0, phi, py, px, t1, t2, dy, dx, l, w, r):
        size = y.shape
        x, y = x.ravel(), y.ravel()
        x, y = np.array(
            [[cos(phi), -sin(phi)], [sin(phi), cos(phi)]]) @ np.vstack((x, y))
        x, y = x.reshape(size), y.reshape(size)
        x, y = np.mod(x - x0, px), np.mod(y - y0, py)
        return SlotModel.gaussian_cell(y, x, my=py/2, mx=px/2, py=py, px=px, t1=t1, t2=t2, dy=dy, dx=dx, l=l, w=w, r=r)

    @staticmethod
    def gaussian_fit(xdata, y0, x0, phi, py, px, t1, t2, dy, dx, l, w, r):
        y, x = xdata
        img = 255 * SlotModel.gaussian_lattice(
            y, x, y0=y0, x0=x0, phi=phi, py=py, px=px, t1=t1, t2=t2, dy=dy, dx=dx, l=l, w=w, r=r)
        return img
