import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import time

# constants
n_tb = 1024
# N = frame time / camera freq (20MHz)
tbsize = 56
tcc = 4
# eta = 0.003
eta = 2e6/128/50000/100/1024
debug = False


def ver():
    print('v2')


class FitModel:
    def __init__(self, xdata, ydata, fitradius, init_params, name):
        self.name = name
        self.xdata = xdata
        self.ydata = ydata
        self.fitradius = fitradius
        self.init_params = init_params

    def model(self, x, N, m, delta_T, tw, tau0, A, b, Z):
        N = int(N)
        P = {tau: 0 for tau in range(-N, N+1)}

        def calculate_P_nonzero(k):
            # return np.sum([eta**2 * (1-eta)**(2*(j-1)+abs(k)) for j in range(1, N-abs(k) + 1)])
            return (N - abs(k))

        def calculate_P_zero(k):
            # return np.sum([eta**2 * (1-eta)**(2*(j-1)) * m*m for j in range(1, N + 1)])
            # return np.sum([eta**2 * (1-eta)**(2*(j-1)) * m for j in range(1, N + 1)])
            return m*N

        for k in P.keys():
            if k == 0:
                P[k] = calculate_P_zero(k)
            else:
                P[k] = calculate_P_nonzero(k)

        def prob(tau_):
            # return 1
            a = (Z-abs(tau_))/n_tb
            return a if a > 0 else 0

        def erf(tau_, tcc_, k_, delta_T):
            # return math.erf((tcc_ + 2*(tau_ + k_*delta_T)) / (math.sqrt(2)*tw/math.sqrt(math.log(2)))) # use gaussian distribution
            return math.atan((2*(tau_ + k_*delta_T) + tcc_) / (2*tw)) # use lorentzian distribution

        def is_even(x):
            return 1 if x % 2 == 0 else -1

        def terms(tau, delta_T, k_):
            result = 0
            alias = 2

            for i in range(-alias, alias+1):
                if i not in (0,):
                    continue
                # if i not in (-1, 1): continue
                # if i not in (-2, 2): continue
                result += (-1 * is_even(i)) * (erf(tau + i*Z, tcc,
                                                k_, delta_T) - erf(tau + i*Z, -tcc, k_, delta_T))

            return abs(result)

        def calculate_total_prob_cc(tau, delta_T):
            return A * prob(tau) * P[0] * terms(tau, delta_T, 0)

        def calculate_total_prob_ac(tau, delta_T):
            return A * prob(tau) * np.sum([(P[k] * terms(tau, delta_T, k)) for k in range(-(N-1), (N-1)+1) if k != 0])

        def calculate_total_prob_noise(tau, delta_T):
            return A * b * prob(tau)

        total_prob_cc = {tau: calculate_total_prob_cc(
            tau, delta_T) for tau in range(-self.fitradius, self.fitradius+1)}
        total_prob_ac = {tau: calculate_total_prob_ac(
            tau, delta_T) for tau in range(-self.fitradius, self.fitradius+1)}
        total_prob_noise = {tau: calculate_total_prob_noise(
            tau, delta_T) for tau in range(-self.fitradius, self.fitradius+1)}
        total_prob_all = {
            tau: total_prob_cc[tau]+total_prob_ac[tau]+total_prob_noise[tau] for tau in range(-self.fitradius, self.fitradius+1)}
        self.total_prob_cc = total_prob_cc
        self.total_prob_ac = total_prob_ac
        self.total_prob_noise = total_prob_noise
        self.total_prob_all = total_prob_all
        self.params = (m, delta_T, tw, tau0, eta, A, b)

        return list(total_prob_all.values())

    def plot_model_sep(self, lim=None):
        if lim is None:
            lim = self.fitradius
        plt.plot(*zip(*[(k, v) for k, v in self.total_prob_cc.items()]))
        plt.plot(*zip(*[(k, v) for k, v in self.total_prob_ac.items()]))
        plt.plot(*zip(*[(k, v) for k, v in self.total_prob_noise.items()]))
        plt.xlim(xmin=-lim, xmax=lim)
        plt.title(f'model sep {self.name}')
        # plt.ylim(ymin=0)
        plt.show()
        plt.close()

    def plot_model_all(self, lim=None):
        if lim is None:
            lim = self.fitradius
        plt.plot(list(range(-self.fitradius, self.fitradius+1)), self.ydata)
        plt.plot(*zip(*[(k, v) for k, v in self.total_prob_all.items()]))
        plt.xlim(xmin=-lim, xmax=lim)
        plt.title(f'model all {self.name}')
        # plt.ylim(ymin=0)
        plt.show()
        plt.close()

    def get_ydata_cc(self):
        ydata_cc = np.array(self.ydata) * np.array(list(self.total_prob_cc.values())
                                                   ) / np.array(list(self.total_prob_all.values()))
        return ydata_cc

    def plot_ydata_cc(self, lim=None):
        if lim is None:
            lim = self.fitradius
        ydata_cc = self.get_ydata_cc()

        plt.plot(list(range(-self.fitradius, self.fitradius+1)), self.ydata)
        plt.plot(list(range(-self.fitradius, self.fitradius+1)), ydata_cc)
        plt.xlim(xmin=-lim, xmax=lim)
        plt.title(f'ydata cc only {self.name}')
        plt.ylim(ymin=0)
        plt.show()
        plt.close()

    def get_ydata_ac(self):
        ydata_ac = np.array(self.ydata) * np.array(list(self.total_prob_ac.values())
                                                   ) / np.array(list(self.total_prob_all.values()))
        return ydata_ac

    def plot_ydata_ac(self, lim=None):
        if lim is None:
            lim = self.fitradius
        ydata_ac = self.get_ydata_ac()

        plt.plot(list(range(-self.fitradius, self.fitradius+1)), self.ydata)
        plt.plot(list(range(-self.fitradius, self.fitradius+1)), ydata_ac)
        plt.xlim(xmin=-lim, xmax=lim)
        plt.title(f'ydata ac only {self.name}')
        plt.ylim(ymin=0)

        plt.show()
        plt.close()

    def get_center_cc(self):
        ydata_cc = np.array(list(self.total_prob_cc.values()))
        return np.sum(self.total_prob_cc[self.fitradius-tcc//2: self.fitradius+tcc//2+1])
        # return ydata_cc[self.fitradius]

    def get_center_ac(self):
        ydata_ac = np.array(list(self.total_prob_ac.values()))
        return np.sum(ydata_ac[self.fitradius-tcc//2: self.fitradius+tcc//2+1])
        # return ydata_cc[self.fitradius]

    def get_subtractAC(self):
        return np.array(self.ydata) - np.array(list(self.total_prob_ac.values())) - np.array(list(self.total_prob_noise.values()))

    def plot_ydata_subtractAC(self, lim=None):
        if lim is None:
            lim = self.fitradius
        ydata_subtractAC = self.get_subtractAC()

        plt.plot(list(range(-self.fitradius, self.fitradius+1)), self.ydata)
        plt.plot(list(range(-self.fitradius, self.fitradius+1)), ydata_subtractAC)
        plt.xlim(xmin=-lim, xmax=lim)
        plt.title(f'ydata subtract ac model {self.name}')
        plt.show()
        plt.close()

    def get_center_subtractAC(self):
        subtractAC = self.get_subtractAC()
        return np.sum(subtractAC[self.fitradius-tcc//2: self.fitradius+tcc//2+1])

    def get_center_ydata(self):
        ydata = self.ydata
        return np.sum(ydata[self.fitradius-tcc//2: self.fitradius+tcc//2+1])

    def print_params(self):
        print('-----------------------------------------------------')
        print('{:10} {}'.format('name', self.name))
        params = zip(('m', 'delta_T', 'tw', 'tau0',
                     'eta', 'A', 'b'), self.params)
        for p in params:
            print('{:10} {}'.format(p[0], p[1]))
        print()
        print('{:10} {}'.format('ctr peak', self.get_center_subtractAC()))
        print('-----------------------------------------------------')

    def get_params(self):
        return self.params
