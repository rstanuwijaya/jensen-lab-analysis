import numpy as np
import matplotlib.pyplot as plt
from math import floor, prod, sqrt, atan2, pi, ceil, log, log10, inf
import random
import os
from PIL import Image
import cv2


class PathGenerator:
    def zigzag(shape, n):
        R = np.zeros(shape, dtype=bool)
        L = dict()
        for i in range(shape[0] * shape[1]):
            u, v = i // shape[1], i % shape[1]
            r, t = u + v, u
            L[(u, v)] = (r, t)
        L = {k: v for k, v in sorted(
            L.items(), key=lambda x: (x[1][0], x[1][1]))}
        for i, (u, v) in enumerate(L.keys()):
            if i < n:
                R[u, v] = True
        return R

    def circular(shape, n):
        R = np.zeros(shape, dtype=bool)
        L = dict()
        for i in range(shape[0] * shape[1]):
            u, v = i // shape[1], i % shape[1]
            r, t = sqrt(u**2 + v**2), atan2(u, v)
            L[(u, v)] = (r, t)
        L = {k: v for k, v in sorted(
            L.items(), key=lambda x: (x[1][0], x[1][1]))}
        for i, (u, v) in enumerate(L.keys()):
            if i < n:
                R[u, v] = True
        return R

    def square(shape, n):
        R = np.zeros(shape, dtype=bool)
        L = dict()
        for i in range(shape[0] * shape[1]):
            u, v = i // shape[1], i % shape[1]
            r = max(u, v)
            t = v if u < v else u + v
            L[(u, v)] = (r, t)
        L = {k: v for k, v in sorted(
            L.items(), key=lambda x: (x[1][0], x[1][1]))}
        for i, (u, v) in enumerate(L.keys()):
            if i < n:
                R[u, v] = True
        return R


class GhostSimulator:
    def __init__(self, path, shape_slm, shape_cam, shape_mac, num_filters, shift=(0, 0), sigma=0, method='zigzag', mode='ideal'):
        self.path = path
        self.shape_slm = shape_slm  # SLM resolution
        self.shape_cam = shape_cam  # camera resolution
        self.shape_mac = shape_mac  # macro pixel shape
        self.sigma = sigma
        self.num_filters = num_filters
        self.shift = shift
        self.method = method
        self.mode = mode
        self.T = None  # 2d target image
        self.h = self.generate_hadamard()  # 2d hadamard matrix
        self.A = None
        self.At = None
        self.R = None
        self.I = None
        self.G2 = None  # 2d reconstructed image

        self.reset_vars()

    def reset_vars(self):
        if self.mode == 'ideal':
            if self.sigma == 0:
                self.A, self.At = self.generate_cali_matrix_ideal()
        else:
            self.A, self.At = self.generate_cali_matrix_gaussian()
        self.R = np.zeros(self.shape_mac)
        self.I = np.empty((prod(self.shape_mac), prod(self.shape_cam)))
        self.I[:] = np.nan
        self.G2 = np.zeros(self.shape_slm)

    def generate_cali_matrix_ideal(self):
        A = np.zeros((prod(self.shape_slm), prod(self.shape_cam)))
        for i in range(prod(self.shape_slm)):
            u, v = i // self.shape_slm[1], i % self.shape_slm[1]
            slm = np.zeros(self.shape_slm)
            slm[u, v] = 1
            cam = cv2.resize(slm, self.shape_cam, interpolation=cv2.INTER_AREA)
            cam = cam * prod(self.shape_slm) / prod(self.shape_cam)
            A[i, :] = cam.flatten()

        S = A.T.sum(axis=1, keepdims=True)
        At = A.T/S
        return A, At

    def generate_cali_matrix_ideal2(self):
        cam_res = self.shape_cam
        slm_res = self.shape_slm
        mac_res = self.shape_mac
        sigma = self.sigma
        At = np.zeros((prod(cam_res), prod(slm_res)))
        for i in range(prod(cam_res)):
            u, v = i // cam_res[1], i % cam_res[1]
            x = np.arange(0, slm_res[0], 1)
            y = np.arange(0, slm_res[1], 1)
            xv, yv = np.meshgrid(x, y)
            z = np.logical_and(abs((xv - (v + 0.5)*mac_res[0] + 0.5)) <= sigma, abs((yv - (u+0.5)*mac_res[1]) + 0.5) <= sigma)
            At[i] = z.flatten()
        # S = At.T.sum(axis=1, keepdims=True)
        # A = At.T/S
        A = At.T
        return A, At


    def generate_cali_matrix_gaussian(self):
        cam_res = self.shape_cam
        slm_res = self.shape_slm
        mac_res = self.shape_mac
        sigma = self.sigma
        At = np.zeros((prod(cam_res), prod(slm_res)))
        for i in range(prod(cam_res)):
            u, v = i // cam_res[1], i % cam_res[1]
            x = np.arange(0, slm_res[0], 1)
            y = np.arange(0, slm_res[1], 1)
            xv, yv = np.meshgrid(x, y)
            xy = (xv - (v+0.5)*mac_res[0] + 0.5)**2 + \
                (yv - (u+0.5)*mac_res[1] + 0.5)**2
            temp = 1/(2*np.pi*sigma**2)*np.exp(-xy/(2*sigma**2))
            At[i] = temp.flatten()
        S = At.T.sum(axis=1, keepdims=True)
        A = At.T/S

        return A, At

    def generate_image_from_file(self, path):
        img = Image.open(path).resize(
            (self.shape_slm[1], self.shape_slm[0]), Image.ANTIALIAS).convert('L')
        img = np.asarray(img)
        return img

    def generate_hadamard(self):
        order = ceil(log(max(self.shape_mac[0], self.shape_mac[1]), 2))
        h = np.array([[1, 1], [1, -1]])
        for i in range(order-1):
            h = np.kron(h, np.array([[1, 1], [1, -1]]))

        # sort hadamard by frequency
        def calculate_flip(H):
            return sum([H[i] != H[i+1] for i in range(H.shape[0]-1)])

        h = np.array(sorted(h, key=lambda row: calculate_flip(row)))
        h = h[:self.shape_mac[0], :self.shape_mac[1]]

        return h

    def generate_partial_filter(self, i):
        # generate filter for one camera pixel
        u, v = i // self.shape_mac[1], i % self.shape_mac[1]
        h = self.generate_hadamard()
        S = h[:, [v]] @ h[[u], :]
        return S

    def generate_filter(self, i):
        # tiled filter
        Si = self.generate_partial_filter(i)
        shape = ceil(self.shape_slm[0] / self.shape_mac[0]
                     ), ceil(self.shape_slm[1] / self.shape_mac[1])
        Si = np.tile(Si, shape)
        Si = Si[:self.shape_slm[0], :self.shape_slm[1]]
        return Si

    def generate_filter2(self, i):
        Si = self.generate_partial_filter(i//(3*3))
        X = np.zeros((3, 3))
        j = i % (3*3)
        X[j//3, j%3] = 1
        X = np.kron(np.ones((ceil(self.shape_cam[0]/3), ceil(self.shape_cam[1]/3))), X)
        X = X[:self.shape_cam[0], :self.shape_cam[1]]
        Si = np.kron(X, Si)
        return Si

    def generate_single_pass_filter(self, i):
        # generate filter for one camera pixel
        u, v = i // self.shape_slm[1], i % self.shape_slm[1]
        S = np.zeros(self.shape_slm)
        S[u, v] = 1
        
        return S

    def run_simulation(self):
        self.reset_vars()
        self.T = self.generate_image_from_file(self.path)  # 2d target image
        k = 0  # count of filters
        G2 = np.zeros(prod(self.shape_slm))
        if self.method == 'zigzag':
            self.R = PathGenerator.zigzag(self.shape_mac, self.num_filters)
        elif self.method == 'circular':
            self.R = PathGenerator.circular(self.shape_mac, self.num_filters)
        elif self.method == 'square':
            self.R = PathGenerator.square(self.shape_mac, self.num_filters)

        Tk = self.T.flatten()
        self.I = np.empty((9*prod(self.shape_mac), prod(self.shape_cam)))

        A, _ = self.generate_cali_matrix_ideal2()
        self.A, self.At = self.generate_cali_matrix_ideal()
        for i in range(9 * prod(self.shape_mac)):
            j = i % prod(self.shape_mac)
            u, v = j // self.shape_mac[1], j % self.shape_mac[1]
            if not self.R[u, v]:
                continue
            k += 1
            Sk = self.generate_filter2(i)  # generate filter pattern
            Sk = np.roll(Sk, self.shift[0], axis=0)
            Sk = np.roll(Sk, self.shift[1], axis=1)
            Sk = Sk.flatten()

            # simulate measurement
            Ik = A.T @ (Tk*Sk)
            self.I[i, :] = Ik.T  # store to the intensity matrix

            # reconstruct image
            Sk = self.generate_filter2(i).flatten()  # generate filter pattern
            G2 += (self.A @ Ik) * Sk

        G2 = G2 / prod(self.shape_mac)
        self.G2 = G2.reshape(self.shape_slm)
        return k

    def calc_rmse(self):
        return sqrt(np.sum((self.T - self.G2)**2) / (self.shape_slm[0] * self.shape_slm[1]))

    def calc_psnr(self):
        if self.calc_rmse() == 0:
            return inf
        return 20*log10(255/self.calc_rmse())


class GhostAnalyser(GhostSimulator):
    def __init__(self, path, shape_slm, shape_cam, shape_mac, crop, num_filters, shift=(0, 0), sigma=0, method='zigzag'):
        super().__init__(path, shape_slm, shape_cam,
                         shape_mac, num_filters, shift, sigma, method)
        self.crop = crop

    def run_simulation(self):
        raise NotImplementedError("Class not for simulation")

    def run_analysis(self):
        self.reset_vars()
        k = 0  # count of filters
        G2 = np.zeros(prod(self.shape_slm))
        if self.method == 'zigzag':
            self.R = PathGenerator.zigzag(self.shape_mac, self.num_filters)
        elif self.method == 'circular':
            self.R = PathGenerator.circular(self.shape_mac, self.num_filters)
        elif self.method == 'square':
            self.R = PathGenerator.square(self.shape_mac, self.num_filters)

        # Tk = self.T.flatten()
        for i in range(self.shape_mac[0] * self.shape_mac[1]):
            u, v = i // self.shape_mac[1], i % self.shape_mac[1]
            if not self.R[u, v]:
                continue
            k += 1

            # generate and shift the filter pattern
            Sk = self.generate_filter(i)
            Sk = np.roll(Sk, self.shift[0], axis=0)
            Sk = np.roll(Sk, self.shift[1], axis=1)
            Sk = Sk.flatten()

            # difference imaging and resizing
            Ik_p = np.loadtxt(self.path + f'{i}p.csv', delimiter=',')
            Ik_m = np.loadtxt(self.path + f'{i}m.csv', delimiter=',')
            Ik = Ik_p - Ik_m
            Ik = Ik[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
            Ik = cv2.resize(Ik, self.shape_cam, interpolation=cv2.INTER_AREA)
            Ik = Ik.flatten()

            # reconstruct image
            G2 += (self.A @ Ik) * Sk

        G2 = G2 / prod(self.shape_mac)
        self.G2 = G2.reshape(self.shape_slm)
        return k


class SLMPattern:
    def __init__(self, lens_size, slm_res, k, theta):
        self.lens_size = lens_size
        self.slm_res = slm_res
        self.xp_size = ceil(slm_res[0] / lens_size)
        self.yp_size = ceil(slm_res[1] / lens_size)
        self.k = k
        self.theta = theta

    def generate_pattern_grad(self, amp_pattern, ph_pattern):
        k, theta = self.k, self.theta
        lens_size = self.lens_size
        xp_size, yp_size = self.xp_size, self.yp_size

        def pixel(a, b, xg, yg): return np.array([[a, b], [b, a]]).repeat(xg, axis=2).repeat(
            yg, axis=3).transpose(2, 0, 3, 1).reshape(2*a.shape[0]*xg, 2*a.shape[1]*yg)

        def p1(ph, amp): return ph + np.arccos(amp)
        def p2(ph, amp): return ph - np.arccos(amp)
        ph_ref = pi/2

        p1_data = p1(ph_pattern + ph_ref, amp_pattern) / (2 * pi)
        p2_data = p2(ph_pattern + ph_ref, amp_pattern) / (2 * pi)

        amp_phase = pixel(p1_data, p2_data, lens_size//2, lens_size//2)
        amp_phase = amp_phase[:self.slm_res[0], :self.slm_res[1]]
        X, Y = np.mgrid[0:lens_size*xp_size, 0:lens_size*yp_size]
        phase_gradient = np.mod(8*k*X*np.sin(theta), 2*pi)

        PG = np.kron(amp_pattern > 0, np.ones(
            (lens_size, lens_size))) * phase_gradient
        PG = PG[:self.slm_res[0], :self.slm_res[1]]

        tot_phase = np.mod(amp_phase + PG, 1)
        return tot_phase

    def generate_pattern_binary(self, amp_pattern, ph_pattern):
        k, theta = self.k, self.theta
        lens_size = self.lens_size
        xp_size, yp_size = self.xp_size, self.yp_size
        def pixel(a, b, xg, yg): return np.array([[a, b], [b, a]]).repeat(xg, axis=2).repeat(
            yg, axis=3).transpose(2, 0, 3, 1).reshape(2*a.shape[0]*xg, 2*a.shape[1]*yg)

        def p1(ph, amp): return ph + np.arccos(amp)
        def p2(ph, amp): return ph - np.arccos(amp)
        ph_ref = pi/2

        p1_data = p1(ph_pattern + ph_ref, amp_pattern) / (2 * pi)
        p2_data = p2(ph_pattern + ph_ref, amp_pattern) / (2 * pi)

        amp_phase = pixel(p1_data, p2_data, lens_size//2, lens_size//2)
        amp_phase = amp_phase[:self.slm_res[0], :self.slm_res[1]]

        tot_phase = amp_phase
        return tot_phase
