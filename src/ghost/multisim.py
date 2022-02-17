import numpy as np
import matplotlib.pyplot as plt
from math import floor, prod, sqrt, atan2, pi, ceil, log, log10, inf
import random
import os
from PIL import Image
import cv2
from scipy.ndimage import convolve


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
    def __init__(self, path, shape_slm, shape_cam, shape_mac, num_filters, spacing=(1, 1), shift=(0, 0), sigma=0, method='zigzag', mode='ideal'):
        self.path = path
        self.shape_slm = shape_slm  # SLM resolution
        self.shape_cam = shape_cam  # camera resolution
        self.shape_mac = shape_mac  # macro pixel shape
        self.sigma = sigma
        self.num_filters = num_filters
        self.shift = shift
        self.method = method
        self.mode = mode
        self.spacing = spacing
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
            self.A, self.At = self.generate_cali_matrix_ideal()
        else:
            self.A, self.At = self.generate_cali_matrix_gaussian()
        self.R = np.zeros(self.shape_mac)
        self.I = np.empty((prod(self.spacing)*prod(self.shape_mac), prod(self.shape_cam)))
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

    def select_active_subpixels(self, i):
        X = np.zeros(self.spacing)
        j = i % prod(self.spacing)
        X[j//self.spacing[1], j%self.spacing[0]] = 1
        X = np.kron(np.ones((ceil(self.shape_cam[0]/self.spacing[0]), ceil(self.shape_cam[1]/self.spacing[1]))), X)
        X = X[:self.shape_cam[0], :self.shape_cam[1]]
        return X

    def generate_filter(self, i):
        # tiled filter
        Si = self.generate_partial_filter(i)
        shape = ceil(self.shape_slm[0] / self.shape_mac[0]
                     ), ceil(self.shape_slm[1] / self.shape_mac[1])
        Si = np.tile(Si, shape)
        Si = Si[:self.shape_slm[0], :self.shape_slm[1]]
        return Si

    def generate_filter2(self, i):
        Si = self.generate_partial_filter(i//prod(self.spacing))
        X = self.select_active_subpixels(i)
        Si = np.kron(X, Si)
        return Si

    def generate_single_pass_filter(self, i):
        # generate filter for one camera pixel
        u, v = i // self.shape_slm[1], i % self.shape_slm[1]
        S = np.zeros(self.shape_slm)
        S[u, v] = 1
        
        return S

    def generate_cali_pattern(self, i):
        # i range: prod(spacing) * prod(shape_mac)
        X = self.select_active_subpixels(i)
        j = i // prod(self.spacing)
        S = np.zeros(self.shape_mac)
        u, v = j // self.shape_mac[1], j % self.shape_mac[1]
        S[u, v] = 1
        X = np.kron(X, np.ones(self.shape_mac))
        S = np.kron(np.ones(self.shape_cam), S)
        S = S * X
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

        for i in range(prod(self.spacing) * prod(self.shape_mac)):
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
            Ik = self.A.T @ (Tk*Sk)
            Ik = Ik.reshape(self.shape_cam)
            Ik = convolve(Ik, np.ones(self.spacing), mode='constant')
            Ik = Ik * self.select_active_subpixels(i)
            Ik = Ik.flatten()

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
    def __init__(self, path, shape_slm, shape_cam, shape_mac, crop, num_filters, spacing=(1, 1), shift=(0, 0), sigma=0, method='zigzag', mode='ideal'):
        super().__init__(path, shape_slm, shape_cam,
                         shape_mac, num_filters, spacing, shift, sigma, method, mode)
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
        for i in range(prod(self.spacing) * prod(self.shape_mac)):
            j = i % prod(self.shape_mac)
            u, v = j // self.shape_mac[1], j % self.shape_mac[1]
            if not self.R[u, v]:
                continue
            k += 1

            # generate and shift the filter pattern
            Sk = self.generate_filter2(i)
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

            Ik = Ik.reshape(self.shape_cam)
            Ik = convolve(Ik, np.ones(self.spacing), mode='constant')
            Ik = Ik * self.select_active_subpixels(i)
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
        phase_gradient = np.mod(8*k*X*np.sin(theta), 2*pi) / (2 * pi)

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

class AlignmentHelper:
    def __init__(self, slm_res):
        self.shape_slm = slm_res
    
    def ones(self):
        M = np.ones(self.shape_slm)
        return M

    def border(self, w):
        shape_slm = self.shape_slm
        M = np.zeros((shape_slm[0], shape_slm[1]))
        M[:w, :] = 1
        M[:, :w] = 1
        M[-w:, :] = 1
        M[:, -w:] = 1
        return M
 
    def arrow(self):
        res = self.shape_slm
        lud, rud = (res[1]//2, res[1]//2), (res[1]//2, res[1]//2)
        lbd, rbd = (res[0] - lud[0], lud[1]), (rud[0], res[0] - rud[1]) 
        RU = np.tri(lud[0])
        LU = np.flip(np.tri(rud[0]), axis=1)
        LB = np.hstack((np.zeros((lbd[0], ceil(lbd[1]/2))), np.ones((lbd[0], floor(lbd[1]/ 2)))))
        RB = np.hstack((np.ones((rbd[0], floor(rbd[1]/2))), np.zeros((rbd[0], ceil(rbd[1]/ 2)))))
        M = np.vstack((np.hstack((LU, RU)), np.hstack((LB, RB))))
        return M

    def squares(self, w, rep):
        shape_slm = self.shape_slm
        lu, ru = (0, 0), (0, shape_slm[1])
        lb, rb = (shape_slm[0], 0), (shape_slm[0], shape_slm[1])
        M = np.zeros((shape_slm[0], shape_slm[1]))

        for i in range(rep):
            M[lu[0]:lb[0], lu[1]:lb[1]+w] = 1
            M[ru[0]:rb[0], ru[1]-w:rb[1]] = 1
            M[lu[0]:ru[0]+w, lu[1]:ru[1]] = 1
            M[lb[0]-w:rb[0], lb[1]:rb[1]] = 1

            lu = (lu[0]+2*w, lu[1]+2*w)
            ru = (ru[0]+2*w, ru[1]-2*w)
            lb = (lb[0]-2*w, lb[1]+2*w)
            rb = (rb[0]-2*w, rb[1]-2*w)

        return M

    def checkerboard(self, w):
        shape_slm = self.shape_slm
        reps = shape_slm[0]//w, shape_slm[1]//w
        tile = np.array([[0, 1], [1, 0]]).repeat(w, axis=0).repeat(w, axis=1)
        M = np.kron(np.ones(reps), tile)  
        M = M[:shape_slm[0], :shape_slm[1]]
        return M
    
    def from_img(self, path, threshold=None):
        img = Image.open(path)
        img = img.resize(self.shape_slm).convert('L')
        M = np.array(img)/255
        if threshold is not None:
            M = M > threshold
        return M