import numpy as np
import matplotlib.pyplot as plt
from math import prod, sqrt, atan2, pi, ceil, log, log10, inf
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
            L[(u,v)] = (r,t)
        L = {k: v for k, v in sorted(L.items(), key=lambda x: (x[1][0], x[1][1]))}
        for i, (u, v) in enumerate(L.keys()):
            if i < n: R[u, v] = True
        return R
    def circular(shape, n):
        R = np.zeros(shape, dtype=bool)
        L = dict()
        for i in range(shape[0] * shape[1]):
            u, v = i // shape[1], i % shape[1]
            r, t = sqrt(u**2 + v**2), atan2(u, v)
            L[(u,v)] = (r,t)
        L = {k: v for k, v in sorted(L.items(), key=lambda x: (x[1][0], x[1][1]))}
        for i, (u, v) in enumerate(L.keys()):
            if i < n: R[u, v] = True
        return R
    def square(shape, n):
        R = np.zeros(shape, dtype=bool)
        L = dict()
        for i in range(shape[0] * shape[1]):
            u, v = i // shape[1], i % shape[1]
            r = max(u, v)
            t = v if u < v else u + v
            L[(u,v)] = (r,t)
        L = {k: v for k, v in sorted(L.items(), key=lambda x: (x[1][0], x[1][1]))}
        for i, (u, v) in enumerate(L.keys()):
            if i < n: R[u, v] = True
        return R
class GhostSimulator:
    def __init__(self, path, shape_slm, shape_cam, num_filters, method='zigzag'):
        self.shape_slm = shape_slm # SLM resolution
        self.shape_cam = shape_cam # camera resolution
        self.shape_mac = (ceil(shape_slm[0] // shape_cam[0]), ceil(shape_slm[1] // shape_cam[1])) # macro pixel shape
        self.num_filters = num_filters
        self.method = method
        self.T = self.generate_image_from_file(path) # flattened image
        self.h = self.generate_hadamard(self.shape_mac)
        self.A = self.generate_cali_matrix()
        self.R = None
        self.I = None
        self.G2 = None

        self.reset_vars()
        
    def reset_vars(self):
        self.R = np.zeros(self.shape_mac)
        self.I = np.empty((prod(self.shape_mac), prod(self.shape_cam)))
        self.I[:] = np.nan
        self.G2 = np.zeros(self.shape_slm)
    
    def generate_cali_matrix(self):
        A = np.zeros((prod(self.shape_slm), prod(self.shape_cam)))
        for i in range(prod(self.shape_slm)):
            u, v = i // self.shape_slm[1], i % self.shape_slm[1]
            slm = np.zeros(self.shape_slm)
            slm[u, v] = 1
            cam = cv2.resize(slm, self.shape_cam, interpolation=cv2.INTER_AREA)
            cam = cam * prod(self.shape_slm) / prod(self.shape_cam)
            if i == 0:
                plt.imshow(cam)
                plt.plot()
            A[i, :] = cam.flatten()
        return A

    def generate_image_from_file(self, path):
        img = Image.open(path).resize(
            (self.shape_slm[1], self.shape_slm[0]), Image.ANTIALIAS).convert('L')
        img = np.asarray(img)
        img = img.flatten()
        return img
    
    @staticmethod
    def generate_hadamard(shape):
        '''
        generate Hadamard matrix of size shape with
        '''
        order = ceil(log(max(shape[0], shape[1]), 2))
        h = np.array([[1, 1], [1, -1]])
        for i in range(order-1):
            h = np.kron(h, np.array([[1, 1], [1, -1]]))
        
        # sort hadamard by frequency
        def calculate_flip(H):
            return sum([H[i] != H[i+1] for i in range(H.shape[0]-1)])

        h = np.array(sorted(h, key=lambda row: calculate_flip(row)))
        h = h[:shape[0], :shape[1]]
        return h

    def generate_partial_filter(self, i):
        # generate filter for one camera pixel
        u, v = i // self.shape_mac[1], i % self.shape_mac[1]
        h = self.h
        S = h[:,[v]] @ h[[u], :]
        return S

    def generate_filter(self, i):
        # tiled filter
        Si = self.generate_partial_filter(i)
        Si = np.tile(Si, self.shape_cam)
        Si = Si[:self.shape_slm[0], :self.shape_slm[1]]
        return Si

    def run_simulation(self):
        self.reset_vars()
        cnt = 0
        G2 = np.zeros(prod(self.shape_slm))
        if self.method == 'zigzag':
            self.R = PathGenerator.zigzag(self.shape_mac, self.num_filters)
        elif self.method == 'circular':
            self.R = PathGenerator.circular(self.shape_mac, self.num_filters)
        elif self.method == 'square':
            self.R = PathGenerator.square(self.shape_mac, self.num_filters)

        for i in range(self.shape_mac[0] * self.shape_mac[1]):
            u, v = i // self.shape_mac[1], i % self.shape_mac[1]
            if not self.R[u, v]: continue
            cnt += 1
            Si = self.generate_filter(i).flatten()
            Ti = self.T.flatten()
            Ii = (Ti*Si) @ self.A
            self.I[i, :] = Ii
            G2 += Ii @ self.A.T * Si
        
        G2 = G2 / cnt
        self.G2 = G2.reshape(self.shape_slm)
        return cnt
    
    def calc_rmse(self):
        return sqrt(np.sum((self.T - self.G2)**2) / (self.shape_slm[0] * self.shape_slm[1]))
    def calc_psnr(self):
        if self.calc_rmse() == 0: return inf
        return 20*log10(255/self.calc_rmse())
