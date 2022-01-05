import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os
from PIL import Image

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
            r, t = math.sqrt(u**2 + v**2), math.atan2(u, v)
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
    def __init__(self, path, shape, num_filters, method='zigzag'):
        self.shape = shape
        self.num_filters = num_filters
        self.method = method
        self.T = self.generate_image_from_file(path)
        self.h = self.generate_hadamard(self.shape)
        self.R = None
        self.I = None
        self.G2 = None

        self.reset_vars()
        
    def reset_vars(self):
        self.R = np.zeros(self.shape)
        self.I = np.empty(self.shape)
        self.I[:] = np.nan
        self.G2 = np.zeros(self.shape)
        
    def generate_image_from_file(self, path):
        img = Image.open(path).resize(
            (self.shape[1], self.shape[0]), Image.ANTIALIAS).convert('L')
        img = np.asarray(img)
        return img
    
    @staticmethod
    def generate_hadamard(shape):
        '''
        generate Hadamard matrix of size shape with
        '''
        order = math.ceil(math.log(max(shape[0], shape[1]), 2))
        h = np.array([[1, 1], [1, -1]])
        for i in range(order-1):
            h = np.kron(h, np.array([[1, 1], [1, -1]]))
        
        # sort hadamard by frequency
        def calculate_flip(H):
            return sum([H[i] != H[i+1] for i in range(H.shape[0]-1)])

        h = np.array(sorted(h, key=lambda row: calculate_flip(row)))
        return h[:shape[0], :shape[1]]

    @staticmethod
    def generate_filter(h, u, v):
        return h[:,[v]] @ h[[u], :]

    def run_simulation(self):
        self.reset_vars()
        cnt = 0
        if self.method == 'zigzag':
            self.R = PathGenerator.zigzag(self.shape, self.num_filters)
        elif self.method == 'circular':
            self.R = PathGenerator.circular(self.shape, self.num_filters)
        elif self.method == 'square':
            self.R = PathGenerator.square(self.shape, self.num_filters)
        
        for i in range(self.shape[0] * self.shape[1]):
            u, v = i // self.shape[1], i % self.shape[1]
            if not self.R[u, v]: continue
            cnt += 1
            Hi = self.generate_filter(self.h, u, v)
            Ii = np.sum((self.T*Hi).flatten())
            self.I[u, v] = Ii
            self.G2 += Hi*Ii
        
        self.G2 = self.G2 / (self.shape[0] * self.shape[1])
        return cnt
    
    def calc_rmse(self):
        return math.sqrt(np.sum((self.T - self.G2)**2) / (self.shape[0] * self.shape[1]))
    def calc_psnr(self):
        if self.calc_rmse() == 0: return math.inf
        return 20*math.log10(255/self.calc_rmse())
