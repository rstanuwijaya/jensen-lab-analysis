from cv2 import accumulate, calibrateCamera
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
        R[0, 0] = True
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
    def __init__(self, path, slm_res, camera_res, num_filters, method='zigzag'):
        self.slm_res = slm_res
        self.num_filters = num_filters
        self.method = method
        self.camera_res = camera_res
        self.T = self.generate_image_from_file(path)
        self.h = self.generate_hadamard(self.slm_res)
        self.R = None
        self.I = None
        self.G2 = None

        self.reset_vars()
        
    def reset_vars(self):
        self.R = np.zeros(self.slm_res)
        self.I = np.empty(self.slm_res, dtype=np.ndarray)
        self.G2 = np.zeros(self.slm_res)
        
    def generate_image_from_file(self, path):
        img = Image.open(path).resize(
            (self.slm_res[1], self.slm_res[0]), Image.ANTIALIAS).convert('L')
        img = np.asarray(img)
        return img
    
    def generate_hadamard(self, shape):
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

    def accumulate_I(self, T, Hi):
        I_ = np.zeros(self.camera_res, dtype=np.ndarray)
        T_ = [np.array_split(a, self.camera_res[1],axis=1) for a in np.array_split(T, self.camera_res[0],axis=0)]
        T_ = sum(T_, [])
        Hi_ = [np.array_split(a, self.camera_res[1],axis=1) for a in np.array_split(Hi, self.camera_res[0],axis=0)]
        Hi_ = sum(Hi_, [])

        for i, (Tij, Hij) in enumerate(zip(T_, Hi_)):
            u, v = i // self.camera_res[1], i % self.camera_res[1]
            I_[u, v] = np.sum((Tij * Hij).flatten())
        return I_
    
    def accumulate_G2(self, Hi, I_):
        I_ = np.reshape(I_, (self.camera_res[0] * self.camera_res[1], -1))
        G2_ = np.zeros(self.camera_res, dtype=np.ndarray)
        Hi_ = [np.array_split(a, self.camera_res[1],axis=1) for a in np.array_split(Hi, self.camera_res[0],axis=0)]
        Hi_ = sum(Hi_, [])

        for i, (Iij, Hij) in enumerate(zip(I_, Hi_)):
            u, v = i // self.camera_res[1], i % self.camera_res[1]
            G2_[u, v] = Hij * Iij
        
        G2 = np.array([])
        for u in range(self.camera_res[0]):
            G2_row = np.array([])
            for v in range(self.camera_res[1]):
                G2_row = np.hstack((G2_row, G2_[u, v])) if G2_row.size else G2_[u, v]
            G2 = np.vstack((G2, G2_row)) if G2.size else G2_row
        
        G2 = np.array(G2, dtype=float)
        return G2

    def run_simulation(self):
        self.reset_vars()
        cnt = 0
        if self.method == 'zigzag':
            self.R = PathGenerator.zigzag(self.slm_res, self.num_filters)
        elif self.method == 'circular':
            self.R = PathGenerator.circular(self.slm_res, self.num_filters)
        elif self.method == 'square':
            self.R = PathGenerator.square(self.slm_res, self.num_filters)
                
        for i in range(self.slm_res[0] * self.slm_res[1]):
            u, v = i // self.slm_res[1], i % self.slm_res[1]
            if not self.R[u, v]: continue
            cnt += 1
            Hi = self.generate_filter(self.h, u, v)
            Ii = self.accumulate_I(self.T, Hi)
            self.I[u, v] = Ii

        for i in range(self.slm_res[0] * self.slm_res[1]):
            u, v = i // self.slm_res[1], i % self.slm_res[1]
            if not self.R[u, v]: continue
            Hi = self.generate_filter(self.h, u, v)
            self.G2 = np.add(self.G2, self.accumulate_G2(Hi, self.I[u,v]))

        self.G2 = self.G2 / (self.slm_res[0] * self.slm_res[1])
        return cnt
    
    def calc_rmse(self):
        return math.sqrt(np.sum((self.T - self.G2)**2) / (self.slm_res[0] * self.slm_res[1]))
