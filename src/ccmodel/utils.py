import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy.sparse as sp
from . import utils

def count_frames_txt(path):
    assert path.endswith('.txt')
    cnt = 0
    with open(path) as handle:
        for line in handle:
            cnt += 1
    return cnt

def count_frames_npz(path):
    assert path.endswith('.npz')
    sparse_matrix =  sp.load_npz(path)
    return sparse_matrix.shape[0]

def count_frames_npy(path):
    assert path.endswith('.npy')
    sparse_matrix =  np.load(path)
    return sparse_matrix.shape[0]

def load_txt(path):
    assert path.endswith('.txt')
    bins = pd.read_csv(path, header=None)\
        .to_numpy()\
        .reshape(-1, 32, 32)\
        .transpose((0,2,1))\
        .astype(np.int16)
    return bins

def load_npz(path):
    assert path.endswith('.npz')
    sparse_matrix =  sp.load_npz(path)
    bins = np.array(sparse_matrix.todense())\
        .reshape((-1, 32, 32))
    return bins

def load_npy(path):
    assert path.endswith('.npy')
    bins = np.load(path)\
        .reshape((-1, 32, 32))
    return bins

def save_npz(dirname, name, bins):
    path = os.path.join(dirname, name + '.npz')
    sparse_matrix = sp.csr_matrix(bins.reshape(-1, 32*32))
    sp.save_npz(path, sparse_matrix, compressed=True)

def save_npy(dirname, name, bins):
    path = os.path.join(dirname, name + '.npy')
    np.save(path, bins)

def save(path, bins, convert=None):
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    name = os.path.splitext(basename)[0]

    if convert == 'npz':
        save_npz(dirname, name, bins)
    elif convert == 'npy':
        save_npy(dirname, name, bins)
    else:
        raise('Wrong convert key (npz/npy):', convert)
        
def load(path, convert='npz'):
    if path.endswith('.txt'):
        bins = load_txt(path)
        save(path, bins, convert)
    elif path.endswith('.npz'):
        bins = load_npz(path)
    elif path.endswith('.npy'):
        bins = load_npy(path)
    else:
        raise('Cannot load path specified: invalid extension')
    return bins

    