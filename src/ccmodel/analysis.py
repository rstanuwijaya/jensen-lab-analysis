import os
import numpy as np

def cc_vs_delay(dirpath, cw):
    os.chdir(dirpath)
    fnames = os.listdir(dirpath)
    positions = list(map(lambda fname: os.path.basename(fname).split('_')[0], fnames))
    counts = []

    for i in range(len(fnames)):
        temp = np.loadtxt(fnames[i])
        cc = sum(temp[1024-cw:1024+cw+1])
        counts.append(cc)
    
    return (positions, counts)

def accumulated_count_vs_delay(dirpath):
    os.chdir(dirpath)
    fnames = os.listdir(dirpath)
    positions = list(map(lambda fname: os.path.basename(fname).split('_')[0], fnames))
    counts = []

    for i in range(len(fnames)):
        temp = np.loadtxt(fnames[i])
        cc = sum(temp[1:])
        counts.append(cc)
    
    return (positions, counts)
