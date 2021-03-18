#%%
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os

os.chdir(os.path.join(os.path.dirname(__file__), 'testdir'))

name = '2690816_Frame5_Exp250_iter1'
densemat = pd.read_csv(name + ".txt", header=None).to_numpy().reshape(50000, 32,32).transpose((0,2,1)).astype(np.int16).reshape(50000, 32*32)
print(densemat.shape)
sparsemat = sp.csr_matrix(densemat)
sp.save_npz(name + '.npz', sparsemat, compressed=False)

loadsparsemat = sp.load_npz(name + '.npz')
print(loadsparsemat.shape)
loaddensemat = np.array(loadsparsemat.todense()).reshape(50000, 32, 32)
print(loaddensemat.shape)
# %%
