#%% testing Filtered F implementation
import numpy as np
import matplotlib.pyplot as plt

height = 3 
width = 3
num_px = height * width
cw = 2
np.random.seed(4)

adjusted_bins = np.random.randint(-10, 10, (height, width), dtype=int)
adjusted_bins[adjusted_bins < 0] = 0

F = np.ones((num_px, num_px), dtype=int)
diag = (np.arange(num_px), np.arange(num_px))
F[diag] = 0
print('F-matrix\n', F)

print('adjusted bins\n', adjusted_bins)

A = adjusted_bins.flatten()
print('A-matrix:\n', A)

D = A.nonzero()
print('D-matrix:\n', D)

A_D = A[D]
print('A_D-matrix:\n', A_D)

F_D = F[D].transpose()[D]
print('F_D-matrix:\n', F_D)

T_D = A_D.reshape(1, -1) - A_D.reshape(-1, 1)
print('T_D-matrix:\n', T_D)

C_D = (np.abs(T_D) <= cw).astype(int)
print('C_D-matrix:\n', C_D)

C_D = C_D & F_D 
print('C_D\'-matrix:\n', C_D)

R_D = np.sum(C_D, axis=1)
print('R_D-matrix:\n', R_D)

R = np.zeros(height*width, dtype=int)
R[D] += R_D
print('R-matrix:\n', R.reshape(height, width))

from fast_histogram import histogram1d
max_tb = 10
N = histogram1d(T_D, bins=max_tb * 2 + 1, range=(-max_tb, max_tb))
print('N-matrix:\n', N)
plt.plot(N)
plt.show()
# %%