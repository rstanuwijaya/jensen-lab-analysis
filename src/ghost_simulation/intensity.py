# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from PIL import Image


def generate_object():
    return np.array([
        [0, 0, 0, 0, 1, 0, ],
        [0, 0, 0, 1, 1, 0, ],
        [0, 0, 1, 0, 1, 0, ],
        [0, 1, 1, 1, 1, 0, ],
        [0, 0, 0, 0, 1, 0, ],
        [0, 0, 0, 0, 1, 0, ],
    ])


def generate_object2():
    size = (50, 50)
    img = Image.open('RC_200.png').resize(
        size, Image.ANTIALIAS).convert('L')
    img = np.asarray(img)
    return img



def run_iteration(p1, m):
    obj = generate_object()
    G2 = np.zeros(obj.shape)
    ns = obj.shape[0]*obj.shape[1]
    p1 = p1
    N = ns*m
    n1 = math.ceil(ns*p1)
    print('N:', N)
    print('passing pixels:', n1)
    print('p1:', p1)
    input = np.array([1 if i < n1 else 0 for i in range(ns)])
    T = obj
    for i in range(N):
        if i % 100 == 0:
            print('\r', i, end='')
        np.random.shuffle(input)
        Si = input.reshape(obj.shape)
        Ii = np.sum((T*Si).flatten())
        G2 += Si*Ii
    G2 = 1/N * G2
    plt.figure(figsize=(5,5))
    plt.imshow(obj, cmap='gray')
    plt.savefig('../out/init2.png')
    # plt.show()
    plt.figure(figsize=(5,5))
    plt.imshow(G2, cmap='gray', vmin=0)
    plt.title(f'p1 = {p1}, N = {N}')
    plt.savefig(f'../out/sim2_p1={p1}_N={N}.png')
    plt.show()


def main():
    for p1 in (0.01, 0.05, 0.1):
        for m in (1, 5, 10, 50, 100):
            run_iteration(p1, m)


if __name__ == '__main__':
    main()
