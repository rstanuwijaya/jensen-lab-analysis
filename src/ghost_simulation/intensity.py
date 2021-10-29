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


def generate_object():
    size = (50, 50)
    img = Image.open('RC_200.png').resize(
        size, Image.ANTIALIAS).convert('L')
    img = np.asarray(img)
    return img


def generate_filter(shape):
    p1 = 0.01
    weights = (1-p1, p1)
    return np.random.choice(2, shape, p=weights)


def run_iteration():
    obj = generate_object()
    list_pairs = []
    G2 = np.zeros(obj.shape)
    N = 10000
    p1 = 0.001
    ns = obj.shape[0]*obj.shape[1]
    n1 = math.ceil(ns*p1)
    print('passing pixels:', n1)
    input = np.array([1 if i < n1 else 0 for i in range(ns)])
    T = obj
    for i in range(N):
        if i % 100 == 0:
            print('\r', i, end='')
        np.random.shuffle(input)
        Si = input.reshape(obj.shape)
        Ii = np.sum((T*Si).flatten())
        G2 += Si*Ii
        pair = (Si, Ii)
        # list_pairs.append(pair)
    # G2 = 1/N * sum([(Si*Ii) for (Si, Ii) in list_pairs])
    G2 = 1/N * G2
    plt.imshow(obj, cmap='gray')
    plt.show()
    plt.imshow(G2, cmap='gray')
    plt.show()


def main():
    run_iteration()


if __name__ == '__main__':
    main()
