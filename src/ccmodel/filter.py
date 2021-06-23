# %%
from fast_histogram import histogram1d
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from numpy.core.einsumfunc import _parse_einsum_input
import pandas as pd

class filter:
    filter_heatmap = None
    filter_type = None
    filter_map = dict()
    config = None

    def select_blank(self) -> np.ndarray:
        n_px = self.config['number_of_pixels']
        blank_image: np.ndarray = np.zeros((n_px, n_px), dtype=bool)
        return blank_image

    def flatten(self, input: np.ndarray) -> np.ndarray:
        return input.reshape(1, -1)
    
    def load_image_model(self) -> np.ndarray:
        image_model_path = self.config.get('image_model', '')   
        return np.loadtxt(image_model_path)

    def select_bright(self) -> np.ndarray:
        threshold = self.config.get('threshold', 0)
        image_model = self.load_image_model()
        image_bright =  image_model > (threshold * np.max(image_model))
        return image_bright

    def select_left(self) -> np.ndarray:
        r_left = self.config['regions']['left']
        image_left = self.select_region(
            up=r_left['up'], bottom=r_left['bottom'], left=r_left['left'], right=r_left['right'])
        return image_left
    
    def select_right(self) -> np.ndarray:
        r_right = self.config['regions']['right']
        image_right = self.select_region(
            up=r_right['up'], bottom=r_right['bottom'], left=r_right['left'], right=r_right['right'])
        return image_right

    def select_region(self, up: int = -1, bottom: int = -1, left: int = -1, right: int = -1) -> np.ndarray:
        n_px = self.config['number_of_pixels']
        blank_image = self.select_blank()

        up = up if up != -1 else 1
        left = left if left != -1 else 1
        bottom = bottom if bottom != -1 else n_px
        right = right if right != -1 else n_px

        blank_image[up-1:bottom, left-1:right] = True
        return blank_image

    def create_base_filter(self) -> np.ndarray:
        n_px = self.config['number_of_pixels']
        image_left = self.select_left()
        image_right = self.select_right()

        flatten_left = self.flatten(image_left)
        flatten_right = self.flatten(image_right)
        flatten_base = flatten_left | flatten_right 

        diag = (np.arange(n_px*n_px), np.arange(n_px*n_px))

        base_filter = flatten_base & flatten_base.transpose()
        base_filter[diag] = False

        return base_filter

    def create_self_filter(self) -> np.ndarray:
        image_left = self.select_left()
        image_right = self.select_right()
        flatten_left = self.flatten(image_left)
        flatten_right = self.flatten(image_right)

        filter_left = flatten_left & flatten_left.transpose()
        filter_right = flatten_right & flatten_right.transpose()

        base_filter = self.create_base_filter()
        self_filter = filter_left | filter_right
        self_filter = self_filter & base_filter

        return self_filter

    def create_cross_filter(self) -> np.ndarray:
        image_left = self.select_left()
        image_right = self.select_right()
        flatten_left = self.flatten(image_left)
        flatten_right = self.flatten(image_right)

        filter_left = flatten_left & flatten_right.transpose()
        filter_right = flatten_right & flatten_left.transpose()

        base_filter = self.create_base_filter()
        cross_filter = filter_left | filter_right
        cross_filter = cross_filter & base_filter

        return cross_filter

    def create_nearby_filter(self, radius=1) -> np.ndarray:
        n_px = self.config['number_of_pixels']
        base_filter = self.create_base_filter()
        nearby_filter = np.zeros((n_px*n_px, n_px*n_px), dtype=bool)

        nearby = np.ones((2*radius + 1, 2*radius+1), dtype=bool)
        for i in range(n_px*n_px):
            temp = np.zeros((n_px, n_px), dtype=bool)
            nearby_args = np.argwhere(nearby) - [[radius, radius]]
            for arg in nearby_args:
                y = arg[0] + i//n_px
                x = arg[1] + i % n_px
                if y < 0 or x < 0 or y >= n_px or x >= n_px:
                    continue
                temp[y, x] = 1
                nearby_filter[i] = temp.flatten()

        nearby_filter = nearby_filter & base_filter

        return nearby_filter

    def create_bright_filter(self) -> np.ndarray:
        base_filter = self.create_base_filter()
        image_bright = self.select_bright()
        flatten_bright = self.flatten(image_bright)
        
        bright_filter = flatten_bright & flatten_bright.transpose()
        bright_filter = bright_filter & base_filter

        return bright_filter

    def get_bright_image_model(self):
        image_model = self.load_image_model()
        selected_region = self.select_bright() & (self.select_left() | self.select_right())
        image_model[np.logical_not(selected_region)] = 0
        return image_model

    def plot_spots(self):
        n_px = self.config['number_of_pixels']
        print('Two spots location:')
        region = np.zeros((n_px, n_px))
        region[self.select_left()] = 1
        region[self.select_right()] = -1
        plt.imshow(region)
        plt.show()

    def plot_filter(self):
        print('Bright pixels:')
        try:
            image_model = self.get_bright_image_model()
            plt.imshow(image_model)
            plt.show()
        except:
            print('Failed to get bright pixels')

    def __init__(self, config:dict):
        self.config = config
# %%
