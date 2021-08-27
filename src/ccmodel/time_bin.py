# %%
'''
About ccmodel:

v3: refactored signifacnt amount of code for optimization.

Analysis tool for the data produced by pf32 camera - picosecond time resolved 32-by-32 pixels camera.
'''
def version():
    print('CCmodel_v3')

import numpy as np
import os
import pandas as pd
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy.sparse as sp
from fast_histogram import histogram1d

from . import utils
from . import filter

class time_bin:
    path = None
    name = None
    config = None
    ext = None
    modes = None
    filter = None
    debug = False
    cmap = None
    raw_bins = None
    jitter_bins = None
    adjusted_bins = None
    pixel_accumulated_count = None
    time_accumulated_count = None
    pixel_coincidence_count = None
    time_coincidence_count = None

    def initialize_raw_bins(self, mode='npz'):
        if self.debug: print(self.name, 'Initializing raw bins')
        try:
            self.raw_bins = utils.load(self.path)
        except Exception as exc:
            raise(exc)

    def initialize_jitter_bins(self):
        n_px = self.config['number_of_pixels']
        if self.debug: print(self.name, 'Initializing jitter bins')
        jitter_path = self.config['jitter_path']
        try:
            self.jitter_bins = pd.read_csv(jitter_path)["Jitter"].to_numpy().round().astype(int).reshape(n_px,n_px)
        except Exception as exc:
            print('Failed to load jitter file')
            raise(exc)

    def initialize_adjusted_bins(self):
        if self.debug: print(self.name, 'Initializing adjusted bins')
        is_good_pixel = self.jitter_bins != 10000
        is_photon_incident = self.raw_bins != 0
        is_not_counted = np.logical_not(np.logical_and(is_good_pixel, is_photon_incident))
        self.adjusted_bins = self.raw_bins - self.jitter_bins
        self.adjusted_bins[is_not_counted] = 0

    def initialize_pixel_accumulated_count(self):
        if self.debug: print(self.name, 'Initializing pixel_accumulated_count')
        self.pixel_accumulated_count = np.count_nonzero(self.adjusted_bins, axis=0)

    def plot_pixel_accumulated_count(self, axs=None):
        '''
        param: None
        return the plot the pixel accumulated count on the given axes.

        ------
        axs = plt axes subplot
        '''
        if self.debug: print(self.name, 'Plotting pixel_accumulated_count')
        if axs == None: axs = plt.axes()
        try:
            img = axs.imshow(self.pixel_accumulated_count, cmap=self.cmap)
        except:
            print('Failed to plot image accumulated count')
            raise
        region = self.config['regions']
        for key in region.keys():
            rect = patches.Rectangle((region[key]['left']-0.5, region[key]['up']-0.6), region[key]['right']-region[key]['left'], region[key]['bottom']-region[key]['up'], linewidth=1, edgecolor='red', facecolor='none')
            axs.add_patch(rect)
        axs.set_xticks(range(0, self.config['number_of_pixels'],4))
        axs.set_yticks(range(0, self.config['number_of_pixels'],4))
        axs.set_title(self.name + '\nPixel accumulated count')
        axs.grid()
        return axs, img

    def initialize_time_accumulated_count(self):
        if self.debug: print(self.name, 'Initializing time_accumulated_count')
        self.time_accumulated_count = histogram1d(self.adjusted_bins, bins=1024, range=(0,1024))

    def plot_time_accumulated_count(self, axs=None):
        '''
        param: None
        return the plot the time accumulated count on the given axes.

        ------
        axs = plt axes subplot
        '''
        if self.debug: print(self.name, 'Plotting time_accumulated_count')
        if axs == None: axs = plt.axes()
        img = axs.plot(self.time_accumulated_count[1:])
        axs.set_title(self.name + '\nTime accumulated count')
        return axs, img
    
    def initialize_coincidence_count(self, pixel_mode=False, time_mode=False):
        if self.debug: print(self.name, 'Initializing pixel and time coincidence_count')
        n_px = self.config['number_of_pixels']
        height = n_px
        width = n_px
        num_px = height * width
        adjusted_bins = self.adjusted_bins
        cw = self.config['coincidence_window']
        max_tb = 1024
        F = self.filter
        if pixel_mode: pixel_coincidence_count = np.zeros(num_px, dtype=int)
        if time_mode: time_coincidence_count = np.zeros(max_tb*2 + 1)
        for i in range(adjusted_bins.shape[0]):
            A = adjusted_bins[i].flatten()
            D = A.nonzero()
            A_D = A[D]
            T_D = A_D.reshape(1, -1) - A_D.reshape(-1, 1)
            F_D = F[D].transpose()[D]
            if pixel_mode:
                C_D = F_D & (np.abs(T_D) <= cw)
                R_D = np.sum(C_D, axis=1)
                R = np.zeros(height*width, dtype=int)
                R[D] += R_D
                pixel_coincidence_count += R
            if time_mode:
                N = histogram1d(T_D[F_D], bins=max_tb * 2 + 1, range=(-max_tb, max_tb))
                time_coincidence_count += N
        if pixel_mode: self.pixel_coincidence_count = pixel_coincidence_count.reshape((height, width))
        if time_mode: self.time_coincidence_count = time_coincidence_count

    def plot_pixel_coincidence_count(self, axs=None):
        '''
        param: None
        return the plot the pixel accumulated count on the given axes.

        ------
        axs = plt axes subplot
        '''
        if self.debug: print(self.name, 'Plotting plot_pixel_coincidence_count')
        if self.pixel_coincidence_count is None: return
        if axs == None: axs = plt.axes()
        img = axs.imshow(self.pixel_coincidence_count, cmap=self.cmap)
        axs.set_xticks(range(0, self.config['number_of_pixels'],4))
        axs.set_yticks(range(0, self.config['number_of_pixels'],4))
        axs.grid()
        axs.set_title(self.name + '\nPixel coincidence count')
        return axs, img

    def plot_time_coincidence_count(self, axs=None):
        '''
        param: None
        return the plot the pixel accumulated count on the given axes.

        ------
        axs = plt axes subplot
        '''
        if self.debug: print(self.name, 'Plotting plot_time_coincidence_count')
        if self.time_coincidence_count is None: return
        if axs == None: axs = plt.axes()
        x = np.arange(-1024, 1024+1)
        y = self.time_coincidence_count
        img = axs.plot(x, y)
        axs.set_title(self.name + '\nTime coincidence count')
        return axs, img

    def write_to_file(self):
        '''
        param: None
        saves all counts to the writing directory.
        '''
        if self.debug: print(self.name, 'Writing to file')
        write_directory = self.config['write_directory']
        
        os.makedirs(write_directory, exist_ok=True)

        def save(save_name, save_data):
            if save_data is None: return
            os.makedirs(os.path.join(write_directory, save_name), exist_ok=True)
            np.savetxt(os.path.join(write_directory, save_name, self.name+'.csv'), save_data)

        save('data_pixel_accumulated_count', self.pixel_accumulated_count)
        save('data_time_accumulated_count', self.time_accumulated_count)
        save('data_pixel_coincidence_count', self.pixel_coincidence_count)
        save('data_time_coincidence_count', self.time_coincidence_count)

    def save_figures(self, figsize=(8, 6), show_only=False):
        '''
        param: None
        generates all figures and save it to the writing directory.
        '''
        if self.debug: print(self.name, 'Saving Figures')
        write_directory = self.config['write_directory']

        os.makedirs(write_directory, exist_ok=True)

        def plot(plot_name, plot_method, plot_data, cbar=False):
            if plot_data is None: return
            os.makedirs(os.path.join(write_directory, plot_name), exist_ok=True)
            fig = plt.figure(figsize=figsize)
            axs = fig.add_subplot(111)
            axs, img = plot_method(axs)
            if cbar == True: fig.colorbar(img, ax=axs)
            if not show_only: 
                plt.savefig(os.path.join(write_directory, plot_name, self.name+'.png'))
            else:
                plt.show()
            plt.close()

        plot('plot_pixel_accumulated_count', self.plot_pixel_accumulated_count, self.pixel_accumulated_count, cbar=True)
        plot('plot_time_accumulated_count', self.plot_time_accumulated_count, self.time_accumulated_count, cbar=False)
        plot('plot_pixel_coincidence_count', self.plot_pixel_coincidence_count, self.pixel_coincidence_count, cbar=True)
        plot('plot_time_coincidence_count', self.plot_time_coincidence_count, self.time_coincidence_count, cbar=False)

    def __init__(self, path, config, filter, debug=False, init_only=False):
        '''
        param: name, config
        initialize the parameters of the model.

        ------
        name: str, name of the file
        config: dict, dictionary containing the constants
        '''
        # pack constructor parameters
        self.path = path
        self.name , self.ext = os.path.splitext(os.path.basename(path))
        self.config = config
        self.debug = debug
        self.filter = filter

        # set working directory
        os.chdir(self.config['working_directory'])
        
        # basic parameter initialization
        self.initialize_raw_bins()
        self.initialize_jitter_bins()
        self.initialize_adjusted_bins()
        if not init_only:
            if self.config['modes']['pixel_accumulated_count']: 
                self.initialize_pixel_accumulated_count()
            if self.config['modes']['time_accumulated_count']:    
                self.initialize_time_accumulated_count()
            if self.config['modes']['pixel_coincidence_count'] or self.config['modes']['time_coincidence_count']:    
                self.initialize_coincidence_count(self.config['modes']['pixel_coincidence_count'], self.config['modes']['time_coincidence_count'])

    def __str__(self):
        self.plot_pixel_accumulated_count()
        return ''      

# %%
