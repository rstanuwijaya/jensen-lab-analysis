
# %%
# ccmodel_v2.py:
'''
About ccmodel:

v2: refactored signifacnt amount of code for optimization.

Analysis tool for the data produced by pf32 camera - picosecond time resolved 32-by-32 pixels camera.
'''
def version():
    print('CCmodel_v2')

import numpy as np
import os
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy.sparse as sp
from fast_histogram import histogram1d

class time_bins:
    name = None
    consts = None
    modes = None
    filt = None
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
        if os.path.isfile(self.name + '.npz'):
            try:
                sparse_matrix =  sp.load_npz(self.name + '.npz')
                self.raw_bins = np.array(sparse_matrix.todense())\
                    .reshape((-1, self.consts['number_of_pixels'][0], self.consts['number_of_pixels'][1]))[:self.consts['number_of_frames']]
            except:
                print('Failed to load:', self.name + '.npz')
        elif os.path.isfile(self.name + '.npy'):
            try:
                self.raw_bins = np.load(self.name + ".npy")
            except:
                print('Failed to load:', self.name + '.npy')
        elif os.path.isfile(self.name + '.txt'):
            try:
                self.raw_bins = pd.read_csv(self.name + ".txt", header=None)\
                    .to_numpy()\
                    .reshape(self.consts['number_of_frames'], self.consts['number_of_pixels'][0], self.consts['number_of_pixels'][1])\
                    .transpose((0,2,1))\
                    .astype(np.int16)
                if mode == 'npz':
                    try:
                        sparse_matrix = sp.csr_matrix(self.raw_bins.reshape(self.consts['number_of_frames'], self.consts['number_of_pixels'][0]*self.consts['number_of_pixels'][1]))
                        sp.save_npz(self.name + '.npz', sparse_matrix, compressed=True)
                    except Exception as exc:
                        print('Failed to save:', self.name + '.npz')
                        raise(exc)
                elif mode == 'npy':
                    try:
                        np.save(self.name + ".npy", self.raw_bins)
                    except:
                        print('Faiselled to save:', self.name + '.npy')
                else:
                    print('Mode', mode, 'is not supported')
            except OSError as exc:
                print('Failed to load:', self.name + '.txt')
                raise exc
        else:
            print('File does not exist:', self.name)

    def initialize_jitter_bins(self):
        if self.debug: print(self.name, 'Initializing jitter bins')
        jitter_path = self.consts['jitter_path']
        try:
            self.jitter_bins = pd.read_csv(jitter_path)["Jitter"].to_numpy().round().astype(int).reshape(32,32)
        except:
            print('Failed to load jitter file')

    def initialize_adjusted_bins(self):
        if self.debug: print(self.name, 'Initializing adjusted bins')
        is_good_pixel = self.jitter_bins != 10000
        is_photon_incident = self.raw_bins != 0
        is_not_counted = np.logical_not(np.logical_and(is_good_pixel, is_photon_incident))
        self.adjusted_bins = self.raw_bins - self.jitter_bins
        self.adjusted_bins[is_not_counted] = 0

    def initialize_pixel_accumulated_count(self):
        if self.debug: print(self.name, 'Initializing pixel_accumulated_count')
        self.pixel_accumulated_count = self.adjusted_bins.sum(0)

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
        region = self.consts['regions']
        for key in region.keys():
            rect = patches.Rectangle((region[key][1][0]-0.5, region[key][0][0]-0.6), region[key][1][1]-region[key][1][0], region[key][0][1]-region[key][0][0], linewidth=1, edgecolor='red', facecolor='none')
            axs.add_patch(rect)
        axs.set_xticks(range(0, self.consts['number_of_pixels'][0],4))
        axs.set_yticks(range(0, self.consts['number_of_pixels'][1],4))
        axs.set_title(self.name + '\nPixel accumulated count')
        axs.grid()
        return axs, img

    def initialize_time_accumulated_count(self):
        if self.debug: print(self.name, 'Initializing time_accumulated_count')
        self.time_accumulated_count = np.histogram(self.adjusted_bins, bins=1024, range=(0,1024))[0]

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
        consts = self.consts
        height = self.consts['number_of_pixels'][0]
        width = self.consts['number_of_pixels'][1]
        num_px = height * width
        adjusted_bins = self.adjusted_bins
        cw = self.consts['coincidence_window']
        max_tb = 1024
        F = self.filt
        if pixel_mode: pixel_coincidence_count = np.zeros(num_px, dtype=int)
        if time_mode: time_coincidence_count = np.zeros(max_tb*2 + 1)
        for i in range(consts['number_of_frames']):
            if pixel_mode:
                A = adjusted_bins[i].flatten()
                D = A.nonzero()
                A_D = A[D]
                T_D = A_D.reshape(1, -1) - A_D.reshape(-1, 1)
                F_D = F[D].transpose()[D]
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
        axs.set_xticks(range(0, self.consts['number_of_pixels'][0],4))
        axs.set_yticks(range(0, self.consts['number_of_pixels'][1],4))
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
        write_directory = self.consts['write_directory']
        
        os.makedirs(write_directory, exist_ok=True)

        def save(save_name, save_data):
            if save_data is None: return
            os.makedirs(os.path.join(write_directory, save_name), exist_ok=True)
            np.savetxt(os.path.join(write_directory, save_name, self.name+'.csv'), save_data)

        save('pixel_accumulated_count', self.pixel_accumulated_count)
        save('time_accumulated_count', self.time_accumulated_count)
        save('pixel_coincidence_count', self.pixel_coincidence_count)
        save('time_coincidence_count', self.time_coincidence_count)

    def save_figures(self, figsize=(8, 6), show_only=False):
        '''
        param: None
        generates all figures and save it to the writing directory.
        '''
        if self.debug: print(self.name, 'Saving Figures')
        write_directory = self.consts['write_directory']

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

    def __init__(self, name, consts, modes, filt, debug=False):
        '''
        param: name, consts
        initialize the parameters of the model.

        ------
        name: str, name of the file
        consts: dict, dictionary containing the constants
        '''
        # pack constructor parameters
        self.consts = consts
        self.debug = debug
        self.name = name
        self.modes = modes
        self.filt = filt

        # set working directory
        os.chdir(self.consts['working_directory'])
        
        # basic parameter initialization
        self.initialize_raw_bins()
        self.initialize_jitter_bins()
        self.initialize_adjusted_bins()
        if self.modes['pixel_accumulated_count']: 
            self.initialize_pixel_accumulated_count()
        if self.modes['time_accumulated_count']:    
            self.initialize_time_accumulated_count()
        if self.modes['pixel_coincidence_count'] or self.modes['time_coincidence_count']:    
            self.initialize_coincidence_count(self.modes['pixel_coincidence_count'], self.modes['time_coincidence_count'])

    def __str__(self):
        self.plot_pixel_accumulated_count()
        return ''      

# %%

class filter_generator:
    filter_type = None
    filter_map = dict()
    consts = None
    def initialize_self_filter_map(self):
        n_px = self.consts['number_of_pixels']
        r_left = self.consts['regions']['left']
        r_right = self.consts['regions']['right']
        filter_self_left = np.zeros((n_px[0], n_px[1]), dtype=bool)
        filter_self_right = np.zeros((n_px[0], n_px[1]), dtype=bool)
        filter_self_left[r_left[0][0]:r_left[0][1], r_left[1][0]:r_left[1][1]] = True
        filter_self_right[r_right[0][0]:r_right[0][1], r_right[1][0]:r_right[1][1]] = True
        # plt.imshow(filter_self_left | filter_self_right)
        # plt.show()
        filter_self_left_map = filter_self_left.reshape(1, -1) & filter_self_left.reshape(-1, 1)
        filter_self_right_map = filter_self_right.reshape(1, -1) & filter_self_right.reshape(-1, 1)
        filter_self_map = filter_self_left_map | filter_self_right_map
        for i in range(n_px[0]*n_px[1]):
            filter_self_map[i][i] = 0
        self.filter_map['self'] = filter_self_map

    def initialize_cross_filter_map(self):
        n_px = self.consts['number_of_pixels']
        r_left = self.consts['regions']['left']
        r_right = self.consts['regions']['right']
        filter_left = np.zeros((n_px[0], n_px[1]), dtype=bool)
        filter_right = np.zeros((n_px[0], n_px[1]), dtype=bool)
        filter_left[r_left[0][0]:r_left[0][1], r_left[1][0]:r_left[1][1]] = True
        filter_right[r_right[0][0]:r_right[0][1], r_right[1][0]:r_right[1][1]] = True
        filter_cross_map = np.zeros((n_px[0]*n_px[1], n_px[0]*n_px[1]), dtype=bool)
        filter_cross_map[filter_left.reshape(1, n_px[0] * n_px[1]) & filter_right.reshape(n_px[0] * n_px[1], 1)] = True 
        filter_cross_map[filter_left.reshape(n_px[0] * n_px[1], 1) & filter_right.reshape(1, n_px[0] * n_px[1])] = True 
        # plt.imshow(filter_cross_map)
        # plt.show()
        self.filter_map['cross'] = filter_cross_map

    def initialize_nearby_filter_map(self, radius = 1):
        nearby = np.ones((2*radius + 1, 2*radius+1), dtype=bool)
        nearby[radius][radius] = 0
        n_px = self.consts['number_of_pixels']
        # print('Nearby region:')
        # plt.imshow(nearby, cmap='gray')
        # plt.show()
        filter_nearby_map = np.zeros((n_px[0]*n_px[1], n_px[0]*n_px[1]), dtype=bool)

        for i in range(n_px[0]*n_px[1]):
            temp = np.zeros((n_px[0], n_px[1]), dtype=bool)
            nearby_args = np.argwhere(nearby) - [[radius, radius]]
            for arg in nearby_args:
                y = arg[0] + i//n_px[1]
                x = arg[1] + i%n_px[1]
                if y < 0 or x < 0 or y >= n_px[0] or x >= n_px[1]: continue
                temp[y, x] = 1
            filter_nearby_map[i] = temp.flatten()
            self.filter_map['nearby'] = filter_nearby_map

    def initialize_all_map(self):
        n_px = self.consts['number_of_pixels']
        filter_all_map = np.ones((n_px[0] * n_px[1], n_px[0] * n_px[1]), dtype=bool)
        diag = (np.arange(1, n_px[0] * n_px[1]), np.arange(1, n_px[0] * n_px[1]))
        filter_all_map[diag] = False 
        self.filter_map['all'] = filter_all_map


    def plot_filter_map(self):
        n_px = self.consts['number_of_pixels']
        r_left = self.consts['regions']['left']
        r_right = self.consts['regions']['right']
        region = np.zeros((n_px[0], n_px[1]), dtype=int)
        region[r_left[0][0]:r_left[0][1], r_left[1][0]:r_left[1][1]] = 1
        region[r_right[0][0]:r_right[0][1], r_right[1][0]:r_right[1][1]] = -1
        print('Two spots location:')
        plt.imshow(region)
        # plt.savefig('testdir/simulation/spot.png')
        plt.show()
        for key in self.filter_map:
            print('Filter for:', key)
            plt.imshow(self.filter_map[key])
            plt.grid()
            # plt.savefig('testdir/simulation/' + key + '.png')
            plt.show()
    
    def __init__(self, consts, nearby_radius=2):
        self.consts = consts
        self.initialize_self_filter_map()
        self.initialize_cross_filter_map()
        self.initialize_nearby_filter_map()
        self.initialize_all_map()
# %%
