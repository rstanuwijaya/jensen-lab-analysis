# %%
# ccmodel.py:
'''
About ccmodel:

Analysis tool for the data produced by pf32 camera - picosecond time resolved 32-by-32 pixels camera.
'''

import numpy as np
import os
import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from fast_histogram import histogram1d

class time_bins:
    raw_bins = None
    jitter_bins = None
    adjusted_bins = None
    pixel_accumulated_count = None
    ctime_accumulated_count = None
    debug = False
    consts = None
    name = None
    coincidence_count = None
    pixel_self_coincidence_count = None
    pixel_cross_coincidence_count = None

    def initialize_raw_bins(self):
        if self.debug: print(self.name, 'Initializing raw bins')
        if os.path.isfile(self.name + '.npy'):
            try:
                self.raw_bins = np.load(self.name + ".npy")
            except:
                print('Failed to load:', self.name + '.npy')
        elif os.path.isfile(self.name + '.txt'):
            try:
                self.raw_bins = pd.read_csv(self.name + ".txt", header=None).to_numpy().reshape(self.consts['number_of_frames'], self.consts['number_of_pixels'][0], self.consts['number_of_pixels'][1]).transpose((0,2,1)).astype(np.int16)
                try:
                    np.save(self.name + ".npy", self.raw_bins)
                except:
                    print('Failed to save:', self.name + '.npy')
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
        axs.imshow(self.pixel_accumulated_count, cmap='gray')
        region = self.consts['regions']
        for key in region.keys():
            rect = patches.Rectangle((region[key][1][0]-0.5, region[key][0][0]-0.6), region[key][1][1]-region[key][1][0], region[key][0][1]-region[key][0][0], linewidth=1, edgecolor='red', facecolor='none')
            axs.add_patch(rect)
        axs.set_xticks(range(0, self.consts['number_of_pixels'][0],4))
        axs.set_yticks(range(0, self.consts['number_of_pixels'][1],4))
        axs.grid()
        return axs

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
        axs.plot(self.time_accumulated_count[1:])
        return axs

    def initialize_coincidence_count(self):
        '''
        param: None
        initialize cross coincidence count array
        '''
        if self.debug: print(self.name, 'Initializing coincidence_count')
        bins = dict()
        region = self.consts['regions']
        for key in region.keys():
            if key != 'left' and key != 'right': continue
            start_x = region[key][1][0]
            end_x   = region[key][1][1]
            start_y = region[key][0][0]
            end_y   = region[key][0][1]
            bins[key] = self.adjusted_bins[:, start_y:end_y, start_x:end_x]
        Z = np.zeros(1024*2 + 1)

        for i in range(50000):
            # if (i%1000) == 0 and self.debug:
            #     print(i)
            L = bins['left'][i].flatten()
            R = bins['right'][i].flatten()
            L = L[L.nonzero()]
            R = R[R.nonzero()]
            L_reps = np.tile(L, R.shape[0])
            R_reps = np.repeat(R, L.shape[0])
            Z += histogram1d(L_reps - R_reps, bins=1024*2 + 1, range=(-1024, 1024))
        
        self.coincidence_count = Z
    
    def initialize_coincidence_count_v2(self, tw):
        '''
        param: tw
        initialize coincidence count array with time window tolerance of tw
        '''
        pass

    def plot_coincidence_count(self, axs=None):
        '''
        param: None
        return the plot the coincidence count on the given axes.

        ------
        axs = plt axes subplot
        '''
        if self.debug: print(self.name, 'Plotting coincidence_count')
        if axs == None: axs = plt.axes()
        x = np.arange(-1024, 1024+1, 1)
        y = self.coincidence_count
        try:
            axs.plot(x, y)
        except:
            print('Unable to plot coincidence count array', self.name)
        return axs

    def initialize_pixel_self_coincidence_count(self, coincidence_window, spot = ['left', 'right']):
        '''
        param: coincidence_window
        Initialize pixel self coincidence count

        -----
        coincidence_window: int, time window for coincidence_count
        '''
        if coincidence_window < 0: raise('Coincidence window must be geq 0')
        if self.debug: print(self.name, 'Initializing pixel_self_coincidence_count')
        bins = dict()
        spot = np.array(spot)
        region = self.consts['regions']
        Z = np.zeros(self.consts['number_of_pixels'])
        for key in region.keys():
            if np.all(spot != key): continue
            start_x = region[key][1][0]
            end_x   = region[key][1][1]
            start_y = region[key][0][0]
            end_y   = region[key][0][1]
            height  = end_y - start_y
            width   = end_x - start_x
            bins[key] = self.adjusted_bins[:, start_y:end_y, start_x:end_x]
            total_coincidence_count = np.zeros(height * width, dtype=int)
            for i in range(self.consts['number_of_frames']):
                time_bin = bins[key][i].flatten()
                detection_arg = time_bin.nonzero()
                detection_time_bin = time_bin[detection_arg]
                r1 = np.tile(detection_time_bin, detection_time_bin.shape[0])
                r2 = np.repeat(detection_time_bin, detection_time_bin.shape[0])
                diff = (r1-r2).reshape(detection_time_bin.shape[0], detection_time_bin.shape[0])
                coincidence_detection_arg = np.abs(diff) <= coincidence_window
                coincidence_detection_count = np.sum(coincidence_detection_arg, axis=1) - 1
                coincidence_count = np.zeros(height*width, dtype=int) 
                coincidence_count[detection_arg] += coincidence_detection_count
                total_coincidence_count += coincidence_count
            Z[start_y:start_y+height, start_x:start_x+width] = total_coincidence_count.reshape(height, width)
            
        self.pixel_self_coincidence_count = Z

    def plot_pixel_self_coincidence_count(self, axs=None):
        '''
        param: None
        return the plot the pixel accumulated count on the given axes.

        ------
        axs = plt axes subplot
        '''
        if self.debug: print(self.name, 'Plotting pixplot_pixel_self_coincidence_count')
        if axs == None: axs = plt.axes()
        axs.imshow(self.pixel_self_coincidence_count, cmap='gray')
        axs.set_xticks(range(0, self.consts['number_of_pixels'][0],4))
        axs.set_yticks(range(0, self.consts['number_of_pixels'][1],4))
        axs.grid()
        return axs

    def initialize_pixel_cross_coincidence_count(self, coincidence_window):
        '''
        param: coincidence_window
        Initialize pixel self coincidence count

        -----
        coincidence_window: int, time window for coincidence_count
        '''
        if self.debug: print(self.name, 'Initializing pixel_cross_coincidence_count')
        bins = dict()
        region = self.consts['regions']
        Z = np.zeros(self.consts['number_of_pixels'])
        left_start_x = region['left'][1][0]
        left_end_x   = region['left'][1][1]
        left_start_y = region['left'][0][0]
        left_end_y   = region['left'][0][1]
        left_height  = left_end_y - left_start_y
        left_width   = left_end_x - left_start_x
        bins['left'] = self.adjusted_bins[:, left_start_y:left_end_y, left_start_x:left_end_x]

        right_start_x = region['right'][1][0]
        right_end_x   = region['right'][1][1]
        right_start_y = region['right'][0][0]
        right_end_y   = region['right'][0][1]
        right_height  = right_end_y - right_start_y
        right_width   = right_end_x - right_start_x
        bins['right'] = self.adjusted_bins[:, right_start_y:right_end_y, right_start_x:right_end_x]
        
        left_total_coincidence_count = np.zeros(left_height * left_width, dtype=int)
        right_total_coincidence_count = np.zeros(right_height * right_width, dtype=int)

        for i in range(self.consts['number_of_frames']):
            left_time_bin = bins['left'][i].flatten()
            right_time_bin = bins['right'][i].flatten()

            left_detection_arg = left_time_bin.nonzero()
            left_detection_time_bin = left_time_bin[left_detection_arg]
            right_detection_arg = right_time_bin.nonzero()
            right_detection_time_bin = right_time_bin[right_detection_arg]


            r1 = np.tile(left_detection_time_bin, right_detection_time_bin.shape[0])
            r2 = np.repeat(right_detection_time_bin, left_detection_time_bin.shape[0])
            diff = (r1-r2).reshape(right_detection_time_bin.shape[0], left_detection_time_bin.shape[0])

            coincidence_detection_arg = np.abs(diff) <= coincidence_window

            left_coincidence_detection_count = np.sum(coincidence_detection_arg, axis=0)
            right_coincidence_detection_count = np.sum(coincidence_detection_arg, axis=1)

            left_coincidence_count = np.zeros(left_height*left_width, dtype=int) 
            right_coincidence_count = np.zeros(right_height*right_width, dtype=int) 
            left_coincidence_count[left_detection_arg] += left_coincidence_detection_count
            right_coincidence_count[right_detection_arg] += right_coincidence_detection_count
            left_total_coincidence_count += left_coincidence_count
            right_total_coincidence_count += right_coincidence_count
            
            Z[left_start_y:left_start_y+left_height, left_start_x:left_start_x+left_width] = left_total_coincidence_count.reshape(left_height, left_width)
            Z[right_start_y:right_start_y+right_height, right_start_x:right_start_x+right_width] = right_total_coincidence_count.reshape(right_height, right_width)

        self.pixel_cross_coincidence_count = Z

    def plot_pixel_cross_coincidence_count(self, axs=None):
        '''
        param: None
        return the plot the pixel accumulated count on the given axes.

        ------
        axs = plt axes subplot
        '''
        if self.debug: print(self.name, 'Plotting pixplot_pixel_cross_coincidence_count')
        if axs == None: axs = plt.axes()
        axs.imshow(self.pixel_cross_coincidence_count, cmap='gray')
        axs.set_xticks(range(0, self.consts['number_of_pixels'][0],4))
        axs.set_yticks(range(0, self.consts['number_of_pixels'][1],4))
        axs.grid()
        return axs


    def write_to_file(self):
        '''
        param: None
        saves all counts to the writing directory.
        '''
        if self.debug: print(self.name, 'Writing to file')
        write_directory = self.consts['write_directory']
        
        os.makedirs(write_directory, exist_ok=True)
        os.makedirs(os.path.join(write_directory, 'pixel_accumulated_count'), exist_ok=True)
        os.makedirs(os.path.join(write_directory, 'time_accumulated_count'), exist_ok=True)
        os.makedirs(os.path.join(write_directory, 'coincidence_count'), exist_ok=True)

        if self.pixel_self_coincidence_count is not None:
            os.makedirs(os.path.join(write_directory, 'pixel_self_coincidence_count'), exist_ok=True)
        if self.pixel_cross_coincidence_count is not None:    
            os.makedirs(os.path.join(write_directory, 'pixel_cross_coincidence_count'), exist_ok=True)

        np.savetxt(write_directory + "/pixel_accumulated_count/" + self.name + ".csv", self.pixel_accumulated_count)
        np.savetxt(write_directory + "/time_accumulated_count/" + self.name + ".csv", self.time_accumulated_count)
        np.savetxt(write_directory + "/coincidence_count/" + self.name + ".csv", self.coincidence_count)
        if self.pixel_self_coincidence_count is not None:
            np.savetxt(write_directory + "/pixel_self_coincidence_count/" + self.name + ".csv", self.time_accumulated_count)
        if self.pixel_cross_coincidence_count is not None:    
            np.savetxt(write_directory + "/pixel_cross_coincidence_count/" + self.name + ".csv", self.time_accumulated_count)

    def save_figures(self):
        '''
        param: None
        generates all figures and save it to the writing directory.
        '''
        if self.debug: print(self.name, 'Saving Figures')
        write_directory = self.consts['write_directory']

        os.makedirs(write_directory, exist_ok=True)
        os.makedirs(os.path.join(write_directory, 'plot_pixel_accumulated_count'), exist_ok=True)
        os.makedirs(os.path.join(write_directory, 'plot_time_accumulated_count'), exist_ok=True)
        os.makedirs(os.path.join(write_directory, 'plot_coincidence_count'), exist_ok=True)
        os.makedirs(os.path.join(write_directory, 'plot_pixel_self_coincidence_count'), exist_ok=True)
        os.makedirs(os.path.join(write_directory, 'plot_pixel_cross_coincidence_count'), exist_ok=True)

        fig_pixel_accumulated_count = plt.figure(1)
        axs_pixel_accumulated_count = fig_pixel_accumulated_count.add_subplot(111)
        self.plot_pixel_accumulated_count(axs_pixel_accumulated_count)
        plt.savefig(write_directory + "/plot_pixel_accumulated_count/" + self.name + ".png")
        plt.close(fig_pixel_accumulated_count)

        fig_time_accumulated_count = plt.figure(2)
        axs_time_accumulated_count = fig_time_accumulated_count.add_subplot(111)
        self.plot_time_accumulated_count(axs_time_accumulated_count)
        plt.savefig(write_directory + "/plot_time_accumulated_count/" + self.name + ".png")
        plt.close(fig_time_accumulated_count)

        fig_coincidence_count = plt.figure(3)
        axs_coincidence_count = fig_coincidence_count.add_subplot(111)
        self.plot_coincidence_count(axs_coincidence_count)
        plt.savefig(write_directory + "/plot_coincidence_count/" + self.name + ".png")
        plt.close(fig_coincidence_count)

        if self.pixel_self_coincidence_count is not None:
            fig_pixel_self_coincidence_count = plt.figure(4)
            axs_pixel_self_coincidence_count = fig_pixel_self_coincidence_count.add_subplot(111)
            self.plot_pixel_self_coincidence_count(axs_pixel_self_coincidence_count)
            plt.savefig(write_directory + "/plot_pixel_self_coincidence_count/" + self.name + ".png")
            plt.close(fig_pixel_self_coincidence_count)

        if self.pixel_cross_coincidence_count is not None:
            fig_pixel_cross_coincidence_count = plt.figure(5)
            axs_pixel_cross_coincidence_count = fig_pixel_cross_coincidence_count.add_subplot(111)
            self.plot_pixel_cross_coincidence_count(axs_pixel_cross_coincidence_count)
            plt.savefig(write_directory + "/plot_pixel_cross_coincidence_count/" + self.name + ".png")
            plt.close(fig_pixel_cross_coincidence_count)


    def __init__(self, name, consts, debug=False):
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

        # set working directory
        os.chdir(self.consts['working_directory'])
        
        # basic parameter initialization
        self.initialize_raw_bins()
        self.initialize_jitter_bins()
        self.initialize_adjusted_bins()
        self.initialize_pixel_accumulated_count()
        self.initialize_time_accumulated_count()
        self.initialize_coincidence_count()

    def __str__(self):
        self.plot_pixel_accumulated_count()
        return ''# %%
# %%
