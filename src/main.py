# %%
# main function for two rings metasurface HOM Scan
# using ccmodel_v2
import sys
import ccmodel_v2 as ccmodel

import os 
import pandas as pd
import importlib
import time
import numpy as np
import re
from dotenv import load_dotenv
dir_path = os.path.dirname(__file__)
load_dotenv(os.path.join(dir_path, '.env'), override=True)

try:
    importlib.reload(ccmodel)   
except:
    print('CCmodel have not been imported')

def backend(query):
    consts = dict()
    modes = dict()

    ### CONSTS CONFIG - file and directory configuration

    consts['number_of_pixels'] = (32, 32)
    consts['regions'] = {
        'left' : ((query['regions']['left'][0][0], query['regions']['left'][0][1]), (query['regions']['left'][1][0], query['regions']['left'][1][1])),
        'right' : ((query['regions']['right'][0][0], query['regions']['right'][0][1]), (query['regions']['right'][1][0], query['regions']['right'][1][1])),
        'all': ((query['regions']['all'][0][0], query['regions']['all'][0][1]), (query['regions']['all'][1][0], query['regions']['all'][1][1])),
    }
    consts['number_of_frames'] = query['number_of_frames']
    consts['jitter_path'] = query['jitter_path']
    consts['working_directory'] = query['working_directory']
    consts['write_directory'] = os.path.join(consts['working_directory'], 'analysis')
    consts['coincidence_window'] = query['coincidence_window']

    ###

    ### MODES CONFIG - modes[var] = True to set auto initailization

    modes['pixel_accumulated_count'] = query['modes']['pixel_accumulated_count']
    modes['time_accumulated_count'] = query['modes']['time_accumulated_count']
    modes['pixel_coincidence_count'] = query['modes']['pixel_coincidence_count']
    modes['time_coincidence_count'] = query['modes']['time_coincidence_count']

    ###

    print('Working Directory at: {}'.format(consts['working_directory']))
    filt_dict = dict()
    filters =  ccmodel.filter_generator(consts)
    filt = filters.filter_map[query['filter_map']]

    ld = os.listdir(path=consts['working_directory'])
    files = set()
    times = np.array([])

    for file in ld:
        match = re.match(r'(.+)\.(txt|npy|npz)$', file)
        if match is None: continue
        file = match[1]
        files.add(file)

    files = list(files)

    start_time = time.time()

    for i in range(len(files)):
        start_iter_time = time.time()

        print(str(i+1) + '/' + str(len(files)), files[i], end=' ')

        try:
            ### MAIN CONFIG
            tb = ccmodel.time_bins(files[i], consts, modes, filt, debug=False)
            # tb.adjusted_bins[:, 24, 16] = 0
            # tb.initialize_pixel_accumulated_count()
            # tb.initialize_time_accumulated_count()
            # tb.initialize_coincidence_count(pixel_mode=True, time_mode=True)
            ###
        except Exception as e:
            print(f'\n{str(e).upper()}\nTERMINATING PROGRAM')
            return

        tb.write_to_file()
        tb.save_figures()

        end_iter_time = time.time()
        iter_time = end_iter_time - start_iter_time
        print('t=' + '%.2f' % iter_time + 's', end=' ')
        times = np.append(times, iter_time)
        elapsed_time = times.sum()
        estimated_total_time = np.average(times)*len(files)
        estimated_time_left = estimated_total_time - elapsed_time
        print('etc=' + '%d' % (estimated_time_left//60) +'m' + '%d' % (estimated_time_left % 60) + 's', end='\n')

    end_time = time.time()
    print('Total Elapsed Time:', '%d' % ((end_time-start_time)//60) +'m' + '%d' % ((end_time-start_time) % 60) + 's')


