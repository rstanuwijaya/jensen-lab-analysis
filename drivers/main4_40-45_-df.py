# %%
# main function for two rings metasurface HOM Scan
# using ccmodel_v2
import sys
sys.path
sys.path.append('..')

from src import ccmodel_v2 as ccmodel

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

consts = dict()
modes = dict()

### CONSTS CONFIG - file and directory configuration

consts['number_of_pixels'] = (32, 32)
consts['regions'] = {
    'left' : ((13, 21), (3, 11)),
    'right' : ((13, 21), (21, 29)),
    'all': ((0, 32), (0, 32))
}
consts['number_of_frames'] = 100000
consts['jitter_path'] = os.getenv('JITTER_PATH')
consts['working_directory'] = r'D:\FTP Server\40-45_-df'
consts['write_directory'] = os.path.join(consts['working_directory'], 'analysis')
consts['coincidence_window'] = 2

###

### MODES CONFIG - modes[var] = True to set auto initailization

modes['pixel_accumulated_count'] = False
modes['time_accumulated_count'] = False
modes['pixel_coincidence_count'] = False
modes['time_coincidence_count'] = False

###

print('cwd:', os.getcwd())

filt_dict = dict()
filters =  ccmodel.filter_generator(consts)
filt = filters.filter_map['nearby']

ld = os.listdir(path=consts['working_directory'])
files = []
times = np.array([])

for file in ld:
    match = re.match(r'(.+)\.txt', file)
    if match is None: continue
    file = match[1]
    files.append(file)

start_time = time.time()

for i in range(len(files)):
    start_iter_time = time.time()

    print(str(i) + '/' + str(len(files)), files[i], end=' ')

    ### MAIN CONFIG

    tb = ccmodel.time_bins(files[i], consts, modes, filt, debug=False)
    # tb.adjusted_bins[:, 24, 16] = 0
    # tb.initialize_pixel_accumulated_count()
    # tb.initialize_time_accumulated_count()
    tb.initialize_coincidence_count(pixel_mode=True, time_mode=True)

    ###

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


