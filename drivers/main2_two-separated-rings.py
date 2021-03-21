# %%
# main function for two rings metasurface HOM Scan
import sys
sys.path
sys.path.append('..')

from src import ccmodel as ccmodel

import os 
import pandas as pd
import importlib
import time
import numpy as np
import re
from dotenv import load_dotenv
from src import utils
dir_path = os.path.dirname(__file__)
load_dotenv(os.path.join(dir_path, '.env'), override=True)

try:
    importlib.reload(ccmodel)   
except:
    print('Module is not imported')

consts = dict()
consts['number_of_pixels'] = (32, 32)
consts['regions'] = {
    'left' : ((13, 21), (3, 11)),
    'right' : ((13, 21), (21, 29)),
    'all': ((0, 32), (0, 32))
}
consts['number_of_frames'] = 50000
consts['jitter_path'] = os.getenv('JITTER_PATH')
consts['working_directory'] = os.getenv('WORKING_DIRECTORY_40-45HOMSCAN')
dirname = 'Both Inc_HWP=302.5_Theta=-89'
# dirname = 'Ref Inc_HWP=302.5_Theta=-89'
# dirname = 'SLM Inc_HWP=302.5_Theta=-89'
consts['working_directory'] = os.path.join(consts['working_directory'], dirname)
print(consts['working_directory'])
consts['write_directory'] = os.path.join(consts['working_directory'], 'analysis')

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
    if files[i] != '2685568_Frame2_Exp100_iter1': continue
    tb = ccmodel.time_bins(files[i], consts)
    tb.initialize_pixel_accumulated_count()
    # tb.initialize_time_accumulated_count()
    tb.initialize_coincidence_count()
    # tb.initialize_pixel_all_coincidence_count(1)
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

# if utils.count_raw_files(consts['working_directory']):
#     utils.delete_raw_files(consts['working_directory'])