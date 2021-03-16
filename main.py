# %%
# main function for two spot
import os 
import pandas as pd
import ccmodel
import importlib
import time
import numpy as np
import re

try:
    importlib.reload(ccmodel)   
except:
    print('Module is not imported')

consts = dict()
consts['number_of_pixels'] = (32, 32)
consts['regions'] = {
    'left' : ((9, 17), (5, 13)),
    'right' : ((9, 17), (18, 26))
}
consts['number_of_frames'] = 250000
consts['jitter_path'] = r"c:\Users\Stefan Tanuwijaya\OneDrive - HKUST Connect\Academics\Jensen Lab\Python codes\ccmodel_v2\_20200926_JitterCali_DropBadPixel.csv"

# consts['working_directory'] = r"c:\Users\Stefan Tanuwijaya\OneDrive - HKUST Connect\Academics\Jensen Lab\Python codes\ccmodel_v2\testdir"
consts['working_directory'] = r'C:\Users\Stefan Tanuwijaya\OneDrive - HKUST Connect\Academics\Jensen Lab\FineScan_Aligned_139kCount10sPerPixel_5usFrame12x12SpotSize'
consts['write_directory'] = r"c:\Users\Stefan Tanuwijaya\OneDrive - HKUST Connect\Academics\Jensen Lab\Python codes\ccmodel_v2\testdir\writedir"

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
    tb = ccmodel.time_bins(files[i], consts)
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
print('Total Elapsed Time:', '%.2f' % (end_time-start_time) + 's')
# %%
# main function for two ring metasurface
import os 
import pandas as pd
import ccmodel
import importlib
import time
import numpy as np
import re

try:
    importlib.reload(ccmodel)   
except:
    print('Module is not imported')

consts = dict()
consts['number_of_pixels'] = (32, 32)
consts['regions'] = {
    'left' : ((9, 17), (5, 13)),
    'right' : ((9, 17), (18, 26))
}
consts['number_of_frames'] = 250000
consts['jitter_path'] = r"c:\Users\Stefan Tanuwijaya\OneDrive - HKUST Connect\Academics\Jensen Lab\Python codes\ccmodel_v2\_20200926_JitterCali_DropBadPixel.csv"

# consts['working_directory'] = r"c:\Users\Stefan Tanuwijaya\OneDrive - HKUST Connect\Academics\Jensen Lab\Python codes\ccmodel_v2\testdir"
consts['working_directory'] = r'C:\Users\Stefan Tanuwijaya\OneDrive - HKUST Connect\Academics\Jensen Lab\FineScan_Aligned_139kCount10sPerPixel_5usFrame12x12SpotSize'
consts['write_directory'] = r"c:\Users\Stefan Tanuwijaya\OneDrive - HKUST Connect\Academics\Jensen Lab\Python codes\ccmodel_v2\testdir\writedir"

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
    tb = ccmodel.time_bins(files[i], consts)
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
print('Total Elapsed Time:', '%.2f' % (end_time-start_time) + 's')
