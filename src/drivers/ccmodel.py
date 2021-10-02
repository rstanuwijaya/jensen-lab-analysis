
#%%
import sys
sys.path.append('.')
sys.path.append('..')

import ccmodel

import os 
import pandas as pd
import time
import numpy as np
import re
import matplotlib.pyplot as plt
import json

config = {
    'number_of_pixels' : 32,
    'regions' : {
        'left' : {
            'up' : 12,
            'bottom' : 16,
            'left' : 9,
            'right' : 13,
        }, 
        'right' : {
            'up' : 12,
            'bottom' : 16,
            'left' : 17,
            'right' : 21,
        }
    },
    'coincidence_window' : 2,
    'jitter_path' : r'C:\Users\stnav\OneDrive - HKUST Connect\Academics\Jensen Lab\Python codes\ccmodel_v3\common\_20200926_JitterCali_DropBadPixel.csv',
    'working_directory' : r'E:\Data\JensenLab\VarySyncFine',
    'write_directory' : r'E:\Data\JensenLab\VarySyncFine\analysis',
    'modes' : {
        'pixel_accumulated_count' : True,
        'time_accumulated_count' : True,
        'pixel_coincidence_count' : False,
        'time_coincidence_count' : True,
    },
    'filters' : ['create_cross_filter', 'create_bright_filter'],
    'threshold' : 0.2,
    # 'image_model' : r'E:\Data\JensenLab\test\analysis\data_pixel_accumulated_count\2690816_Frame5_Exp250_iter1.csv'
}     

start_time = time.time()

ld = os.listdir(path=config['working_directory'])
files = set()
paths = list()
times = np.array([])

for file in ld:
    match = re.match(r'(.+)\.(txt|npy|npz)$', file)
    if match is None: continue
    file = match[1]
    files.add(file)

files = sorted(list(files))

for file in files:
    for ext in ('.npz', '.npy', '.txt'):
        if file + ext in ld:
            break
    path = os.path.join(os.path.abspath(config['working_directory']), file+ext) 
    paths.append(path)

# prepare filter
filter = ccmodel.filter(config)

if 'image_model' not in config:
    tb = ccmodel.time_bin(paths[0], config, filter.create_base_filter(), debug=False, init_only=True)
    tb.initialize_pixel_accumulated_count()
    tb.write_to_file()
    tb.save_figures()
    config['image_model'] = config['write_directory'] + '/data_pixel_accumulated_count/' + os.path.basename(paths[0]).split('.')[0] + '.csv'

custom_filter = filter.create_base_filter() 
for f in config['filters']:
    custom_filter = custom_filter & getattr(filter, f)()
    
fig = plt.figure(figsize=(8, 6))
plt.imshow(filter.get_bright_image_model())
plt.savefig(os.path.join(config['write_directory'], 'filtered.png'))
plt.close()
# filter.plot_filter()

# record config
os.makedirs(config['write_directory'], exist_ok=True)
with open(os.path.join(config['write_directory'], 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

for i in range(len(paths)):
    start_iter_time = time.time()

    print(str(i+1) + '/' + str(len(files)), files[i], end=' ')

    tb = ccmodel.time_bin(paths[i], config, custom_filter)
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

delay_cc = ccmodel.analysis.cc_vs_delay(config['write_directory'] + '/data_time_coincidence_count', config['coincidence_window'])
plt.scatter(*delay_cc)
plt.savefig(config['write_directory'] + '/delay_cc.png')
plt.close()
# %%
