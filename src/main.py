
#%%
import sys
sys.path.append('..')

import ccmodel

import os 
import pandas as pd
import time
import numpy as np
import re
import matplotlib.pyplot as plt
import json

consts = {
    'number_of_pixels' : 32,
    'regions' : {
        'left' : {
            'up' : 8,
            'bottom' : 26,
            'left' : 4,
            'right' : 14,
        }, 
        'right' : {
            'up' : 8,
            'bottom' : 26,
            'left' : 17,
            'right' : 27,
        }
    },
    'coincidence_window' : 2,
    'jitter_path' : r'C:\Users\stnav\OneDrive - HKUST Connect\Academics\Jensen Lab\Python codes\ccmodel_v3\common\_20200926_JitterCali_DropBadPixel.csv',
    'working_directory' : r'E:\Data\JensenLab\test',
    'write_directory' : r'E:\Data\JensenLab\test\analysis',
    'modes' : {
        'pixel_accumulated_count' : True,
        'time_accumulated_count' : True,
        'pixel_coincidence_count' : True,
        'time_coincidence_count' : True,
    },
    'filters' : ['create_cross_filter', 'create_bright_filter'],
    'threshold' : 0.75,
    # 'image_model' : r'E:\Data\JensenLab\test\analysis\data_pixel_accumulated_count\2690816_Frame5_Exp250_iter1.csv'
}     

start_time = time.time()

ld = os.listdir(path=consts['working_directory'])
files = set()
paths = list()
times = np.array([])

for file in ld:
    match = re.match(r'(.+)\.(txt|npy|npz)$', file)
    if match is None: continue
    file = match[1]
    files.add(file)

files = list(files)

for file in files:
    for ext in ('.npz', '.npy', '.txt'):
        if file + ext in ld:
            break
    path = os.path.join(consts['working_directory'], file+ext) 
    paths.append(path)

# prepare filter
filter = ccmodel.filter(consts)

if 'image_model' not in consts:
    tb = ccmodel.time_bin(paths[0], consts, filter.create_base_filter(), debug=False, init_only=True)
    tb.initialize_pixel_accumulated_count()
    tb.write_to_file()
    tb.save_figures()
    consts['image_model'] = consts['write_directory'] + '/data_pixel_accumulated_count/' + os.path.basename(paths[0]).split('.')[0] + '.csv'

custom_filter = filter.create_base_filter() 
for f in consts['filters']:
    custom_filter = custom_filter & getattr(filter, f)()
    
fig = plt.figure(figsize=(8, 6))
plt.imshow(filter.get_bright_image_model())
plt.savefig(os.path.join(consts['write_directory'], 'filtered.png'))
filter.plot_filter()

# record config
os.makedirs(consts['write_directory'], exist_ok=True)
with open(os.path.join(consts['write_directory'], 'config.json'), 'w') as f:
    json.dump(consts, f, indent=2)

for i in range(len(paths)):
    start_iter_time = time.time()

    print(str(i+1) + '/' + str(len(files)), files[i], end=' ')

    tb = ccmodel.time_bin(paths[i], consts, custom_filter)
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


# %%

# %%
