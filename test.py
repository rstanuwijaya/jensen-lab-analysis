# %%
import os 
import pandas as pd
import ccmodel
import importlib
import time
import numpy as np
try:
    importlib.reload(ccmodel)   
except:
    print('module have not not imported')


consts = dict()
consts['number_of_pixels'] = (32, 32)
consts['regions'] = {
    'left' : ((9, 17), (5, 13)),
    'right' : ((9, 17), (18, 26))
}
consts['regions'] = {
    'left' : ((0, 32), (0, 13)),
    'right' : ((0, 32), (21, 32))
}
consts['number_of_frames'] = 50000
consts['jitter_path'] = r"c:\Users\Stefan Tanuwijaya\OneDrive - HKUST Connect\Academics\Jensen Lab\Python codes\ccmodel_v2\_20200926_JitterCali_DropBadPixel.csv"
consts['working_directory'] = r"c:\Users\Stefan Tanuwijaya\OneDrive - HKUST Connect\Academics\Jensen Lab\Python codes\ccmodel_v2\testdir"
consts['write_directory'] = r"c:\Users\Stefan Tanuwijaya\OneDrive - HKUST Connect\Academics\Jensen Lab\Python codes\ccmodel_v2\testdir\writedir2"

print('cwd:', os.getcwd())

start = time.time()

a = ccmodel.time_bins(r'Theta=0_2677165_Frame2_Exp500_Bothinc', consts, debug=True)
a.initialize_pixel_self_coincidence_count(2)
a.initialize_pixel_cross_coincidence_count(2)

# a.write_to_file()
# a.save_figures()

end = time.time()
print('Elapsed Time:', end-start)
# %%
import numpy as np
handle = open('test3.txt', 'w')
for i in range(4):
    for j in range(32*32):
        handle.write(str(np.random.randint(0, 1024)))
        if j != 1023:
            handle.write(', ')
    handle.write('\n')
# %%
file = r'2690816_Frame5_Exp250_iter1.txt'
handle = open(file)
cnt = 0 
for line in handle:
    cnt += 1
print(cnt)
# %%
x = np.array([1, 2, 0, 4])
y = np.array([5, 0, 7])

x = x[x.nonzero()]
y = y[y.nonzero()]

x_reps = np.tile(x, y.shape[0])
y_reps = np.repeat(y, x.shape[0])

print(x_reps, y_reps)

z = x_reps - y_reps
print(z)
# %%
