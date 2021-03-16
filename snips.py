# %%
# import os
# import re

# file_path = '2690816_Frame5_Exp250_iter1.txt'

# os.chdir(file_path)
# ls = os.listdir()

# for file in ls:
#     if re.match('.+\.txt', file):
#         handle = open(file)
#         cnt = 0
#         for line in handle:
#             cnt += 1
#         print(file, cnt)

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
file = r'Theta=0_2677165_Frame2_Exp500_Bothinc.txt'
# file = '2690816_Frame5_Exp250_iter1.txt'
handle = open(file)
cnt = 0 
for line in handle:
    cnt += 1
print(cnt)


# %%
