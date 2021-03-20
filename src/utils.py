# %% utility functions
import os

testdir = '' # input directory path

dirname = os.path.join(os.path.dirname(__file__), 'testdir')

def count_raw_files(dirname):
    print('Counting the number of files')
    print('Directory:', dirname)
    files = os.listdir(dirname)
    count_txt = 0
    for file in files:
        if file.endswith('.txt'):
            count_txt += 1
    count_file = 0
    for file in files:
        if os.path.isfile(file):
            count_file +=1
    print('Count of total files', count_file)
    print('Count of .txt files', count_txt)
    if count_file == 2*count_txt:
        print('Raw flies fully converted')
        return True
    else:
        print('Warning! the raw files might not be fully converted')
        return False

def delete_raw_files(dirname):
    print('Deleting all .txt files in this directory')
    print('Directory:', dirname)
    agree = input('WARNING THIS OPERATION WILL DELETE ALL DATA IN DIRECTORY\n Press \'y\' to continue: ')
    if agree != 'y': return
    files = os.listdir(dirname)
    for file in files:
        if file.endswith('.txt'):
            os.remove(file)
            print('Deleted', file)
# %%
def frames_counter(file):
    handle = open(file)
    cnt = 0 
    for line in handle:
        cnt += 1
    print(cnt)
