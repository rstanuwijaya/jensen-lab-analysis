# %% removing raw files
import os

testdir = '' # input directory path

dirname = os.path.join(os.path.dirname(__file__), 'testdir')

def count_raw_files(dirname):
    os.chdir(dirname)
    print('Counting the number of files')
    print('Directory:', dirname)
    files = os.listdir(dirname)
    count_txt = 0
    for file in files:
        if file.endswith('.txt'):
            count_txt += 1
    count_npz = 0
    for file in files:
        if file.endswith('.npz'):
            count_npz += 1
    print('Count of total files', count_npz)
    print('Count of .txt files', count_txt)
    if count_npz == count_txt:
        print('Raw flies fully converted')
        return True
    else:
        # print('Warning! the raw files might not be fully converted')
        # return False
        raise AssertionError('Warning! the raw files might not be fully converted')

def delete_raw_files(dirname):
    print('Deleting all .txt files in this directory')
    print('Directory:', dirname)
    agree = input('WARNING THIS OPERATION WILL DELETE ALL DATA IN DIRECTORY\n Press \'y\' to continue: ')
    if agree != 'y': return
    files = os.listdir(dirname)
    cnt = 0
    for file in files:
        if file.endswith('.txt'):
            os.remove(file)
            print('Deleted', file)
            cnt += 1
    print('Deleted a total of', cnt, 'files')
# %% frame counter
def frames_counter(file):
    handle = open(file)
    cnt = 0 
    for line in handle:
        cnt += 1
    print(cnt)

# %%