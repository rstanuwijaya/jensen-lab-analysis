import os
import time
import numpy as np
import math
import matplotlib.pyplot as plt

from meta import MetaAnalyzer

def main():
    path = os.path.abspath("/home/stnav/jensen-lab/sample/Align_P300L135_003.tif")
    analysis = MetaAnalyzer(path)

if __name__ == '__main__':
    main()