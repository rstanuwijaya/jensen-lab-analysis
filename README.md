# PF-32 Python Analysis
Python implementation of the data produced by PF32, 32-by-32 pixels TCSPC (Time Correlated Single Photon Counting) SPAD (Single Photon Avalanche Diode) camera, produced and distributed by Photon Force. 

## Definitions
### 1. Adjusted Time Bin
Adjusted time bin is the Raw time bin - jitting time and drop bad pixels

### 2. Pixel Accumulated Count 
Pixel accumulated count/intensity is the total number of incident photon for each pixel

### 3. Time Accumulated Count
Time accumulated count is the total number of incident photon for each time bin

### 4. Filter Matrix
Filter matrix determines the correlation between two pixel, i.e. F[i][j] = 1 if i and j are correlated. Features: symmetric across diagonal, diagonal = 0

### 5. Coincidence Window:
Coincidence window determines if two photons are coincidence. i.e. a photon within the same frame and time arrival difference within $\pm$ coincidence window are coincidence. Must be smaller than the central peak in the time coincidence count. Defualt value = 2 time bins.

### 5. Pixel Coincidence Count
Pixel coincidence count is the number of coincidence within a pixel. Detailed demo can be found in /demo/counting_method.py

### 6. Time Coincidence Count
Time coincidence count is the number of photon pair with time difference of $\Delta t$

## Coincidence Counting Method
Can be found in demo/counting_method.py
