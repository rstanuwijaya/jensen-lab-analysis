# PF-32 Python Analysis
Collections of tools for quantum metasurface analysis using Python3. This repository includes:
1. Metasurface image processing tool to capture metasurface lattice constants.
2. Computational simulation for single-pixel ghost imaging.
3. Iterative fitting model for coincidence count using LMFIT library.
4. Vectorized coincidence count analysis tool for data produced by PF32.

# Getting started and contributing
## Prerequisites
To get started, download the latest [Git](https://git-scm.com/downloads), [VSCode](https://code.visualstudio.com/), and [Docker](https://www.docker.com/get-started)
Install VSCode extension `Remote - Containers` and verify the docker service is running. 

## Clone the repository and run in VSCode
    git clone https://github.com/rstanuwijaya/pf32-python-analysis
    code .

## Configure mounted directory
The container has its own local filesystem, thus the data must be mounted to the container. Modify the `.devcontainer/devcontainer.json` `"mount"` properties to mount needed directory to the container. 

## Run in dev container
Open command pallete by pressing `F1` and run `Remote-Containers: Build Container` to build the container. The VSCode will restart shortly and run in the dev container.

## Using the notebooks
The notebooks can be used directly from the `notebooks` folder to reproduce the needed analysis.

# Contributing
To contribute, feel free to fork the project and open a pull request on Github. Contributions are greatly appreciated. 

# Special Thanks
Special thanks to the Professor Jensen and the members of JensenLab @ HKUST.