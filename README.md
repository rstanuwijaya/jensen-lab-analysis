﻿# Jensen Lab Analysis
Collections of tools for quantum metasurface analysis using Python3.

# Getting started and contributing
## Prerequisites
To get started, download the latest [Git](https://git-scm.com/downloads), [VSCode](https://code.visualstudio.com/), and [Docker](https://www.docker.com/get-started) (Optional).
Install VSCode extension `Remote - Containers` and verify the docker service is running. 

## Clone the repository and run in VSCode
    git clone https://github.com/rstanuwijaya/pf32-python-analysis
    code .

## Run locally without docker
To install the dependencies locally, you can either use Miniconda to provide dependencies encapsulation or just install the dependencies directly by using pip.

    pip install --user -r requirements.txt # Directly install the dependencies

## Setup virtual environment (Optional)
If you are setting up for the first time, you can install the dependencies inside a virtual environement by using the following commands

    python -m venv env # Create an virtual environment
    source env/bin/activate # To activate the virtual environment
    pip install --user -r requirements.txt # Install the dependencies inside the virtual environement

## Run in Docker dev container (Optional)
The container has its own local filesystem, thus the data must be mounted to the container. Modify the `.devcontainer/devcontainer.json` `"mount"` properties to mount required directory to the container. 

Open command pallete by pressing `F1` in VSCode and run `Remote-Containers: Build Container` to build the container. The VSCode will restart shortly and run in the dev container.

## Using the notebooks
The notebooks can be used directly from the `notebooks` folder to reproduce the analysis.

# Contributing
To contribute, feel free to fork the project and open a pull request on Github. Contributions are greatly appreciated. 

# Content
1. Computational simulation for single-pixel ghost imaging. [Live Demo](https://colab.research.google.com/github/rstanuwijaya/pf32-python-analysis/blob/main/demo/gdi_simulation/gdi_simulation.ipynb)

<img src="docs/single_pixel_imaging_sim_6x6.png" alt="Single pixel imaging" width="700"/>

2. Metasurface image processing tool to capture metasurface lattice constants.

<img src="./docs/meta_result.png" alt="Metasurface result" width="700"/>

3. Iterative fitting model for coincidence count using LMFIT library.

<img src="docs/fitmodel_result.png" alt="Fitmodel" width="700"/>

4. Vectorized coincidence count analysis tool for data produced by PF32.

<img src="docs/cc_result.png" alt="CC result" width="700">



# Special Thanks
Special thanks to the Professor Jensen and the group members of JensenLab @ HKUST.
