#!/bin/bash
#
#
# REQUIRES: installed (Ana)conda virtualization framework
#
# below calls create virtual environments for running and using the code in this repo. pick the one suitable for your hardware.
# comment/uncomment the desired conda setup calls



# Option A: CPU Only
# CREATES: an environment called "gait", for running the code on CPU hardware
conda create -n gait -c conda-forge python=3.* numpy matplotlib prettytable flake8 pylint rope scikit-learn scikit-image scipy natsort




# Option B: GPU Support
# CREATES: an environment called gait-gpu, for running the code on NVIDIA-hardware
# conda create -n gait-gpu -c conda-forge python=3.* numpy cupy matplotlib prettytable flake8 pylint rope scikit-learn scikit-image scipy natsort




# After setting up the required environments, just activate the environment and run the script(s), e.g.
# conda activate gait-gpu ; python gait_experiments.py