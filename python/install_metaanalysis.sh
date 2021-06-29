#!/bin/bash

# install corelay + dependecies for meta analyses based on xai
pip install corelay
#pip install corelay[umap,hdbscan] # importing umap makes everything slow. we do not need it.
pip install h5py
pip install termcolor



