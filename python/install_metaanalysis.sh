#!/bin/bash

# install corelay + dependecies for meta analyses based on xai
pip install 'git+https://github.com/virelay/corelay'
#pip install 'git+git://github.com/virelay/corelay#egg=corelay[umap,hdbscan]' # importing umap makes everything slow. we do not need it.
pip install h5py
pip install termcolor



