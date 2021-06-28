#!/bin/bash

for model in Cnn1DC8 ; do
for fold in {0..9} ; do
    python main_metaanalysis.py -cmaps tab20 -f $fold -gi 0 -tp 12 -m $model -sr # -s # no showing
    python main_metaanalysis.py -cmaps tab20 -f $fold -gi 1 -tp 30 -m $model -sr # -s # no showing
done
done

