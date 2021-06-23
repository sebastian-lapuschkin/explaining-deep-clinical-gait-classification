#!/bin/bash
for tp in 12 ; do # pair tp 12 with gi 0, tp 30 with gi 1 , ie scales the visualized embedding wrt number of involved datapoints per class
for gi in 0 ; do
for model in Cnn1DC8 ; do
for fold in {0..9} ; do
    python main_metaanalysis.py -tp $tp -cmaps tab20 -f $fold -gi $gi -m $model -sr # -s # no showing
done
done
done
done
