#!/bin/bash
echo $1
if [ $1 = 'L' ]
then
    ROOTFOLDER=BASELINE-LINEAR-S1234
    ARCHLIST="Linear"

elif [ $1 = '2' ]
then
    ROOTFOLDER=2017-09-15-S1234
    ARCHLIST="2LayerFCNN-64 2LayerFCNN-256 2LayerFCNN-1024"

elif [ $1 = '3' ]
then
    ROOTFOLDER=2017-09-14-S1234
    ARCHLIST="3LayerFCNN-64 3LayerFCNN-256 3LayerFCNN-1024"

elif [ $1 = 'A' ]
then
    ROOTFOLDER=2017-10-05-S1234
    ARCHLIST="CNN-A CNN-A3 CNN-A6"

elif [ $1 = 'C' ]
then
    ROOTFOLDER=2017-10-05-S1234
    ARCHLIST="CNN-C3 CNN-C6"

else
    echo 'invalid argument. How to use:'
    echo './perturb_and_pach.sh A'
    echo ''
    echo 'A = L : Linear architectures'
    echo 'A = 2 : 2-Layer FCNNs'
    echo 'A = 3 : 3-Layer FCNNs'
    echo 'A = A : CNN-A-Architectures'
    echo 'A = C : CNN-C-Architectures'
    echo 'Now try again!'
    echo ''
fi


for DATA in GRF_AV GRF_JV ;  do
for TARGET in Subject Gender ; do
for ARCH in $ARCHLIST ; do
    python perturbations.py folder=$ROOTFOLDER arch=$ARCH data=$DATA target=$TARGET
done #ARCH
done #TARGET
done #DATA

echo 'packing results for' $ROOTFOLDER
find $ROOTFOLDER -type f -name perturbations.mat -print0 | tar -czvf $ROOTFOLDER-PERTURBATIONS --null -T -
