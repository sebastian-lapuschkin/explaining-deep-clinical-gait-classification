#!/bin/bash

if [$1 == 'L']
then

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


ROOTFOLDER=BASELINE-LINEAR-S1234
for DATA in GRF_AV GRF_JV do
for TARGET in Subject Gender do
for ARCH in Linear do
    python perturbations.py folder=$ROOTFOLDER arch=$ARCH data=$DATA target=$TARGET
done #ARCH
done #TARGET
done #DATA

echo packing results for $ROOTFOLDER
find $ROOTFOLDER -type f -name perturbations.mat -print0 | tar -czvf $ROOTFOLDER-PERTURBATIONS --null -T -
