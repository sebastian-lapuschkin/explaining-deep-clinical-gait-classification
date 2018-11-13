#!/bin/bash
echo $1 $2
if [ $1 = 'L' ]
then
    ROOTFOLDER=BASELINE-LINEAR-S1234
    ARCHLIST="Linear"

elif [ $1 = 'LSVMC0' ]
then
    ROOTFOLDER=BASELINE-LINEAR-S1234
    ARCHLIST="LinearSVM-L2C0p1SquareHinge"

elif [ $1 = 'LSVMC1' ]
then
    ROOTFOLDER=BASELINE-LINEAR-S1234
    ARCHLIST="LinearSVM-L2C1SquareHinge LinearSVM-L2C1SquareHinge-0p5randn LinearSVM-L2C1SquareHinge-1p0randn"

elif [ $1 = 'LSVMC10' ]
then
    ROOTFOLDER=BASELINE-LINEAR-S1234
    ARCHLIST="LinearSVM-L2C10SquareHinge LinearSVM-L2C10SquareHinge-0p5randn LinearSVM-L2C10SquareHinge-1p0randn"

elif [ $1 = '2' ]
then
    ROOTFOLDER=2017-09-15-S1234
    ARCHLIST="2LayerFCNN-64 2LayerFCNN-256 2LayerFCNN-1024"

elif [ $1 = '2+' ]
then
    ROOTFOLDER=2017-09-15-S1234
    ARCHLIST="2LayerFCNN-128 2LayerFCNN-512"

elif [ $1 = '3' ]
then
    ROOTFOLDER=2017-09-14-S1234
    ARCHLIST="3LayerFCNN-64 3LayerFCNN-256 3LayerFCNN-1024"

elif [ $1 = '3+' ]
then
    ROOTFOLDER=2017-09-14-S1234
    ARCHLIST="3LayerFCNN-128 3LayerFCNN-512"

elif [ $1 = 'A' ]
then
    ROOTFOLDER=2017-10-05-S1234
    ARCHLIST="CNN-A CNN-A3 CNN-A6"

elif [ $1 = 'C3' ]
then
    ROOTFOLDER=2017-10-05-S1234
    ARCHLIST="CNN-C3"

elif [ $1 = 'C6' ]
then
    ROOTFOLDER=2017-10-05-S1234
    ARCHLIST="CNN-C6"

else
    echo 'invalid argument. How to use:'
    echo './perturb_and_pach.sh A [B]'
    echo ''
    echo 'A = L   : Linear architectures'
    echo 'A = 2   : 2-Layer FCNNs'
    echo 'A = 2+  : more 2-Layer FCNNs'
    echo 'A = 3   : 3-Layer FCNNs'
    echo 'A = 3+  : more 3-Layer FCNNs'
    echo 'A = A   : CNN-A-Architectures'
    echo 'A = C   : CNN-C-Architectures'
    echo 'A = C-3 : CNN-C-3-Architectures'
    echo 'A = C-6 : CNN-C-6-Architectures'
    echo 'B = P   : Only pack results'
    echo 'Now try again!'
    echo ''
fi

#if [ $2 = 'P' ]
#then
#   ARCHLIST=
#fi

PIDS=()

#for DATA in GRF_AV GRF_JV ;  do
for DATA in JA_X_Lower JA_X_Full JA_Lower JA_Full; do
for TARGET in Gender Subject ; do
for ARCH in $ARCHLIST ; do
    python perturbations.py folder=$ROOTFOLDER arch=$ARCH data=$DATA target=$TARGET debug=1 &
    PIDS+=($!) #collect process IDs of stuff run in the background
done #ARCH
done #TARGET
done #DATA

#wait for processes to finish
for pid in ${PIDS[*]}; do
    echo "waiting for" $pid ; wait $pid ;  echo $pid "finished".
done

if [ $2 = 'P' ]
then
    echo 'packing results for' $ROOTFOLDER
    find $ROOTFOLDER -type f -name perturbations.mat -print0 | tar -czvf $ROOTFOLDER-PERTURBATIONS.tar.gz --null -T -
    du -sh $ROOTFOLDER-PERTURBATIONS.tar.gz
fi

#alternatively: just pack everything within the linear models folder.
#tar -czvf $ROOTFOLDER-MODELS+PERTURBATIONS.tar.gz $ROOTFOLDER
#du -sh $ROOTFOLDER-MODELS+PERTURBATIONS.tar.gz
