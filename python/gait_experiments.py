
'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 08.09.2017
@version: 1.0
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import datetime
import os
import numpy as np # the de facto numerics package for python
import scipy.io as scio # scientific python package, which supports mat-file IO within python
import helpers
import training
import eval_score_logs
import sys

SKIPTHISMANY=0
ROOTFOLDER='.'
MODELSTOEVALUATE = ['cnnA']
# skip to just do nothing and leave the results as is
# load to load the model and reevaluate, recompute heatmaps
# retrain to overwrite the old model and results
DOTHISIFMODELEXISTS = 'skip'

for param in sys.argv:
    if not '=' in param:
        continue
    else:
        k,v = param.split('=')
        if 'skip' in k:
            print('setting skip param to', v)
            SKIPTHISMANY = int(v)
        elif 'root' in k:
            print('setting root folder param to', v)
            ROOTFOLDER = v
        elif 'model' in k:
            print('setting model to evaluate to', v)
            MODELSTOEVALUATE = v # TODO REWORK handling of this THIS

        #TODO: add "what if model already exists" to list of passable arguments for behavioral control


################################
#           "Main"
################################


def trim_empty_classes(Y):
    # expects an input array shaped Y x C. removes label columns for classes without samples.
    n_per_col = Y.sum(axis=0)
    empty_cols = n_per_col == 0
    if np.any(empty_cols):
        print(('{} Empty columns detected in label matrix shaped {}. Columns are: {}. Removing.'.format(empty_cols.sum(), Y.shape, np.where(empty_cols)[0])))
        Y = Y[:,~empty_cols]
        print(('    shape is {} post column removal.'.format(Y.shape)))
        return Y
    else:
        print(('No empty columns detected in label matrix shaped {}'.format(Y.shape)))

#load matlab data as dictionary using scipy
gaitdata = scio.loadmat('{}/data/DatasetC_Classification_Norm_5_Normal-Ankle-Hip-Knee.mat'.format(ROOTFOLDER)) #TODO make dataset passable parameter

# Feature -> Bodenreaktionskraft
X_GRF_AV = gaitdata['Feature']                       # 1142 x 101 x 6
Label_GRF_AV = gaitdata['Feature_GRF_AV_Label']             # 1 x 1 x 6 channel label

# Targets -> Subject labels und gender labels
Y_Subject = gaitdata['Target_Subject']                    # 1142 x 57, binary labels
Y_Injury = gaitdata['Target_Injury']                        # 1142 x 1 , binary labels

#split data for experiments.
#Injury identification: create 8 splits with the injury split evenly over all partitions, but pool
#the samples per subject in only one bin: avoid prediction based on personal characteristics
#use a random seed to make partitioning deterministic
RANDOMSEED = 1234

Y_Injury = trim_empty_classes(Y_Injury)
Y_Subject = trim_empty_classes(Y_Subject)
SubjectIndexSplits, InjuryIndexSplits, Permutation = helpers.create_index_splits(Y_Subject, Y_Injury, splits=8, seed=RANDOMSEED)

#apply the permutation to the given data for the inputs and labels to match the splits again
X_GRF_AV = X_GRF_AV[Permutation, ...]
Y_Injury = Y_Injury[Permutation, ...]
Y_Subject = Y_Subject[Permutation, ...]

#transposing axes, to obtain N x time x channel axis ordering, as in Horst et al. 2019
X_GRF_AV = np.transpose(X_GRF_AV, [0, 2, 1])


#create dictionaries for easier batch access for training and testing.
X = {'GRF_AV': X_GRF_AV}

Y = {'Injury': Y_Injury}

L = {'GRF_AV': Label_GRF_AV}

S = {'Injury': InjuryIndexSplits,
     'Subject': SubjectIndexSplits}





if 'linear' in MODELSTOEVALUATE:
    DAYFOLDER = './Normal-Ankle-Hip-Knee-2019-08-05-S{}'.format(RANDOMSEED)
    training.run_linear(X, Y, L, S, DAYFOLDER, ifModelExists=DOTHISIFMODELEXISTS)
    training.run_linear_SVM_L2_C1_SquareHinge(X, Y, L, S, DAYFOLDER, ifModelExists=DOTHISIFMODELEXISTS)
    training.run_linear_SVM_L2_C0p1_SquareHinge(X, Y, L, S, DAYFOLDER, ifModelExists=DOTHISIFMODELEXISTS)
    training.run_linear_SVM_L2_C10_SquareHinge(X, Y, L, S, DAYFOLDER, ifModelExists=DOTHISIFMODELEXISTS)
    eval_score_logs.run(DAYFOLDER)

if '3Layer' in MODELSTOEVALUATE:
    DAYFOLDER = './Normal-Ankle-Hip-Knee-2019-08-05-S{}'.format(RANDOMSEED)
    training.run_3layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=64, ifModelExists=DOTHISIFMODELEXISTS)
    training.run_3layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=128, ifModelExists=DOTHISIFMODELEXISTS)
    training.run_3layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=256, ifModelExists=DOTHISIFMODELEXISTS)
    training.run_3layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=512, ifModelExists=DOTHISIFMODELEXISTS)
    training.run_3layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=1024, ifModelExists=DOTHISIFMODELEXISTS)
    eval_score_logs.run(DAYFOLDER)

if '2Layer' in MODELSTOEVALUATE:
    DAYFOLDER = './Normal-Ankle-Hip-Knee-2019-08-05-S{}'.format(RANDOMSEED)
    training.run_2layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=64, ifModelExists=DOTHISIFMODELEXISTS)
    training.run_2layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=128,ifModelExists=DOTHISIFMODELEXISTS)
    training.run_2layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=256, ifModelExists=DOTHISIFMODELEXISTS)
    training.run_2layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=512, ifModelExists=DOTHISIFMODELEXISTS)
    training.run_2layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=1024,ifModelExists=DOTHISIFMODELEXISTS)
    eval_score_logs.run(DAYFOLDER)

if 'cnnC' in MODELSTOEVALUATE:
    DAYFOLDER = './Normal-Ankle-Hip-Knee-2019-08-05-S1234'.format(ROOTFOLDER)
    CURRENTMODEL ='cnnC3'
    training.run_cnn_C3(X, Y, L, S, DAYFOLDER, ifModelExists=DOTHISIFMODELEXISTS) # C3 - mode. classical 1-stride convolutions in either direction with filter size 3 in time and channel direction

    CURRENTMODEL ='cnnC33'
    training.run_cnn_C3_3(X, Y, L, S, DAYFOLDER, ifModelExists=DOTHISIFMODELEXISTS)

    CURRENTMODEL ='cnnC6'
    training.run_cnn_C6(X, Y, L, S, DAYFOLDER, ifModelExists=DOTHISIFMODELEXISTS)

if 'cnnA' in MODELSTOEVALUATE:
    DAYFOLDER = './Normal-Ankle-Hip-Knee-2019-08-05-S1234'.format(ROOTFOLDER)
    MODELSTOEVALUATE ='cnnA'
    training.run_cnn_A(X, Y, L, S, DAYFOLDER, ifModelExists=DOTHISIFMODELEXISTS) # A - mode uses ALL features in each convolution and slides over time. filters are square in shape

    CURRENTMODEL ='cnnA6'
    training.run_cnn_A6(X, Y, L, S, DAYFOLDER, ifModelExists=DOTHISIFMODELEXISTS)

    CURRENTMODEL ='cnnA3'
    training.run_cnn_A3(X, Y, L, S, DAYFOLDER, ifModelExists=DOTHISIFMODELEXISTS)


eval_score_logs.run(DAYFOLDER)
