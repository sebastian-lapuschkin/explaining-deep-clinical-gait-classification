
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







################################
#           "Main"
################################

gaitdata = scio.loadmat('Gait_GRF_JA_Label.mat') #load matlab data as dictionary using scipy

# Bodenreaktionskraft mit zwei verschiedenen Datennormalisierungen
X_GRF_AV = gaitdata['Feature_GRF_AV']                       # 1142 x 101 x 6
Label_GRF_AV = gaitdata['Feature_GRF_AV_Label']             # 1 x 1 x 6 channel label

X_GRF_JV = gaitdata['Feature_GRF_JV']                       # 1142 x 101 x 6
Label_GRF_JV = gaitdata['Feature_GRF_JV_Label']             # 1 x 1 x 6 channel label

# Gelenkwinkel in drei Drehachsen fuer den gesamten Koerper
X_JA_Full = gaitdata['Feature_JA_Full']                     # 1142 x 101 x 33
Label_JA_Full = gaitdata['Feature_JA_Full_Label']           # 1 x 1 x 33 channel label

# ... fuer den Unterkoerper
X_JA_Lower = gaitdata['Feature_JA_Lower']                # 1142 x 101 x 10
Label_JA_Lower = gaitdata['Feature_JA_Lower_Label']      # 1 x 1 x 10 channel label

# Gelenkwinkel lediglich in der Hauptdrehachse fuer den gesamten Koerper
X_JA_X_Full = gaitdata['Feature_JA_X_Full']
Label_JA_X_Full = gaitdata['Feature_JA_X_Full_Label']

# ... und den Unterkoerper
X_JA_X_Lower = gaitdata['Feature_JA_X_Lower']
Label_JA_X_Lower = gaitdata['Feature_JA_X_Lower_Label']

#Target subject labels und gender labels
Y_Subject = gaitdata['Target_SubjectID']                    # 1142 x 57, binary labels
Y_Gender = gaitdata['Target_Gender']                        # 1142 x 1 , binary labels
#-> create uniform label structure for gender
Y_Gender = np.repeat(Y_Gender, 2, axis=1)
Y_Gender[:, 1] = 1-Y_Gender[:, 0]                           # 1142 x 2, binary labels  (562 vs 580)

#split data for experiments.
#Subject identification: create 10 splits with the samples of each subject spread over all splits.
#Gender identification: create 10 splits with the genders split evenly over all partitions, but pool
#the samples per subject in only one bin: avoid prediction based on personal characteristics
#use a random seed to make partitioning deterministic
SubjectIndexSplits, GenderIndexSplits, Permutation = helpers.create_index_splits(Y_Subject, Y_Gender, splits=10, seed=1234)

#apply the permutation to the given data for the inputs and labels to match the splits again
X_GRF_AV = X_GRF_AV[Permutation, ...]
X_GRF_JV = X_GRF_JV[Permutation, ...]
X_JA_Full = X_JA_Full[Permutation, ...]
X_JA_Lower = X_JA_Lower[Permutation, ...]
X_JA_X_Full = X_JA_X_Full[Permutation, ...]
X_JA_X_Lower = X_JA_X_Lower[Permutation, ...]
Y_Gender = Y_Gender[Permutation, ...]
Y_Subject = Y_Subject[Permutation, ...]

#create dictionaries for easier batch access for training and testing.
X = {'GRF_AV': X_GRF_AV,
     'GRF_JV': X_GRF_JV,
     'JA_Full': X_JA_Full,
     'JA_Lower': X_JA_Lower,
     'JA_X_Full': X_JA_X_Full,
     'JA_X_Lower': X_JA_X_Lower}

Y = {'Gender': Y_Gender,
     'Subject': Y_Subject}

L = {'GRF_AV': Label_GRF_AV,
     'GRF_JV': Label_GRF_JV,
     'JA_Full': Label_JA_Full,
     'JA_Lower': Label_JA_Lower,
     'JA_X_Full': Label_JA_X_Full,
     'JA_X_Lower': Label_JA_X_Lower}

S = {'Gender':GenderIndexSplits,
     'Subject':SubjectIndexSplits}

#create folder for today's experiments.
DAYFOLDER = './' + str(datetime.datetime.now()).split()[0]


#run some experiments
training.run_3layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=256)
training.run_3layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=512)
training.run_3layer_fcnn(X, Y, L, S, DAYFOLDER, n_hidden=1024)

eval_score_logs.run(DAYFOLDER)
