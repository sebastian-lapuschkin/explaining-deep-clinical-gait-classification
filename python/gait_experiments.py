
'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 08.09.2017
@version: 1.0
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''


import os
import modules
import model_io
import render
import numpy as np # the de facto numerics package for python
import scipy.io as scio # scientific python package, which supports mat-file IO within python
import matplotlib.pyplot as plt #realizes plotting capabilities similar to what matlab does natively
import helpers







################################
#           "Main"
################################

gaitdata = scio.loadmat('Gait_GRF_JA_Label.mat') #load matlab data as dictionary using scipy

# Bodenreaktionskraft mit zwei verschiedenen Datennormalisierungen
X_GRF_AV = gaitdata['Feature_GRF_AV']                       # 1142 x 101 x 6
Label_GRF_AV = gaitdata['Feature_GRF_AV_Label']             # 1 x 1 x 6 channel label

X_Feature_GRF_JV = gaitdata['Feature_GRF_JV']               # 1142 x 101 x 6
Label_Feature_GRF_JV = gaitdata['Feature_GRF_JV_Label']     # 1 x 1 x 6 channel label

# Gelenkwinkel in drei Drehachsen fuer den gesamten Koerper
X_JA_Full = gaitdata['Feature_JA_Full']                     # 1142 x 101 x 33
Label_JA_Full = gaitdata['Feature_JA_Full_Label']           # 1 x 1 x 33 channel label

# Gelenkwinkel lediglich in der Hauptdrehachse fuer den Unterkoerper
X_JA_X_Lower = gaitdata['Feature_JA_X_Full']                # 1142 x 101 x 10
Label_JA_X_Lower = gaitdata['Feature_JA_X_Full_Label']      # 1 x 1 x 10 channel label

#Target subject labels und gender labels
Y_Subject = gaitdata['Target_SubjectID']                    # 1142 x 57, binary labels
Y_Gender = gaitdata['Target_Gender']                        # 1142 x 1 , binary labels
#-> create uniform label structure for gender
Y_Gender = np.repeat(Y_Gender, 2, axis=1)
Y_Gender[:, 1] = 1-Y_Gender[:, 0]                           # 1142 x 2, binary labels  (562 vs 580)
#print Y_Gender.sum(axis=0)/20.


print helpers.create_index_splits(Y_Subject, Y_Gender, seed=1234)
