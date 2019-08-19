
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
import sys

import numpy
import numpy as numpy # no cupy import here, stay on the CPU in the main script.

import scipy.io as scio # scientific python package, which supports mat-file IO within python
import helpers
import eval_score_logs

from model import *             #import all known model architectures
from model.training import *    # import all known nn-training-programmes
import train_test_cycle         #import main loop


ROOTFOLDER='.'
OUTPUT_DIR='./test_output'
DO_IF_MODEL_EXISTS='evaluate'

################################
#           "Main"
################################

#load matlab data as dictionary using scipy
gaitdata = scio.loadmat('{}/data/DatasetC_Classification_Norm_5_Normal-Ankle-Hip-Knee.mat'.format(ROOTFOLDER))

# Feature -> Bodenreaktionskraft
X_GRF_AV = gaitdata['Feature']                          # 1142 x 101 x 6
Label_GRF_AV = gaitdata['Feature_GRF_AV_Label'][0][0]   # x 6 channel label

#transposing axes, to obtain N x time x channel axis ordering, as in Horst et al. 2019
X_GRF_AV = numpy.transpose(X_GRF_AV, [0, 2, 1])

# Targets -> Subject labels und gender labels
Y_Subject = gaitdata['Target_Subject']                  # 1142 x 57, binary labels
Y_Injury = gaitdata['Target_Injury']                    # 1142 x 1 , binary labels

#split data for experiments.
#Injury identification: create 8 splits with the injury split evenly over all partitions, but pool
#the samples per subject in only one bin: avoid prediction based on personal characteristics
#use a random seed to make partitioning deterministic
RANDOMSEED = 1234

Y_Injury_trimmed = helpers.trim_empty_classes(Y_Injury)
Y_Subject_trimmed = helpers.trim_empty_classes(Y_Subject) #TODO careful with that!
SubjectIndexSplits, InjuryIndexSplits, Permutation = helpers.create_index_splits(Y_Subject_trimmed, Y_Injury_trimmed, splits=8, seed=RANDOMSEED)

#apply the permutation to the given data for the inputs and labels to match the splits again
X_GRF_AV = X_GRF_AV[Permutation, ...]
Y_Injury_trimmed = Y_Injury_trimmed[Permutation, ...]
Y_Subject_trimmed = Y_Subject_trimmed[Permutation, ...]

#specify which architectures should be processed. caution: selecting ALL models at once might exceed GPU memory.
#cupy apparently does not support mark-and-sweep garbage collection
#I recommend executing one-model-at-a-time. Do you need a command line argument parser for that?
architectures = []
architectures += [SvmLinearL2C1e0, SvmLinearL2C1em1, SvmLinearL2C1ep1]
#architectures += [MlpLinear, Mlp2Layer64Unit, Mlp2Layer128Unit, Mlp2Layer256Unit, Mlp2Layer512Unit, Mlp2Layer768Unit]
#architectures += [Mlp3Layer64Unit, Mlp3Layer128Unit, Mlp3Layer256Unit]
#architectures +=  [Mlp3Layer512Unit, Mlp3Layer768Unit]
#architectures += [CnnA3, CnnA6]
#architectures += [CnnC3, CnnC6]
#architectures += [CnnC3_3]

for arch in architectures:
    # this load of parameters could also be packed into a dict and thenn passed as **param_dict, if this were to be automated further.
    train_test_cycle.run_train_test_cycle(
        X=X_GRF_AV,
        Y=Y_Injury_trimmed,
        L=Label_GRF_AV,
        LS=Y_Subject,
        S=InjuryIndexSplits,
        model_class=arch,
        output_root_dir=OUTPUT_DIR,
        data_name='GRF_AV',
        target_name='Injury',
        training_programme=None, #model training behavio can be exchanged (for NNs), eg by using NeuralNetworkTrainingQuickTest instead of None. define new behaviors in mode.training.py!
        do_this_if_model_exists=DO_IF_MODEL_EXISTS,
        force_device_for_evaluation='gpu' #computing heatmaps on gpu is always worth it for any model. requires a gpu, obviously
        )
eval_score_logs.run(OUTPUT_DIR)
