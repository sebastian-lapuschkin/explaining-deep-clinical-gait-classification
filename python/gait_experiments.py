
'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 08.09.2017
@version: 1.0
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

import argparse
import datetime
import os
import sys

import numpy
import numpy as numpy # no cupy import here, stay on the CPU in the main script.

import scipy.io as scio # scientific python package, which supports mat-file IO within python
import helpers
import eval_score_logs
import datetime

import model
from model import *
from model.base import ModelArchitecture, ModelTraining
import train_test_cycle         #import main loop

current_datetime = datetime.datetime.now()
#setting up an argument parser for controllale command line calls
import argparse
parser = argparse.ArgumentParser(description="Train and evaluate Models on human gait recordings!")
parser.add_argument('-d', '--data_path', type=str, default='./data/DatasetC_Classification_Norm_5_Normal-Ankle-Hip-Knee.mat', help='Sets the path to the dataset mat-file to be processed')
parser.add_argument('-o', '--output_dir', type=str, default='./output', help='Sets the output directory root for models and results. Default: "./output"')
parser.add_argument('-me', '--model_exists', type=str, default='skip', help='Sets the behavior of the code in case a model file has been found at the output location. "skip" (default) skips remaining execution loop and does nothing. "retrain" trains the model anew. "evaluate" only evaluates the model with test data')
parser.add_argument('-rs', '--random_seed', type=int, default=1234, help='Sets a random seed for the random number generator. Default: 1234')
parser.add_argument('-s', '--splits', type=int, default=5, help='The number of splits to divide the data into. Default: 5')
parser.add_argument('-a', '--architecture', type=str, default='SvmLinearL2C1e0', help='The name of the model architecture to use/train/evaluate. Can be any joint-specialization of model.base.ModelArchitecture and model.base.ModelTraining. Default: SvmLinearL2C1e0 ')
parser.add_argument('-tp', '--training_programme', type=str, default=None, help='The training regime for the (NN) model to follow. Can be any class from model.training or any class implementing model.base.ModelTraining. The default value None executes the training specified for the NN model as part of the class definition.')
parser.add_argument('-dn', '--data_name', type=str, default='GRF_AV', help='The feature name of the data behind --data_path to be processed. Default: GRF_AV')
parser.add_argument('-tn', '--target_name', type=str, default='Injury', help='The target type of the data behind --data_path to be processed. Default: Injury')
parser.add_argument('-sd', '--save_data', type=bool, default=True, help='Whether to save the training and split data at the output directory root or not. Default: True')
parser.add_argument('-ft', '--force_training_device', type=str, default=None, help='Force training to be performed on a specific device, despite the default chosen numeric backend? Options: cpu, gpu, None. Default: None: Pick as defined in model definition.')
parser.add_argument('-fe', '--force_evaluation_device', type=str, default=None, help='Force evaluat to be performed on a specific device, despite the default chosen numeric backend? Options: cpu, gpu, None. Default: None. NOTE: Execution on GPU is beneficial in almost all cases, due to the massive across-batch-parallelism.')
parser.add_argument('-rc', '--record_call', type=bool, default=False, help='Whether to record the current call to this script in an ouput file specified via -rf or --record-file. Default: False. Only records in case the script terminates gracefully')
parser.add_argument('-rf', '--record_file', type=str, default='./command_history.txt', help='Determines the file name into which the current call to this script is recorded')
ARGS = parser.parse_args()


################################
#           "Main"
################################

#TODO: ISOLATE DATA LOADING INTO CLASS

#load matlab data as dictionary using scipy
gaitdata = scio.loadmat(ARGS.data_path)

# Feature -> Bodenreaktionskraft
X_GRF_AV = gaitdata['Feature']
Label_GRF_AV = gaitdata['Feature_GRF_AV_Label'][0][0]   # x 6 channel label

#transposing axes, to obtain N x time x channel axis ordering, as in Horst et al. 2019
X_GRF_AV = numpy.transpose(X_GRF_AV, [0, 2, 1])         # N x 101 x 6

# Targets -> Subject labels und gender labels
Y_Subject = gaitdata['Target_Subject']                  # 1142 x 57, binary labels
Y_Injury = gaitdata['Target_Injury']                    # 1142 x 1 , binary labels

#split data for experiments.
Y_Injury_trimmed = helpers.trim_empty_classes(Y_Injury)
Y_Subject_trimmed = helpers.trim_empty_classes(Y_Subject)
SubjectIndexSplits, InjuryIndexSplits, Permutation = helpers.create_index_splits(Y_Subject_trimmed, Y_Injury_trimmed, splits=ARGS.splits, seed=ARGS.random_seed)

#apply the permutation to the given data for the inputs and labels to match the splits again
X_GRF_AV = X_GRF_AV[Permutation, ...]
Y_Injury_trimmed = Y_Injury_trimmed[Permutation, ...]
Y_Subject_trimmed = Y_Subject_trimmed[Permutation, ...]

#specify which architectures should be processed. caution: selecting ALL models at once might exceed GPU memory.
#cupy apparently does not support mark-and-sweep garbage collection
#I recommend executing one-model-at-a-time. Do you need a command line argument parser for that?
#architectures = []
#architectures += [SvmLinearL2C1e0, SvmLinearL2C1em1, SvmLinearL2C1ep1]
#architectures += [MlpLinear, Mlp2Layer64Unit, Mlp2Layer128Unit, Mlp2Layer256Unit, Mlp2Layer512Unit, Mlp2Layer768Unit]
#architectures += [Mlp3Layer64Unit, Mlp3Layer128Unit, Mlp3Layer256Unit]
#architectures +=  [Mlp3Layer512Unit, Mlp3Layer768Unit]
#architectures += [CnnA3, CnnA6]
#architectures += [CnnC3, CnnC6]
#architectures += [CnnC3_3]

arch = ARGS.architecture
if isinstance(arch, ModelArchitecture) and isinstance(arch, ModelTraining):
    pass # already a valid class
elif isinstance(arch,str):
    #try to get class from string name
    arch = model.get_architecture(arch)
else:
    raise ValueError('Invalid command line argument type {} for "architecture'.format(type(arch)))

training_regime =  ARGS.training_programme
if training_regime is None or isinstance(training_regime, ModelTraining):
    pass #default training behavior of the architecture class, or training class
elif isinstance(training_regime, str):
    if training_regime.lower() == 'none':
        training_regime = None #default training behavior of the architecture class, or training class
    else:
        training_regime = model.training.get_training(training_regime)
    #try to get class from string name

#register and then select available features
#TODO: REFACTOR INTO A DATA LOADING CLASS
X, X_channel_labels = {'GRF_AV': (X_GRF_AV, Label_GRF_AV)}[ARGS.data_name]

#register and then select available targets
#TODO: REFACTOR INTO A DATA LOADING CLASS
Y, Y_splits = {'Injury': (Y_Injury_trimmed, InjuryIndexSplits) , 'Subject': (Y_Subject_trimmed, SubjectIndexSplits)}[ARGS.target_name]

# this load of parameters could also be packed into a dict and thenn passed as **param_dict, if this were to be automated further.
train_test_cycle.run_train_test_cycle(
        X=X,
        Y=Y,
        L=X_channel_labels,
        LS=Y_Subject,
        S=Y_splits,
        P=Permutation,
        model_class=arch,
        output_root_dir=ARGS.output_dir,
        data_name=ARGS.data_name,
        target_name=ARGS.target_name,
        save_data_in_output_dir=ARGS.save_data,
        training_programme=training_regime, #model training behavio can be exchanged (for NNs), eg by using NeuralNetworkTrainingQuickTest instead of None. define new behaviors in mode.training.py!
        do_this_if_model_exists=ARGS.model_exists,
        force_device_for_training=ARGS.force_training_device,
        force_device_for_evaluation=ARGS.force_evaluation_device#computing heatmaps on gpu is always worth it for any model. requires a gpu, obviously
)
eval_score_logs.run(ARGS.output_dir)

#record function call and parameters if we arrived here

if ARGS.record_call:
    print('Recording current call configuration to {}'.format(ARGS.record_file))
    helpers.ensure_dir_exists(os.path.dirname(ARGS.record_file))
    with open(ARGS.record_file, 'a') as f:
        argline = ' '.join(['--{} {}'.format(a, getattr(ARGS,a)) for a in vars(ARGS)])
        line = '{} : python {} {}'.format(current_datetime,
                                       sys.modules[__name__].__file__,
                                       argline)
        f.write('{}\n\n'.format(line))