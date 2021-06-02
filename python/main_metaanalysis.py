
'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de
@version: 1.0
@copyright: Copyright (c)  2021, Sebastian Lapuschkin
@license : BSD-2-Clause
'''

# %%
print('importing packages and modules, defining functions...')
import numpy as np
import scipy.io
import time
import h5py
import argparse

from corelay.processor.base import Processor
from corelay.processor.flow import Sequential, Parallel
from corelay.processor.affinity import SparseKNN
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.clustering import KMeans
from corelay.processor.embedding import TSNEEmbedding, EigenDecomposition
from corelay.io.storage import HashedHDF5
import matplotlib.pyplot as plt

# %%
# custom processors for corelay
class Flatten(Processor):
    def function(self, data):
        return data.reshape(data.shape[0], np.prod(data.shape[1:]))

class Normalize(Processor):
    def function(self, data):
        data = data / data.sum(keepdims=True)
        return data

# constants
ANALYSIS_GROUPING = ['ground_truth', 'as_predicted', 'all']
ATTRIBUTION_TYPES = ['dom', 'act']
MODELS = ['Cnn1DC8', 'Mlp3Layer768Unit', 'MlpLinear', 'SvmLinearL2C1em1']
FOLDS = [str(f) for f in range(10)] + ['all']

# parameterizable data loader
def load_analysis_data(model, fold, attribution_type, analysis_groups):
    """
        loads and prepares data. for expected input parameters, call script with --help parameter.
        outputs a list of sets of values; each entry in the list is its own input for a SpRAy analysis.
    """
    assert model in MODELS, 'Invalid model argument "{}". Pick from: {}'.format(model, MODELS)
    assert fold in FOLDS, 'Invalid model argument "{}". Pick from: {}'.format(fold, FOLDS)
    assert attribution_type in ATTRIBUTION_TYPES, 'Invalid model argument "{}". Pick from: {}'.format(attribution_type, ATTRIBUTION_TYPES)
    assert analysis_groups in ANALYSIS_GROUPING, 'Invalid analysis_groups argument "{}". Pick from: {}'.format(analysis_groups, ANALYSIS_GROUPING)

    #load precomputed model outputs (predictions, attributions)
    targets_health = scipy.io.loadmat('./data_metaanalysis/2019_frontiers_small_dataset_v3_aff-unaff-atMM_1-234_/targets.mat')
    targets_injurytypes = scipy.io.loadmat('./data_metaanalysis/2019_frontiers_small_dataset_v3_aff-unaff-atMM_1-234_/targets_injurytypes.mat')
    splits = scipy.io.loadmat('./data_metaanalysis/2019_frontiers_small_dataset_v3_aff-unaff-atMM_1-234_/splits.mat')

    if fold == 'all':
        fold = [int(f) for f in FOLDS if f != 'all']
    else:
        fold = [int(fold)]

    split_indices = np.concatenate([splits['S'][0,f][0] for f in fold],axis=0)
    y_pred = []; R = []

    for f in fold:
        model_outputs = scipy.io.loadmat('./data_metaanalysis/2019_frontiers_small_dataset_v3_aff-unaff-atMM_1-234_/Injury/GRF_AV/{}/part-{}/outputs.mat'.format(model, f))
        y_pred.append(model_outputs['y_pred'])
        R.append(model_outputs['R_pred_{}_epsilon'.format(attribution_type)])
    y_pred = np.concatenate(y_pred, axis=0)
    R = np.concatenate(R, axis=0)
    R = np.reshape(R, [-1, np.prod(R.shape[1::])]) # get relevance maps into uniform and flattened shape

    true_injury_sublabels = targets_injurytypes['Y'][split_indices]
    true_health_labels = targets_health['Y'][split_indices]

    if analysis_groups == 'as_predicted':
        y = np.argmax(y_pred, axis=1) # analyze as predicted, y = ypred
    else:
        y = np.argmax(targets_health['Y'][split_indices], axis=1) # analyze as actual label groups stuff, y = ytrue

    # split data into inputs for multiple experiments, grouped by label assignment strategy for y
    evaluation_groups = []
    for cls in np.unique(y):
        evaluation_groups.append({  'cls':cls,
                                    'y':y[y == cls],
                                    'R':R[y == cls],
                                    'y_injury_type':true_injury_sublabels[y == cls], # true injury sublabels
                                    'y_health_type':true_health_labels[y == cls], #healthy or not?
                                    })

    # some debug prints
    # for e in evaluation_groups:
    #    print(e['cls'], e['y'].shape, e['R'].shape, e['y_injury_type'].shape, e['y_health_type'].shape)
    return evaluation_groups



# main module doing most of the things.
def main():

    print('parsing command line arguments...')
    parser = argparse.ArgumentParser(description="Use Spectral Relevance Analysis via CoReLay to analyze patterns in the model's behavior.")
    parser.add_argument('-rs', '--random_seed', type=str, default='0xDEADBEEF', help='seed for the numpy random generator')
    parser.add_argument('-ag', '--analysis_groups', type=str, default='ground_truth', help='How to handle/group data for analysis. Possible inputs from: {}'.format(ANALYSIS_GROUPING))
    parser.add_argument('-at', '--attribution_type', type=str, default='act', help='Determines the attribution scores wrt either the DOMinant prediction or the ACTual class of a sample. Possible inputs from: {}'.format(ATTRIBUTION_TYPES))
    parser.add_argument('-m', '--model', type=str, default='Cnn1DC8', help='For which model(s precomputed attribution scores) should the analysis be performed? Possible inputs from: {}'.format(MODELS))
    parser.add_argument('-f', '--fold', type=str, default='0', help='Which (test9 data split/fold should be analyzed? Possible inputs from: {} '.format(FOLDS))
    ARGS = parser.parse_args()

    print('setting random seed...')
    np.random.seed(int(ARGS.random_seed,0))

    print('loading and preparing data as per specification...')
    evaluation_groups = load_analysis_data(ARGS.model, ARGS.fold, ARGS.attribution_type, ARGS.analysis_groups)

    # TODO SpRAy below






# %%
if __name__ == '__main__':
    main()
    exit()


# %%
    print('init')
    np.random.seed(0xDEADBEEF)


    #class_selection = 'ground_truth'
    class_selection = 'as_predicted'

    #R_to_analyze = 'R_pred_act_epsilon'
    R_to_analyze = 'R_pred_dom_epsilon'

    #some settings for data loading
    models = ['Cnn1DC8', 'Mlp3Layer768Unit', 'MlpLinear', 'SvmLinearL2C1em1']
    folds = [i for i in range(10)] #
    for model in models:
        for fold in folds:
            print('\n'*2)

            #load model outputs
            model_output_path = './data_metaanalysis/2019_frontiers_small_dataset_v3_aff-unaff-atMM_1-234_/Injury/GRF_AV/{}/part-{}/outputs.mat'.format(model, fold)
            targets_health = scipy.io.loadmat('./data_metaanalysis/2019_frontiers_small_dataset_v3_aff-unaff-atMM_1-234_/targets.mat')
            targets_injurytypes = scipy.io.loadmat('./data_metaanalysis/2019_frontiers_small_dataset_v3_aff-unaff-atMM_1-234_/targets_injurytypes.mat')
            splits = scipy.io.loadmat('./data_metaanalysis/2019_frontiers_small_dataset_v3_aff-unaff-atMM_1-234_/splits.mat')


            true_injury_sublabels = targets_injurytypes['Y'][splits['S'][0,fold][0]]


            print('data loading (%s)' % model_output_path)
            model_outputs = scipy.io.loadmat(model_output_path)
            #print(model_outputs.keys())

            if class_selection == 'as_predicted':
                #NOTE: first attempt: analyze as predicted, y = ypred
                y = np.argmax(model_outputs['y_pred'], axis=1)

            elif class_selection  == 'ground_truth':
                #NOTE: second attempt: analyze as actual label groups stuff, y = ytrue
                y = np.argmax(targets_health['Y'][splits['S'][0,fold][0]], axis=1)



            for cls in np.unique(y):
                print('process {} for class {} as per {}'.format(R_to_analyze, cls, model))

                # get relevance maps for predicted class (in uniform flattened shape)
                R = model_outputs[R_to_analyze][y == cls]
                R = np.reshape(R, [-1, np.prod(R.shape[1::])])

                #get true injury state labels for that class
                y_true_injury = true_injury_sublabels[y == cls,:]


                ## SPRAY STUFF, ADAPTED FROM CORELAY EXAMPLES
                fpath = 'test-cls-{}.analysis.h5'.format(cls)
                with h5py.File(fpath, 'a') as fd:
                    # HashedHDF5 is an io-object that stores outputs of Processors based on hashes in hdf5
                    iobj = HashedHDF5(fd.require_group('proc_data'))

                    # generate some exemplary data
                    n_clusters = range(3, 8)

                    # SpectralClustering is an Example for a pre-defined Pipeline
                    pipeline = SpectralClustering(
                        #affinity = SparseKNN(n_neighbors=4, symmetric=True), #optional, overwrites default
                        affinity = SparseKNN(n_neighbors=2, symmetric=True), #optional, overwrites default

                        # processors, such as EigenDecomposition, can be assigned to pre-defined tasks
                        embedding=EigenDecomposition(n_eigval=8, io=None),
                        # flow-based Processors, such as Parallel, can combine multiple Processors
                        # broadcast=True copies the input as many times as there are Processors
                        # broadcast=False instead attempts to match each input to a Processor
                        clustering=Parallel([
                            Parallel([
                                KMeans(n_clusters=k, io=None) for k in n_clusters
                            ], broadcast=True),
                            # io-objects will be used during computation when supplied to Processors
                            # if a corresponding output value (here identified by hashes) already exists,
                            # the value is not computed again but instead loaded from the io object
                            TSNEEmbedding(io=None)
                        ], broadcast=True, is_output=True)
                    )
                    # Processors (and Params) can be updated by simply assigning corresponding attributes
                    pipeline.preprocessing = Sequential([
                        #SumChannel(),
                        Normalize(),
                        Flatten()
                    ])

                    start_time = time.perf_counter()

                    # Processors flagged with "is_output=True" will be accumulated in the output
                    # the output will be a tree of tuples, with the same hierachy as the pipeline
                    # (i.e. clusterings here contains a tuple of the k-means outputs)
                    clusterings, tsne = pipeline(R)

                    # since we memoize our results in a hdf5 file, subsequent calls will not compute
                    # the values (for the same inputs), but rather load them from the hdf5 file
                    # try running the script multiple times
                    duration = time.perf_counter() - start_time
                    print(f'Pipeline execution time: {duration:.4f} seconds')


                    #print(clusterings, tsne)
                    #print(len(clusterings))

                    fig = plt.figure(figsize=(2*(len(clusterings)+1),2))

                    #true injury sublabel plots
                    ax = plt.subplot(1, len(clusterings)+1, 1)
                    ax.scatter(tsne[:,0], tsne[:,1],c=np.argmax(y_true_injury,axis=1))
                    ax.set_title('gt injury labels')

                    for i in range(1,len(clusterings)):
                        ax = plt.subplot(1, len(clusterings)+1, i+1)
                        ax.scatter(tsne[:,0], tsne[:,1],c=clusterings[i])
                        #ax.set_title('clustered')

                    plt.suptitle('xai clusters; model: {}, fold: {}, {}: {}'.format(model, fold, class_selection,  cls))
                    #plt.savefig(fpath + '-tsne-{}.pdf'.format(plcs))
                    print('number of samples: {}'.format(tsne.shape[0]))
                    plt.show()













# %%
