
'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de
@version: 1.0
@copyright: Copyright (c)  2021, Sebastian Lapuschkin
@license : BSD-2-Clause
'''

# %%
# rough draft of what to do, before we build a nice script
print('imports')
import numpy as np
import scipy.io

import time

import h5py
import numpy as np

from corelay.base import Param
from corelay.processor.base import Processor
from corelay.processor.flow import Sequential, Parallel
from corelay.processor.affinity import SparseKNN
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.clustering import KMeans
from corelay.processor.embedding import TSNEEmbedding, EigenDecomposition
from corelay.io.storage import HashedHDF5

import matplotlib.pyplot as plt


# custom processors can be implemented by defining a function attribute
class Flatten(Processor):
    def function(self, data):
        return data.reshape(data.shape[0], np.prod(data.shape[1:]))


class Normalize(Processor):
    def function(self, data):
        data = data / data.sum(keepdims=True)
        return data

# %%
if __name__ == '__main__':

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
            #permutation = scipy.io.loadmat('./data_metaanalysis/2019_frontiers_small_dataset_v3_aff-unaff-atMM_1-234_/permutation.mat')

            true_injury_sublabels = targets_injurytypes['Y'][splits['S'][0,fold][0]]
            # print(true_injury_sublabels)


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
