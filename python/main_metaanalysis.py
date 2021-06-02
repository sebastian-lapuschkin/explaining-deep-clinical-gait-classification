
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
import os
from natsort import natsorted

from corelay.processor.base import Processor
from corelay.processor.flow import Sequential, Parallel
from corelay.processor.affinity import SparseKNN
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.clustering import KMeans
from corelay.processor.embedding import TSNEEmbedding, EigenDecomposition
from corelay.io.storage import HashedHDF5
import matplotlib.pyplot as plt

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


def args_to_stuff(ARGS):
    # reads command line arguments and creates a folder name for figures and info for reproducing the call
    relevant_keys = ['random_seed', 'analysis_groups', 'attribution_type',
                    'model', 'fold', 'min_clusters', 'max_clusters',
                    'neighbors_affinity', 'cmap_injury', 'cmap_clustering']
    relevant_keys = natsorted(relevant_keys)

    foldername = '-'.join(['{}'.format(getattr(ARGS,k)) for k in relevant_keys])
    args_string = '  '.join(['--{} {}'.format(k, getattr(ARGS,k)) for k in relevant_keys])
    return foldername, args_string


# main module doing most of the things.
def main():

    print('parsing command line arguments...')
    parser = argparse.ArgumentParser(   description="Use Spectral Relevance Analysis via CoReLay to analyze patterns in the model's behavior.",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-rs', '--random_seed', type=str, default='0xDEADBEEF', help='seed for the numpy random generator')
    parser.add_argument('-ag', '--analysis_groups', type=str, default='ground_truth', help='How to handle/group data for analysis. Possible inputs from: {}'.format(ANALYSIS_GROUPING))
    parser.add_argument('-at', '--attribution_type', type=str, default='act', help='Determines the attribution scores wrt either the DOMinant prediction or the ACTual class of a sample. Possible inputs from: {}'.format(ATTRIBUTION_TYPES))
    parser.add_argument('-m', '--model', type=str, default='Cnn1DC8', help='For which model(s precomputed attribution scores) should the analysis be performed? Possible inputs from: {}'.format(MODELS))
    parser.add_argument('-f', '--fold', type=str, default='0', help='Which (test9 data split/fold should be analyzed? Possible inputs from: {} '.format(FOLDS))
    parser.add_argument('-mc','--min_clusters', type=int, default=3, help='Minimum number of clusters for cluster label assignment in the analysis.' )
    parser.add_argument('-MC','--max_clusters', type=int, default=8, help='Maximum number of clusters for cluster label assignment in the analysis.' )
    parser.add_argument('-na','--neighbors_affinity', type=int, default=3, help='Number of nearest neighbors to considef for affinity graph computation')
    parser.add_argument('-neig', '--number_eigen', type=int, default=8, help='Number of eigenvalues to consider for the spectral embedding, ie, the number of eigenvectors spanning the spectral space, ie, the dimensionalty of the computed spectral embedding')
    parser.add_argument('-cmapi','--cmap_injury', type=str, default='Set1', help='Color map for drawing the ground truth injury labels. Any valid matplotlib colormap name is can be given')
    parser.add_argument('-cmapc','--cmap_clustering', type=str, default='Set2', help='Color map for drawing the cluster labels inferred by SpRAy. Any valid matplotlib colormap name is can be given')
    parser.add_argument('-o', '--output', type=str, default='./output_metaanalysis', help='Output root directory for the computed results. Figures and embedding coordinates, etc, will be stored here in parameter-dependently named sub-folders')
    parser.add_argument('-s','--show', action='store_true', help='Show intermediate figures?')
    ARGS = parser.parse_args()



    print('setting random seed...')
    np.random.seed(int(ARGS.random_seed,0))

    print('loading and preparing data as per specification...')
    evaluation_groups = load_analysis_data(ARGS.model, ARGS.fold, ARGS.attribution_type, ARGS.analysis_groups)

    print('Starting Spectral Relevance Analysis...')
    for e in evaluation_groups:

        cls = e['cls']
        R = e['R']
        y_true_injury = e['y_injury_type']
        n_clusters = range(ARGS.min_clusters, ARGS.max_clusters+1) # +1, because range is max value exclusive

        print('    process "{}" relevance for class {} ({}) as per {}'.format(ARGS.attribution_type, cls, ARGS.analysis_groups, ARGS.model))

        pipeline = SpectralClustering(
            #optional, overwrites default settings of SpectralClustering class
            affinity  = SparseKNN(n_neighbors=ARGS.neighbors_affinity, symmetric=True),
            embedding = EigenDecomposition(n_eigval=ARGS.number_eigen),
            clustering=Parallel([
                Parallel([
                    KMeans(n_clusters=k) for k in n_clusters
                ], broadcast=True),
                TSNEEmbedding() #use default parameters for TSNE
            ], broadcast=True, is_output=True)
        )
        # Data (ie relevance) preprocessors for above pipeline
        pipeline.preprocessing = Sequential([
            Normalize(),    # normnalization to compare the structure in the relevance, not the overall magnitude scaling (which depends on f(x))
            Flatten()       # redundant.
        ])

        start_time = time.perf_counter()

        # Run the pipeline
        # Processors flagged with "is_output=True" will be accumulated in the output
        # the output will be a tree of tuples, with the same hierachy as the pipeline
        # (i.e. clusterings here contains a tuple of the k-means outputs)
        clusterings, tsne_embedding = pipeline(R)

        duration = time.perf_counter() - start_time
        print('    Pipeline execution time: {:.4f} seconds with {} input samples'.format(duration, tsne_embedding.shape[0]))

        # drawing figures of results
        fig = plt.figure(figsize=(2*(len(clusterings)+1),2))

        #true injury sublabel plots
        ax = plt.subplot(1, len(clusterings)+1, 1)
        ax.scatter( tsne_embedding[:,0],
                    tsne_embedding[:,1],
                    c=np.argmax(y_true_injury,axis=1),
                    cmap=ARGS.cmap_injury) #color code injury labels TODO CHOOSABLE COLOR MAP DIFFERENT FROM CLUSTER ASSIGNMENTS
        ax.set_title('gt injury labels')

        for i in range(1,len(clusterings)):
            ax = plt.subplot(1, len(clusterings)+1, i+1)
            ax.scatter( tsne_embedding[:,0],
                        tsne_embedding[:,1],
                        c=clusterings[i],
                        cmap=ARGS.cmap_clustering)
            #ax.set_title('clustered')

        plt.suptitle('Relevance Clusters; model: {}, fold: {}, {}: {}'.format(ARGS.model, ARGS.fold, ARGS.analysis_groups, cls))


        if os.path.isfile(ARGS.output):
            print('Can not save results in "{}", exists as FILE already!'.format(ARGS.output))
        else:
            output_dir, args_string = args_to_stuff(ARGS)
            output_dir = '{}/{}'.format(ARGS.output,output_dir)
            print(output_dir, args_string)

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            print('    saving figure, args and clusterings/embedding in {}'.format(output_dir))
            plt.savefig('{}/cls-{}.svg'.format(output_dir, cls))
            plt.savefig('{}/cls-{}.pdf'.format(output_dir, cls))
            with open('{}/callparams.args'.format(output_dir), 'wt') as f: f.write(args_string)
            np.save('{}/emb-{}.npy'.format(output_dir, cls), tsne_embedding)
            np.save('{}/clust-{}.npy'.format(output_dir, cls), clusterings)

        if ARGS.show:
            plt.show()





#####################
# ENTRY POINT
#####################
if __name__ == '__main__':
    main()

