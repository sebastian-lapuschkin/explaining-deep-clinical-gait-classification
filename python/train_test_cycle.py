# this module realizes the basic training and test cycle common to all models and possibly data variants.

import helpers
import scipy
import time
import types
import sys
from model.base import ModelTraining


def run_train_test_cycle(X, Y, L, LS, S, P, model_class,
                        output_root_dir, data_name, target_name,
                        training_programme=None,
                        do_this_if_model_exists='skip', save_data_in_output_dir=True,
                        force_device_for_training=None, force_device_for_evaluation=None):
    """
    This script trains and evaluates a model using the given data X,Y over all splits as determined in S

    Parameters:
    -----------

    X : np.ndarray - An numpy.ndarray shaped (N, T, C), where N is the number of samples, T is the number
        of time points in the data and C is the number of channels per time point.

    Y : np.ndarray - An numpy.ndarray shaped (N, L), where N is the number of samples and L is the number of classes/labels

    L : list - a list of channel labels of length C, where C is the number of channels in the data.
        L holds textual descriptions of the data's channels

    LS: np.array - An numpy.ndarray shaped (N, S), where N is the number of samples and S is the number of existing subjects.
        Identifies the subject belonging to each datum X.
        Should run in parallel to the training labels Y

    S : list of lists - Contains indices determining the partitioning of the data.
        The outer lists groups the splits (ie len(S) groups of data) and each list element of S contains the indices of those lists.

    P : np.ndarray - An numpy.ndarray shaped (N,) describing the permutation applied to the input data X and the target labels Y.
        This allows referencing LS to Y and X.

    model: model_db.Model - a CLASS providing a set of required functions and the model architecture for executing the training and evaluation loop

    output_root_dir: str - a string pointing towards the root folder for writing results into.

    data_name: str - what is the data/feature type called? e.g. GRF or JA_X_Lower, ...

    target_name: str - what is the prediction target called? e.g. Subject, Gender or Injury, ...

    training_programme: (optional) ModelTraining class - If this parameter is not None, the model's default training regime will be overwritten
        with the passed ModelTraining class' train_model() function

    do_this_if_model_exists: str - variable controlling the training/evaluation behaviour if a trained model already exists
        at the model output location. options:
        retrain (do everything from scratch)
        load (load model and skip training, perform evaluation only)
        skip (completely skip, do nothing)

    save_data_in_output_dir: bool - controls wheter to save the experimental data (X, Y, L, LS, S) in the output directory

    force_device_for_training: str - values can be either gpu or cpu. force the use of this device during training.

    force_device_for_evaluation: str - values can either gpu or cpu. force the use of this device during evaluaton.
        here, the use of the GPU is almost always recommended due to the large batch size to be processed.
    """

    # some basic sanity checks
    assert Y.shape[0] == X.shape[0] == LS.shape[0], 'Number of samples differ between labels Y (n={}), data X (n={}) and subject labels LS (n={})'.format(L.shape[0], X.shape[0], LS.shape[0])
    assert len(L) == X.shape[2], 'Number of provided channel names/labels in L (c={}) differs from number of channels in data X(c={})'.format(len(L), X.shape[2])
    assert sum([len(s) for s in S]) == X.shape[0], 'Number of samples distributed over splits in S (n={}) differs from number of samples in X ({})'.format(sum([len(s) for s in S]), X.shape[0])


    # save data, labels and split information in output directory.
    if save_data_in_output_dir:
        print('Saving training and evaluation data to {}'.format(output_root_dir))
        helpers.ensure_dir_exists(output_root_dir)
        scipy.io.savemat('{}/data.mat'.format(output_root_dir), {'X':X})
        scipy.io.savemat('{}/targets.mat'.format(output_root_dir), {'Y':Y})
        scipy.io.savemat('{}/channel_labels.mat'.format(output_root_dir), {'L':L})
        scipy.io.savemat('{}/subject_labels.mat'.format(output_root_dir), {'LS':LS})
        scipy.io.savemat('{}/splits.mat'.format(output_root_dir), {'S':S})
        scipy.io.savemat('{}/permutation.mat'.format(output_root_dir), {'P':P})

    #prepare log to append anything happending in this session. kinda deprecated.
    logfile = open('{}/log.txt'.format(output_root_dir), 'a')

    # start main loop and execute training/evaluation for all the splits definied in S
    for split_index in range(len(S)):
        model = model_class(output_root_dir, data_name, target_name, split_index)
        model_dir = model.path_dir()
        helpers.ensure_dir_exists(model_dir)

        # this case: do nothing.
        if model.exists() and do_this_if_model_exists == 'skip':
            print('Model already exists at {}. skipping'.format(model_dir))
            continue #skip remaining code, there is nothing to be done. please move along.

        # other cases: split data in any case. measure time. set output log
        t_start = time.time()

        # collect data indices from split table
        j_test = split_index;           i_test = S[j_test]
        j_val = (split_index+1)%len(S); i_val = S[j_val]
        j_train = list(set(range(len(S))) - {j_test, j_val})
        i_train = []
        for j in j_train: i_train.extend(S[j])

        # collect data from indices
        x_train = X[i_train, ...]; y_train = Y[i_train, ...]
        x_test  = X[i_test, ...] ; y_test  = Y[i_test, ...]
        x_val   = X[i_val, ...]  ; y_val   = Y[i_val, ...]

        # remember shape of test data as originally given
        x_test_shape_orig  = x_test.shape

        # model-specific data processing
        x_train, x_val, x_test, y_train, y_val, y_test =\
            model.preprocess_data(x_train, x_val, x_test, y_train, y_val, y_test)


        if not model.exists() or (model.exists() and do_this_if_model_exists == 'retrain'):
            model.build_model(x_train.shape, y_train.shape)
            if training_programme is not None:
                #this instance-based monkey-patching is not the best way to do it, but probably the most flexible one.
                model.train_model = types.MethodType(training_programme.train_model, model)
            model.train_model(x_train, y_train, x_val, y_val, force_device=force_device_for_training)
            model.save_model()
        else:
            model.load_model()

        # compute test scores and relevance maps for model.
        results = model.evaluate_model(x_test, y_test,
                                       force_device=force_device_for_evaluation,
                                       lower_upper=helpers.get_channel_wise_bounds(x_train)) # compute and give data bounds computed from training data.

        # measure time for training/evaluation cycle
        t_end = time.time()

        # write report for terminal printing
        report  = '\n{}\n'.format(model.path_dir().replace('/', ' '))
        report += 'test accuracy : {}\n'.format(results['acc'])
        report += 'test loss (l1): {}\n'.format(results['loss_l1'])
        report += 'train-evaluation-sequence done after {}s\n\n'.format(t_end-t_start)
        print(report)

        #dump results to output of this run
        with open('{}/scores.txt'.format(model.path_dir()), 'w') as f: f.write(report)

        #also write results to parsable log file for eval_score_logs module
        logfile.write(report); logfile.flush()

        #dump evaluation results to mat file
        scipy.io.savemat('{}/outputs.mat'.format(model.path_dir()), results)

# end of train_test_cycle


