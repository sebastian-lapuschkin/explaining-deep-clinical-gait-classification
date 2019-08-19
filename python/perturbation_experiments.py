import scipy.io as scio
import model_io
import time
import sys
import numpy as numpy
import time
from modules import SoftMax



###############################
# HELPER FUNCTIONS
###############################

def acc(Y,Ypred):
    return numpy.mean(numpy.argmax(Y,axis=1) == numpy.argmax(Ypred,axis=1))


def reshape_data(X,Y,modelpath):
    #reshape data according to loaded architecture. return as X,Y again
    N,T,C = X.shape

    if 'Linear' in modelpath or 'LayerFCNN' in modelpath:
        #flatten data
        X = numpy.reshape(X, [N,-1])

    elif 'CNN-' in modelpath and not 'LayerFCNN' in modelpath:
        #special 3-stride cnn
        if 'CNN-C3-3' in modelpath:
            #in order to realize full 3-convs with stride 3 in the time axis, we need to pad with one zero
            X = numpy.concatenate([X, numpy.zeros([N,1,C],dtype=X.dtype)],axis=1)
        #regular cnns: just add extra axis
        X = X[...,None]

    return X, Y


###############################
# NOISE FUNCTIONS
###############################
"""
helper functions f(X,p)
do receive some data X which is manipulated according to some param(s) p
and then returned.
"""

def gaussian_noise(X, sigma):
    #additive gaussian noise
    return X + sigma * numpy.random.randn(X.size)

def shot_noise(X, p):
    #random shot noise. set pixels to -1 and 1
    return (numpy.random.rand(X.size) > 0).astype(numpy.float) * 2 - 1

def pepper_noise(X,p):
    #set selected pixels to black (0)
    return numpy.zeros_like(X)

def salt_noise(X,p):
    #set selected pixels to white (1)
    return numpy.ones_like(X)

def negative_salt_noise(X,p):
    return -numpy.ones_like(X)


###############################
# ORDER FUNCTIONS
###############################
"""
order functions f(X,R)
receive some data X shaped M x D and some measure R
by which perturbation orders are computed and returned as
as a list of M lists with D entries or a M x D array
"""
def random_order(X, R):
    M, D = X.shape
    return [numpy.random.permutation(D) for m in range(M)]

def relevance_ascending(X, R):
    # sort by relevance ascendingly, e.g. the least important (and contradicting)
    # dimensions are at the head of the list.
    # this can be used to "repair" broken parts of the data, or for a sanity check.

    M, D = X.shape
    N, E = R.shape
    assert M == N # make sure the shapes match
    assert D == E

    order = [None]*M
    for m in range(M):
        order[m] = R[m,:].argsort()
    return order

def relevance_descending(X,R):
    # reversed ascending order.
    # most relevant parts are at the beginning of the list.
    return [o[::-1] for o in relevance_ascending(X,R)]





###############################
# MAIN PERTURBATION FUNCTIONS
###############################

def perturbations(nn, X, Y, R, CHANGE, repetitions, orderfxn, noisefxn, noiseparam):
    """
    This is a general, parameterizable perturbation method.

    """

    N, nclasses = Y.shape
    M = X[0, ...].size # 'size' of each sample in number of pixels/dimensions
    YpredPerturbed = numpy.zeros([N, nclasses, len(CHANGE), repetitions]) # container for gradual perturbations across all repetitions

    for c in range(nclasses):
        # do all classes separately to keep the memory footprint low.
        IcurrentClass = Y[:, c] > 0
        for r in range(repetitions):   # repeat for some times
            Xt = X[IcurrentClass,...]   # get a fresh copy of the 'clean' data.
            Xtshape = Xt.shape          # get the shape constant for later reconstruction
            Xt = numpy.reshape(Xt, [-1, M])# reshape into 1-dim samples, which makes perturbation across all models easier.
            if not R is None:                       # same for relevance scores, if given
                Rt = R[IcurrentClass,...]
                Rt = numpy.reshape(Rt, [-1, M])
            else:
                Rt = R
            ORDERS = orderfxn(Xt, Rt)    # compute a perturbation order, provided the given method.

            for t in range(len(CHANGE)):
                # iteratively change a given percentage of the data at a time
                change_start = 0 if t == 0 else int(CHANGE[t-1]*M/100.)
                change_end = int(CHANGE[t]*M/100.)
                #print change_start, change_end
                for i in range(IcurrentClass.sum()):
                    Xt[i,ORDERS[i][change_start:change_end]] = noisefxn(Xt[i, ORDERS[i][change_start:change_end]], noiseparam)


                Yp = nn.forward(numpy.reshape(Xt, Xtshape))                                # reconstruct original shape and predict
                Yp = nn.modules[-2].Y if nn.modules[-1].__class__ == SoftMax else Yp    # collect presoftmax-prediction, if we have a softmax layer. Softmax can be added later manually, if needed
                YpredPerturbed[IcurrentClass,:,t,r] = Yp                                # save the current results in the output tensor

    #return YpredPerturbed
    return YpredPerturbed.astype(numpy.float16) # reduce the memory footprint a bit.









def run(workerparams):

    t_start = time.time()

    REPS = 10                                           # repeat the experiment ten times
    CHANGEPERCENT = list(range(0,50,1))                       # change up to 50% of the data

    #unpack parameters
    S = workerparams['split_no']                        # an int. the current split
    modelpath = workerparams['model']                   # path of the model to load
    relevancepath = workerparams['relevances']          # path of the relevance mat file to load
    outputfolder = workerparams['outputfolder']         # path to the output folder
    outputfile = outputfolder + '/perturbations.mat'    # the file to store the results in
    X = workerparams['Xtest']                           # N x T x C test data
    Y = workerparams['Ytest']                           # N x 2 or N x 57 test labels. binary

    print('split', S, ': [1] loading model [time: {}]'.format(time.time() - t_start))
    nn = model_io.read(modelpath)

    print('split', S, ': [2] loading precomputed model results [time: {}]'.format(time.time() - t_start))
    modeloutputs = scio.loadmat(relevancepath)

    print('split', S, ': [3] (re)shaping data to fit model inputs [time: {}]'.format(time.time() - t_start))
    X,Y = reshape_data(X,Y,modelpath)

    print('split', S, ': [4] prediction performance sanity check [time: {}]'.format(time.time() - t_start))
    Ypred = nn.forward(X)

    #compare computed and precomputed prediction scores before doing anything else
    assert acc(Ypred, modeloutputs['Ypred']) == 1.0                                                                                                                         #computed and stored predictions should match.
    assert acc(Ypred, Y) == acc(modeloutputs['Ypred'],Y), "{} {} {}".format(acc(Ypred, Y), acc(modeloutputs['Ypred'], Y), acc(Y, modeloutputs['Ypred'])- acc(Y, Ypred))     #second test for that
    numpy.testing.assert_allclose(Ypred, modeloutputs['Ypred'])                                                                                                                #third, more detailed test.

    print('    split', S, ': [5] sanity check passed. model performance is at {}% [time: {}]'.format(100*acc(Ypred, Y), time.time() - t_start))


    print('split', S, ': [6] random additive gaussian random permutations on the data [time: {}]'.format(time.time() - t_start))
    p_gaussian_random_sigma05   = perturbations(nn=nn, X=X, Y=Y, R=None, CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=gaussian_noise, noiseparam=0.5)
    p_gaussian_random_sigma1    = perturbations(nn=nn, X=X, Y=Y, R=None, CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=gaussian_noise, noiseparam=1)
    p_gaussian_random_sigma2    = perturbations(nn=nn, X=X, Y=Y, R=None, CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=gaussian_noise, noiseparam=2)


    print('split', S, ': [7] different random shot noise variants on the data [time: {}]'.format(time.time() - t_start))
    p_shot_random               = perturbations(nn=nn, X=X, Y=Y, R=None, CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=shot_noise, noiseparam=None)
    p_pepper_random             = perturbations(nn=nn, X=X, Y=Y, R=None, CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=pepper_noise, noiseparam=None)
    p_salt_random               = perturbations(nn=nn, X=X, Y=Y, R=None, CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=salt_noise, noiseparam=None)
    p_negative_salt_random      = perturbations(nn=nn, X=X, Y=Y, R=None, CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=negative_salt_noise, noiseparam=None)


    print('split', S, ': [8] different gaussian noise variants wrt eps-LRP order on the data [time: {}]'.format(time.time() - t_start))
    p_gaussian_reps_sigma05     = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredAct'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=relevance_descending, noisefxn=gaussian_noise, noiseparam=0.5)
    p_gaussian_reps_sigma1      = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredAct'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=relevance_descending, noisefxn=gaussian_noise, noiseparam=1)
    p_gaussian_reps_sigma2      = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredAct'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=relevance_descending, noisefxn=gaussian_noise, noiseparam=2)


    print('split', S, ': [9] different gaussian noise variants wrt composite-LRP order on the data [time: {}]'.format(time.time() - t_start))
    p_gaussian_rcomp_sigma05    = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredActComp'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=relevance_descending, noisefxn=gaussian_noise, noiseparam=0.5)
    p_gaussian_rcomp_sigma1     = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredActComp'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=relevance_descending, noisefxn=gaussian_noise, noiseparam=1)
    p_gaussian_rcomp_sigma2     = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredActComp'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=relevance_descending, noisefxn=gaussian_noise, noiseparam=2)


    print('split', S, ': [10] different shot noise variants wrt eps-LRP order on the data [time: {}]'.format(time.time() - t_start))
    p_shot_reps                 = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredAct'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=shot_noise, noiseparam=None)
    p_pepper_reps               = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredAct'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=pepper_noise, noiseparam=None)
    p_salt_reps                 = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredAct'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=salt_noise, noiseparam=None)
    p_negative_salt_reps        = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredAct'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=negative_salt_noise, noiseparam=None)


    print('split', S, ': [11] different shot noise variants wrt composite-LRP order on the data [time: {}]'.format(time.time() - t_start))
    p_shot_rcomp                = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredActComp'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=shot_noise, noiseparam=None)
    p_pepper_rcomp              = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredActComp'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=pepper_noise, noiseparam=None)
    p_salt_rcomp                = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredActComp'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=salt_noise, noiseparam=None)
    p_negative_salt_rcomp       = perturbations(nn=nn, X=X, Y=Y, R=modeloutputs['RPredActComp'], CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=negative_salt_noise, noiseparam=None)


    print('split', S, ': [12] packing results for {} [time: {}]'.format(outputfile, time.time() - t_start))
    matdict = {   'p_gaussian_random_sigma05' : p_gaussian_random_sigma05,
                    'p_gaussian_random_sigma1' : p_gaussian_random_sigma1,
                    'p_gaussian_random_sigma2' : p_gaussian_random_sigma2,
                    #
                    'p_shot_random' : p_shot_random,
                    'p_pepper_random' : p_pepper_random,
                    'p_salt_random' : p_salt_random,
                    'p_negative_salt_random' : p_negative_salt_random,
                    #
                    'p_gaussian_reps_sigma05' : p_gaussian_reps_sigma05,
                    'p_gaussian_reps_sigma1' : p_gaussian_reps_sigma1,
                    'p_gaussian_reps_sigma2' : p_gaussian_reps_sigma2,
                    #
                    'p_gaussian_rcomp_sigma05' : p_gaussian_rcomp_sigma05,
                    'p_gaussian_rcomp_sigma1' : p_gaussian_rcomp_sigma1,
                    'p_gaussian_rcomp_sigma2' : p_gaussian_rcomp_sigma2,
                    #
                    'p_shot_reps' : p_shot_reps,
                    'p_pepper_reps' : p_pepper_reps,
                    'p_salt_reps' : p_salt_reps,
                    'p_negative_salt_reps' : p_negative_salt_reps,
                    #
                    'p_shot_rcomp' : p_shot_rcomp,
                    'p_pepper_rcomp' : p_pepper_rcomp,
                    'p_salt_rcomp' : p_salt_rcomp,
                    'p_negative_salt_rcomp' : p_negative_salt_rcomp

                }

    #scio.savemat(outputfile, matdict)
    print('split', S, ': [13] done [time: {}]'.format(time.time() - t_start))
    return (outputfile, matdict)

