import scipy.io as scio
import model_io
import time
import sys
import numpy as np
import time
from modules import SoftMax



###############################
# HELPER FUNCTIONS
###############################

def acc(Y,Ypred):
    return np.mean(np.argmax(Y,axis=1) == np.argmax(Ypred,axis=1))


def reshape_data(X,Y,modelpath):
    #reshape data according to loaded architecture. return as X,Y again
    N,T,C = X.shape

    if 'Linear' in modelpath or 'LayerFCNN' in modelpath:
        #flatten data
        X = np.reshape(X, [N,-1])

    elif 'CNN-' in modelpath and not 'LayerFCNN' in modelpath:
        #special 3-stride cnn
        if 'CNN-C3-3' in modelpath:
            #in order to realize full 3-convs with stride 3 in the time axis, we need to pad with one zero
            X = np.concatenate([X, np.zeros([N,1,C],dtype=X.dtype)],axis=1)
        #regular cnns: just add extra axis
        X = X[...,None]

    return X, Y


###############################
# HELPER FUNCTIONS
###############################
"""
helper functions f(X,p)
do receive some data X which is manipulated according to some param(s) p
and then returned.
"""

def gaussian_noise(X, sigma):
    return X + sigma * np.random.randn(X.size)



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
    M,D = X.shape
    return [np.random.permutation(D) for m in xrange(M)]

###############################
# MAIN PERTURBATION FUNCTIONS
###############################

def perturbations(nn, X, Y, R, CHANGE, repetitions, orderfxn, noisefxn, noiseparam):
    """
    This is a general, parameterizable perturbation method.

    """

    N, nclasses = Y.shape
    M = X[0, ...].size # 'size' of each sample in number of pixels/dimensions
    YpredPerturbed = np.zeros([N, nclasses, len(CHANGE), repetitions]) # container for gradual perturbations across all repetitions

    for c in xrange(nclasses):
        # do all classes separately to keep the memory footprint low.
        IcurrentClass = Y[:, c] > 0
        for r in xrange(repetitions):   # repeat for some times
            Xt = X[IcurrentClass,...]   # get a fresh copy of the 'clean' data.
            Xtshape = Xt.shape          # get the shape constant for later reconstruction
            Xt = np.reshape(Xt, [-1, M])# reshape into 1-dim samples, which makes perturbation across all models easier.
            ORDERS = orderfxn(Xt, R)    # compute a perturbation order, provided the given method.

            for t in xrange(len(CHANGE)):
                # iteratively change a given percentage of the data at a time
                change_start = 0 if t == 0 else int(CHANGE[t-1]*M/100.)
                change_end = int(CHANGE[t]*M/100.)
                print change_start, change_end
                for i in xrange(IcurrentClass.sum()):
                    Xt[i,ORDERS[i][change_start:change_end]] = noisefxn(Xt[i, ORDERS[i][change_start:change_end]], noiseparam)


                Yp = nn.forward(np.reshape(Xt, Xtshape))                                # reconstruct original shape and predict
                Yp = nn.modules[-2].Y if nn.modules[-1].__class__ == SoftMax else Yp    # collect presoftmax-prediction, if we have a softmax layer. Softmax can be added later manually, if needed
                YpredPerturbed[IcurrentClass,:,t,r] = Yp                                # save the current results in the output tensor

    return YpredPerturbed









def run(workerparams):

    t_start = time.time()

    REPS = 10                                   # repeat the experiment ten times
    CHANGEPERCENT = range(0,50,1)               # change up to 50% of the data

    #unpack parameters
    S = workerparams['split_no']                # an int. the current split
    modelpath = workerparams['model']           # path of the model to load
    relevancepath = workerparams['relevances']  # path of the relevance mat file to load
    X = workerparams['Xtest']                   # N x T x C test data
    Y = workerparams['Ytest']                   # N x 2 or N x 57 test labels. binary

    print 'split', S, ': [1] loading model [time: {}]'.format(time.time() - t_start)
    nn = model_io.read(modelpath)

    print 'split', S, ': [2] loading precomputed model results [time: {}]'.format(time.time() - t_start)
    modeloutputs = scio.loadmat(relevancepath)

    print 'split', S, ': [3] (re)shaping data to fit model inputs [time: {}]'.format(time.time() - t_start)
    X,Y = reshape_data(X,Y,modelpath)

    print 'split', S, ': [4] prediction performance sanity check [time: {}]'.format(time.time() - t_start)
    Ypred = nn.forward(X)

    #compare computed and precomputed prediction scores before doing anything else
    assert acc(Ypred, modeloutputs['Ypred']) == 1.0                                                                                                                         #computed and stored predictions should match.
    assert acc(Ypred, Y) == acc(modeloutputs['Ypred'],Y), "{} {} {}".format(acc(Ypred, Y), acc(modeloutputs['Ypred'], Y), acc(Y, modeloutputs['Ypred'])- acc(Y, Ypred))     #second test for that
    np.testing.assert_allclose(Ypred, modeloutputs['Ypred'])                                                                                                                #third, more detailed test.

    print '    split', S, ': [5] sanity check passed. model performance for is at {}% [time: {}]'.format(100*acc(Ypred, Y), time.time() - t_start)


    print 'split', S, ': [6] random additive gaussian permutations on the data [time: {}]'.format(time.time() - t_start)
    result = perturbations(nn=nn, X=X, Y=Y, R=None, CHANGE=CHANGEPERCENT, repetitions=REPS, orderfxn=random_order, noisefxn=gaussian_noise, noiseparam=1)




    #YP = random_perturbations(nn,X,Y,CHANGE,repetitions,0.1)


    #after here the actual work
    RVARIANTS = ['RpredAct', 'RpredActComp'] #relevance variants to compare
    # PLUS RANDOM PERTURBATIONS




    #change 10 percent max
    # in steps of 1 percent (rounded up))

    #type of exp:
        #ypred x 10
        #relevance x 10 / X

    return
