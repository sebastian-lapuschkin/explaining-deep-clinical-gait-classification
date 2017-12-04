import scipy.io as scio
import model_io
import time
import numpy as np
import time


def acc(Y,Ypred):
    return np.mean(np.argmax(Y,axis=1) == np.argmax(Ypred,axis=1))

def random_perturbations(nn,X,Y,CHANGE,repetitions,sigma):
    #additive random permutations on the data.

    N,nclasses = Y.shape
    M = X[0,...].size # 'size' of each sample in number of pixels/dimensions

    YpredPerturbed = np.zeros([N,nclasses,len(CHANGE),repetitions]) #do NOT return all reps, but the average.

    # labels to numeric indicesprint 'split', S, ': [3] (re)shaping data to fit model inputs [time: {}]'.format(time.time() - t_start)
    for c in xrange(nclasses): # do all classes separately
        IcurrentClass = Y[:,0] > 0
        for r in xrange(repetitions): # repeat for some times
            Xt = X[IcurrentClass,...]
            Xtshape = Xt.shape
            Xt = np.reshape(Xt, [-1,M]) # get all samples of that class as 'clean copy' in the beginning. reshape into 1-dim samples
            ORDERS = [np.random.permutation(M) for s in xrange(N)] #create random permutation arrays for all samples

            for t in xrange(len(CHANGE)): #change in percent of the available samples per datapoint
                #process iterative change here:
                #  move over N and change samples
                #  reshape back for forward pass.
                cstart = 0 if t == 0 else int(CHANGE[t-1]*M/100.)
                cend = int(CHANGE[t]*M/100.)
                for i in xrange(IcurrentClass.sum()):
                    Xt[i,ORDERS[i][cstart:cend]] += sigma * np.random.randn(cend-cstart)

                Yp = nn.forward(np.reshape(Xt, Xtshape))
                #Yp = nn.modules[-2].Y if len(nn.modules) > 1 else Yp # get presoftmax score, if we have a softmax layer
                YpredPerturbed[IcurrentClass,:,t,r] = Yp

                print acc(Y[IcurrentClass], Yp) , 'after', t , 'changes for class', c, (cstart, cend)
                #print acc(Y[IcurrentClass],Y[IcurrentClass])
                #print acc(Yp,Yp)
            exit()


    #return the average perturbed predictions over the repetitions
    print np.mean(YpredPerturbed, axis=3)
    return np.mean(YpredPerturbed, axis=3)





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




def run(workerparams):

    t_start = time.time()

    REPS = 10
    CHANGEPERCENT = range(0,50,1)
    print CHANGEPERCENT

    #unpack parameters
    S = workerparams['split_no'] # an int. the current split
    modelpath = workerparams['model'] # path of the model to load
    relevancepath = workerparams['relevances'] # path of the relevance mat file to load
    X = workerparams['Xtest'] # N x T x C test data
    Y = workerparams['Ytest'] # N x 2 or N x 57 test labels. binary

    print 'split', S, ': [1] loading model [time: {}]'.format(time.time() - t_start)
    nn = model_io.read(modelpath)

    print 'split', S, ': [2] loading precomputed model results [time: {}]'.format(time.time() - t_start)
    modeloutputs = scio.loadmat(relevancepath)

    print 'split', S, ': [3] (re)shaping data to fit model inputs [time: {}]'.format(time.time() - t_start)
    X,Y = reshape_data(X,Y,modelpath)

    print 'split', S, ': [4] random permutations on the data [time: {}]'.format(time.time() - t_start)
    result = random_perturbations(nn, X, Y, CHANGEPERCENT, REPS, sigma=1)


    #YP = random_perturbations(nn,X,Y,CHANGE,repetitions,0.1)


    #after here the actual work
    RVARIANTS = ['RpredAct', 'RpredActComp'] #relevance variants to compare
    # PLUS RANDOM PERTURBATIONS




    #change 10 percent max
    # in steps of 1 percent (rounded up))

    #type of exp:
        #ypred x 10
        #relevance x 10 / X
