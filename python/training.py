import sys
import time
import os
import numpy as np
import scipy
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import modules
from modules import Convolution, Linear
import model_io


def test_model(nn, Xtest, Ytest,  Nte, T, C):
    """
    receives a neural network model and some test data and runs some test and lrp diagnostics

    nn is a neural network model
    Xtest is the test data
    Nte = number test samples of original input shape
    T = number time points of original input shape
    C = number channels of original input shape

    returns:

    Ypred -- the model predictions
    Rpred -- the relevance maps for all predictions with eps lrp
    RPredPressoftmax -- the presoftmax lrp predictions with eps lrp
    Ract -- the lrp predictions for the actual sample class only
    RPredAct -- the presoftmax lrp for the actual class
    RPredDom -- the presoftmax lrp for the dominant class
    RPredActComp -- the presoftmax lrp for the actual class with the known optimal decomposition parametrization
    RPredDomComp -- the presoftmax lrp for the actual class with the known optimal decomposition parametrization
    """

    #presoftmaxindex. the linear model does not have a softmax output.
    iP = -1 if len(nn.modules) == 1 else -2
    print '  forward'
    Ypred = nn.forward(Xtest,lrp_aware=False)
    YpredPresoftmax = nn.modules[iP].Y
    amax = np.argmax(YpredPresoftmax,axis=1)

    Ydom = np.zeros_like(YpredPresoftmax)
    #loop over all samples (since I do not know any better solution right now) and set Ydom
    for i in xrange(Nte):
        Ydom[i,amax[i]] = YpredPresoftmax[i,amax[i]]



    print '  Rpred'
    Rpred =             nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
    print '  RpredPresoftmax'
    RpredPresoftmax =   nn.lrp(YpredPresoftmax, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
    print '  Ract'
    Ract =              nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

    print '  PRPredAct'
    RPredAct =          nn.lrp(Ytest * YpredPresoftmax, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
    print '  RPredDom'
    RPredDom =          nn.lrp(Ydom, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)


    #preconfigure lrp for all layers
    for i in xrange(len(nn.modules)):
        m = nn.modules[i]
        if m.__class__ == Convolution:
            m.set_lrp_parameters(lrp_var='alpha',  param=2.0)
            print 'setting lrp parameters to alpha=2 for module {} of type {}'.format(i,m.__class__)
        elif m.__class__ == Linear:
            m.set_lrp_parameters(lrp_var='epsilon',  param=1e-5)
            print 'setting lrp parameters to epsilon=1e-5 for module {} of type {}'.format(i,m.__class__)

    print '  RPredActComp'
    RPredActComp = nn.lrp(Ytest * YpredPresoftmax).reshape(Nte, T, C)
    print '  RPRedDomComp'
    RPredDomComp = nn.lrp(Ydom).reshape(Nte, T, C)


    return Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp


def run_cnn_C3_3(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip', SKIPTHISMANY=-1):
    """
    Trains a CNN model. The architecture of the model adapts to the dimensions of the data.
    This Type "C3" CNN uses classical convolution masks of size 3 in either directions and multiple layers
    Stride along both axes is 3.

    This method only trains for the full angle data sets

    X is a dictionary of DataName -> np.array , containing raw input data
    Y is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    #prepare model output
    MODELNAME = 'CNN-C3-3'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?

                if SKIPTHISMANY > 0:
                    print 'skipping {} due to request by parameter.\n\n'.format(modelfile)
                    SKIPTHISMANY-=1
                    continue

                if not xname in ['JA_Lower', 'JA_Full']:
                    print 'skipping', xname, 'data for this model'
                    continue # skip all non-relevant models

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #in order to realize full 3-convs with stride 3 in the time axis, we need to pad with one zero (because I am too lazy to implement support for padding)
                Xtrain = np.concatenate([Xtrain, np.zeros([Xtrain.shape[0],1,Xtrain.shape[2]],dtype=Xtrain.dtype)],axis=1)
                Xtest = np.concatenate([Xtest, np.zeros([Xtest.shape[0],1,Xtest.shape[2]],dtype=Xtest.dtype)],axis=1)
                Xval = np.concatenate([Xval, np.zeros([Xval.shape[0],1,Xval.shape[2]],dtype=Xval.dtype)],axis=1)

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #attach artificial channel axis.
                Xtrain = Xtrain[..., None]
                Xval = Xval[..., None]
                Xtest = Xtest[..., None]

                #number of target labels
                L = Ytrain.shape[1]

                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    print 'incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists)
                    exit()

                if modelExists and ifModelExists == 'skip':
                    print '{} exists. skipping.\n\n'.format(modelfile)
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    print '{} exists. loading model, re-evaluating. \n\n'.format(modelfile)
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here

                    if xname == 'JA_Full':
                        #samples are shaped 102 x 33 x 1

                        # I: 102 x 33 x 1
                        h1 = modules.Convolution(filtersize=(3,3,1,64), stride=(3,3))
                        # H1: 34 x 11 x 64
                        h2 = modules.Convolution(filtersize=(3,3,64,64), stride=(1,1))
                        # H2: 32 x 9 x 64
                        h3 = modules.Convolution(filtersize=(3,3,64,32), stride=(1,1))
                        # H3: 30 x 7 x 32 = 6720
                        h4 = modules.Linear(6720,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), h3, modules.Rect(), modules.Flatten(), h4, modules.SoftMax()])

                    elif xname == 'JA_Lower':
                        #samples are shaped 102 x 18 x 1

                        # I: 101 x 18 x 1
                        h1 = modules.Convolution(filtersize=(3,3,1,64), stride=(3,3))
                        # H1: 34 x 6 x 64
                        h2 = modules.Convolution(filtersize=(3,3,64,64), stride=(1,1))
                        # H2: 32 x 4 x 64
                        h3 = modules.Convolution(filtersize=(3,3,64,32), stride=(1,1))
                        # H3: 30 x 2 x 32 = 1920
                        h4 = modules.Linear(1920,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), h3, modules.Rect(), modules.Flatten(), h4, modules.SoftMax()])

                    else:
                        print 'No architecture defined for data named', xname
                        exit()


                    #print 'starting {} {}'.format(xname, yname)
                    #nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005, convergence=10,iters=10) # train the model
                    #print '    {} {} ok\n'.format(xname, yname)
                    #continue

                    print 'starting training for {}'.format(modeldir)
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005, convergence=10) # train the model
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.001, convergence=10) # slower training once the model has converged somewhat
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.0005, convergence=10)# one last epoch


                #test the model
                #Ypred = nn.forward(Xtest)
                #Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                #RpredPresoftmax = nn.lrp(nn.modules[-2].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                #Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                print message

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})

                #return -1 # we have done a training. this should suffice.
    return SKIPTHISMANY
    LOG.close()


def run_cnn_C6(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip', SKIPTHISMANY=-1):
    """
    Trains a CNN model. The architecture of the model adapts to the dimensions of the data.
    This Type "C6" CNN uses classical convolution masks of size 6 in either directions and multiple layers
    Stride along both axes is 1.

    X is a dictionary of DataName -> np.array , containing raw input data
    Y is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    #prepare model output
    MODELNAME = 'CNN-C6'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?

                if SKIPTHISMANY > 0:
                    print 'skipping {} due to request by parameter.\n\n'.format(modelfile)
                    SKIPTHISMANY-=1
                    continue

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #attach artificial channel axis.
                Xtrain = Xtrain[..., None]
                Xval = Xval[..., None]
                Xtest = Xtest[..., None]

                #number of target labels
                L = Ytrain.shape[1]

                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    print 'incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists)
                    exit()

                if modelExists and ifModelExists == 'skip':
                    print '{} exists. skipping.\n\n'.format(modelfile)
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    print '{} exists. loading model, re-evaluating. \n\n'.format(modelfile)
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here

                    if 'GRF_' in xname or xname == 'JA_X_Lower':
                        #samples are shaped 101 x 6 x 1

                        # I: 101 x 6 x 1
                        h1 = modules.Convolution(filtersize=(6,6,1,32), stride=(1,1))
                        # H3: 96 x 1 x 32 = 3072
                        h2 = modules.Linear(3072,L)
                        nn = modules.Sequential([h1, modules.Rect(), modules.Flatten(), h2, modules.SoftMax()])


                    elif xname == 'JA_Full':
                        #samples are shaped 101 x 33 x 1

                        # I: 101 x 33 x 1
                        h1 = modules.Convolution(filtersize=(6,6,1,32), stride=(1,1))
                        # H1: 96 x 28 x 32
                        h2 = modules.Convolution(filtersize=(6,6,32,32), stride=(1,1))
                        # H2: 91 x 23 x 32
                        h3 = modules.Convolution(filtersize=(6,6,32,16), stride=(1,1))
                        # H3: 86 x 18 x 32 = 24768
                        h4 = modules.Linear(24768,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), h3, modules.Rect(), modules.Flatten(), h4, modules.SoftMax()])

                    elif xname == 'JA_Lower':
                        #samples are shaped 101 x 18 x 1

                        # I: 101 x 18 x 1
                        h1 = modules.Convolution(filtersize=(6,6,1,32), stride=(1,1))
                        # H1: 96 x 13 x 32
                        h2 = modules.Convolution(filtersize=(6,6,32,32), stride=(1,1))
                        # H2: 91 x 8 x 32
                        h3 = modules.Convolution(filtersize=(6,6,32,16), stride=(1,1))
                        # H3: 86 x 3 x 16 = 4128
                        h4 = modules.Linear(4128,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), h3, modules.Rect(), modules.Flatten(), h4, modules.SoftMax()])

                    elif xname == 'JA_X_Full':
                        #samples are shaped 101 x 10 x 1

                        # I: 101 x 10 x 1
                        h1 = modules.Convolution(filtersize=(6,6,1,32), stride=(1,1))
                        # H1: 96 x 5 x 32
                        h2 = modules.Convolution(filtersize=(3,3,32,32), stride=(1,1))
                        # H2: 94 x 3 x 32
                        h3 = modules.Convolution(filtersize=(3,3,32,32), stride=(1,1))
                        # H3: 92 x 1 x 32 = 2944
                        h4 = modules.Linear(2944,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), h3, modules.Rect(), modules.Flatten(), h4, modules.SoftMax()])


                    else:
                        print 'No architecture defined for data named', xname
                        exit()

                    #DEBUG TESTS
                    #print 'starting {} {}'.format(xname, yname)
                    #nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005, convergence=10,iters=10) # train the model
                    #print '    {} {} ok\n'.format(xname, yname)
                    #continue

                    print 'starting training for {}'.format(modeldir)
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005, convergence=10) # train the model
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.001, convergence=10) # slower training once the model has converged somewhat
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.0005, convergence=10)# one last epoch


                #test the model
                #Ypred = nn.forward(Xtest)
                #Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                #RpredPresoftmax = nn.lrp(nn.modules[-2].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                #Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                print message

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})

                #return -1 # we have done a training. this should suffice.
    return SKIPTHISMANY
    LOG.close()

def run_cnn_C3(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip', SKIPTHISMANY=-1):
    """
    Trains a CNN model. The architecture of the model adapts to the dimensions of the data.
    This Type "C3" CNN uses classical convolution masks of size 3 in either directions and multiple layers
    Stride along both axes is 1.

    X is a dictionary of DataName -> np.array , containing raw input data
    Y is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    #prepare model output
    MODELNAME = 'CNN-C3'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?

                if SKIPTHISMANY > 0:
                    print 'skipping {} due to request by parameter.\n\n'.format(modelfile)
                    SKIPTHISMANY-=1
                    continue

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #attach artificial channel axis.
                Xtrain = Xtrain[..., None]
                Xval = Xval[..., None]
                Xtest = Xtest[..., None]

                #number of target labels
                L = Ytrain.shape[1]

                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    print 'incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists)
                    exit()

                if modelExists and ifModelExists == 'skip':
                    print '{} exists. skipping.\n\n'.format(modelfile)
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    print '{} exists. loading model, re-evaluating. \n\n'.format(modelfile)
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here

                    if 'GRF_' in xname or xname == 'JA_X_Lower':
                        #samples are shaped 101 x 6 x 1

                        # I: 101 x 6 x 1
                        h1 = modules.Convolution(filtersize=(3,3,1,32), stride=(1,1))
                        # H1: 99 x 4 x 32
                        h2 = modules.Convolution(filtersize=(3,3,32,32), stride=(1,1))
                        # H2: 97 x 2 x 32
                        h3 = modules.Convolution(filtersize=(2,2,32,32), stride=(1,1))
                        # H3: 96 x 1 x 32 = 3072
                        h4 = modules.Linear(3072,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), h3, modules.Rect(), modules.Flatten(), h4, modules.SoftMax()])


                    elif xname == 'JA_Full':
                        #samples are shaped 101 x 33 x 1

                        # I: 101 x 33 x 1
                        h1 = modules.Convolution(filtersize=(3,3,1,32), stride=(1,1))
                        # H1: 99 x 31 x 32
                        h2 = modules.Convolution(filtersize=(3,3,32,32), stride=(1,1))
                        # H2: 97 x 29 x 32
                        h3 = modules.Convolution(filtersize=(3,3,32,16), stride=(1,1))
                        # H3: 95 x 27 x 16 = 41040
                        h4 = modules.Linear(41040,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), h3, modules.Rect(), modules.Flatten(), h4, modules.SoftMax()])

                    elif xname == 'JA_Lower':
                        #samples are shaped 101 x 18 x 1

                        # I: 101 x 18 x 1
                        h1 = modules.Convolution(filtersize=(3,3,1,32), stride=(1,1))
                        # H1: 99 x 16 x 32
                        h2 = modules.Convolution(filtersize=(3,3,32,32), stride=(1,1))
                        # H2: 97 x 14 x 32
                        h3 = modules.Convolution(filtersize=(3,3,32,16), stride=(1,1))
                        # H3: 95 x 12 x 16 = 18240
                        h4 = modules.Linear(18240,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), h3, modules.Rect(), modules.Flatten(), h4, modules.SoftMax()])

                    elif xname == 'JA_X_Full':
                        #samples are shaped 101 x 10 x 1

                        # I: 101 x 10 x 1
                        h1 = modules.Convolution(filtersize=(3,3,1,32), stride=(1,1))
                        # H1: 99 x 8 x 32
                        h2 = modules.Convolution(filtersize=(3,3,32,32), stride=(1,1))
                        # H2: 97 x 6 x 32
                        h3 = modules.Convolution(filtersize=(3,3,32,32), stride=(1,1))
                        # H3: 95 x 4 x 32 = 12160
                        h4 = modules.Linear(12160,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), h3, modules.Rect(), modules.Flatten(), h4, modules.SoftMax()])


                    else:
                        print 'No architecture defined for data named', xname
                        exit()


                    #print 'starting {} {}'.format(xname, yname)
                    #nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005, convergence=10,iters=10) # train the model
                    #print '    {} {} ok\n'.format(xname, yname)
                    #continue

                    print 'starting training for {}'.format(modeldir)
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005, convergence=10) # train the model
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.001, convergence=10) # slower training once the model has converged somewhat
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.0005, convergence=10)# one last epoch


                #test the model
                #Ypred = nn.forward(Xtest)
                #Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                #RpredPresoftmax = nn.lrp(nn.modules[-2].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                #Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                print message

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})

                #return -1 # we have done a training. this should suffice.
    return SKIPTHISMANY
    LOG.close()

def run_cnn_A(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip', SKIPTHISMANY=-1):
    """
    Trains a CNN model. The architecture of the model adapts to the dimensions of the data.
    This Type "A" CNN sees all variables at once and slides its filters across the time axis.
    Stride along the time axis is 1.

    T is an int, defining the amount of time steps seen at a time.
    X is a dictionary of DataName -> np.array , containing raw input data
    Y is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    #prepare model output
    MODELNAME = 'CNN-A'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)


    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?

                if SKIPTHISMANY > 0:
                    print 'skipping {} due to request by parameter.\n\n'.format(modelfile)
                    SKIPTHISMANY-=1
                    continue

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #attach artificial channel axis.
                Xtrain = Xtrain[..., None]
                Xval = Xval[..., None]
                Xtest = Xtest[..., None]

                #number of target labels
                L = Ytrain.shape[1]

                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    print 'incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists)
                    exit()

                if modelExists and ifModelExists == 'skip':
                    print '{} exists. skipping.\n\n'.format(modelfile)
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    print '{} exists. loading model, re-evaluating. \n\n'.format(modelfile)
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here

                    if 'GRF_' in xname or xname == 'JA_X_Lower':
                        #samples are shaped 101 x 6 x 1

                        # I: 101 x 6 x 1
                        h1 = modules.Convolution(filtersize=(6,6,1,32), stride=(1,1))
                        # H1: 96 x 1 x 32 = 3072
                        h2 = modules.Linear(3072,L)
                        nn = modules.Sequential([h1, modules.Rect(), modules.Flatten(), h2, modules.SoftMax()])


                    elif xname == 'JA_Full':
                        #samples are shaped 101 x 33 x 1
                        # I: 101 x 33 x 1
                        h1 = modules.Convolution(filtersize=(33,33,1,64), stride=(1,1))
                        # H1: 69 x 1 x 64 = 4416
                        h2 = modules.Linear(4416, L)
                        nn = modules.Sequential([h1, modules.Rect(), modules.Flatten(), h2, modules.SoftMax()])

                    elif xname == 'JA_Lower':
                        #samples are shaped 101 x 18 x 1
                        # I: 101 x 18 x 1
                        h1 = modules.Convolution(filtersize=(18,18,1,64), stride=(1,1))
                        # H1: 84 x 1 x 64 = 5376
                        h2 = modules.Linear(5376, L)
                        nn = modules.Sequential([h1, modules.Rect(), modules.Flatten(), h2, modules.SoftMax()])

                    elif xname == 'JA_X_Full':
                        #samples are shaped 101 x 10 x 1
                        # I: 101 x 10 x 1
                        h1 = modules.Convolution(filtersize=(10,10,1,32), stride=(1,1))
                        # H1: 92 x 1 x 32 = 2944
                        h2 = modules.Linear(2944, L)
                        nn = modules.Sequential([h1, modules.Rect(), modules.Flatten(), h2, modules.SoftMax()])


                    else:
                        print 'No architecture defined for data named', xname
                        exit()

                    print 'starting training for {}'.format(modeldir)
                    #STDOUT.write('starting {} {}'.format(xname, yname))
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005, convergence=10) # train the model
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.001, convergence=10) # slower training once the model has converged somewhat
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.0005, convergence=10)# one last epoch
                    #STDOUT.write('    {} {} ok\n'.format(xname, yname))

                #test the model
                #Ypred = nn.forward(Xtest)
                #Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                #RpredPresoftmax = nn.lrp(nn.modules[-2].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                #Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                print message

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})

                #return -1 # we have done a training. this should suffice.
    return SKIPTHISMANY
    LOG.close()


def run_cnn_A6(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip', SKIPTHISMANY=-1):
    """
    Trains a CNN model. The architecture of the model adapts to the dimensions of the data.
    This Type "A" CNN sees all variables at once and slides its filters across the time axis.
    Stride along the time axis is 1.

    T is an int, defining the amount of time steps seen at a time.
    X is a dictionary of DataName -> np.array , containing raw input data
    Y is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    #prepare model output
    MODELNAME = 'CNN-A6'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)


    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?

                if SKIPTHISMANY > 0:
                    print 'skipping {} due to request by parameter.\n\n'.format(modelfile)
                    SKIPTHISMANY-=1
                    continue

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #attach artificial channel axis.
                Xtrain = Xtrain[..., None]
                Xval = Xval[..., None]
                Xtest = Xtest[..., None]

                #number of target labels
                L = Ytrain.shape[1]

                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    print 'incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists)
                    exit()

                if modelExists and ifModelExists == 'skip':
                    print '{} exists. skipping.\n\n'.format(modelfile)
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    print '{} exists. loading model, re-evaluating. \n\n'.format(modelfile)
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here

                    if 'GRF_' in xname or xname == 'JA_X_Lower':
                        #samples are shaped 101 x 6 x 1

                        # I: 101 x 6 x 1
                        h1 = modules.Convolution(filtersize=(6,6,1,32), stride=(1,1))
                        # H1: 96 x 1 x 32
                        h2 = modules.Convolution(filtersize=(6,1,32,32), stride=(1,1))
                        # H2: 91 x 1 x 32 = 2912
                        h3 = modules.Linear(2912,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), modules.Flatten(), h3, modules.SoftMax()])


                    elif xname == 'JA_Full':
                        #samples are shaped 101 x 33 x 1
                        # I: 101 x 33 x 1
                        h1 = modules.Convolution(filtersize=(6,33,1,64), stride=(1,1))
                        # H1: 96 x 1 x 64
                        h2 = modules.Convolution(filtersize=(6,1,64,32), stride=(1,1))
                        # H2: 91 x 1 x 32 = 2912
                        h3 = modules.Linear(2912, L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), modules.Flatten(), h3, modules.SoftMax()])

                    elif xname == 'JA_Lower':
                        #samples are shaped 101 x 18 x 1
                        # I: 101 x 18 x 1
                        h1 = modules.Convolution(filtersize=(6,18,1,64), stride=(1,1))
                        # H1: 96 x 1 x 64
                        h2 = modules.Convolution(filtersize=(6,1,64,32), stride=(1,1))
                        # H2: 91 x 1 x 32 = 2912
                        h3 = modules.Linear(2912, L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), modules.Flatten(), h3, modules.SoftMax()])

                    elif xname == 'JA_X_Full':
                        #samples are shaped 101 x 10 x 1
                        # I: 101 x 10 x 1
                        h1 = modules.Convolution(filtersize=(6,10,1,32), stride=(1,1))
                        # H1: 96 x 1 x 32
                        h2 = modules.Convolution(filtersize=(6,1,32,32), stride=(1,1))
                        # H2: 91 x 1 x 32 = 2912
                        h3 = modules.Linear(2912, L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), modules.Flatten(), h3, modules.SoftMax()])


                    else:
                        print 'No architecture defined for data named', xname
                        exit()


                    #print 'starting {} {}'.format(xname, yname)
                    #nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005, convergence=10,iters=10) # train the model
                    #print '    {} {} ok\n'.format(xname, yname)
                    #continue


                    #STDOUT.write('starting {} {}'.format(xname, yname))
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005, convergence=10) # train the model
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.001, convergence=10) # slower training once the model has converged somewhat
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.0005, convergence=10)# one last epoch
                    #STDOUT.write('    {} {} ok\n'.format(xname, yname))

                #test the model
                #Ypred = nn.forward(Xtest)
                #Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                #RpredPresoftmax = nn.lrp(nn.modules[-2].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                #Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                print message

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})

                #return -1 # we have done a training. this should suffice.
    return SKIPTHISMANY
    LOG.close()


def run_cnn_A3(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip', SKIPTHISMANY=-1):
    """
    Trains a CNN model. The architecture of the model adapts to the dimensions of the data.
    This Type "A" CNN sees all variables at once and slides its filters across the time axis.
    Stride along the time axis is 1.

    T is an int, defining the amount of time steps seen at a time.
    X is a dictionary of DataName -> np.array , containing raw input data
    Y is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    #prepare model output
    MODELNAME = 'CNN-A3'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)


    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?

                if SKIPTHISMANY > 0:
                    print 'skipping {} due to request by parameter.\n\n'.format(modelfile)
                    SKIPTHISMANY-=1
                    continue

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #attach artificial channel axis.
                Xtrain = Xtrain[..., None]
                Xval = Xval[..., None]
                Xtest = Xtest[..., None]

                #number of target labels
                L = Ytrain.shape[1]

                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    print 'incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists)
                    exit()

                if modelExists and ifModelExists == 'skip':
                    print '{} exists. skipping.\n\n'.format(modelfile)
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    print '{} exists. loading model, re-evaluating. \n\n'.format(modelfile)
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here

                    if 'GRF_' in xname or xname == 'JA_X_Lower':
                        #samples are shaped 101 x 6 x 1

                        # I: 101 x 6 x 1
                        h1 = modules.Convolution(filtersize=(3,6,1,32), stride=(1,1))
                        # H1: 99 x 1 x 32
                        h2 = modules.Convolution(filtersize=(3,1,32,32), stride=(1,1))
                        # H2: 97 x 1 x 32 = 3104
                        h3 = modules.Linear(3104,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), modules.Flatten(), h3, modules.SoftMax()])


                    elif xname == 'JA_Full':
                        #samples are shaped 101 x 33 x 1
                        # I: 101 x 33 x 1
                        h1 = modules.Convolution(filtersize=(3,33,1,64), stride=(1,1))
                        # H1: 99 x 1 x 64
                        h2 = modules.Convolution(filtersize=(3,1,64,32), stride=(1,1))
                        # H2: 97 x 1 x 32 = 3104
                        h3 = modules.Linear(3104,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), modules.Flatten(), h3, modules.SoftMax()])

                    elif xname == 'JA_Lower':
                        #samples are shaped 101 x 18 x 1
                        # I: 101 x 18 x 1
                        h1 = modules.Convolution(filtersize=(3,18,1,64), stride=(1,1))
                        # H1: 99 x 1 x 64
                        h2 = modules.Convolution(filtersize=(3,1,64,32), stride=(1,1))
                        # H2: 97 x 1 x 32 = 3104
                        h3 = modules.Linear(3104,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), modules.Flatten(), h3, modules.SoftMax()])

                    elif xname == 'JA_X_Full':
                        #samples are shaped 101 x 10 x 1
                        # I: 101 x 10 x 1
                        h1 = modules.Convolution(filtersize=(3,10,1,32), stride=(1,1))
                        # H1: 99 x 1 x 32
                        h2 = modules.Convolution(filtersize=(3,1,32,32), stride=(1,1))
                        # H2: 97 x 1 x 32 = 3104
                        h3 = modules.Linear(3104,L)
                        nn = modules.Sequential([h1, modules.Rect(), h2, modules.Rect(), modules.Flatten(), h3, modules.SoftMax()])


                    else:
                        print 'No architecture defined for data named', xname
                        exit()


                    #print 'starting {} {}'.format(xname, yname)
                    #nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005, convergence=10,iters=10) # train the model
                    #print '    {} {} ok\n'.format(xname, yname)
                    #continue


                    #STDOUT.write('starting {} {}'.format(xname, yname))
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005, convergence=10) # train the model
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.001, convergence=10) # slower training once the model has converged somewhat
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.0005, convergence=10)# one last epoch
                    #STDOUT.write('    {} {} ok\n'.format(xname, yname))

                #test the model
                #Ypred = nn.forward(Xtest)
                #Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                #RpredPresoftmax = nn.lrp(nn.modules[-2].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                #Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                print message

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})

                #return -1 # we have done a training. this should suffice.
    return SKIPTHISMANY
    LOG.close()

def run_pca(X,Y,S):

    """
    Runs PCA for given data.
    X is a dictionary of DataName -> np.array , containing raw input data
    Y is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    if not os.path.isdir('./PCA'):
        os.mkdir('./PCA')

    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            plt.clf()
            for i in xrange(len(targetSplits)): #the splits for this target
                t_start = time.time()
                #set output log to capture all prints
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])

                #input dims and output dims
                D = Xtrain.shape[1]

                pca = PCA(n_components=None)
                pca.fit(Xtrain)

                t_end = time.time()
                print '{}-{} set {} done after: {}s '.format(xname, yname, i, t_end-t_start)
                plt.plot(pca.singular_values_, alpha=0.5)

            plt.xlabel('components')
            plt.ylabel('singular values')
            figpath = './PCA/pca-{}-{}.pdf'.format(xname, yname)
            print 'saving figure', figpath
            plt.savefig(figpath)


def run_linear(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip'):
    """
    Trains a linear model.
    X is a dictionary of DataName -> np.array , containing raw input data
    X is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    #prepare model output
    MODELNAME = 'Linear'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    STDOUT = sys.stdout
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints
                sys.stdout = open('{}/log.txt'.format(modeldir), 'wb')

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])
                Xval = np.reshape(Xval, [Nv, -1])
                Xtest = np.reshape(Xtest, [Nte, -1])

                #input dims and output dims
                D = Xtrain.shape[1]
                L = Ytrain.shape[1]


                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    STDOUT.write('incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists))
                    exit()

                if modelExists and ifModelExists == 'skip':
                    STDOUT.write('{} exists. skipping.\n\n'.format(modelfile))
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    STDOUT.write('{} exists. loading model, re-evaluating. \n\n'.format(modelfile))
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here
                    nn = modules.Sequential([modules.Linear(D, L)])
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005) # train the model
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.001) # slower training once the model has converged somewhat
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.0005)# one last epoch

                #test the model
                #Ypred = nn.forward(Xtest)
                #Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                #RpredPresoftmax = nn.lrp(nn.modules[-1].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                #Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                STDOUT.write(message)

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()


def run_linear_SVM_L2_C1_SquareHinge(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip'):
    """
    Trains a linear model.
    X is a dictionary of DataName -> np.array , containing raw input data
    X is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    import sklearn
    #prepare model output
    MODELNAME = 'LinearSVM-L2C1SquareHinge'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    STDOUT = sys.stdout
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?
                #print modelfile, modelExists, yname, i

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints
                sys.stdout = open('{}/log.txt'.format(modeldir), 'wb')

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])
                Xval = np.reshape(Xval, [Nv, -1])
                Xtest = np.reshape(Xtest, [Nte, -1])

                #encode labels as required by sklearn
                YtrainSVM = np.argmax(Ytrain, axis=1)
                YtestSVM = np.argmax(Ytest, axis=1)
                YvalSVM = np.argmax(Yval, axis=1)

                #input dims and output dims
                D = Xtrain.shape[1]
                L = Ytrain.shape[1]


                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    STDOUT.write('incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists))
                    exit()

                if modelExists and ifModelExists == 'skip':
                    STDOUT.write('{} exists. skipping.\n\n'.format(modelfile))
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    STDOUT.write('{} exists. loading model, re-evaluating. \n\n'.format(modelfile))
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here
                    STDOUT.write('    training SVM model\n')
                    model = sklearn.svm.LinearSVC() #use default options: square hinge loss, l2 penalty on the weights, C = 1.0
                    model.fit(Xtrain, YtrainSVM)

                    STDOUT.write('    converting SVM model to Toolbox NN model\n')
                    #convert SKLearn-Models to  Toolbox NN models
                    if np.unique(YtrainSVM).size == 2:
                        #make a multi-output model
                        L = Linear(D, L)
                        L.W = np.concatenate([-model.coef_.T, model.coef_.T], axis=1)
                        L.B = np.concatenate([-model.intercept_, model.intercept_], axis=0)
                        nn = modules.Sequential([L])
                    else:
                        #just copy the learned parameters
                        L = Linear(D, L)
                        L.W = model.coef_.T
                        L.B = model.intercept_
                        nn = modules.Sequential([L])

                    STDOUT.write('    sanity checking model conversion\n')
                    #sanity check model conversion.
                    YpredSVM = model.decision_function(Xtest)
                    YpredNN = nn.forward(Xtest)

                    rtol=1e-7
                    if np.unique(YtrainSVM).size == 2:
                        np.testing.assert_allclose(YpredSVM, -YpredNN[:,0], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        np.testing.assert_allclose(YpredSVM, YpredNN[:,1], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        STDOUT.write('    sanity check passed (2-Class).\n')
                    else:
                        np.testing.assert_allclose(YpredSVM, YpredNN, rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal!')
                        STDOUT.write('    sanity check passed (Multiclass).\n')

                #test the model
                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                STDOUT.write(message)

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()

def run_linear_SVM_L2_C0p1_SquareHinge(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip'):
    """
    Trains a linear model.
    X is a dictionary of DataName -> np.array , containing raw input data
    X is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    import sklearn
    #prepare model output
    MODELNAME = 'LinearSVM-L2C0p1SquareHinge'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    STDOUT = sys.stdout
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?
                #print modelfile, modelExists, yname, i

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints
                sys.stdout = open('{}/log.txt'.format(modeldir), 'wb')

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])
                Xval = np.reshape(Xval, [Nv, -1])
                Xtest = np.reshape(Xtest, [Nte, -1])

                #encode labels as required by sklearn
                YtrainSVM = np.argmax(Ytrain, axis=1)
                YtestSVM = np.argmax(Ytest, axis=1)
                YvalSVM = np.argmax(Yval, axis=1)

                #input dims and output dims
                D = Xtrain.shape[1]
                L = Ytrain.shape[1]


                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    STDOUT.write('incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists))
                    exit()

                if modelExists and ifModelExists == 'skip':
                    STDOUT.write('{} exists. skipping.\n\n'.format(modelfile))
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    STDOUT.write('{} exists. loading model, re-evaluating. \n\n'.format(modelfile))
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here
                    STDOUT.write('    training SVM model\n')
                    model = sklearn.svm.LinearSVC(C=0.1) #use default options: square hinge loss, l2 penalty on the weights, C = 1.0
                    model.fit(Xtrain, YtrainSVM)

                    STDOUT.write('    converting SVM model to Toolbox NN model\n')
                    #convert SKLearn-Models to  Toolbox NN models
                    if np.unique(YtrainSVM).size == 2:
                        #make a multi-output model
                        L = Linear(D, L)
                        L.W = np.concatenate([-model.coef_.T, model.coef_.T], axis=1)
                        L.B = np.concatenate([-model.intercept_, model.intercept_], axis=0)
                        nn = modules.Sequential([L])
                    else:
                        #just copy the learned parameters
                        L = Linear(D, L)
                        L.W = model.coef_.T
                        L.B = model.intercept_
                        nn = modules.Sequential([L])

                    STDOUT.write('    sanity checking model conversion\n')
                    #sanity check model conversion.
                    YpredSVM = model.decision_function(Xtest)
                    YpredNN = nn.forward(Xtest)

                    rtol=1e-7
                    if np.unique(YtrainSVM).size == 2:
                        np.testing.assert_allclose(YpredSVM, -YpredNN[:,0], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        np.testing.assert_allclose(YpredSVM, YpredNN[:,1], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        STDOUT.write('    sanity check passed (2-Class).\n')
                    else:
                        np.testing.assert_allclose(YpredSVM, YpredNN, rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal!')
                        STDOUT.write('    sanity check passed (Multiclass).\n')

                #test the model
                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                STDOUT.write(message)

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()

def run_linear_SVM_L2_C10_SquareHinge(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip'):
    """
    Trains a linear model.
    X is a dictionary of DataName -> np.array , containing raw input data
    X is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    import sklearn
    #prepare model output
    MODELNAME = 'LinearSVM-L2C10SquareHinge'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    STDOUT = sys.stdout
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?
                #print modelfile, modelExists, yname, i

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints
                sys.stdout = open('{}/log.txt'.format(modeldir), 'wb')

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])
                Xval = np.reshape(Xval, [Nv, -1])
                Xtest = np.reshape(Xtest, [Nte, -1])

                #encode labels as required by sklearn
                YtrainSVM = np.argmax(Ytrain, axis=1)
                YtestSVM = np.argmax(Ytest, axis=1)
                YvalSVM = np.argmax(Yval, axis=1)

                #input dims and output dims
                D = Xtrain.shape[1]
                L = Ytrain.shape[1]


                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    STDOUT.write('incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists))
                    exit()

                if modelExists and ifModelExists == 'skip':
                    STDOUT.write('{} exists. skipping.\n\n'.format(modelfile))
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    STDOUT.write('{} exists. loading model, re-evaluating. \n\n'.format(modelfile))
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here
                    STDOUT.write('    training SVM model\n')
                    model = sklearn.svm.LinearSVC(C=10) #use default options: square hinge loss, l2 penalty on the weights, C = 1.0
                    model.fit(Xtrain, YtrainSVM)

                    STDOUT.write('    converting SVM model to Toolbox NN model\n')
                    #convert SKLearn-Models to  Toolbox NN models
                    if np.unique(YtrainSVM).size == 2:
                        #make a multi-output model
                        L = Linear(D, L)
                        L.W = np.concatenate([-model.coef_.T, model.coef_.T], axis=1)
                        L.B = np.concatenate([-model.intercept_, model.intercept_], axis=0)
                        nn = modules.Sequential([L])
                    else:
                        #just copy the learned parameters
                        L = Linear(D, L)
                        L.W = model.coef_.T
                        L.B = model.intercept_
                        nn = modules.Sequential([L])

                    STDOUT.write('    sanity checking model conversion\n')
                    #sanity check model conversion.
                    YpredSVM = model.decision_function(Xtest)
                    YpredNN = nn.forward(Xtest)

                    rtol=1e-7
                    if np.unique(YtrainSVM).size == 2:
                        np.testing.assert_allclose(YpredSVM, -YpredNN[:,0], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        np.testing.assert_allclose(YpredSVM, YpredNN[:,1], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        STDOUT.write('    sanity check passed (2-Class).\n')
                    else:
                        np.testing.assert_allclose(YpredSVM, YpredNN, rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal!')
                        STDOUT.write('    sanity check passed (Multiclass).\n')

                #test the model
                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                STDOUT.write(message)

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()


#### LINEAR SVMS PLUS NOISE of 0.5 randn

def run_linear_SVM_L2_C1_SquareHinge_plus_0p5randn(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip'):
    """
    Trains a linear model.
    X is a dictionary of DataName -> np.array , containing raw input data
    X is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    import sklearn
    #prepare model output
    MODELNAME = 'LinearSVM-L2C1SquareHinge-0p5randn'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    STDOUT = sys.stdout
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?
                #print modelfile, modelExists, yname, i

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints
                sys.stdout = open('{}/log.txt'.format(modeldir), 'wb')

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])
                Xval = np.reshape(Xval, [Nv, -1])
                Xtest = np.reshape(Xtest, [Nte, -1])

                #add some random noise to the training data
                Xtrain += 0.5 + np.random.randn(Xtrain.shape[0], Xtrain.shape[1])

                #encode labels as required by sklearn
                YtrainSVM = np.argmax(Ytrain, axis=1)
                YtestSVM = np.argmax(Ytest, axis=1)
                YvalSVM = np.argmax(Yval, axis=1)

                #input dims and output dims
                D = Xtrain.shape[1]
                L = Ytrain.shape[1]


                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    STDOUT.write('incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists))
                    exit()

                if modelExists and ifModelExists == 'skip':
                    STDOUT.write('{} exists. skipping.\n\n'.format(modelfile))
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    STDOUT.write('{} exists. loading model, re-evaluating. \n\n'.format(modelfile))
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here
                    STDOUT.write('    training SVM model\n')
                    model = sklearn.svm.LinearSVC() #use default options: square hinge loss, l2 penalty on the weights, C = 1.0
                    model.fit(Xtrain, YtrainSVM)

                    STDOUT.write('    converting SVM model to Toolbox NN model\n')
                    #convert SKLearn-Models to  Toolbox NN models
                    if np.unique(YtrainSVM).size == 2:
                        #make a multi-output model
                        L = Linear(D, L)
                        L.W = np.concatenate([-model.coef_.T, model.coef_.T], axis=1)
                        L.B = np.concatenate([-model.intercept_, model.intercept_], axis=0)
                        nn = modules.Sequential([L])
                    else:
                        #just copy the learned parameters
                        L = Linear(D, L)
                        L.W = model.coef_.T
                        L.B = model.intercept_
                        nn = modules.Sequential([L])

                    STDOUT.write('    sanity checking model conversion\n')
                    #sanity check model conversion.
                    YpredSVM = model.decision_function(Xtest)
                    YpredNN = nn.forward(Xtest)

                    rtol=1e-7
                    if np.unique(YtrainSVM).size == 2:
                        np.testing.assert_allclose(YpredSVM, -YpredNN[:,0], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        np.testing.assert_allclose(YpredSVM, YpredNN[:,1], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        STDOUT.write('    sanity check passed (2-Class).\n')
                    else:
                        np.testing.assert_allclose(YpredSVM, YpredNN, rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal!')
                        STDOUT.write('    sanity check passed (Multiclass).\n')

                #test the model
                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                STDOUT.write(message)

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()

def run_linear_SVM_L2_C0p1_SquareHinge_plus_0p5randn(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip'):
    """
    Trains a linear model.
    X is a dictionary of DataName -> np.array , containing raw input data
    X is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    import sklearn
    #prepare model output
    MODELNAME = 'LinearSVM-L2C0p1SquareHinge-0p5randn'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    STDOUT = sys.stdout
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?
                #print modelfile, modelExists, yname, i

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints
                sys.stdout = open('{}/log.txt'.format(modeldir), 'wb')

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])
                Xval = np.reshape(Xval, [Nv, -1])
                Xtest = np.reshape(Xtest, [Nte, -1])

                #add some random noise to the training data
                Xtrain += 0.5 + np.random.randn(Xtrain.shape[0], Xtrain.shape[1])

                #encode labels as required by sklearn
                YtrainSVM = np.argmax(Ytrain, axis=1)
                YtestSVM = np.argmax(Ytest, axis=1)
                YvalSVM = np.argmax(Yval, axis=1)

                #input dims and output dims
                D = Xtrain.shape[1]
                L = Ytrain.shape[1]


                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    STDOUT.write('incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists))
                    exit()

                if modelExists and ifModelExists == 'skip':
                    STDOUT.write('{} exists. skipping.\n\n'.format(modelfile))
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    STDOUT.write('{} exists. loading model, re-evaluating. \n\n'.format(modelfile))
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here
                    STDOUT.write('    training SVM model\n')
                    model = sklearn.svm.LinearSVC(C=0.1) #use default options: square hinge loss, l2 penalty on the weights, C = 1.0
                    model.fit(Xtrain, YtrainSVM)

                    STDOUT.write('    converting SVM model to Toolbox NN model\n')
                    #convert SKLearn-Models to  Toolbox NN models
                    if np.unique(YtrainSVM).size == 2:
                        #make a multi-output model
                        L = Linear(D, L)
                        L.W = np.concatenate([-model.coef_.T, model.coef_.T], axis=1)
                        L.B = np.concatenate([-model.intercept_, model.intercept_], axis=0)
                        nn = modules.Sequential([L])
                    else:
                        #just copy the learned parameters
                        L = Linear(D, L)
                        L.W = model.coef_.T
                        L.B = model.intercept_
                        nn = modules.Sequential([L])

                    STDOUT.write('    sanity checking model conversion\n')
                    #sanity check model conversion.
                    YpredSVM = model.decision_function(Xtest)
                    YpredNN = nn.forward(Xtest)

                    rtol=1e-7
                    if np.unique(YtrainSVM).size == 2:
                        np.testing.assert_allclose(YpredSVM, -YpredNN[:,0], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        np.testing.assert_allclose(YpredSVM, YpredNN[:,1], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        STDOUT.write('    sanity check passed (2-Class).\n')
                    else:
                        np.testing.assert_allclose(YpredSVM, YpredNN, rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal!')
                        STDOUT.write('    sanity check passed (Multiclass).\n')

                #test the model
                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                STDOUT.write(message)

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()

def run_linear_SVM_L2_C10_SquareHinge_plus_0p5randn(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip'):
    """
    Trains a linear model.
    X is a dictionary of DataName -> np.array , containing raw input data
    X is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    import sklearn
    #prepare model output
    MODELNAME = 'LinearSVM-L2C10SquareHinge-0p5randn'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    STDOUT = sys.stdout
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?
                #print modelfile, modelExists, yname, i

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints
                sys.stdout = open('{}/log.txt'.format(modeldir), 'wb')

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])
                Xval = np.reshape(Xval, [Nv, -1])
                Xtest = np.reshape(Xtest, [Nte, -1])

                #add some random noise to the training data
                Xtrain += 0.5 + np.random.randn(Xtrain.shape[0], Xtrain.shape[1])

                #encode labels as required by sklearn
                YtrainSVM = np.argmax(Ytrain, axis=1)
                YtestSVM = np.argmax(Ytest, axis=1)
                YvalSVM = np.argmax(Yval, axis=1)

                #input dims and output dims
                D = Xtrain.shape[1]
                L = Ytrain.shape[1]


                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    STDOUT.write('incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists))
                    exit()

                if modelExists and ifModelExists == 'skip':
                    STDOUT.write('{} exists. skipping.\n\n'.format(modelfile))
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    STDOUT.write('{} exists. loading model, re-evaluating. \n\n'.format(modelfile))
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here
                    STDOUT.write('    training SVM model\n')
                    model = sklearn.svm.LinearSVC(C=10) #use default options: square hinge loss, l2 penalty on the weights, C = 1.0
                    model.fit(Xtrain, YtrainSVM)

                    STDOUT.write('    converting SVM model to Toolbox NN model\n')
                    #convert SKLearn-Models to  Toolbox NN models
                    if np.unique(YtrainSVM).size == 2:
                        #make a multi-output model
                        L = Linear(D, L)
                        L.W = np.concatenate([-model.coef_.T, model.coef_.T], axis=1)
                        L.B = np.concatenate([-model.intercept_, model.intercept_], axis=0)
                        nn = modules.Sequential([L])
                    else:
                        #just copy the learned parameters
                        L = Linear(D, L)
                        L.W = model.coef_.T
                        L.B = model.intercept_
                        nn = modules.Sequential([L])

                    STDOUT.write('    sanity checking model conversion\n')
                    #sanity check model conversion.
                    YpredSVM = model.decision_function(Xtest)
                    YpredNN = nn.forward(Xtest)

                    rtol=1e-7
                    if np.unique(YtrainSVM).size == 2:
                        np.testing.assert_allclose(YpredSVM, -YpredNN[:,0], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        np.testing.assert_allclose(YpredSVM, YpredNN[:,1], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        STDOUT.write('    sanity check passed (2-Class).\n')
                    else:
                        np.testing.assert_allclose(YpredSVM, YpredNN, rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal!')
                        STDOUT.write('    sanity check passed (Multiclass).\n')

                #test the model
                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                STDOUT.write(message)

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()


#### LINEAR SVMS PLUS NOISE of 1 randn

def run_linear_SVM_L2_C1_SquareHinge_plus_1randn(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip'):
    """
    Trains a linear model.
    X is a dictionary of DataName -> np.array , containing raw input data
    X is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    import sklearn
    #prepare model output
    MODELNAME = 'LinearSVM-L2C1SquareHinge-1p0randn'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    STDOUT = sys.stdout
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?
                #print modelfile, modelExists, yname, i

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints
                sys.stdout = open('{}/log.txt'.format(modeldir), 'wb')

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])
                Xval = np.reshape(Xval, [Nv, -1])
                Xtest = np.reshape(Xtest, [Nte, -1])

                #add some random noise to the training data
                Xtrain += 0.5 + np.random.randn(Xtrain.shape[0], Xtrain.shape[1])

                #encode labels as required by sklearn
                YtrainSVM = np.argmax(Ytrain, axis=1)
                YtestSVM = np.argmax(Ytest, axis=1)
                YvalSVM = np.argmax(Yval, axis=1)

                #input dims and output dims
                D = Xtrain.shape[1]
                L = Ytrain.shape[1]


                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    STDOUT.write('incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists))
                    exit()

                if modelExists and ifModelExists == 'skip':
                    STDOUT.write('{} exists. skipping.\n\n'.format(modelfile))
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    STDOUT.write('{} exists. loading model, re-evaluating. \n\n'.format(modelfile))
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here
                    STDOUT.write('    training SVM model\n')
                    model = sklearn.svm.LinearSVC() #use default options: square hinge loss, l2 penalty on the weights, C = 1.0
                    model.fit(Xtrain, YtrainSVM)

                    STDOUT.write('    converting SVM model to Toolbox NN model\n')
                    #convert SKLearn-Models to  Toolbox NN models
                    if np.unique(YtrainSVM).size == 2:
                        #make a multi-output model
                        L = Linear(D, L)
                        L.W = np.concatenate([-model.coef_.T, model.coef_.T], axis=1)
                        L.B = np.concatenate([-model.intercept_, model.intercept_], axis=0)
                        nn = modules.Sequential([L])
                    else:
                        #just copy the learned parameters
                        L = Linear(D, L)
                        L.W = model.coef_.T
                        L.B = model.intercept_
                        nn = modules.Sequential([L])

                    STDOUT.write('    sanity checking model conversion\n')
                    #sanity check model conversion.
                    YpredSVM = model.decision_function(Xtest)
                    YpredNN = nn.forward(Xtest)

                    rtol=1e-7
                    if np.unique(YtrainSVM).size == 2:
                        np.testing.assert_allclose(YpredSVM, -YpredNN[:,0], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        np.testing.assert_allclose(YpredSVM, YpredNN[:,1], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        STDOUT.write('    sanity check passed (2-Class).\n')
                    else:
                        np.testing.assert_allclose(YpredSVM, YpredNN, rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal!')
                        STDOUT.write('    sanity check passed (Multiclass).\n')

                #test the model
                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                STDOUT.write(message)

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()

def run_linear_SVM_L2_C0p1_SquareHinge_plus_1randn(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip'):
    """
    Trains a linear model.
    X is a dictionary of DataName -> np.array , containing raw input data
    X is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    import sklearn
    #prepare model output
    MODELNAME = 'LinearSVM-L2C0p1SquareHinge-1p0randn'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    STDOUT = sys.stdout
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?
                #print modelfile, modelExists, yname, i

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints
                sys.stdout = open('{}/log.txt'.format(modeldir), 'wb')

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])
                Xval = np.reshape(Xval, [Nv, -1])
                Xtest = np.reshape(Xtest, [Nte, -1])

                #add some random noise to the training data
                Xtrain += 0.5 + np.random.randn(Xtrain.shape[0], Xtrain.shape[1])

                #encode labels as required by sklearn
                YtrainSVM = np.argmax(Ytrain, axis=1)
                YtestSVM = np.argmax(Ytest, axis=1)
                YvalSVM = np.argmax(Yval, axis=1)

                #input dims and output dims
                D = Xtrain.shape[1]
                L = Ytrain.shape[1]


                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    STDOUT.write('incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists))
                    exit()

                if modelExists and ifModelExists == 'skip':
                    STDOUT.write('{} exists. skipping.\n\n'.format(modelfile))
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    STDOUT.write('{} exists. loading model, re-evaluating. \n\n'.format(modelfile))
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here
                    STDOUT.write('    training SVM model\n')
                    model = sklearn.svm.LinearSVC(C=0.1) #use default options: square hinge loss, l2 penalty on the weights, C = 1.0
                    model.fit(Xtrain, YtrainSVM)

                    STDOUT.write('    converting SVM model to Toolbox NN model\n')
                    #convert SKLearn-Models to  Toolbox NN models
                    if np.unique(YtrainSVM).size == 2:
                        #make a multi-output model
                        L = Linear(D, L)
                        L.W = np.concatenate([-model.coef_.T, model.coef_.T], axis=1)
                        L.B = np.concatenate([-model.intercept_, model.intercept_], axis=0)
                        nn = modules.Sequential([L])
                    else:
                        #just copy the learned parameters
                        L = Linear(D, L)
                        L.W = model.coef_.T
                        L.B = model.intercept_
                        nn = modules.Sequential([L])

                    STDOUT.write('    sanity checking model conversion\n')
                    #sanity check model conversion.
                    YpredSVM = model.decision_function(Xtest)
                    YpredNN = nn.forward(Xtest)

                    rtol=1e-7
                    if np.unique(YtrainSVM).size == 2:
                        np.testing.assert_allclose(YpredSVM, -YpredNN[:,0], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        np.testing.assert_allclose(YpredSVM, YpredNN[:,1], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        STDOUT.write('    sanity check passed (2-Class).\n')
                    else:
                        np.testing.assert_allclose(YpredSVM, YpredNN, rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal!')
                        STDOUT.write('    sanity check passed (Multiclass).\n')

                #test the model
                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                STDOUT.write(message)

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()

def run_linear_SVM_L2_C10_SquareHinge_plus_1randn(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip'):
    """
    Trains a linear model.
    X is a dictionary of DataName -> np.array , containing raw input data
    X is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    import sklearn
    #prepare model output
    MODELNAME = 'LinearSVM-L2C10SquareHinge-1p0randn'
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    STDOUT = sys.stdout
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?
                #print modelfile, modelExists, yname, i

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints
                sys.stdout = open('{}/log.txt'.format(modeldir), 'wb')

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])
                Xval = np.reshape(Xval, [Nv, -1])
                Xtest = np.reshape(Xtest, [Nte, -1])

                #add some random noise to the training data
                Xtrain += 0.5 + np.random.randn(Xtrain.shape[0], Xtrain.shape[1])

                #encode labels as required by sklearn
                YtrainSVM = np.argmax(Ytrain, axis=1)
                YtestSVM = np.argmax(Ytest, axis=1)
                YvalSVM = np.argmax(Yval, axis=1)

                #input dims and output dims
                D = Xtrain.shape[1]
                L = Ytrain.shape[1]


                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    STDOUT.write('incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists))
                    exit()

                if modelExists and ifModelExists == 'skip':
                    STDOUT.write('{} exists. skipping.\n\n'.format(modelfile))
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    STDOUT.write('{} exists. loading model, re-evaluating. \n\n'.format(modelfile))
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here
                    STDOUT.write('    training SVM model\n')
                    model = sklearn.svm.LinearSVC(C=10) #use default options: square hinge loss, l2 penalty on the weights, C = 1.0
                    model.fit(Xtrain, YtrainSVM)

                    STDOUT.write('    converting SVM model to Toolbox NN model\n')
                    #convert SKLearn-Models to  Toolbox NN models
                    if np.unique(YtrainSVM).size == 2:
                        #make a multi-output model
                        L = Linear(D, L)
                        L.W = np.concatenate([-model.coef_.T, model.coef_.T], axis=1)
                        L.B = np.concatenate([-model.intercept_, model.intercept_], axis=0)
                        nn = modules.Sequential([L])
                    else:
                        #just copy the learned parameters
                        L = Linear(D, L)
                        L.W = model.coef_.T
                        L.B = model.intercept_
                        nn = modules.Sequential([L])

                    STDOUT.write('    sanity checking model conversion\n')
                    #sanity check model conversion.
                    YpredSVM = model.decision_function(Xtest)
                    YpredNN = nn.forward(Xtest)

                    rtol=1e-7
                    if np.unique(YtrainSVM).size == 2:
                        np.testing.assert_allclose(YpredSVM, -YpredNN[:,0], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        np.testing.assert_allclose(YpredSVM, YpredNN[:,1], rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal! (2-Class-Case)')
                        STDOUT.write('    sanity check passed (2-Class).\n')
                    else:
                        np.testing.assert_allclose(YpredSVM, YpredNN, rtol, err_msg='Predictions of Trained SVM model and converted NN model are NOT equal!')
                        STDOUT.write('    sanity check passed (Multiclass).\n')

                #test the model
                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                STDOUT.write(message)

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()


def run_2layer_fcnn(X,Y,L,S,outputfolder='./tmp', n_hidden=512, ifModelExists='skip'):
    """
    Trains a 2-layer fully connected net with ReLU activations
    X is a dictionary of DataName -> np.array , containing raw input data
    X is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    #prepare model output
    MODELNAME = '2LayerFCNN-{}'.format(n_hidden)
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    STDOUT = sys.stdout
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)


                t_start = time.time()
                #set output log to capture all prints
                sys.stdout = open('{}/log.txt'.format(modeldir), 'wb')

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])
                Xval = np.reshape(Xval, [Nv, -1])
                Xtest = np.reshape(Xtest, [Nte, -1])

                #input dims and output dims
                D = Xtrain.shape[1]
                L = Ytrain.shape[1]

                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    STDOUT.write('incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists))
                    exit()

                if modelExists and ifModelExists == 'skip':
                    STDOUT.write('{} exists. skipping.\n\n'.format(modelfile))
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    STDOUT.write('{} exists. loading model, re-evaluating. \n\n'.format(modelfile))
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here
                    nn = modules.Sequential([modules.Linear(D, n_hidden), modules.Rect(), modules.Linear(n_hidden, L), modules.SoftMax()])
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005) # train the model
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.001) # slower training once the model has converged somewhat
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.0005)# one last epoch

                #test the model
                #Ypred = nn.forward(Xtest)
                #Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                #RpredPresoftmax = nn.lrp(nn.modules[-2].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                #Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                STDOUT.write(message)

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()

def run_3layer_fcnn(X,Y,L,S,outputfolder='./tmp', n_hidden=512, ifModelExists='skip'):
    """
    Trains a 3-layer fully connected net with ReLU activations
    X is a dictionary of DataName -> np.array , containing raw input data
    X is a dictionary of Targetname -> np.array , containing binary labels
    L is a dictionary of DataName -> channel labels
    S is a dictionary of TargetName -> prepared index splits
    """

    #prepare model output
    MODELNAME = '3LayerFCNN-{}'.format(n_hidden)
    #and output folder
    if not os.path.isdir(outputfolder):
        os.mkdir(outputfolder)
    #grab stdout to relay all prints to a log file
    STDOUT = sys.stdout
    LOG = open(outputfolder + '/log.txt', 'ab') #append (each model trained this day)

    #write out data and stuff used in this configuration. we just keep the same seed every time to ensure reproducibility
    scipy.io.savemat(outputfolder+'/data.mat', X)
    scipy.io.savemat(outputfolder+'/targets.mat', Y)
    scipy.io.savemat(outputfolder+'/labels.mat', L)
    scipy.io.savemat(outputfolder+'/splits.mat', S)


    #loop over all possible combinatinos of things
    for xname, x in X.iteritems():
        for yname, y in Y.iteritems(): #target name, i.e. pick a label in name and data
            targetSplits = S[yname]
            for i in xrange(len(targetSplits)): #the splits for this target
                #create output directory for this run
                modeldir = '{}/{}/{}/{}/part-{}'.format(outputfolder, yname, xname, MODELNAME, i)
                modelfile = '{}/model.txt'.format(modeldir)
                modelExists = os.path.isfile(modelfile) # is there an already pretrained model?

                if not os.path.isdir(modeldir):
                    os.makedirs(modeldir)

                t_start = time.time()
                #set output log to capture all prints
                sys.stdout = open('{}/log.txt'.format(modeldir), 'wb')

                iTest = targetSplits[i] #get split for validation and testing
                iVal = targetSplits[(i+1)%len(targetSplits)]
                iTrain = []
                for j in [r % len(targetSplits) for r in range(i+2, (i+2)+(len(targetSplits)-2))]: #pool remaining data into training set.
                    iTrain.extend(targetSplits[j])

                #format the data for this run
                Xtrain = x[iTrain, ...]
                Ytrain = y[iTrain, ...]

                Xval = x[iVal, ...]
                Yval = y[iVal, ...]

                Xtest = x[iTest, ...]
                Ytest = y[iTest, ...]

                #get original data shapes
                Ntr, T, C = Xtrain.shape
                Nv = Xval.shape[0]
                Nte = Xtest.shape[0]

                #reshape for fully connected inputs
                Xtrain = np.reshape(Xtrain, [Ntr, -1])
                Xval = np.reshape(Xval, [Nv, -1])
                Xtest = np.reshape(Xtest, [Nte, -1])

                #input dims and output dims
                D = Xtrain.shape[1]
                L = Ytrain.shape[1]

                #how to handle existing model files
                if modelExists and ifModelExists not in ['retrain', 'skip', 'load']:
                    STDOUT.write('incorrect instruction "{}" for handling preexisting model. aborting.\n\n'.format(ifModelExists))
                    exit()

                if modelExists and ifModelExists == 'skip':
                    STDOUT.write('{} exists. skipping.\n\n'.format(modelfile))
                    continue #ok, let us skip existing results again, as long as a model file exists. assume the remaining results exist as well

                elif modelExists and ifModelExists == 'load':
                    STDOUT.write('{} exists. loading model, re-evaluating. \n\n'.format(modelfile))
                    nn = model_io.read(modelfile)

                else: # model does not exist or parameter is retrain.
                    #create and train the model here
                    nn = modules.Sequential([modules.Linear(D, n_hidden), modules.Rect(), modules.Linear(n_hidden,n_hidden), modules.Rect(), modules.Linear(n_hidden, L), modules.SoftMax()])
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005) # train the model
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.001) # slower training once the model has converged somewhat
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.0005)# one last epoch

                #test the model
                #Ypred = nn.forward(Xtest)
                #Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                #RpredPresoftmax = nn.lrp(nn.modules[-2].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                #Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

                Ypred, Rpred, RpredPresoftmax, Ract, RPredAct, RPredDom, RPredActComp, RPredDomComp = test_model(nn, Xtest, Ytest, Nte, T, C)

                #measure test performance
                l1loss = np.abs(Ypred - Ytest).sum()/Nte
                predictions = np.argmax(Ypred, axis=1)
                groundTruth = np.argmax(Ytest, axis=1)
                acc = np.mean((predictions == groundTruth))

                t_end = time.time()

                #print results to terminal and log file
                message = '\n'
                message += '{} {}\n'.format(modeldir.replace('/', ' '),':')
                message += 'test accuracy: {}\n'.format(acc)
                message += 'test loss (l1): {}\n'.format(l1loss)
                message += 'train-test-sequence done after: {}s\n\n'.format(t_end-t_start)

                LOG.write(message)
                LOG.flush()
                STDOUT.write(message)

                #write out the model
                model_io.write(nn, modelfile)

                #write out performance
                with open('{}/scores.txt'.format(modeldir), 'wb') as f:
                    f.write('test loss (l1): {}\n'.format(l1loss))
                    f.write('test accuracy : {}'.format(acc))


                #write out matrices for prediction, GT heatmaps and prediction heatmaps
                scipy.io.savemat('{}/outputs.mat'.format(modeldir),
                                 {'Ypred': Ypred,
                                  'Rpred': Rpred,
                                  'RpredPresoftmax': RpredPresoftmax,
                                  'Ract': Ract,
                                  'RPredAct' : RPredAct,
                                  'RPredDom' : RPredDom,
                                  'RPredActComp' : RPredActComp,
                                  'RPredDomComp' : RPredDomComp,
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()
