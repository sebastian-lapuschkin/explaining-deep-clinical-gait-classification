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
import model_io


def run_cnn_A(X,Y,L,S,outputfolder='./tmp', ifModelExists='skip'):
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

                #attach artificial channel axis.
                Xtrain = Xtrain[..., None]
                Xval = Xval[..., None]
                Xtest = Xtest[..., None]

                #number of target labels
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
                        return

                    #STDOUT.write('starting {} {}'.format(xname, yname))
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.005) # train the model
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.001) # slower training once the model has converged somewhat
                    nn.train(Xtrain, Ytrain, Xval=Xval, Yval=Yval, batchsize=5, lrate=0.0005)# one last epoch
                    #STDOUT.write('    {} {} ok\n'.format(xname, yname))

                #test the model
                Ypred = nn.forward(Xtest)
                Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                RpredPresoftmax = nn.lrp(nn.modules[-2].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

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
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
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
                Ypred = nn.forward(Xtest)
                Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                RpredPresoftmax = nn.lrp(nn.modules[-1].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

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
                Ypred = nn.forward(Xtest)
                Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                RpredPresoftmax = nn.lrp(nn.modules[-2].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

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
                Ypred = nn.forward(Xtest)
                Rpred = nn.lrp(Ypred, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C) #reshape data into original input shape
                RpredPresoftmax = nn.lrp(nn.modules[-2].Y, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)
                Ract = nn.lrp(Ytest, lrp_var='epsilon', param=1e-5).reshape(Nte, T, C)

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
                                  'l1loss': l1loss,
                                  'acc': acc})


                #reinstate original sys.stdout
                sys.stdout.close()
                sys.stdout = STDOUT

    sys.stdout = STDOUT
    LOG.close()
