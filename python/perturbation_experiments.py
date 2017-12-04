import scipy.io as scio
import model_io
import time
import numpy as np

def run(workerparams):

    #unpack parameters
    S = workerparams['split_no'] # an int. the current split
    modelpath = workerparams['model'] # path of the model to load
    relevancepath = workerparams['relevances'] # path of the relevance mat file to load
    X = workerparams['Xtest'] # N x T x C test data
    Y = workerparams['Ytest'] # N x 2 or N x 57 test labels. binary

    print 'split',S, ': [1] loading model'
    nn = model_io.read(modelpath)


    #after here the actual work





