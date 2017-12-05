import os
import sys
import scipy.io as scio
import model_io
import time
import numpy as np
from multiprocessing import Pool
import perturbation_experiments









if __name__ == '__main__':

    DAYFOLDER = 'X'
    ARCHITECTURE = 'X'
    TARGET = 'X'
    DATA = 'X'
    DEBUG = 0

    for param in sys.argv:
        if not '=' in param:
            continue
        else:
            print 'parsing parameter pair', param
            k,v = param.split('=')
            if 'folder' in k:
                if os.path.isdir(v):
                    DAYFOLDER = v
                else:
                    print '    folder',v,'does not exist.'
                    exit()
            elif 'arch' in k:
                ARCHITECTURE = v
            elif 'target' in k:
                if not v in ['Gender', 'Subject']:
                    print '    target must be "gender" or "subject"'
                    exit()
                else:
                    TARGET = v
            elif 'data' in k:
                DATA = v
            elif 'debug' in k:
                DEBUG = int(v)
            else:
                print 'unknown parameter key', k
                exit()


    # assert selection. and paths. load stuff.
    assert os.path.isdir(DAYFOLDER), '{} does not exist'.format(DAYFOLDER)

    print 'loading input data', DATA
    data = DAYFOLDER + '/data.mat'
    assert os.path.isfile(data), '{} does not exist'.format(data)
    data = scio.loadmat(data) # dictionary datatype -> nparray
    #select relevant input data
    assert DATA in data, 'data typ {} not available in dataset. all targets: {}'.format(DATA, data.keys())
    data = data[DATA]

    print 'loading target', TARGET
    targets = DAYFOLDER + '/targets.mat'
    assert os.path.isfile(targets), '{} does not exist'.format(targets)
    targets = scio.loadmat(targets) # dictionary class type -> nparray
    #select relevant stuff.
    assert TARGET in targets, 'prediction target {} not available in dataset. all targets: {}'.format(TARGET, targets.keys())
    targets = targets[TARGET]

    print 'loading splits for', TARGET
    splits = DAYFOLDER + '/splits.mat'
    assert os.path.isfile(splits), '{} does not exist'.format(splits)
    splits = scio.loadmat(splits) # dictionary target -> nparray
    #unpack splits according to prediction target
    splits = splits[TARGET] # 1 x 10 object type numpy array.


    print 'locating resources / preparing worker packages'
    #check paths
    targetfolder = DAYFOLDER + '/' + TARGET
    assert os.path.isdir(targetfolder), '{} does not exist'.format(targetfolder)
    datafolder = targetfolder + '/' + DATA
    assert os.path.isdir(datafolder), '{} does not exist'.format(datafolder)
    modelfolder = datafolder + '/' + ARCHITECTURE
    assert os.path.isdir(modelfolder), '{} does not exist'.format(modelfolder)

    N = splits.size
    modelpaths = [None]*N
    relevancepaths =[None]*N
    outputfolder = [None]*N

    #check model file availability and collect paths for splits/workers
    for i in xrange(N):
        splitfolder = modelfolder + '/part-{}'.format(i)
        assert os.path.isdir(splitfolder), '{} does not exist'.format(splitfolder)
        outputfolder[i] = splitfolder

        modelfile = splitfolder + '/model.txt'
        assert os.path.isfile(modelfile), '{} does not exist'.format(modelfile)
        modelpaths[i] = modelfile

        modeloutputs = splitfolder + '/outputs.mat'
        assert os.path.isfile(modeloutputs), '{} does not exist'.format(modeloutputs)
        relevancepaths[i] = modeloutputs

    workerparams = [None]*N
    #pack parameters for workers
    for i in xrange(N):

        testsplit = splits[0,i][0]
        workerparams[i] = { 'split_no':i,
                            'model':modelpaths[i],
                            'relevances':relevancepaths[i],
                            'outputfolder':outputfolder[i],
                            'Xtest':data[testsplit],
                            'Ytest':targets[testsplit]}




    t_start = time.time()
    if DEBUG:
        print 'debugging single run'
        for i in xrange(len(workerparams)):
            perturbation_experiments.run(workerparams[i])
    else:
        print 'creating worker pool of size', N
        pool = Pool(N)

        print 'executing jobs'
        pool.map(perturbation_experiments.run, workerparams)


    print 'done after {}s'.format(time.time() - t_start)


#python perturbations.py folder=BASELINE-LINEAR-S1234 arch=Linear data=GRF_AV target=Gender debug=1
