
'''
@author: Sebastian Lapuschkin
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 08.09.2017
@version: 1.0
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''


import os
import modules
import model_io
import render
import numpy as np # the de facto numerics package for python
import scipy.io as scio # scientific python package, which supports mat-file IO within python
import matplotlib.pyplot as plt #realizes plotting capabilities similar to what matlab does natively


def create_index_splits(Y_Subject, Y_Gender, splits = 10, seed=None):
    """ this method subdivides the given labels into optimal groups

        for the subject prediction labels, it divides the indices into equally sized groups,
        each containing equally many samples of each person.

        for the gender prediction labels (gender is linked to subject obviously) the
        data is split into partitions where no subject can reoccur
    """

    assert splits > 3, 'At least three splits required'

    #number of samples
    assert Y_Subject.shape[0] == Y_Gender.shape[0], 'Number of Subject and Gender sample labels differ.'

    N, P = Y_Subject.shape
    _, G = Y_Gender.shape

    #create global permutation sequence
    Permutation = np.arange(N)

    if seed is not None: #reseed the random generator
        np.random.seed(seed)
        Permutation = np.random.permutation(Permutation)

    #permute label matrices. also return this thing!
    Y_Subject = Y_Subject[Permutation,...]
    Y_Gender = Y_Gender[Permutation,...]

    #initialize index lists
    SubjectIndexSplits = [None]*splits

    #1) create a split over subject labels first by iterating over all person labels and subdividing them as equally as possible.
    for i in xrange(P):
        pIndices = np.where(Y_Subject[:,i] == 1)[0]

        #compute an approx equally sized partitioning.
        partitioning = np.linspace(0, len(pIndices), splits+1, dtype=int)
        for si in xrange(splits):
            #make sure index lists exist
            if SubjectIndexSplits[si] is None:
                SubjectIndexSplits[si] = []

            #spread subject label across those index lists
            if si == splits-1:
                #the last group.
                SubjectIndexSplits[si].extend(pIndices[partitioning[si]:])
            else:
                SubjectIndexSplits[si].extend(pIndices[partitioning[si]:partitioning[si+1]])


    #2) create a split over gender labels, balancing gender as good as possible but by avoiding the same subject label in more than one bin.
    #for gender recognition, we want to avoid the model to learn gait criteria of subjects and classify by that bias.
    #first split into gender groups and use them as queues
    gender0 = np.where(Y_Gender[:, 0] == 1)[0].tolist()
    gender1 = np.where(Y_Gender[:, 1] == 1)[0].tolist()
    genderQueues = [gender0, gender1]
    GenderIndexSplits = [None]*splits
    currentSplit = 0

    #alternatingly move through gender lists and place people into splits accordingly.
    #remove those people from the gender queues accordingly.
    while sum([len(gQ) for gQ in genderQueues]) > 0:
        #make sure the split is populated
        if GenderIndexSplits[currentSplit] is None:
            GenderIndexSplits[currentSplit] = []

        #for each gender get next person, if this gender is not yet exhausted.
        for gQ in genderQueues:
            if len(gQ) == 0:
                continue

            #process lists/subjects:
            #find out who the next person is. get all those entries.
            pindex = np.where(Y_Subject[gQ[0], :])[0]
            #get all the indices for that person.
            pIndices = np.where(Y_Subject[:, pindex])[0]

            #remove this person from its respective queue
            for p in pIndices:
                gQ.remove(p)

            #and add it to its split group
            GenderIndexSplits[currentSplit].extend(pIndices)

        #move split position
        currentSplit = (currentSplit + 1) % splits

    #return the indices for the subject recognition training, the gender recognition training and the original permutation to be applied on the data.
    return SubjectIndexSplits, GenderIndexSplits, Permutation




################################
#           "Main"
################################

gaitdata = scio.loadmat('Gait_GRF_JA_Label.mat') #load matlab data as dictionary using scipy

# Bodenreaktionskraft mit zwei verschiedenen Datennormalisierungen
X_GRF_AV = gaitdata['Feature_GRF_AV']                   # 1142 x 101 x 6
Label_GRF_AV = gaitdata['Feature_GRF_AV_Label']         # 1 x 1 x 6 channel label

X_Feature_GRF_JV = gaitdata['Feature_GRF_JV']           # 1142 x 101 x 6
Label_Feature_GRF_JV = gaitdata['Feature_GRF_JV_Label'] # 1 x 1 x 6 channel label

# Gelenkwinkel in drei Drehachsen fuer den gesamten Koerper
X_JA_Full = gaitdata['Feature_JA_Full']                 # 1142 x 101 x 33
Label_JA_Full = gaitdata['Feature_JA_Full_Label']       # 1 x 1 x 33 channel label

# Gelenkwinkel lediglich in der Hauptdrehachse fuer den Unterkoerper
X_JA_X_Lower = gaitdata['Feature_JA_X_Full']            # 1142 x 101 x 10
Label_JA_X_Lower = gaitdata['Feature_JA_X_Full_Label']  # 1 x 1 x 10 channel label

#Target subject labels und gender labels
Y_Subject = gaitdata['Target_SubjectID']                # 1142 x 57, binary labels
Y_Gender = gaitdata['Target_Gender']                    # 1142 x 1 , binary labels
#-> create uniform label structure for gender
Y_Gender = np.repeat(Y_Gender,2,axis=1)
Y_Gender[:,1] = 1-Y_Gender[:,0]                         # 1142 x 2, binary labels  (562 vs 580)
#print Y_Gender.sum(axis=0)/20.


create_index_splits(Y_Subject, Y_Gender, seed=1234)
