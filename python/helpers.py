import numpy as numpy

def create_index_splits(Y_Subject, Y_Injury, splits = 10, seed=None):
    """ this method subdivides the given labels into optimal groups

        for the subject prediction labels, it divides the indices into equally sized groups,
        each containing equally many samples of each person.

        for the gender prediction labels (gender is linked to subject obviously) the
        data is split into partitions where no subject can reoccur
    """

    assert splits > 3, 'At least three splits required'

    #number of samples
    assert Y_Subject.shape[0] == Y_Injury.shape[0], 'Number of Subject and Gender sample labels differ: {} vs {}'.format(Y_Subject.shape, Y_Injury.shape)

    N, P = Y_Subject.shape
    _, I = Y_Injury.shape

    #create global permutation sequence
    Permutation = numpy.arange(N)

    if seed is not None: #reseed the random generator
        numpy.random.seed(seed)
        Permutation = numpy.random.permutation(Permutation)

    #permute label matrices. also return this thing!
    Y_Subject = Y_Subject[Permutation,...]
    Y_Injury = Y_Injury[Permutation,...]

    #initialize index lists
    SubjectIndexSplits = [None]*splits

    #1) create a split over subject labels first by iterating over all person labels and subdividing them as equally as possible.
    for i in range(P):
        pIndices = numpy.where(Y_Subject[:,i] == 1)[0]

        #compute an approx equally sized partitioning.
        partitioning = numpy.linspace(0, len(pIndices), splits+1, dtype=int)
        for si in range(splits):
            #make sure index lists exist
            if SubjectIndexSplits[si] is None:
                SubjectIndexSplits[si] = []

            #spread subject label across those index lists
            if si == splits-1:
                #the last group.
                SubjectIndexSplits[si].extend(pIndices[partitioning[si]:])
            else:
                SubjectIndexSplits[si].extend(pIndices[partitioning[si]:partitioning[si+1]])


    #2) create a split over injury labels, balancing injury as good as possible but by avoiding the same subject label in more than one bin.
    #for injury recognition, we want to avoid the model to learn gait criteria of subjects and classify by that bias.
    #first split into injury groups and use them as queues
    injuryQueues = [numpy.where(Y_Injury[:, i] == 1)[0].tolist() for i in range(I)]
    InjuryIndexSplits = [None]*splits
    currentSplit = 0

    #alternatingly move through injury lists and place people into splits accordingly.
    #remove those people from the injury queues accordingly.
    while sum([len(iQ) for iQ in injuryQueues]) > 0:
        #make sure the split is populated
        if InjuryIndexSplits[currentSplit] is None:
            InjuryIndexSplits[currentSplit] = []

        #for each injury get next person, if this injury is not yet exhausted.
        for iQ in injuryQueues:
            if len(iQ) == 0:
                continue

            #process lists/subjects:
            #find out who the next person is. get all those entries.
            pindex = numpy.where(Y_Subject[iQ[0], :])[0]
            #get all the indices for that person.
            pIndices = numpy.where(Y_Subject[:, pindex])[0]

            #remove this person from its respective queue
            for p in pIndices:
                iQ.remove(p)

            #and add it to its split group
            InjuryIndexSplits[currentSplit].extend(pIndices)

        #move split position
        currentSplit = (currentSplit + 1) % splits

    #return the indices for the subject recognition training, the gender recognition training and the original permutation to be applied on the data.
    return SubjectIndexSplits, InjuryIndexSplits, Permutation


def convIOdims(D,F,S):
    #helper method for computing output dims of 2d convolutions when giving D as data shape, F as filter shape and S as stride
    #D, F and S are expected to be scalar values
    D = float(D)
    F = float(F)
    S = float(S)
    return (D-F)/S + 1
