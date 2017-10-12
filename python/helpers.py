import numpy as np

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


def convIOdims(D,F,S):
    D = float(D)
    F = float(F)
    S = float(S)
    return (D-F)/S + 1
