import glob
import os
import sys
import numpy as np
import natsort
from prettytable import PrettyTable

def run(EXPERIMENTS='./BASELINE-NO-TRAINING'):
    results = []
    with open(EXPERIMENTS + '/log.txt', 'rb') as f:
        log = f.read().split('\n')
        for i in xrange(len(log)):
            line = log[i]
            if len(line) > 0:
                if 'part' in line: #header
                    lineparts = line.strip('.').strip(':').strip().split()
                    target = lineparts[-4]
                    data = lineparts[-3]
                    model = lineparts[-2]
                    part = lineparts[-1]

                    acc = log[i+1].split()[-1]
                    thisresult = [target, data, model, float(acc)]
                    results.append(thisresult)

    results = np.array(results)
    #get unique field entries
    targets = natsort.natsorted(np.unique(results[:, 0]))
    print targets
    data = natsort.natsorted(np.unique(results[:, 1]))
    print data
    models = natsort.natsorted(np.unique(results[:, 2]))
    print models
    print ''


    for t in targets:
        tab = PrettyTable([t] + data)

        for m in models:
            row = [m]
            for d in data:
                ii = np.where((results[:,0] == t) * (results[:,1] == d) * (results[:,2] == m))[0]
                #print t, m, d, ii

                scores = [float(s) for s in results[ii,3]]
                scores = '{:.2f} +- {:.2f}'.format(np.mean(scores)*100, np.std(scores)*100)
                row.append(scores)

            tab.add_row(row)

        print tab
    print ''


if __name__ == '__main__':

    # can now be called from command line as
    # python eval_score_logs.py <resultfolder>
    # the script intends to load and parse <resultfolder>/log.txt

    if len(sys.argv) < 2:
        run()
    else:
        run(sys.argv[1])


