import glob
import os
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
                    target = lineparts[1]
                    data = lineparts[2]
                    model = lineparts[3]
                    part = lineparts[4]

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
        tab = PrettyTable([''] + data)

        for m in models:
            row = [m]
            for d in data:
                ii = np.where((results[:,0] == t) * (results[:,1] == d) * (results[:,2] == m))[0]
                #print t, m, d, ii

                scores = [float(s) for s in results[ii,3]]
                scores = '{:.2f} +- {:.2f}'.format(np.mean(scores)*100, np.std(scores)*100)
                row.append(scores)

            tab.add_row(row)

        print t
        print tab
    print ''

if __name__ == '__main__':
    run()


