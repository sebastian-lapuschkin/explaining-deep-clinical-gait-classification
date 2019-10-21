""" this script will analyze log dirs. """

import os
import sys
import fnmatch
import natsort
import config
import numpy as np




if __name__ == '__main__':

    #read in sge config
    config.read_config_file('sge-sebastian.cfg')

    if len(sys.argv) > 1:
        print 'restricting analysis/result output to "{}"'.format(sys.argv[1])


    #by defailt, analyze anyting. extend for some command line calls when necessary
    logdir = config.gridengine.log_location
    reports = {} #dictionary taskname -> report

    for currentdir, dirs, files in os.walk(logdir):
        if len(dirs) == 0: #we are at the bottom of the log file system tree. only log files here.
            taskname = os.path.basename(currentdir) # the executed task
            if len(sys.argv) > 1 and sys.argv[1] != taskname:
                continue

            report = []
            jobs_successful = 0
            failed_jobs = []
            failed_jobs_nodes = []
            failed_jobs_logs =  []
            empty_jobs = []

            #split logs in error and output logs
            files = natsort.natsorted(files)
            stdoutlogs = []
            stderrlogs = []
            for f in files:
                if fnmatch.fnmatch(f, '*.e*'):
                    stderrlogs.append(f)
                elif fnmatch.fnmatch(f, '*.o*'):
                    stdoutlogs.append(f)
                else:
                    raise ValueError('Unmatchable log file ' + f)

            #loop over files and check contents. error files should be empty to be ok
            if len(stderrlogs) != len(stdoutlogs):
                report.append('#stdout logs: {} != #stderr logs: {} . Incomplete task?'.format(len(stdoutlogs), len(stderrlogs)))
            else:
                for i in xrange(len(stderrlogs)):
                    ofile = stdoutlogs[i]
                    efile = stderrlogs[i]

                    #assert these files describe the results for the same job
                    assert os.path.splitext(os.path.basename(ofile))[0] == os.path.splitext(os.path.basename(efile))[0], 'Job names differ! {} {}'.format(ofile, efile)
                    assert os.path.splitext(os.path.basename(ofile))[1][2::] == os.path.splitext(os.path.basename(efile))[1][2::], 'Job IDs differ! {} {}'.format(ofile, efile)

                    jobname = os.path.splitext(os.path.basename(efile))[0]
                    osize = os.path.getsize(currentdir + '/' + ofile)
                    esize = os.path.getsize(currentdir + '/' + efile)
                    if osize > 0 and esize == 0:
                        jobs_successful += 1

                    elif osize == 0 and esize == 0:
                        empty_jobs.append(jobname)

                    else: # esize > 0 and (osize > 0  or osize == 0)
                        hostname = stacktrace = None
                        with open(currentdir + '/' + ofile, 'rb') as f:
                            hostname = f.readline() #assume we call hostname before executing the actual cmd on each script
                        with open(currentdir + '/' + efile, 'rb') as f:
                            stracktrace = f.read()

                        failed_jobs.append(jobname)
                        failed_jobs_logs.append('{}:\n{}'.format(hostname, stracktrace))
                        failed_jobs_nodes.append(hostname)

            #compile report text
            text = '{} REPORT:\n'.format(taskname)
            text += '    {} jobs OK\n'.format(jobs_successful)
            text += '    {} job logs empty\n'.format(len(empty_jobs)) if len(empty_jobs) > 0 else ''
            text += '    {} jobs FAILED\n'.format(len(failed_jobs)) if len(failed_jobs) > 0 else ''
            text += 'VERDICT : {}\n'.format('OK' if len(failed_jobs) == 0 else 'MEH                !!!!!!!!                  !!!!!!!!')

            #write short report to terminal.
            print text

            #extend report for file printed version
            if len(empty_jobs) > 0:
                text += '\n'
                text += 'Empty job files:\n'
                text += '\n'.join(empty_jobs)

            if len(failed_jobs) > 0:
                text += '\n'
                text += 'Nodes with failing jobs:\n'
                unique_nodes = np.unique(failed_jobs_nodes)
                for u in natsort.natsorted(unique_nodes.tolist()):
                    text += '{}'.format(u)

            if len(failed_jobs) > 0:
                text += '\n'
                text += 'Failed jobs:\n'
                for i in xrange(len(failed_jobs)):
                    text += '{}:\n'.format(failed_jobs[i])
                    text += '{}\n'.format(failed_jobs_logs[i])

            text += '\n\n'

            #write long report to file
            with open('{}/{}-report.txt'.format(config.gridengine.log_location, taskname), 'wb') as f:
                f.write(text)











