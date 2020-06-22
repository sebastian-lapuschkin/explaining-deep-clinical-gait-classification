"""
Created on Aug 09 09:32:28 2017

@author: helle
"""

import argparse
import sys
import os
import getpass
import shutil
import subprocess
import re
import urllib.parse

MEM = '20G'
THREADS = 20
CONDA = 'conda'
ENV = 'gait'
HERE = os.getcwd()
JOBNAME = None
LOGDIR = '/home/fe/lapuschkin/logs-gait'
SCRIPTDIR = '/home/fe/lapuschkin/scripts-gait'


if __name__ == '__main__':

    ARGSFILE = None
    SCRIPTFILES = []
    if len(sys.argv) < 2:
        print('Not enough arguments. passing of args file path required.')
        print('Example: python3 sge_job_simple.py ./path/to/arg_lines.txt')
        print('arg_lines.args should be a file where all the command line args are given in one line')
    else:
        ARGSFILE = sys.argv[1]

    if not os.path.isdir(LOGDIR):
        print('Creating {}'.format(LOGDIR))
        os.makedirs(LOGDIR)

    if not os.path.isdir(SCRIPTDIR):
        print('Creating {}'.format(SCRIPTDIR))
        os.makedirs(SCRIPTDIR)

    all_arg_lines = []
    with open(ARGSFILE, 'rt') as args:
        all_arg_lines = args.read().split('\n')

    for i in range(len(all_arg_lines)):
        args = all_arg_lines[i]
        print('Generating and submitting jobs for:')
        print('    {}'.format(args))

        #generate script
        execthis    =  ['#!/bin/bash']
        execthis    =  ['source /home/fe/lapuschkin/.bashrc']           # enables conda for bash
        execthis    += ['cd {}/../python'.format(HERE)]                    # go to python root
        execthis    += ['{} activate {}'.format(CONDA, ENV)]            # enter venv
        execthis    += ['python3 gait_experiments.py {}'.format(args)]  # call script with parameters.
        execthis    += ['{} deactivate'.format(CONDA)]                  # leave venv
        execthis = '\n'.join(execthis)

        JOBNAME ='[{}_of_{}]_of_"{}"'.format(i+1,len(all_arg_lines), os.path.basename(ARGSFILE).replace(' ','_'))
        SCRIPTFILE='{}/{}-{}'.format(SCRIPTDIR, os.path.basename(ARGSFILE), i+1)
        print('    as file: {}'.format(SCRIPTFILE))
        print('    as name: {}'.format(JOBNAME))


        #write to script
        with open(SCRIPTFILE,'w') as f:
            f.write(execthis)
            print('    job written to {}'.format(SCRIPTFILE))
            os.system('chmod 777 {}'.format(SCRIPTFILE))

        #job submission
        cmd  = [ 'qsub' ]
        cmd += [ '-N', JOBNAME ]
        cmd += [ '-pe', 'multi', str(THREADS) ]
        cmd += [ '-e', LOGDIR ]
        cmd += [ '-o', LOGDIR ]
        cmd += [ '-l', 'h_vmem=' + MEM ]
        cmd += [SCRIPTFILE]
        print('    preparing to submit via command: {}'.format(cmd))

        
        p = subprocess.Popen( cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
        ( stdout, stderr ) = p.communicate()
        print('    ', 'P:', p, 'O:', stdout, 'E:',  stderr)
        if ( p.returncode == 0 ):
            id_job = stdout.decode('utf-8').split()[2]
        else:
            id_job = None

        if len(stderr) > 0:
            print('ERROR during job submission? "{}"'.format(stderr))
            exit()

        print('    dispatched job {0} with id {1}'.format(JOBNAME, id_job))
        

        print('    submitted job executes:')
        print('\n'.join(['>>>> ' + e for e in execthis.split('\n')]))
        print()
        

