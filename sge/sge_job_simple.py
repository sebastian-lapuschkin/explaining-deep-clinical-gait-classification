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
THREADS = 10
CONDA = 'conda'
ENV = 'gait'
HERE = os.getcwd()
JOBNAME = 'GAIT-'
LOGDIR = '/home/fe/lapuschkin/logs-gait'
SCRIPTDIR = '/home/fe/lapuschkin/scripts-gait'

ARGSFILE = None
if len(sys.argv) <= 2:
    print('Not enough arguments. passing of args file path required.)
    print('Example: python3 sge_job_simple.py ./path/to/arg_lines.txt')
    print('arg_lines.txt should be a file where all the command line args are given in one line')
else:
    ARGSFILE = sys.argv[1]

if not os.path.isdir(LOGDIR):  os.makedirs(LOGDIR)
if not os.path.isdir(SCRIPTDIR): os.makedirs(SCRIPTDIR)

all_arg_lines = []
with open(ARGSFILE, 'rt') as args:
    all_arg_lines = args.read().split('\n')

for args in all_arg_lines:
    print('Generating and submitting jobs for:')
    print('    {}'.format(args))
exit()




#generate script
execthis    =  ['#!/bin/bash']
execthis    =  ['source /home/fe/lapuschkin/.bashrc'] #enables conda for bash
execthis    += ['cd {}/../python'.format(HERE)] #go to python root
execthis    += ['{} activate {}'.format(CONDA, ENV)]
execthis    += ['python3 THISFILE.py']
execthis    += ['{} deactivate'.format(CONDA)]
execthis = '\n'.join(execthis)

#write to script
with open(SCRIPTFILE,'w') as f:
    f.write(execthis)
    print('Job written to {}'.format(SCRIPTFILE))
    os.system('chmod 777 {}'.format(SCRIPTFILE))

#job submission
cmd  = [ 'qsub' ]
cmd += [ '-N', JOBNAME ]
cmd += [ '-pe', 'multi', str(THREADS) ]
cmd += [ '-e', LOGDIR ]
cmd += [ '-o', LOGDIR ]
cmd += [ '-l', 'h_vmem=' + MEM ]
cmd += [SCRIPTFILE]
print(cmd)

#p = subprocess.Popen( cmd, cwd = nameOsWD, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
#print(' '.join(cmd))
p = subprocess.Popen( cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE )
( stdout, stderr ) = p.communicate()
print(p, stdout, stderr)
if ( p.returncode == 0 ):
    id_job = stdout.decode('utf-8').split()[2]
else:
    id_job = None

print('Dispatched job {0} with id {1}'.format(JOBNAME, id_job))
print()
print('Submitted job executes:')
print(execthis)

