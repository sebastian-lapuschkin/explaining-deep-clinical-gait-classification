#!/usr/bin/python

import subprocess
from datetime import datetime


def query():
    QTEXT = subprocess.check_output(['qstat -u "*"'], shell=True)
    QTEXT = QTEXT.decode("utf-8")
    # print(QTEXT)
    QTEXT = QTEXT.split('\n')

    running = 0
    waiting = 0
    groups = 0
    load = len(QTEXT)

    oldest_running = None
    newest_running = None

    #filter out user-jobs
    QTEXT = [qline for qline in QTEXT if 'lapuschkin' in qline]

    for qline in QTEXT:
        if ' r ' in qline:
            running += 1
            newest_running = qline
            if oldest_running is None:
                oldest_running = qline

        elif ' qw ' in qline:
            waiting += 1

        elif ' hqw ' in qline:
            groups += 1

    return running, waiting, groups, load, oldest_running, newest_running

def status():
    running, waiting, groups, load, oldest_running, newest_running = query()

    print ('JOB STATUS AT', datetime.now())
    print ('')
    print ('running:', running)
    print ('waiting:', waiting)
    print ('groups:', groups)
    print ('cluster load:', load, '!'*((load-90000)/1000 + 1) if load > 90000 else '')

    print ('')
    print ('oldest running:', oldest_running)
    print ('|')
    print ('newest_running:', newest_running)

if __name__ == '__main__':
    status()
