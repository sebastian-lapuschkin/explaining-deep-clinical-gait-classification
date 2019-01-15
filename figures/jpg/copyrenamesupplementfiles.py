import glob
import shutil

#load file names
filenames = glob.glob('Figure-0-*')
for f in filenames:
    if ' ' in f:
        fnew = f.replace(' ','-')
        print 'cp {} -> {}'.format(f, fnew)
        shutil.copyfile(f, fnew)
