""" module of static functions for drawing figurs 1 to 5 """
import os
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import colors
from matplotlib import gridspec
import PIL


def save_figure(filename, fmt, dpi):
    """
        plots the current figure and applies compression and stuff.
    """

    if not os.path.isdir(fmt):
        print('creating output folder "{}"'.format(fmt))
        os.makedirs(fmt)

    #add corresponding author name to file name
    filename +=  ' (horst)'

    filepath = '{}/{}.{}'.format(fmt,filename, fmt)
    print('writing figure to', filepath)
    plt.savefig(filepath, dpi=dpi)

    size_extensions = ['B', 'K', 'M', 'G']
    size_in_bytes = os.path.getsize(filepath)
    divisions=0
    while size_in_bytes > 1024 and divisions < 3:
        size_in_bytes = size_in_bytes/1024.
        divisions += 1
    print('   (size: {}{})'.format(int(size_in_bytes), size_extensions[divisions]))


def colormap(x, **kwargs):
    #NOTE define your color mapping function here.
    return firered(x, darken=0.7, **kwargs)


def firered(x, darken=1.0):
    """ a color mapping function, receiving floats in [-1,1] and returning
        rgb color arrays from (-1 ) lightblue over blue, black (center = 0), red to brightred (1)

        x : float array  in [-1,1], assuming 0 = center
        darken: float in ]0,1]. multiplicative factor for darkening the color spectrum a bit
    """

    #print 'color mapping {} shaped into array to "firered" color spectrum'.format(x.shape)
    assert darken > 0
    assert darken <= 1

    x *= darken

    hrp  = np.clip(x-0.00,0,0.25)/0.25
    hgp = np.clip(x-0.25,0,0.25)/0.25
    hbp = np.clip(x-0.50,0,0.50)/0.50

    hbn = np.clip(-x-0.00,0,0.25)/0.25
    hgn = np.clip(-x-0.25,0,0.25)/0.25
    hrn = np.clip(-x-0.50,0,0.50)/0.50

    colors =  np.concatenate([(hrp+hrn)[...,None],(hgp+hgn)[...,None],(hbp+hbn)[...,None]],axis = 2)
    #print '    returning rgb mask of size {}'.format(colors.shape)
    return colors


def interpolate_all_samples(Y, fold=10):
    # assumes a N x C matrix Y
    #interpolates along N axis to the fold amount of points.
    N, D = Y.shape
    x = np.arange(N)
    xinterp = np.linspace(0,N-1, num=N*fold)
    Yinterp = np.concatenate([np.interp(xinterp, x, Y[:,d])[:,None] for d in range(D)], axis=1)
    #print Y.shape, xinterp.shape, xinterp[-1], Yinterp.shape
    return xinterp, Yinterp



def draw_fig0 (fmt, variant='a',dpi=300):
    #this one is a bit hardcodey
    print('drawing figure(s) 0...')
    print('   loading data')
    print('   variant', variant)

    data = scipy.io.loadmat('Overview_Figure_15_46_47.mat')
    shortname = 'GRF'
    linewidth=1
    dotsize=1.5

    bodyparts = []
    for f in ['aff.', 'unaff.']:
        #for xyz in ['F_{ML}', 'F_{AP}', 'F_{V}']:
        for xyz in ['GRF_{ML}', 'GRF_{AP}', 'GRF_{V}']:
            #bodyparts.append('{} foot {}'.format(f, xyz))
            bodyparts.append('${}~{}$'.format(f, xyz))

    #subject specific plots
   #for subject in [15, 46, 47]:
    for subject in [46]:
        inputs = data['Feature_{}'.format(subject)]     # N x 606
        relevance = data['Rpred_{}'.format(subject)]    # N x 606

        xticks = np.arange(7)*101
        xlim = xticks[[0,-1]]

        #draw inputs
        maxamp = 1
        #fig = plt.figure(figsize=(6, 0.8))
        fig = plt.figure(figsize=(4, 0.8))
        ax = plt.subplot(111)

        for i in range(inputs.shape[0]):
            values = inputs[i]
            #markersize < 1 transforms dots to circles. stupid.
            plotcolor = 'blue'
            ax.plot(values, color=plotcolor, linestyle='-', linewidth=linewidth)

        ax.set_ylim([-maxamp, maxamp])
        ax.set_yticks([-maxamp, 0, maxamp])
        ticklabs = [str(s) for s in[-maxamp, 0, maxamp]]
        ticklabs = [' '*(2-len(s)) + s for s in ticklabs]
        ax.set_yticklabels(ticklabs, fontsize=8)
        ax.set_xticks([50.5 + 101*ii for ii in range(6)],minor=True)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_xticklabels(bodyparts, minor=True, fontsize=5)
        ax.set_xticks(xticks, minor=False)
        ax.set_xticklabels(xticks, minor=False, fontsize=6)
        ax.yaxis.tick_right()
        ax.set_ylabel('Input'.format(shortname),fontsize=8, color=plotcolor)
        ax.grid(True, linewidth=0.5, axis='x')
        ax.set_xlim(xlim)
        #ax.set_title(dataname)
        plt.subplots_adjust(bottom=0.24,top=0.95, left=0.04, right=0.94)

        save_figure('Figure-0-{}-{}-inputs'.format(shortname,subject), fmt=fmt, dpi=dpi)
        plt.close()


        #draw relevances 1
        maxamp = 0
        #fig = plt.figure(figsize=(6, 0.8))
        fig = plt.figure(figsize=(4, 0.8))
        ax = plt.subplot(111)
        rvals = relevance.T
        x = np.arange(rvals.shape[0])
        x, rvals = interpolate_all_samples(rvals, fold=50)
        nrel = rvals / np.max(np.abs(rvals))
        rel_colors = colormap(nrel)
        for j in range(rvals.shape[1]):
            rel = rvals[:,j]
            maxamp = max(maxamp, np.abs(rel).max())
            #draw foot contact shading
            ax.scatter(x=x,
                        y=rel,
                        c=rel_colors[:,j,:],
                        s=dotsize,
                        marker = '.',
                        edgecolor='none')


        maxamp *= 1.1
        ax.set_ylim([-maxamp, maxamp])
        ax.yaxis.tick_right()
        ax.set_yticks([0]) #show single vertical grid line thresholding zero
        ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)
        ax.set_xticks([50.5 + 101*ii for ii in range(6)],minor=True)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_xticklabels(bodyparts, minor=True, fontsize=5)
        ax.set_xticks(xticks, minor=False)
        ax.set_xticklabels(xticks, minor=False, fontsize=6)
        ax.yaxis.tick_right()
        ax.set_ylabel('Relevance'.format(shortname),fontsize=8, color=plotcolor)
        ax.grid(True, linewidth=0.5, axis='x')
        ax.set_xlim(xlim)
        #ax.set_title(dataname)
        plt.subplots_adjust(bottom=0.24,top=0.95, left=0.04, right=0.94)

        save_figure('Figure-0-{}-{}-Rel1'.format(shortname, subject), fmt=fmt, dpi=dpi)
        plt.close()


        #draw relevances 2
        maxamp = 1
        #fig = plt.figure(figsize=(6, 0.8))
        fig = plt.figure(figsize=(4, 0.8))
        ax = plt.subplot(111)
        rvals = relevance.T
        inpts = inputs.T
        x = np.arange(rvals.shape[0])
        x, rvals = interpolate_all_samples(rvals, fold=50)
        x, inpts = interpolate_all_samples(inpts, fold=50)
        nrel = rvals / np.max(np.abs(rvals))
        rel_colors = colormap(nrel)
        for j in range(rvals.shape[1]):
            rel = rvals[:,j]
            inp = inpts[:,j]
            #maxamp = max(maxamp, np.abs(rel).max())
            #draw foot contact shading
            ax.scatter(x=x,
                       y=inp,
                       c=rel_colors[:,j,:],
                       s=dotsize,
                       marker = '.',
                       edgecolor='none')

        #maxamp *= 1.1

        ax.set_ylim([-maxamp, maxamp])
        #ax2 = ax.twinx()

        ax.yaxis.tick_right()
        ax.set_yticks([0]) #show single vertical grid line thresholding zero
        ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)

        ax.set_xticks([50.5 + 101*ii for ii in range(6)],minor=True)
        #ax.set_xticks([],minor=True)
        ax.tick_params(axis='x', which='minor', length=0)

        ax.set_xticklabels(bodyparts, minor=True, fontsize=5)
        #ax.set_xticklabels([], minor=True, fontsize=6)

        ax.set_xticks(xticks, minor=False)
        #ax.set_xticks([], minor=False)
        ax.set_xticklabels(xticks, minor=False, fontsize=6)
        #ax.set_xticklabels([], minor=False, fontsize=7)
        #ax.tick_params(axis='x', which='major', length=0)

        ax.set_ylabel('Relevance'.format(shortname),fontsize=8, color=plotcolor)
        ax.grid(True, linewidth=0.5, axis='x')
        ax.set_xlim(xlim)

        #ax2.set_yticks([], minor=False)
        #ax2.set_ylabel('on Input'.format(shortname),fontsize=8, color=plotcolor)
        #ax.set_ylabel('Releva'.format(shortname),fontsize=8, color=plotcolor)
        #ax.set_title(dataname)
        plt.subplots_adjust(bottom=0.24,top=0.95, left=0.04, right=0.94)

        save_figure('Figure-0-{}-{}-Rel2'.format(shortname, subject), fmt=fmt, dpi=dpi)
        plt.close()

