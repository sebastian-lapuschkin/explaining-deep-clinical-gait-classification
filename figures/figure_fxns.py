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
        print 'creating output folder "{}"'.format(fmt)
        os.makedirs(fmt)

    #add corresponding author name to file name
    filename +=  ' (horst)'

    filepath = '{}/{}.{}'.format(fmt,filename, fmt)
    print 'writing figure to', filepath
    plt.savefig(filepath, dpi=dpi)

    size_extensions = ['B', 'K', 'M', 'G']
    size_in_bytes = os.path.getsize(filepath)
    divisions=0
    while size_in_bytes > 1024 and divisions < 3:
        size_in_bytes = size_in_bytes/1024.
        divisions += 1
    print '   (size: {}{})'.format(int(size_in_bytes), size_extensions[divisions])

    #apply compression, if it makes sense.
    #if fmt == 'tiff':
    #    print ' attempting lzw compression for tiff'
    #    image = PIL.Image.open(filepath)
    #    image.save(filepath, format=fmt, dpi=(dpi, dpi), compression="tiff_lzw")
    #
    #    size_in_bytes = os.path.getsize(filepath)
    #    divisions=0
    #    while size_in_bytes > 1024 and divisions < 3:
    #        size_in_bytes = size_in_bytes/1024.
    #        divisions += 1
    #    print '   (size: {}{})'.format(int(size_in_bytes), size_extensions[divisions])

    #elif fmt == 'pdf':
    #
    # TODO: deal with that later.
    #









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
    Yinterp = np.concatenate([np.interp(xinterp, x, Y[:,d])[:,None] for d in xrange(D)], axis=1)
    #print Y.shape, xinterp.shape, xinterp[-1], Yinterp.shape
    return xinterp, Yinterp



def draw_fig2(fmt, dpi=300, **kwargs):
    print 'drawing figure 2...'
    mincoord = 0
    maxcoord = 256
    x = np.linspace(-1,1, maxcoord)[None, ...]
    r = colormap(x)
    #r = np.repeat(r, 10, axis=0)  #standalone figure
    r = np.repeat(r, 15, axis=0)  #integrated into overview

    #print(plt.rcParams.get('figure.figsize'))
    #plt.figure(figsize=(6.4, 0.7)) #standalone figure
    plt.figure(figsize=(4, 0.7)) #integrated into overview
    ax = plt.subplot(111)
    ax.imshow(r, interpolation='none')

    ax.set_yticks([])
    ax.set_xticks([mincoord, maxcoord/2. , maxcoord])
    ax.tick_params(axis='x', which='major', length=0)
    #rc('text', usetex=True)
    #rc('font', family='serif')
    ax.set_xticklabels([r"$R \ll 0$", r"$R \approx 0$", r"$R \gg 0$"])#, fontsize=6)


    plt.title('Colour Spectrum for Relevance Visualisation')
    plt.tight_layout()
    save_figure('Figure-2', fmt=fmt, dpi=dpi)
    plt.close()
    #rc('text', usetex=False)




def draw_fig1(fmt, variant='a', dpi=300):
    print 'drawing figure 1...'
    print '   loading data'
    data = scipy.io.loadmat('Fig1.mat')
    mean = data['Acc_StepsOfChange_Output_Mean']   # 7 x 50
    sd = data['Acc_StepsOfChange_Output_SD']       # 7 x 50
    files = data['Acc_StepsOfChange_Output_Files'] # 7 x 1, string

    print '   sanitizing legend'
    legend = []
    colors = []
    linestyles = []
    #first, clean up label info and extract model names for the figure legend.
    for i in xrange(files.size):
        string = files[i][0][0][0]
        string = string.split('GRF_AV')[1].split('_ACC_')[0]

        if 'CNN-A' in string:
            legend.append(string)
            colors.append('red')
            linestyles.append('-')
        elif 'CNN-C3' in string:
            legend.append(string)
            colors.append('salmon')
            linestyles.append('--')
        elif '1024' in string:
            legend.append('MLP (3, 1024)')
            colors.append('blue')
            linestyles.append('-')
        elif '256' in string:
            legend.append('MLP (3, 256)')
            colors.append('cornflowerblue')
            linestyles.append('--')
        elif '64' in string:
            legend.append('MLP (3, 64)')
            colors.append('lightsteelblue')
            linestyles.append(':')
        elif 'SVM' in string:
            legend.append('Linear (SVM)')
            colors.append('forestgreen')
            linestyles.append('-')
        else:
            legend.append('Linear (SGD)')
            colors.append('lightgreen')
            linestyles.append('--')
    print '   plotting'



    #plt.figure(figsize=(4,3))
    x = np.arange(mean.shape[1])
    for i in [6, 5, 2, 1, 0, 3, 4]: #manual plotting order so legend is nice
        #plot mean
        p = plt.plot(x, mean[i], label=legend[i], linewidth=2, color=colors[i], linestyle=linestyles[i])
        color = p[0].get_color()

        #plot error bars (or not)
        plt.fill_between(x,
                         y1=mean[i]-sd[i],
                         y2=mean[i]+sd[i],
                         edgecolor='None',
                         facecolor=color,
                         alpha=0.0,         #deactivate. too much stdev
                         linestyle='-')

    plt.legend()
    plt.ylim([0, 1])
    plt.xlim([0, 50])
    plt.ylabel('prediction accuracy')
    plt.xlabel('% of perturbed inputs')
    plt.tight_layout()

    save_figure('Figure-1', fmt=fmt, dpi=dpi)
    plt.close()




def draw_fig6(fmt ,variant='a',dpi=300):
    print 'drawing figure 6..'
    print '   loading data'
    data = scipy.io.loadmat('Fig2.mat')
    inputs =  data['Feature_Output_Export'][0] #7 x (606 x 19)
    relevance =  data['RpredAct_Output_Export'][0] #7 x (606 x 19)
    files =  data['Files_Export'][0] # 7, string

    #print files
    #print inputs.shape

    #assert that all inputs are the same for the initial index
    for i in xrange(1,inputs.size):
        np.testing.assert_array_equal(inputs[0], inputs[i])

    #order relevance data and labels
    rdata = []
    labels = []
    for i in xrange(len(files)):
        if 'Linear.mat' in files[i][0]: rdata.append(relevance[i]) ; labels.append('Linear\n(SGD)') ; break
    for i in xrange(len(files)):
        if 'LinearSVM' in files[i][0]: rdata.append(relevance[i]) ; labels.append('Linear\n(SVM)') ; break
    for i in xrange(len(files)):
        if '64' in files[i][0]: rdata.append(relevance[i]) ; labels.append('MLP\n(3, 64)') ; break
    for i in xrange(len(files)):
        if '256' in files[i][0]: rdata.append(relevance[i]) ; labels.append('MLP\n(3, 256)') ; break
    for i in xrange(len(files)):
        if '1024' in files[i][0]: rdata.append(relevance[i]) ; labels.append('MLP\n(3, 1024)') ; break
    for i in xrange(len(files)):
        if 'CNN-A' in files[i][0]: rdata.append(relevance[i]) ; labels.append('CNN-A') ; break
    for i in xrange(len(files)):
        if 'CNN-C3' in files[i][0]: rdata.append(relevance[i]) ; labels.append('CNN-C3') ; break


    fig = plt.figure()
    gs = gridspec.GridSpec(len(labels)+1,1)

    xticks = np.arange(7)*101
    xlim = xticks[[0,-1]]

    #draw inputs
    maxamp = 1
    ax = plt.subplot(gs[0,:])
    for i in xrange(inputs[0].shape[1]):
        values = inputs[0][:,i]
        #markersize < 1 transforms dots to circles. stupid.
        ax.plot(values,
                color='blue',
                linestyle='-',
                linewidth = 0.2)
    ax.set_ylim([-maxamp, maxamp])
    ax.set_yticks([-maxamp, 0, maxamp])
    ticklabs = [str(s) for s in[-maxamp, 0, maxamp]]
    ticklabs = [' '*(2-len(s)) + s for s in ticklabs]
    ax.set_yticklabels(ticklabs)
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.yaxis.tick_right()
    ax.set_ylabel('Input',fontsize=7)
    ax.grid(True, linewidth=0.5, axis='x')
    ax.set_xlim(xlim)

    ax.set_title('Ground Reaction Force - Subject 28:\nInput and Relevance Per Model, All Samples')

    #loop over models and draw relevance values
    for i in xrange(len(labels)):
        ax = plt.subplot(gs[i+1,:])
        rvals = rdata[i]
        maxamp = 0

        x = np.arange(rvals.shape[0])
        if variant in ['b','c']:
            x, rvals = interpolate_all_samples(rvals, fold=50)
            nrel = rvals / np.max(np.abs(rvals))
            rel_colors = colormap(nrel)

        for j in xrange(rvals.shape[1]):
            rel = rvals[:,j]
            maxamp = max(maxamp, np.abs(rel).max())

            if variant == 'a':
                ax.plot(x,
                        rel,
                        color='k',
                        linestyle='-',
                        linewidth=0.2,
                )
            elif variant in ['b', 'c']:
                ax.scatter(x=x,
                        y=rel,
                        c=rel_colors[:,j,:],
                        s=0.3,
                        marker = '.',
                        edgecolor='none')
            else:
                print 'unknown figure variant {} . aborting'.format(variant)
                exit(-1)


            #mark max relevance peak by location
            maxrelpos = np.argmax(rel)
            #print maxrelpos, rel[maxrelpos]
            #ax.scatter(x[maxrelpos], rel[maxrelpos],
            #           marker = 'o',
            #           c = 'none',
            #           edgecolor='r',
            #           linewidth=0.5)

            if variant == 'a':
                variantcolor='r'
                ax.plot(x[maxrelpos], rel[maxrelpos], 'o', fillstyle='none', color=variantcolor)
            elif variant == 'b':
                ax.plot(x[maxrelpos], rel[maxrelpos], 'o', fillstyle='none', color=rel_colors[maxrelpos,j,:])

                    #markersize=3)


            #break # debug
        #break #debug

        #print maxamp
        maxamp *= 1.1
        ax.set_ylim([-maxamp, maxamp])
        ax.set_xticks(xticks)
        #print xticks
        if i < len(labels)-1:
            ax.set_xticklabels([]) #deactivate tick labels for all but the lowest row
        ax.yaxis.tick_right()
        ax.set_yticks([0]) #show single vertical grid line thresholding zero
        ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)#show
        ax.set_ylabel(labels[i],fontsize=7)
        ax.grid(True, linewidth=0.5, axis='x')

        if i == len(labels)-1:
            #use minor tick labels as x axis legend
            ax.set_xticks([50.5 + 101*i for i in range(6)],minor=True)
            ax.tick_params(axis='x', which='minor', length=0)
            ax.set_xticklabels(['right fore-aft', 'right med-lat', 'right vert', 'left fore-aft', 'left med-lat', 'left vert'], minor=True, fontsize=6)



        ax.set_axisbelow(True) # grid lines behind all other graph elements
        ax.set_xlim(xlim)


    if variant == 'a':
        variant = ''

    save_figure('Figure-6{}'.format(variant), fmt=fmt, dpi=dpi)
    plt.close()






def draw_fig5(fmt,variant='a', dpi=300):
    print 'drawing figure 5..'
    print '   loading data'
    data = scipy.io.loadmat('Fig3.mat')
    mean_inputs =  data['Feature_Output_Mean'][0] # 4 x (1 x 606)
    mean_relevance =  data['Rpred_Output_Mean'][0] # 4 x (1 x 606)
    files =  data['files'][0] # 4, string

    #order relevance data and labels
    rdata = []
    inputs = []
    labels = []
    for i in xrange(len(files)):
        if 'Linear.mat' in files[i][0]: rdata.append(mean_relevance[i]) ; inputs.append(mean_inputs[i]); labels.append('Linear\n(SGD)') ; break
    for i in xrange(len(files)):
        if 'LinearSVM' in files[i][0]: rdata.append(mean_relevance[i]) ; inputs.append(mean_inputs[i]); labels.append('Linear\n(SVM)') ; break
    for i in xrange(len(files)):
        if '64' in files[i][0]: rdata.append(mean_relevance[i]) ; inputs.append(mean_inputs[i]); labels.append('MLP\n(3, 64)') ; break
    for i in xrange(len(files)):
        if '256' in files[i][0]: rdata.append(mean_relevance[i]) ; inputs.append(mean_inputs[i]); labels.append('MLP\n(3, 256)') ; break
    for i in xrange(len(files)):
        if '1024' in files[i][0]: rdata.append(mean_relevance[i]) ; inputs.append(mean_inputs[i]); labels.append('MLP\n(3, 1024)') ; break
    for i in xrange(len(files)):
        if 'CNN-A' in files[i][0]: rdata.append(mean_relevance[i]) ; inputs.append(mean_inputs[i]); labels.append('CNN-A') ; break
    for i in xrange(len(files)):
        if 'CNN-C3' in files[i][0]: rdata.append(mean_relevance[i]) ; inputs.append(mean_inputs[i]); labels.append('CNN-C3') ; break


    fig = plt.figure()
    gs = gridspec.GridSpec(len(labels),1)

    xticks = np.arange(7)*101
    xlim = xticks[[0,-1]]

    maxamp = 1
    #loop over models and draw relevance values
    for i in xrange(len(labels)):
        ax = plt.subplot(gs[i,:])
        vals = inputs[i]
        rels = rdata[i]
        x = np.arange(vals.size)

        #interpolate measurements and colorize
        x, rels = interpolate_all_samples(rels.T, fold=100)
        _, vals = interpolate_all_samples(vals.T, fold=100)
        nrel = rels / np.max(np.abs(rels))
        rel_colors = colormap(nrel)[:,0,:]

        vals = vals.flatten()
        x = x.flatten()


        ax.scatter(x=x,
                y=vals,
                c=rel_colors,
                s=5*2,
                marker = '.',
                edgecolor='none')

        if i == 0:
            ax.set_title('Ground Reaction Force - Subject 57:\nRelevance of Input Per Model')


        ax.set_ylim([-maxamp, maxamp])
        ax.set_xticks(xticks)
        #print xticks
        if i < len(labels)-1:
            ax.set_xticklabels([]) #deactivate tick labels for all but the lowest row
        if i == 0:
            ax.set_yticks([-maxamp, 0, maxamp])
            ticklabs = [str(s) for s in[-maxamp, 0, maxamp]]
            ticklabs = [' '*(2-len(s)) + s for s in ticklabs]
            ax.set_yticklabels(ticklabs)
        else:
            ax.set_yticks([0]) #show single vertical grid line thresholding zero
            ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)#show
        ax.yaxis.tick_right()
        ax.set_ylabel(labels[i],fontsize=7)
        ax.grid(True, linewidth=0.5, axis='x')
        #ax.spines['top'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True) # grid lines behind all other graph elements
        ax.set_xlim(xlim)


        maxrelpos = [np.argmax(rels[:rels.size/2]), rels.size/2 + np.argmax(rels[rels.size/2:]) ]#find left and right body side symmetry
        if variant == 'a':
            variantcolor='r'
            ax.plot(x[maxrelpos], vals[maxrelpos], 'o', fillstyle='none', color=variantcolor)
        elif variant == 'b':
            ax.plot(x[maxrelpos], vals[maxrelpos], 'o', fillstyle='none', color=rel_colors[maxrelpos,:])

        if i == len(labels)-1:
            #use minor tick labels as x axis legend
            ax.set_xticks([50.5 + 101*i for i in range(6)],minor=True)
            ax.tick_params(axis='x', which='minor', length=0)
            ax.set_xticklabels(['right fore-aft', 'right med-lat', 'right vert', 'left fore-aft', 'left med-lat', 'left vert'], minor=True, fontsize=6)



    save_figure('Figure-5', fmt=fmt, dpi=dpi)
    plt.close()




def draw_fig3(fmt,variant='a',dpi=300):
    #this one is a bit hardcodey
    print 'drawing figure 3..'
    print '   loading data'
    data = scipy.io.loadmat('Fig4.mat')

    inputs =  data['Feature_Output_Mean'][0][0]# 7 x 606
    relevance =  data['Rpred_Output_Mean'][0][0] # 7 x 606
    files =  data['files'][0][0][0] # string

    #order relevance data and labels
    subjectlabels = [21, 26, 28, 39, 42, 55, 57]
    model = 'CNN-A'
    data = 'Ground Reaction Force'

    #delete and ignore subject 26
    subjectindex = 1
    del subjectlabels[subjectindex]
    inputs = np.delete(inputs, subjectindex, 0)
    relevance = np.delete(relevance, subjectindex, 0)

    fig = plt.figure()
    gs = gridspec.GridSpec(inputs.shape[0],1)

    xticks = np.arange(7)*101
    xlim = xticks[[0,-1]]

    maxamp = 1
    #loop over models and draw relevance values
    for i in xrange(len(subjectlabels)):
        ax = plt.subplot(gs[i,:])
        vals = inputs[i:i+1]
        rels = relevance[i:i+1]
        x = np.arange(vals.size)

        #interpolate measurements and colorize
        x, rels = interpolate_all_samples(rels.T, fold=100)
        _, vals = interpolate_all_samples(vals.T, fold=100)
        nrel = rels / np.max(np.abs(rels))
        rel_colors = colormap(nrel)[:,0,:]

        vals = vals.flatten()
        x = x.flatten()


        ax.scatter(x=x,
                y=vals,
                c=rel_colors,
                s=5*2,
                marker = '.',
                edgecolor='none')

        if i == 0:
            ax.set_title('{} - {}:\nRelevance of Input Per Subject'.format(data, model))


        ax.set_ylim([-maxamp, maxamp])
        ax.set_xticks(xticks)
        #print xticks
        if i < len(subjectlabels)-1:
            ax.set_xticklabels([]) #deactivate tick labels for all but the lowest row
        if i == 0:
            ax.set_yticks([-maxamp, 0, maxamp])
            ticklabs = [str(s) for s in[-maxamp, 0, maxamp]]
            ticklabs = [' '*(2-len(s)) + s for s in ticklabs]
            ax.set_yticklabels(ticklabs)
        else:
            ax.set_yticks([0]) #show single vertical grid line thresholding zero
            ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)#show

        ax.yaxis.tick_right()
        ax.set_ylabel('Subj. {}'.format(subjectlabels[i]),fontsize=7)
        ax.grid(True, linewidth=0.5, axis='x')
        #ax.spines['top'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True) # grid lines behind all other graph elements
        ax.set_xlim(xlim)

        if i == len(subjectlabels)-1:
            #use minor tick labels as x axis legend
            ax.set_xticks([50.5 + 101*i for i in range(6)],minor=True)
            ax.tick_params(axis='x', which='minor', length=0)
            ax.set_xticklabels(['right fore-aft', 'right med-lat', 'right vert', 'left fore-aft', 'left med-lat', 'left foot vert'], minor=True, fontsize=6)


        maxrelpos = [np.argmax(rels[:rels.size/2]), rels.size/2 + np.argmax(rels[rels.size/2:]) ]#find left and right body side symmetry
        if variant == 'a':
            variantcolor='r'
            ax.plot(x[maxrelpos], vals[maxrelpos], 'o', fillstyle='none', color=variantcolor)
        elif variant == 'b':
            ax.plot(x[maxrelpos], vals[maxrelpos], 'o', fillstyle='none', color=rel_colors[maxrelpos,:])



    save_figure('Figure-3', fmt=fmt, dpi=dpi)
    plt.close()


def draw_fig3_alt(fmt,variant='a',dpi=300):
    #this one is a bit hardcodey
    print 'drawing figure 3..'
    print '   loading data'
    data = scipy.io.loadmat('Fig4_2017-10-05-S1234SubjectGRF_AVCNN-A.mat')

    inputs =  data['Feature_Output'][0]# 57, 20 x 606
    relevance =  data['RpredAct_Output'][0] # 57, 20 x 606

    #order relevance data and labels
    subjectlabels = [21, 26, 28, 39, 42, 55, 57]
    subjectindices = [l-1 for l in subjectlabels]
    model = 'CNN-A'
    data = 'Ground Reaction Force'

    #get data.
    inputs = inputs[subjectindices]
    relevance = relevance[subjectindices]


    #delete and ignore subject 26
    subjectindex = 1
    del subjectlabels[subjectindex]
    inputs = np.delete(inputs, subjectindex, 0)
    relevance = np.delete(relevance, subjectindex, 0)

    fig = plt.figure()
    gs = gridspec.GridSpec(inputs.shape[0],1)

    xticks = np.arange(7)*101
    xlim = xticks[[0,-1]]

    maxamp = 1
    #loop over models and draw relevance values
    for i in xrange(len(subjectlabels)):
        ax = plt.subplot(gs[i,:])
        vals = inputs[i] # 20 x 606
        rels = relevance[i] # 20 x 606
        #x = np.arange(vals.size)

        #interpolate measurements and colorize
        x, rels = interpolate_all_samples(rels.T, fold=50)
        _, vals = interpolate_all_samples(vals.T, fold=50)

        x = np.repeat(x[...,None] ,vals.shape[1], axis=1).flatten()
        vals = vals.flatten()
        rels = rels


        nrel = rels / np.max(np.abs(rels))
        rel_colors = np.reshape(colormap(nrel) , [-1, 3])


        #optimize z-order for drawing: highlight relevances with high magnitudes
        print rel_colors.shape, x.shape, vals.shape
        I = np.argsort(np.abs(nrel.flatten()))
        vals = vals[I]
        x=x[I]
        rel_colors=rel_colors[I,:]
        print rel_colors.shape, x.shape, vals.shape


        ax.scatter(x=x,
                y=vals,
                c=rel_colors,
                s=0.5,
                marker = '.',
                edgecolor='none')

        if i == 0:
            ax.set_title('{} - {}:\nRelevance of Input Per Subject'.format(data, model))


        ax.set_ylim([-maxamp, maxamp])
        ax.set_xticks(xticks)
        #print xticks
        if i < len(subjectlabels)-1:
            ax.set_xticklabels([]) #deactivate tick labels for all but the lowest row
        if i == 0:
            ax.set_yticks([-maxamp, 0, maxamp])
            ticklabs = [str(s) for s in[-maxamp, 0, maxamp]]
            ticklabs = [' '*(2-len(s)) + s for s in ticklabs]
            ax.set_yticklabels(ticklabs)
        else:
            ax.set_yticks([0]) #show single vertical grid line thresholding zero
            ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)#show

        ax.yaxis.tick_right()
        ax.set_ylabel('Subj. {}'.format(subjectlabels[i]),fontsize=7)
        ax.grid(True, linewidth=0.5, axis='x')
        #ax.spines['top'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True) # grid lines behind all other graph elements
        ax.set_xlim(xlim)

        if i == len(subjectlabels)-1:
            #use minor tick labels as x axis legend
            ax.set_xticks([50.5 + 101*i for i in range(6)],minor=True)
            ax.tick_params(axis='x', which='minor', length=0)
            ax.set_xticklabels(['right foot x', 'right foot y', 'right foot z', 'left foot x', 'left foot y', 'left foot z'], minor=True, fontsize=6)


        #maxrelpos = [np.argmax(rels[:rels.size/2]), rels.size/2 + np.argmax(rels[rels.size/2:]) ]#find left and right body side symmetry
        #if variant == 'a':
        #    variantcolor='r'
        #    ax.plot(x[maxrelpos], vals[maxrelpos], 'o', fillstyle='none', color=variantcolor)
        #elif variant == 'b':
        #    ax.plot(x[maxrelpos], vals[maxrelpos], 'o', fillstyle='none', color=rel_colors[maxrelpos,:])



    save_figure('Figure-3-alt', fmt=fmt, dpi=dpi)
    plt.close()


def draw_fig4(fmt,variant='a',dpi=300):
    print 'drawing figure 4..'
    print '   loading data'
    data = scipy.io.loadmat('Fig5.mat')

    inputs =  data['Feature_Output_Mean'][0][0]# 7 x 606
    relevance =  data['Rpred_Output_Mean'][0][0] # 7 x 606
    files =  data['files'][0][0][0] # string

    #order relevance data and labels
    subjectlabels = [6, 18, 23, 32, 37, 47, 55]
    model = 'CNN-A'
    data = 'Lower-Body Joint Angles (flexion-extension)'

    #delete and ignore subject 18
    subjectindex = 1
    del subjectlabels[subjectindex]
    inputs = np.delete(inputs, subjectindex, 0)
    relevance = np.delete(relevance, subjectindex, 0)

    fig = plt.figure()
    gs = gridspec.GridSpec(inputs.shape[0],1)

    xticks = np.arange(7)*101
    xlim = xticks[[0,-1]]


    #['left ankle x', 'left knee x', 'left hip x', 'right ankle x', 'right knee x', 'right hip x']
    #
    #also der rechte bodenkontakt ist fuer subject 6
    #Fabian Horst: von Frame 1:54 und 94:101
    #Fabian Horst: der linke ist von Frame 1:7 und 48:101
    rightfootshading = [(.5,53), (93,99.5)]
    leftfootshading = [(.5,6),(47,99.5)]
    tmp = [leftfootshading]*3 + [rightfootshading]*3
    footongroundshading = []
    #relative to absolute coordinates
    for i in xrange(len(tmp)):
        blockfeatures = tmp[i]
        for b in blockfeatures:
            footongroundshading.append((b[0]+(i*101), b[1]+(i*101)))
    print footongroundshading

    maxamp = 1
    #loop over models and draw relevance values
    for i in xrange(len(subjectlabels)):
        ax = plt.subplot(gs[i,:])
        vals = inputs[i:i+1] # 20 x 660
        rels = relevance[i:i+1] # 20 x 606
        x = np.arange(vals.size)

        #interpolate measurements and colorize
        x, rels = interpolate_all_samples(rels.T, fold=100)
        _, vals = interpolate_all_samples(vals.T, fold=100)
        nrel = rels / np.max(np.abs(rels))
        rel_colors = colormap(nrel)[:,0,:]

        vals = vals.flatten()
        x = x.flatten()

        #draw foot contact shading
        for fs in footongroundshading:
            ax.fill_between(np.arange(fs[0],fs[1]), maxamp, -maxamp, color=[0.9]*3,  zorder=-1)

        #draw relevance values
        ax.scatter(x=x,
                y=vals,
                c=rel_colors,
                s=5*2,
                marker = '.',
                edgecolor='none')

        if i == 0:
            ax.set_title('{} - {}:\nRelevance of Input Per Subject'.format(data, model))


        ax.set_ylim([-maxamp, maxamp])
        ax.set_xticks(xticks)
        #print xticks
        if i < len(subjectlabels)-1:
            ax.set_xticklabels([]) #deactivate tick labels for all but the lowest row
        if i == 0:
            ax.set_yticks([-maxamp, 0, maxamp])
            ticklabs = [str(s) for s in[-maxamp, 0, maxamp]]
            ticklabs = [' '*(2-len(s)) + s for s in ticklabs]
            ax.set_yticklabels(ticklabs)
            #ax.get_yaxis().set_tick_coords(606,1)
        else:
            ax.set_yticks([0]) #show single vertical grid line thresholding zero
            ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)#show

        ax.yaxis.tick_right()

        ax.set_ylabel('Subj. {}'.format(subjectlabels[i]),fontsize=7)
        ax.grid(True, linewidth=0.5, axis='x')
        #ax.spines['top'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True) # grid lines behind all other graph elements
        ax.set_xlim(xlim)

        if i == len(subjectlabels)-1:
            #use minor tick labels as x axis legend
            ax.set_xticks([50.5 + 101*i for i in range(6)],minor=True)
            ax.tick_params(axis='x', which='minor', length=0)
            ax.set_xticklabels(['left ankle', 'left knee', 'left hip', 'right ankle', 'right knee', 'right hip'], minor=True, fontsize=6)

        maxrelpos = [np.argmax(rels[:rels.size/2]), rels.size/2 + np.argmax(rels[rels.size/2:]) ]#find left and right body side symmetry
        if variant == 'a':
            variantcolor='r'
            ax.plot(x[maxrelpos], vals[maxrelpos], 'o', fillstyle='none', color=variantcolor)
        elif variant == 'b':
            ax.plot(x[maxrelpos], vals[maxrelpos], 'o', fillstyle='none', color=rel_colors[maxrelpos,:])


    save_figure('Figure-4', fmt=fmt, dpi=dpi)
    plt.close()







def draw_fig4_alt(fmt,variant='a',dpi=300):
    print 'drawing figure 4..'
    print '   loading data'
    data = scipy.io.loadmat('Fig5_2017-10-05-S1234SubjectJA_X_LowerCNN-A.mat')

    inputs =  data['Feature_Output'][0]# 57, 20 x 606
    relevance =  data['RpredAct_Output'][0] # 57, 20 x 606

    #order relevance data and labels
    subjectlabels = [6, 18, 23, 32, 37, 47, 55]
    subjectindices = [l-1 for l in subjectlabels]
    model = 'CNN-A'
    data = 'Lower-Body Joint Angles'

    #get data.
    inputs = inputs[subjectindices]
    relevance = relevance[subjectindices]


    #delete and ignore subject 18
    subjectindex = 1
    del subjectlabels[subjectindex]
    inputs = np.delete(inputs, subjectindex, 0)
    relevance = np.delete(relevance, subjectindex, 0)

    fig = plt.figure()
    gs = gridspec.GridSpec(inputs.shape[0],1)

    xticks = np.arange(7)*101
    xlim = xticks[[0,-1]]

    maxamp = 1
    #loop over models and draw relevance values
    for i in xrange(len(subjectlabels)):
        ax = plt.subplot(gs[i,:])
        vals = inputs[i] # 20 x 660
        rels = relevance[i] # 20 x 606
        x = np.arange(vals.size)

        #interpolate measurements and colorize
        x, rels = interpolate_all_samples(rels.T, fold=50)
        _, vals = interpolate_all_samples(vals.T, fold=50)

        x = np.repeat(x[...,None] ,vals.shape[1], axis=1).flatten()
        vals = vals.flatten()
        rels = rels


        nrel = rels / np.max(np.abs(rels))
        rel_colors = np.reshape(colormap(nrel) , [-1, 3])


        #optimize z-order for drawing: highlight relevances with high magnitudes
        print rel_colors.shape, x.shape, vals.shape
        I = np.argsort(np.abs(nrel.flatten()))
        vals = vals[I]
        x=x[I]
        rel_colors=rel_colors[I,:]
        print rel_colors.shape, x.shape, vals.shape


        ax.scatter(x=x,
                y=vals,
                c=rel_colors,
                s=0.5,
                marker = '.',
                edgecolor='none')

        if i == 0:
            ax.set_title('{} - {}:\nRelevance of Input Per Subject'.format(data, model))


        ax.set_ylim([-maxamp, maxamp])
        ax.set_xticks(xticks)
        #print xticks
        if i < len(subjectlabels)-1:
            ax.set_xticklabels([]) #deactivate tick labels for all but the lowest row
        if i == 0:
            ax.set_yticks([-maxamp, 0, maxamp])
            ticklabs = [str(s) for s in[-maxamp, 0, maxamp]]
            ticklabs = [' '*(2-len(s)) + s for s in ticklabs]
            ax.set_yticklabels(ticklabs)
            #ax.get_yaxis().set_tick_coords(606,1)
        else:
            ax.set_yticks([0]) #show single vertical grid line thresholding zero
            ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)#show

        ax.yaxis.tick_right()

        ax.set_ylabel('Subj. {}'.format(subjectlabels[i]),fontsize=7)
        ax.grid(True, linewidth=0.5, axis='x')
        #ax.spines['top'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True) # grid lines behind all other graph elements
        ax.set_xlim(xlim)

        if i == len(subjectlabels)-1:
            #use minor tick labels as x axis legend
            ax.set_xticks([50.5 + 101*i for i in range(6)],minor=True)
            ax.tick_params(axis='x', which='minor', length=0)
            ax.set_xticklabels(['left ankle x', 'left knee x', 'left hip x', 'right ankle x', 'right knee x', 'right hip x'], minor=True, fontsize=6)

        #maxrelpos = [np.argmax(rels[:rels.size/2]), rels.size/2 + np.argmax(rels[rels.size/2:]) ]#find left and right body side symmetry
        #if variant == 'a':
        #    variantcolor='r'
        #    ax.plot(x[maxrelpos], vals[maxrelpos], 'o', fillstyle='none', color=variantcolor)
        #elif variant == 'b':
        #    ax.plot(x[maxrelpos], vals[maxrelpos], 'o', fillstyle='none', color=rel_colors[maxrelpos,:])


    save_figure('Figure-4-alt', fmt=fmt, dpi=dpi)
    plt.close()







def draw_fig0 (fmt, variant='a',dpi=300):
    #this one is a bit hardcodey
    print 'drawing figure(s) 0...'
    print '   loading data'
    print '   variant', variant

    models = 'CNN-A'
    datanames = ['Ground Reaction Force', 'Lower-Body Joint Angles']
    shortnames = ['GRF', 'LBJAX']
    datas = ['P6.GRF_AV.CNN-A.mat', 'P6.JA_X_Lower.CNN-A.mat']

    for d in xrange(len(datanames)):
        dataname = datanames[d]
        shortname = shortnames[d]
        data = scipy.io.loadmat(datas[d])



        if shortname == 'GRF':
            drawshading=False
            bodyparts = []
            for f in ['right', 'left']:
                for xyz in ['fore-aft', 'med-lat', 'vert']:
                    #bodyparts.append('{} foot {}'.format(f, xyz))
                    bodyparts.append('{} {}'.format(f, xyz))
        elif shortname  == 'LBJAX':
            drawshading=True
            bodyparts = []
            for lr in ['left', 'right']:
                for akn in ['ankle', 'knee', 'hip']:
                    bodyparts.append('{} {}'.format(lr, akn))

            #['left ankle x', 'left knee x', 'left hip x', 'right ankle x', 'right knee x', 'right hip x']
            #
            #also der rechte bodenkontakt ist fuer subject 6
            #Fabian Horst: von Frame 1:54 und 94:101
            #Fabian Horst: der linke ist von Frame 1:7 und 48:101
            rightfootshading = [(.5,53), (93,99.5)]
            leftfootshading = [(.5,6),(47,99.5)]
            tmp = [leftfootshading]*3 + [rightfootshading]*3
            footongroundshading = []
            #relative to absolute coordinates
            for i in xrange(len(tmp)):
                blockfeatures = tmp[i]
                for b in blockfeatures:
                    footongroundshading.append((b[0]+(i*101), b[1]+(i*101)))
            print footongroundshading


        inputs = data['Feature_Output'][0][5]       # 21 x 606
        ypred = data['Ypred_Output'][0][5][:,5]     # 21 x 1
        #relevance = data['RpredDom_Output'][0][5]   # 21 x 606
        relevance = data['RpredAct_Output'][0][5]   # 21 x 606


        pred = np.argmax(data['Ypred_Output'][0][5], axis=1)
        tprate = np.mean(pred == 5)

        print  5, shortname, pred, tprate

        xticks = np.arange(7)*101
        xlim = xticks[[0,-1]]



        #draw inputs
        maxamp = 1
        fig = plt.figure(figsize=(6, 0.8))
        ax = plt.subplot(111)
        #draw foot contact shading
        if drawshading:
            for fs in footongroundshading:
                ax.fill_between(np.arange(fs[0],fs[1]), maxamp, -maxamp, color=[0.9]*3,  zorder=-1)
        for i in xrange(inputs.shape[0]):
            values = inputs[i]
            #markersize < 1 transforms dots to circles. stupid.
            plotcolor = 'blue' if shortname == 'GRF' else 'green'
            ax.plot(values, color=plotcolor, linestyle='-', linewidth = 0.2)

        ax.set_ylim([-maxamp, maxamp])
        ax.set_yticks([-maxamp, 0, maxamp])
        ticklabs = [str(s) for s in[-maxamp, 0, maxamp]]
        ticklabs = [' '*(2-len(s)) + s for s in ticklabs]
        ax.set_yticklabels(ticklabs)
        ax.set_xticks([50.5 + 101*ii for ii in range(6)],minor=True)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_xticklabels(bodyparts, minor=True, fontsize=6)
        ax.set_xticks(xticks, minor=False)
        ax.set_xticklabels(xticks, minor=False, fontsize=7)
        ax.yaxis.tick_right()
        ax.set_ylabel('{}\nInput'.format(shortname),fontsize=8, color=plotcolor)
        ax.grid(True, linewidth=0.5, axis='x')
        ax.set_xlim(xlim)
        #ax.set_title(dataname)
        plt.subplots_adjust(bottom=0.24,top=0.95, left=0.05, right=0.95)

        save_figure('Figure-0-{}-inputs'.format(shortname), fmt=fmt, dpi=dpi)
        plt.close()

        #draw relevances 1
        maxamp = 0
        fig = plt.figure(figsize=(6, 0.8))
        ax = plt.subplot(111)
        rvals = relevance.T
        x = np.arange(rvals.shape[0])
        x, rvals = interpolate_all_samples(rvals, fold=50)
        nrel = rvals / np.max(np.abs(rvals))
        rel_colors = colormap(nrel)
        for j in xrange(rvals.shape[1]):
            rel = rvals[:,j]
            maxamp = max(maxamp, np.abs(rel).max())
            #draw foot contact shading
            if drawshading:
                for fs in footongroundshading:
                    ax.fill_between(np.arange(fs[0],fs[1]), maxamp, -maxamp, color=[0.9]*3,  zorder=-1)
            ax.scatter(x=x,
                       y=rel,
                       c=rel_colors[:,j,:],
                       s=0.3,
                       marker = '.',
                       edgecolor='none')
        maxamp *= 1.1
        ax.set_ylim([-maxamp, maxamp])
        ax.yaxis.tick_right()
        ax.set_yticks([0]) #show single vertical grid line thresholding zero
        ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)
        ax.set_xticks([50.5 + 101*ii for ii in range(6)],minor=True)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_xticklabels(bodyparts, minor=True, fontsize=6)
        ax.set_xticks(xticks, minor=False)
        ax.set_xticklabels(xticks, minor=False, fontsize=7)
        ax.yaxis.tick_right()
        ax.set_ylabel('{}\nRelevance'.format(shortname),fontsize=8, color=plotcolor)
        ax.grid(True, linewidth=0.5, axis='x')
        ax.set_xlim(xlim)
        #ax.set_title(dataname)
        plt.subplots_adjust(bottom=0.24,top=0.95, left=0.05, right=0.95)

        save_figure('Figure-0-{}-Rel1'.format(shortname), fmt=fmt, dpi=dpi)
        plt.close()


        #draw relevances 2
        maxamp = 1
        fig = plt.figure(figsize=(6, 0.8))
        ax = plt.subplot(111)
        rvals = relevance.T
        inpts = inputs.T
        x = np.arange(rvals.shape[0])
        x, rvals = interpolate_all_samples(rvals, fold=50)
        x, inpts = interpolate_all_samples(inpts, fold=50)
        nrel = rvals / np.max(np.abs(rvals))
        rel_colors = colormap(nrel)
        for j in xrange(rvals.shape[1]):
            rel = rvals[:,j]
            inp = inpts[:,j]
            #maxamp = max(maxamp, np.abs(rel).max())
            #draw foot contact shading
            if drawshading:
                for fs in footongroundshading:
                    ax.fill_between(np.arange(fs[0],fs[1]), maxamp, -maxamp, color=[0.9]*3,  zorder=-1)
            ax.scatter(x=x,
                       y=inp,
                       c=rel_colors[:,j,:],
                       s=0.3,
                       marker = '.',
                       edgecolor='none')
        #maxamp *= 1.1
        ax.set_ylim([-maxamp, maxamp])
        ax.yaxis.tick_right()
        ax.set_yticks([0]) #show single vertical grid line thresholding zero
        ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)
        ax.set_xticks([50.5 + 101*ii for ii in range(6)],minor=True)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_xticklabels(bodyparts, minor=True, fontsize=6)
        ax.set_xticks(xticks, minor=False)
        ax.set_xticklabels(xticks, minor=False, fontsize=7)
        ax.yaxis.tick_right()
        ax.set_ylabel('{}\nRelevance'.format(shortname),fontsize=8, color=plotcolor)
        ax.grid(True, linewidth=0.5, axis='x')
        ax.set_xlim(xlim)
        #ax.set_title(dataname)
        plt.subplots_adjust(bottom=0.24,top=0.95, left=0.05, right=0.95)

        save_figure('Figure-0-{}-Rel2'.format(shortname), fmt=fmt, dpi=dpi)
        plt.close()

    return # ADDED THAT RETURN TO SAVE TIME AND NOT COMPUTE IRRELEVANT STUFF




    #### COPY-PASTED STUFF FROM OTHER FXN BELOW


    reps = range(inputs.shape[0])
    fig = plt.figure()
    gs = gridspec.GridSpec(inputs.shape[0],1)

    xticks = np.arange(7)*101
    xlim = xticks[[0,-1]]


    maxamp = 1
    #loop over models and draw relevance values
    for i in xrange(len(reps)):
        ax = plt.subplot(gs[i,:])
        vals = inputs[i:i+1]
        rels = relevance[i:i+1]
        x = np.arange(vals.size)

        #interpolate measurements and colorize
        x, rels = interpolate_all_samples(rels.T, fold=100)
        _, vals = interpolate_all_samples(vals.T, fold=100)
        nrel = rels / np.max(np.abs(rels))
        rel_colors = colormap(nrel)[:,0,:]

        vals = vals.flatten()
        x = x.flatten()


        ax.scatter(x=x,
                y=vals,
                c=rel_colors,
                s=5*2,
                marker = '.',
                edgecolor='none')

        if i == 0:
            ax.set_title('{} - {}:\nRelevance of Input Per Sample'.format(dataname, model))


        ax.set_ylim([-maxamp, maxamp])
        ax.set_xticks(xticks)
        #print xticks
        if i < len(reps)-1:
            ax.set_xticklabels([]) #deactivate tick labels for all but the lowest row
        if i == 0:
            ax.set_yticks([-maxamp, 0, maxamp])
            ticklabs = [str(s) for s in[-maxamp, 0, maxamp]]
            ticklabs = [' '*(2-len(s)) + s for s in ticklabs]
            ax.set_yticklabels(ticklabs)
        else:
            ax.set_yticks([0]) #show single vertical grid line thresholding zero
            ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)#show

        ax.yaxis.tick_right()
        ax.set_ylabel('Sample {}'.format(reps[i]),fontsize=7)
        ax.grid(True, linewidth=0.5, axis='x')
        #ax.spines['top'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True) # grid lines behind all other graph elements
        ax.set_xlim(xlim)

        if i == len(reps)-1:
            #use minor tick labels as x axis legend
            ax.set_xticks([50.5 + 101*ii for ii in range(6)],minor=True)
            ax.tick_params(axis='x', which='minor', length=0)
            ax.set_xticklabels(['right foot x', 'right foot y', 'right foot z', 'left foot x', 'left foot y', 'left foot z'], minor=True, fontsize=6)



    #plt.show()
    save_figure('Figure-0', fmt=fmt, dpi=dpi)
    plt.close()




def draw_fig0_supp (fmt, variant='a',dpi=300):
    #this one is a bit hardcodey
    print 'drawing figure(s) 0 (supplement version)...'
    print '   loading data'
    print '   variant', variant

    models = 'CNN-A'
    datanames = ['Ground Reaction Force', 'Lower-Body Joint Angles']
    shortnames = ['GRF', 'LBJAX']
    datas = ['P6.GRF_AV.CNN-A.mat', 'P6.JA_X_Lower.CNN-A.mat']

    for d in xrange(len(datanames)):
        dataname = datanames[d]
        shortname = shortnames[d]
        data = scipy.io.loadmat(datas[d])


        if shortname == 'GRF':
            drawshading=False
            bodyparts = []
            for f in ['right', 'left']:
                for xyz in ['fore-aft', 'med-lat', 'vert']:
                    #bodyparts.append('{} foot {}'.format(f, xyz))
                    bodyparts.append('{} {}'.format(f, xyz))
        elif shortname  == 'LBJAX':
            drawshading=True
            bodyparts = []
            for lr in ['left', 'right']:
                for akn in ['ankle', 'knee', 'hip']:
                    bodyparts.append('{} {}'.format(lr, akn))

            #['left ankle x', 'left knee x', 'left hip x', 'right ankle x', 'right knee x', 'right hip x']
            #
            #also der rechte bodenkontakt ist fuer subject 6
            #Fabian Horst: von Frame 1:54 und 94:101
            #Fabian Horst: der linke ist von Frame 1:7 und 48:101
            rightfootshading = [(.5,53), (93,99.5)]
            leftfootshading = [(.5,6),(47,99.5)]
            tmp = [leftfootshading]*3 + [rightfootshading]*3
            footongroundshading = []
            #relative to absolute coordinates
            for i in xrange(len(tmp)):
                blockfeatures = tmp[i]
                for b in blockfeatures:
                    footongroundshading.append((b[0]+(i*101), b[1]+(i*101)))
            print footongroundshading



        probanden = [20, 49, 53, 55] #subjects 21, 50, 54, 56 (1-based matlab indexing)
        for subject in probanden:
            inputs = data['Feature_Output'][0][subject]       # 21 x 606
            ypred = data['Ypred_Output'][0][subject][:,subject]     # 21 x 1
            #relevance = data['RpredDom_Output'][0][subject]   # 21 x 606
            relevance = data['RpredAct_Output'][0][subject]   # 21 x 606

            pred = np.argmax(data['Ypred_Output'][0][subject], axis=1)
            tprate = np.mean(pred == subject)

            #NOTE: reinstating matlab-counting used for figure file names!"!!!"
            subject+=1

            print  subject, shortname, pred, tprate



            xticks = np.arange(7)*101
            xlim = xticks[[0,-1]]

            #draw inputs
            maxamp = 1
            fig = plt.figure(figsize=(6, 0.8))
            ax = plt.subplot(111)
            if drawshading:
                for fs in footongroundshading:
                    ax.fill_between(np.arange(fs[0],fs[1]), maxamp, -maxamp, color=[0.9]*3,  zorder=-1)
            for i in xrange(inputs.shape[0]):
                values = inputs[i]
                #markersize < 1 transforms dots to circles. stupid.
                plotcolor = 'blue' if shortname == 'GRF' else 'green'
                ax.plot(values, color=plotcolor, linestyle='-', linewidth = 0.2)

            ax.set_ylim([-maxamp, maxamp])
            ax.set_yticks([-maxamp, 0, maxamp])
            ticklabs = [str(s) for s in[-maxamp, 0, maxamp]]
            ticklabs = [' '*(2-len(s)) + s for s in ticklabs]
            ax.set_yticklabels(ticklabs)
            ax.set_xticks([50.5 + 101*ii for ii in range(6)],minor=True)
            ax.tick_params(axis='x', which='minor', length=0)
            ax.set_xticklabels(bodyparts, minor=True, fontsize=6)
            ax.set_xticks(xticks, minor=False)
            ax.set_xticklabels(xticks, minor=False, fontsize=7)
            ax.yaxis.tick_right()
            ax.set_ylabel('{}\nInput'.format(shortname),fontsize=8, color=plotcolor)
            ax.grid(True, linewidth=0.5, axis='x')
            ax.set_xlim(xlim)
            #ax.set_title(dataname)
            plt.subplots_adjust(bottom=0.24,top=0.95, left=0.05, right=0.95)

            save_figure('Figure-0-SUB{}-{}-inputs'.format(subject,shortname), fmt=fmt, dpi=dpi)
            plt.close()

            #draw relevances 1
            maxamp = 0
            fig = plt.figure(figsize=(6, 0.8))
            ax = plt.subplot(111)
            rvals = relevance.T
            x = np.arange(rvals.shape[0])
            x, rvals = interpolate_all_samples(rvals, fold=50)
            nrel = rvals / np.max(np.abs(rvals))
            rel_colors = colormap(nrel)
            for j in xrange(rvals.shape[1]):
                rel = rvals[:,j]
                maxamp = max(maxamp, np.abs(rel).max())
                if drawshading:
                    for fs in footongroundshading:
                        ax.fill_between(np.arange(fs[0],fs[1]), maxamp, -maxamp, color=[0.9]*3,  zorder=-1)
                ax.scatter(x=x,
                        y=rel,
                        c=rel_colors[:,j,:],
                        s=0.3,
                        marker = '.',
                        edgecolor='none')
            maxamp *= 1.1
            ax.set_ylim([-maxamp, maxamp])
            ax.yaxis.tick_right()
            ax.set_yticks([0]) #show single vertical grid line thresholding zero
            ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)
            ax.set_xticks([50.5 + 101*ii for ii in range(6)],minor=True)
            ax.tick_params(axis='x', which='minor', length=0)
            ax.set_xticklabels(bodyparts, minor=True, fontsize=6)
            ax.set_xticks(xticks, minor=False)
            ax.set_xticklabels(xticks, minor=False, fontsize=7)
            ax.yaxis.tick_right()
            ax.set_ylabel('{}\nRelevance'.format(shortname),fontsize=8, color=plotcolor)
            ax.grid(True, linewidth=0.5, axis='x')
            ax.set_xlim(xlim)
            #ax.set_title(dataname)
            plt.subplots_adjust(bottom=0.24,top=0.95, left=0.05, right=0.95)

            save_figure('Figure-0-SUB{}-{}-Rel1'.format(subject,shortname), fmt=fmt, dpi=dpi)
            plt.close()


            #draw relevances 2
            maxamp = 1
            fig = plt.figure(figsize=(6, 0.8))
            ax = plt.subplot(111)
            rvals = relevance.T
            inpts = inputs.T
            x = np.arange(rvals.shape[0])
            x, rvals = interpolate_all_samples(rvals, fold=50)
            x, inpts = interpolate_all_samples(inpts, fold=50)
            nrel = rvals / np.max(np.abs(rvals))
            rel_colors = colormap(nrel)
            for j in xrange(rvals.shape[1]):
                rel = rvals[:,j]
                inp = inpts[:,j]
                #maxamp = max(maxamp, np.abs(rel).max())
                if drawshading:
                    for fs in footongroundshading:
                        ax.fill_between(np.arange(fs[0],fs[1]), maxamp, -maxamp, color=[0.9]*3,  zorder=-1)
                ax.scatter(x=x,
                        y=inp,
                        c=rel_colors[:,j,:],
                        s=0.3,
                        marker = '.',
                        edgecolor='none')
            #maxamp *= 1.1
            ax.set_ylim([-maxamp, maxamp])
            ax.yaxis.tick_right()
            ax.set_yticks([0]) #show single vertical grid line thresholding zero
            ax.set_yticklabels([]) #deactivate tick label texts (no info gained from this)
            ax.set_xticks([50.5 + 101*ii for ii in range(6)],minor=True)
            ax.tick_params(axis='x', which='minor', length=0)
            ax.set_xticklabels(bodyparts, minor=True, fontsize=6)
            ax.set_xticks(xticks, minor=False)
            ax.set_xticklabels(xticks, minor=False, fontsize=7)
            ax.yaxis.tick_right()
            ax.set_ylabel('{}\nRelevance'.format(shortname),fontsize=8, color=plotcolor)
            ax.grid(True, linewidth=0.5, axis='x')
            ax.set_xlim(xlim)
            #ax.set_title(dataname)
            plt.subplots_adjust(bottom=0.24,top=0.95, left=0.05, right=0.95)

            save_figure('Figure-0-SUB{}-{}-Rel2'.format(subject,shortname), fmt=fmt, dpi=dpi)
            plt.close()

    return # ADDED THATbecause below code runs for ages RETURN FOR SOME REASON
