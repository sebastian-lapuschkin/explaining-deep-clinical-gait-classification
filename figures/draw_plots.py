"""
Call this script to draw plots for Figures 1 to 5

How to use:
python draw_plots.py [optionalParameters] [help]

"""

import sys
import json
import figure_fxns

#default values:
figureindex=range(7)
figureformat='jpg'
variant='a'
dpi=300

#help message
def print_howto():
    print ''
    print 'dear user, to run this script, call:'
    print '    pyhton draw_plots [optionalParameters]'
    print ' where optionalParameters are =-separated key value pairs'
    print ' k=v '
    print ''
    print 'possible parameters (and default values) are:'
    print 'figure={} or a value from [1,2,3,4,5] . may also be a list.'.format(figureindex)
    print 'format={} or from any matplotlib-supported format, e.g. [pdf, eps, tiff, jpg, png]'.format(figureformat)
    print 'variant={} from "a" or "b" if applicable.'.format(variant)
    print 'dpi={} from any meaningful value for dpi.'.format(dpi)
    print 'help (as an exception, which is not a key-value-pair'
    print 'now try again!'
    print ''


if __name__ == "__main__":
    #parse sys.argv
    if 'help' in sys.argv:
        print_howto()
        exit()

    for arg in sys.argv:
        if '=' in arg:
            k,v = arg.split('=')
            if k == 'figure':
                figureindex = json.loads(v)
                print 'setting figure to', figureindex
            elif k == 'format':
                figureformat = v
                print 'setting figureformat to', figureformat
            elif k == 'variant':
                variant = v
                print 'setting figure variant to', variant
            elif k == 'dpi':
                dpi = int(v)
                print 'setting figure output dpi to', dpi
            else:
                raise ValueError('Unknown parameter key {} in kex value pair {}'.format(k, arg))

    if isinstance(figureindex, int):
        figureindex = [figureindex]
    for i in figureindex:
        if not isinstance(i,int):
            print 'parameter figure must be an int or a list of ints but contains/was:', i, 'of type', type(i)
            exit(-1)
    if figureformat not in ['eps', 'tiff', 'jpg', 'pdf', 'png']:
        print 'figure format not within acceptable parameters'
        exit(-1)

    #figureindex=[3] #debug
    #variant='b'
    fxndict = { 1:figure_fxns.draw_fig1,
                2:figure_fxns.draw_fig2,
                3:figure_fxns.draw_fig3,
                #3:figure_fxns.draw_fig3_alt,
                4:figure_fxns.draw_fig4,
                #4:figure_fxns.draw_fig4_alt,
                5:figure_fxns.draw_fig5,
                6:figure_fxns.draw_fig6,
                0:figure_fxns.draw_fig0,
                -1:figure_fxns.draw_fig0_supp}

    #call all requirested figure drawing calls
    for i in figureindex:
        fxndict[i](fmt=figureformat, variant=variant, dpi=dpi)



