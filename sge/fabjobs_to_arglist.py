import sys
import os

if len(sys.argv) < 2:
    print(sys.argv)
    print('Not enough command line args. Requires 1 arg!')
    print('Example: python fabjobs_to_arglist.py path/to/joblist/by/fabian')
    exit()

fabjob_file = sys.argv[1]
argument_list = []
with open(fabjob_file,  'rt') as f:
    lines = f.read().split('\n')
    for line in lines:
        if not 'runfile' in line:
            continue

        #get args from call.
        line_components = line.split(',')
        arg_components = [c for c in line_components if 'args=' in c][0]
        arg_components = arg_components.split('=')[1]

        #fix data paths and file names.
        arg_components = [a for a in arg_components.replace("'","").split(' ') if len(a) > 0]
        for i in range(len(arg_components)):
            if arg_components[i] == '-d':
                arg_components[i+1] = arg_components[i+1] + '.mat'

        #reassemble and collect
        argument_list.append(' '.join(arg_components))

#write out to file.
output_file = os.path.splitext(os.path.basename(fabjob_file))[0] + '.args'
print('writing converted arguments to {}'.format(output_file))
with open(output_file, 'wt') as f:
    f.write('\n'.join(argument_list))

