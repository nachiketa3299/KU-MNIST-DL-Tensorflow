import os

def makeDir(_filename):
    return os.path.join(os.path.curdir, 'runs', _filename)

filenames = os.listdir(os.path.join(os.path.curdir, 'runs'))

# tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2
command_names = []
for filename in filenames:
    name = f'{filename}'
    ddir = f'{makeDir(filename)}'
    name += ':' + ddir
    command_names.append(name)

command = f'tensorboard --logdir='
command += ','.join(command_names)
command += ' --host localhost'

os.system(f'{command}')


