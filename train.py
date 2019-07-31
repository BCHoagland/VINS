import os, sys, importlib
from termcolor import colored

try:
    f = sys.argv[1]
    if '.py' in f: f = f[:-3]
    script = 'scripts.' + f

    com = colored(f, 'cyan')
    print(f'Running {com}')
    importlib.import_module(script)

except (IndexError, ModuleNotFoundError):
    com = colored('python train.py {script_name}', 'yellow')
    print('You need to specify which script to execute as so:\n' + com)

    scripts = [f[:-3] for f in os.listdir('scripts') if '.py' in f]
    print('\n' + colored('Available Scripts:', 'cyan'))
    for script in scripts:
        print(' + ' + script)

except KeyboardInterrupt:
    com = colored('Cancelled'.ljust(50), 'red')
    sys.stdout.write(f'\r{com}\n')

except:
    raise
