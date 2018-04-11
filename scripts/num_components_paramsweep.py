"""
Gregory Way 2018
scripts/num_components_param_sweep.py

The script uses two configuration files: 1) parameter and 2) config files.

1) The parameter file (required) is a tab separated file with a header:

header = ["parameter", "sweep_values"]

Where different values are given for specific hyperparameters. The different
values are separated by commas. The possible hyperparameters currently include:

num_components - the dimensionality of the compressed latent space
learning_rates - the step size for each gradient update
batch_sizes - the number of samples to include in each weight update iteration
epochs - how many times to cycle through all batches of data
kappas - the amount of "warmup" applied to VAE models
sparsities - a weight regularization parameter enforcing zero weights in ADAGE
noises - the proportion of dropout weights in ADAGE models

2) The tab sep config file has parameters for training on PMACS with a header:

header = ["variable", "assign"]

The values include:

queue - the queue that will schedule and run jobs
num_gpus - how many GPUs to request (if "gpu" queue requested)
num_gpus_shared - how many of these GPUs run concurrent jobs
walltime - how long to request for each job to run

NOTE: Either a PMACS configuration file or a `--local` flag must be provided.

Usage: Run in command line:

python scripts/num_components_param_sweep.py

     with required command arguments:

       --parameter_file     filepath pointing to tab separated parameters

     and optional arguments:

       --config_file        filepath pointing to PMACS configuration file
       --algorithm          a string indicating which algorithm to sweep over
                              default: 'tybalt' (i.e. variational autoencoder)
       --python_path        absolute path of PMACS python in select environment
                              default: '~/.conda/envs/tybalt-gpu/bin/python'
       --param_folder       filepath of where to save the results
                              default: 'param_sweep'
       --local              if provided, sweep will be run locally instead

Output:
Submit a job to the PMACS cluster training a distinct VAE with a different
combination of hyper parameters.
"""

import os
import argparse
import pandas as pd
from bsub_helper import bsub_help


def get_param(param):
    try:
        sweep = parameter_df.loc[param, 'sweep_values']
    except:
        sweep = ''
        print("Warning! No parameter detected of name: {}".format(param))
    return sweep.split(',')

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--parameter_file',
                    help='location of tab separated parameter file to sweep')
parser.add_argument('-c', '--config_file', default='config/pmacs_config.tsv',
                    help='location of the configuration file for PMACS')
parser.add_argument('-a', '--algorithm',
                    help='which algorithm to sweep hyperparameters over')
parser.add_argument('-s', '--python_path',
                    help='absolute path of python version',
                    default='~/.conda/envs/tybalt-gpu/bin/python')
parser.add_argument('-d', '--param_folder',
                    help='folder to store param sweep results',
                    default='param_sweep')
parser.add_argument('-t', '--script',
                    help='path the script to run the parameter sweep over',
                    default='scripts/vae_pancancer.py')
parser.add_argument('-l', '--local', action='store_true',
                    help='decision to run models locally instead of on PMACS')
args = parser.parse_args()

parameter_file = args.parameter_file
config_file = args.config_file
algorithm = args.algorithm
python_path = args.python_path
param_folder = args.param_folder
script = args.script
local = args.local

# Required to update shell for `subprocess` module if running locally
if local:
    shell = False
else:
    shell = True

if algorithm == 'adage':
    script = 'scripts/adage_pancancer.py'

if not os.path.exists(param_folder):
    os.makedirs(param_folder)

# Load data
parameter_df = pd.read_table(parameter_file, index_col=0)
config_df = pd.read_table(config_file, index_col=0)

# Retrieve hyperparameters to sweep over
num_components = get_param('num_components')
learning_rates = get_param('learning_rate')
batch_sizes = get_param('batch_size')
epochs = get_param('epochs')
kappas = get_param('kappa')
sparsities = get_param('sparsity')
noises = get_param('noise')
adage_weights = get_param('weights')[0]

# Retrieve PMACS configuration
queue = config_df.loc['queue']['assign']
num_gpus = config_df.loc['num_gpus']['assign']
num_gpus_shared = config_df.loc['num_gpus_shared']['assign']
walltime = config_df.loc['walltime']['assign']

# Build lists of job commands depending on input algorithm
all_commands = []
if algorithm == 'tybalt':
    for z in num_components:
        for lr in learning_rates:
            for bs in batch_sizes:
                for e in epochs:
                    for k in kappas:
                        f = 'paramsweep_{}z_{}lr_{}bs_{}e_{}k.tsv'.format(z, lr, bs, e, k)
                        f = os.path.join(param_folder, f)
                        params = ['--num_components', z,
                                  '--learning_rate', lr,
                                  '--batch_size', bs,
                                  '--epochs', e,
                                  '--kappa', k,
                                  '--output_filename', f]
                        final_command = [python_path, script] + params
                        all_commands.append(final_command)
elif algorithm == 'adage':
    for z in num_components:
        for lr in learning_rates:
            for bs in batch_sizes:
                for e in epochs:
                    for s in sparsities:
                        for n in noises:
                            f = 'paramsweep_{}z_{}lr_{}bs_{}e_{}s_{}n.tsv'.format(z, lr, bs, e, s, n)
                            f = os.path.join(param_folder, f)
                            params = ['--num_components', z,
                                      '--learning_rate', lr,
                                      '--batch_size', bs,
                                      '--epochs', e,
                                      '--sparsity', s,
                                      '--noise', n,
                                      '--output_filename', f]
                            final_command = [python_path, script] + params
                            if adage_weights == 'untied':
                                final_command += ['--untied_weights']
                            all_commands.append(final_command)

# Submit commands
for command in all_commands:
    b = bsub_help(command=command,
                  local=local,
                  shell=shell)
    b.submit_command()
