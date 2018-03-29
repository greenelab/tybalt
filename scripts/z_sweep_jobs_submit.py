"""
Gregory Way 2018
scripts/z_sweep_jobs_submit.py

This script will submit several jobs to the PMACS cluster at the University of
Pennsylvania. Each job will train a model for a prespecified number of
components, which corresponds to the latent space dimensionality.

The script uses two configuration files: 1) parameter and 2) config files.

1) The parameter file (required) is a tab separated file. The columns of the
file indicate number of dimensions to constrain the model to learning. The
index of the file indicate separate hyperparameters required as input to
`train_models_given_z.py`. The parameters include:

vae_epochs - number of epochs for Tybalt model
dae_epochs - number of epochs for ADAGE model
vae_lr - learning rate (step size) for Tybalt model
dae_lr - learning rate (step size) for ADAGE model
vae_batch_size - batch size for Tybalt model
dae_batch_size - batch size for ADAGE model
dae_noise - noise injected into ADAGE model training
dae_sparsity - sparsity penalty regularization for ADAGE model training
vae_kappa - warm up parameter for Tybalt model

2) The tab sep config file has parameters for training on PMACS with a header:
header = ["variable", "assign"]
The values include:
queue - the queue that will schedule and run jobs
num_gpus - how many GPUs to request (if "gpu" queue requested)
num_gpus_shared - how many of these GPUs run concurrent jobs
walltime - how long to request for each job to run

NOTE: Either a PMACS configuration file or a `--local` flag must be provided.

Usage: Run in command line: python scripts/latent_space_sweep_submit.py

     with required command arguments:

       --components         space separated dimensionalities to submit jobs for
                            e.g. "--num_components 2 5 10" will submit 3 jobs
                            fitting 2, 5, and 10 latent space dimensions
       --param_config       location of tsv file (param by z dimension) for the
                            specific parameter combination for each z dimension
       --out_dir            filepath of where to save the results

     and optional arguments:

       --pmacs_config       filepath pointing to PMACS configuration file
       --python_path        absolute path of PMACS python in select environment
                              default: '~/.conda/envs/tybalt-gpu/bin/python'
       --num_seeds          how many models to build (random seeds to set)
                              default: 5
       --local              if provided, sweep will be run locally instead

Output:
Submit jobs given certain number of latent space dimensionality
"""

import os
import argparse
import pandas as pd
from bsub_helper import bsub_help

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--components', help='dimensionality to sweep over',
                    nargs='+')
parser.add_argument('-y', '--param_config',
                    help='locaiton of the parameter configuration')
parser.add_argument('-d', '--out_dir', help='folder to store results')
parser.add_argument('-c', '--pmacs_config', default='config/pmacs_config.tsv',
                    help='location of the configuration file for PMACS')
parser.add_argument('-p', '--python_path', help='absolute path of python',
                    default='python')
parser.add_argument('-s', '--num_seeds', default=5,
                    help='number of different seeds to run on current data')
parser.add_argument('-l', '--local', action='store_true',
                    help='decision to run models locally instead of on PMACS')
args = parser.parse_args()

pmacs_config_file = args.pmacs_config
param_config_file = args.param_config
out_dir = args.out_dir
python_path = args.python_path
num_seeds = args.num_seeds
components = args.components
local = args.local

# Required to update shell for `subprocess` module if running locally
if local:
    shell = False
else:
    shell = True

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Load data
config_df = pd.read_table(pmacs_config_file, index_col=0)

# Retrieve PMACS configuration
queue = config_df.loc['queue']['assign']
num_gpus = config_df.loc['num_gpus']['assign']
num_gpus_shared = config_df.loc['num_gpus_shared']['assign']
walltime = config_df.loc['walltime']['assign']

# Set default parameter combination
conda = ['conda', 'activate', 'tybalt-gpu']
default_params = ['--param_config', param_config_file,
                  '--out_dir', out_dir,
                  '--num_seeds', num_seeds]

# Build lists of job commands depending on input algorithm
all_commands = []
for z in components:
    z_command = [python_path, 'train_models_given_z.py', '--num_components', z]
    if local:
        z_command += default_params
    else:
        z_command = conda + z_command + default_params
    all_commands.append(z_command)

# Submit the jobs to PMACS
for command in all_commands:
    print(command)
    b = bsub_help(command=command,
                  queue=queue,
                  num_gpus=num_gpus,
                  num_gpus_shared=num_gpus_shared,
                  walltime=walltime,
                  local=local,
                  shell=shell)
    b.submit_command()
