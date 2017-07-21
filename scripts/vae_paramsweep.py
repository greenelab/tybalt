"""
Gregory Way 2017
Variational Autoencoder - Pan Cancer
scripts/vae_paramsweep.py

Usage: Run in command line with required command arguments:

        python scripts/vae_paramsweep.py --parameter_file <parameter-filepath>
                                         --config_file <configuration-filepath>

Output:
TBD
"""

import argparse
import pandas as pd
from bsub_helper import bsub_help


def get_param(param):
    sweep = parameter_df.loc[param, 'sweep']
    return sweep.split(',')

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--parameter_file',
                    help='location of tab separated parameter file to sweep')
parser.add_argument('-c', '--config_file',
                    help='location of the configuration file for PMACS')
parser.add_argument('-s', '--python_path',
                    help='absolute path of python version',
                    default='~/.conda/envs/vae_pancancer/bin/python')
args = parser.parse_args()

parameter_file = args.parameter_file
config_file = args.config_file
python_path = args.python_path

parameter_df = pd.read_csv(parameter_file, index_col=0)
config_df = pd.read_csv(config_file, index_col=0)

learning_rates = get_param('learning_rate')
batch_sizes = get_param('batch_size')
epochs = get_param('epochs')

queue = config_df.loc['queue']['assign']
num_gpus = config_df.loc['num_gpus']['assign']
num_gpus_shared = config_df.loc['num_gpus_shared']['assign']
walltime = config_df.loc['walltime']['assign']

for lr in learning_rates:
    for bs in batch_sizes:
        for e in epochs:
            params = ['--learning_rate', lr, '--batch_size', bs, '--epochs', e]
            final_command = [python_path, 'scripts/vae_pancancer.py'] + params

            b = bsub_help(command=final_command,
                          queue=queue,
                          num_gpus=num_gpus,
                          num_gpus_shared=num_gpus_shared,
                          walltime=walltime)
            b.submit_command()
