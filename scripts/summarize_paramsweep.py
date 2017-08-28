"""
Gregory Way 2017
Variational Autoencoder - Pan Cancer
scripts/summarize_paramsweep.py

Usage: Run in command line:

        python scripts/summarize_paramsweep.py

Output:
Large dataframe to use in downstream visualizations
"""

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--results_directory',
                    help='location where all results are stored',
                    default='param_sweep')
parser.add_argument('-f', '--output_filename',
                    help='The name of the file to store results',
                    default='parameter_sweep_full_results.tsv')
args = parser.parse_args()

# Set hyper parameters
results_dir = args.results_directory
output_filename = args.output_filename

param_files = os.listdir(results_dir)
df_list = [pd.read_table(os.path.join(results_dir, x)) for x in param_files]

param_df = pd.concat(df_list, axis=0)
param_df.rename(columns={'Unnamed: 0': 'train_epoch'}, inplace=True)

full_result_file = os.path.join('results', output_filename)
param_df.to_csv(full_result_file, index=False, sep='\t')
