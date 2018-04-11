"""
Gregory Way 2018
scripts/util/aggregate_simulation_results.py

Compile individual simulation results into a single results file

Usage: Run in command line:

    python scripts/util/aggregate_simulation_results.py

    With the following command line arguments:

    --simulation_directory - the folder where all simulation results are stored
    --output_filename      - the name of the file to save compiled results to

Output:
Write a single file to results summarizing results of the simulation
"""

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--simulation_directory',
                    help='location where all results are stored',
                    default='results/simulation')
parser.add_argument('-f', '--output_filename',
                    help='The name of the file to store results',
                    default='results/parameter_sweep_full_results.tsv')
args = parser.parse_args()

# Load command args
results_dir = args.results_directory
output_filename = args.output_filename

simulation_results_files = os.listdir(results_dir)

all_sim_list = []
for sim_file in simulation_results_files:
    full_file_name = os.path.join(results_dir, sim_file)
    sim_df = pd.read_table(full_file_name)
    sim_df = sim_df.rename(columns={'Unnamed: 0': 'algorithm'})
    all_sim_list.append(sim_df)

all_sim_df = pd.concat(all_sim_list).reset_index().drop('index', axis=1)
all_sim_df.to_csv(output_filename, sep='\t', index=False)
