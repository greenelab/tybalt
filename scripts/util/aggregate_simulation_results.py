"""
Gregory Way 2018
scripts/util/aggregate_simulation_results.py

Compile individual simulation results into a single results file

Usage: Run in command line:

    python scripts/util/aggregate_simulation_results.py

Output:
Write a single file to results summarizing results of the simulation
"""

import os
import pandas as pd

simulation_folder = os.path.join('results', 'simulation')
simulation_results_files = os.listdir(simulation_folder)

all_sim_list = []
for sim_file in simulation_results_files:
    full_file_name = os.path.join(simulation_folder, sim_file)
    sim_df = pd.read_table(full_file_name)
    sim_df = sim_df.rename(columns={'Unnamed: 0': 'algorithm'})
    all_sim_list.append(sim_df)

all_sim_file = os.path.join('results', 'all_simulation_results.tsv')
all_sim_df = pd.concat(all_sim_list).reset_index().drop('index', axis=1)
all_sim_df.to_csv(all_sim_file, sep='\t', index=False)
