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
import pandas as pd

param_files = os.listdir('param_sweep')
df_list = [pd.read_table(os.path.join('param_sweep', x)) for x in param_files]

param_df = pd.concat(df_list, axis=0)
param_df.rename(columns={'Unnamed: 0': 'train_epoch'}, inplace=True)

full_result_file = os.path.join('results', 'parameter_sweep_full_results.tsv')
param_df.to_csv(full_result_file, index=False, sep='\t')
