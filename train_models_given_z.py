"""
2018 Gregory Way
train_models_given_z.py

This script will train various compression models given a specific z dimension.
Each model will train several times with different initializations.

The script pulls hyperparameters from a parameter file that was determined
after initial hyperparameter sweeps testing latent dimensionality.

Output:

The script will save associated weights and z matrices for each permutation as
well as reconstruction costs for all algorithms and training histories for
Tybalt and ADAGE models. The population of models (ensemble) will be used,
across dimensions z, for follow-up evaluations

Usage:

    python run_model_with_input_dimensions.py

    With required command line arguments:

        --num_components    The z dimensionality we're testing
        --param_config      A tsv file (param by z dimension) indicating the
                            specific parameter combination for the z dimension
        --out_dir           The directory to store the output files

    And optional command line arguments

        --num_seeds         The number of specific models to generate
                                default: 5
"""

import os
import argparse
import numpy as np
import pandas as pd
from tybalt.data_models import DataModel

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_components', help='dimensionality of z')
parser.add_argument('-p', '--param_config',
                    help='text file optimal hyperparameter assignment for z')
parser.add_argument('-o', '--out_dir', help='where to save the output files')
parser.add_argument('-s', '--num_seeds', default=5,
                    help='number of different seeds to run on current data')
parser.add_argument('-r', '--shuffle', action='store_true',
                    help='randomize gene expression data for negative control')
args = parser.parse_args()

# Load command arguments
num_components = int(args.num_components)
param_config = args.param_config
out_dir = args.out_dir
num_seeds = int(args.num_seeds)
shuffle = args.shuffle

# Extract parameters from parameter configuation file
param_df = pd.read_table(param_config, index_col=0)

vae_epochs = param_df.loc['vae_epochs', str(num_components)]
dae_epochs = param_df.loc['dae_epochs', str(num_components)]
vae_lr = param_df.loc['vae_lr', str(num_components)]
dae_lr = param_df.loc['dae_lr', str(num_components)]
vae_batch_size = param_df.loc['vae_batch_size', str(num_components)]
dae_batch_size = param_df.loc['dae_batch_size', str(num_components)]
dae_noise = param_df.loc['dae_noise', str(num_components)]
dae_sparsity = param_df.loc['dae_sparsity', str(num_components)]
vae_kappa = param_df.loc['vae_kappa', str(num_components)]

# Set output directory and file names
train_dir = os.path.join(out_dir, 'training_history')
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if shuffle:
    file_pre = '{}_components_shuffled_'.format(num_components)
else:
    file_pre = '{}_components_'.format(num_components)

recon_file = os.path.join(train_dir, '{}reconstruction.tsv'.format(file_pre))
tybalt_hist_file = os.path.join(train_dir,
                                '{}tybalt_training_hist.tsv'.format(file_pre))
adage_hist_file = os.path.join(train_dir,
                               '{}adage_training_hist.tsv'.format(file_pre))

# Load Data
rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv.gz')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)

# Initialize DataModel class with pancancer data
dm = DataModel(df=rnaseq_df)

# Set seed and list of algorithms for compression
random_seeds = np.random.randint(0, high=1000000, size=num_seeds)
algorithms = ['pca', 'ica', 'nmf', 'adage', 'tybalt']

# Save population of models in specific folder
comp_out_dir = os.path.join(out_dir, 'components_{}'.format(num_components))
if not os.path.exists(comp_out_dir):
    os.makedirs(comp_out_dir)

reconstruction_results = []
tybalt_training_histories = []
adage_training_histories = []
for seed in random_seeds:
    seed_file = os.path.join(comp_out_dir, 'model_{}'.format(seed))

    if shuffle:
        seed_file = '{}_shuffled'.format(seed_file)

        # randomly permute genes of each sample in the rnaseq matrix
        shuf_df = rnaseq_df.apply(lambda x: np.random.permutation(x.tolist()),
                                  axis=1)

        # Initiailze a new DataModel, with different shuffling each permutation
        dm = DataModel(df=shuf_df)

    # Fit models
    dm.pca(n_components=num_components)
    dm.ica(n_components=num_components)
    dm.nmf(n_components=num_components)

    dm.nn(n_components=num_components,
          model='tybalt',
          loss='binary_crossentropy',
          epochs=int(vae_epochs),
          batch_size=int(vae_batch_size),
          learning_rate=float(vae_lr),
          separate_loss=True,
          verbose=False)

    dm.nn(n_components=num_components,
          model='adage',
          loss='binary_crossentropy',
          epochs=int(dae_epochs),
          batch_size=int(dae_batch_size),
          learning_rate=float(dae_lr),
          noise=float(dae_noise),
          sparsity=float(dae_sparsity),
          verbose=False)

    # Obtain z matrix (sample scores per latent space feature) for all models
    full_z_matrix = dm.combine_models()

    # Obtain weight matrices (gene by latent space feature) for all models
    full_weight_matrix = dm.combine_weight_matrix()

    # Store reconstruction costs at training end
    full_reconstruction = dm.compile_reconstruction()

    # Store training histories for neural network models
    tybalt_train_history = dm.tybalt_fit.history_df
    tybalt_train_history = tybalt_train_history.assign(seed=seed)
    tybalt_train_history = tybalt_train_history.assign(shuffled=shuffle)
    adage_train_history = dm.adage_fit.history_df
    adage_train_history = adage_train_history.assign(seed=seed)
    adage_train_history = adage_train_history.assign(shuffled=shuffle)

    # Store z and weight matrices for each seed (the population of models)
    full_z_file = '{}_z_matrix.tsv'.format(seed_file)
    full_z_matrix.to_csv(full_z_file, sep='\t')

    full_weight_file = '{}_weight_matrix.tsv'.format(seed_file)
    full_weight_matrix.to_csv(full_weight_file, sep='\t')

    reconstruction_results.append(full_reconstruction)
    tybalt_training_histories.append(tybalt_train_history)
    adage_training_histories.append(adage_train_history)

# Save reconstruction and neural network training results
reconstruction_df = pd.concat(reconstruction_results)
reconstruction_df.index = range(0, num_seeds)
tybalt_history_df = pd.concat(tybalt_training_histories)
adage_history_df = pd.concat(adage_training_histories)

# Output files
reconstruction_df.to_csv(recon_file, sep='\t')
tybalt_history_df.to_csv(tybalt_hist_file, sep='\t')
adage_history_df.to_csv(adage_hist_file, sep='\t')
