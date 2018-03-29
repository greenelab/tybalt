"""
2018 Gregory Way
train_models_given_z.py

This script will loop through different combinations of latent space
dimensionality, train a distinct model, and save associated decoder weights
and z matrices. The script will need to pull appropriate hyperparameters from
respective files after several initial sweeps testing various hyperparameter
combinations with different z dimensionality.

The script will fit several different compression algorithms and output
results. The results include the weight matrices and z matrices of the models
with the lowest reconstruction loss. The total stability of reconstruction
losses are represented through the variability associated with each model after
retraining several times given the argument for number of seeds. We also
report the determinants of correlation matrices for all iterations. These
values measure the stability of either latent space (z) matrices or weight
matrices across several rounds of fitting. In this case, as the determinants
approach zero, the more stable solutions are.

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


def get_lowest_loss(matrix_list, reconstruction_df,
                    algorithms=['pca', 'ica', 'nmf', 'dae', 'vae']):
    """
    Determine the specific model with the lowest loss using reconstruction cost

    Arguments:
    matrix_list - list of matrices (either weight or z matrices)
    reconstruction_df - pandas DataFrame (seed by reconstruction error)
    algorithms - which algorithms to consider

    Output:
    Single matrix of the "best" alternative matrices by lowest recon error
    """
    final_matrix_list = []
    for alg in algorithms:
        # Get lowest reconstruction error across iterations for an algorithm
        min_recon_subset = reconstruction_df.loc[:, alg].idxmin()

        # subset the matrix to the minimum loss
        best_matrix = matrix_list[min_recon_subset]

        # Extract the algorithm specific columns from the concatenated matrix
        use_cols = best_matrix.columns.str.startswith(alg)
        best_matrix_subset = best_matrix.loc[:, use_cols]

        # Build the final matrix that will eventually be output
        final_matrix_list.append(best_matrix_subset)

    return pd.concat(final_matrix_list, axis=1)


def get_corr_determinants(matrix_list,
                          algorithms=['pca', 'ica', 'nmf', 'dae', 'vae']):
    """
    Computes the determinant of the correlation matrix within each algorithm
    across all fit iterations. The determinant of the correlation matrix
    is used as a measurement of "how correlated" are solutions across
    iterations. The determinant will exist between -1 and 1 and values
    approaching 0 indicate high correlation. A value of 1 indicates no
    correlation.

    Arguments:
    matrix_list - list of matrices (either weight or z matrices)
    algorithms - which algorithms to consider

    Output:
    pandas DataFrame of all correlation matrix determinants by algorithm
    """
    matrix_df = pd.concat(matrix_list, axis=1)
    all_corr_det = {}

    for alg in algorithms:
        # subset the full matrix across seeds to the algorithm of interest
        use_cols = matrix_df.columns.str.startswith(alg)
        alg_matrix_df = matrix_df.loc[:, use_cols]

        # Get the correlation of latent space features
        alg_matrix_corr_df = alg_matrix_df.corr()
        alg_corr_det = np.linalg.det(alg_matrix_corr_df)
        all_corr_det[alg] = [alg_corr_det]

    return pd.DataFrame(all_corr_det)


def full_correlation_determinant(best_matrix):
    """
    Computes the determinant of final correlation matrix. Useful for comparing
    correlated solutions across algorithms.
    """
    return np.linalg.det(best_matrix.corr())

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_components', help='dimensionality of z')
parser.add_argument('-p', '--param_config',
                    help='text file optimal hyperparameter assignment for z')
parser.add_argument('-o', '--out_dir', help='where to save the output files')
parser.add_argument('-s', '--num_seeds', default=5,
                    help='number of different seeds to run on current data')
args = parser.parse_args()

# Load command arguments
num_components = int(args.num_components)
param_config = args.param_config
out_dir = args.out_dir
num_seeds = int(args.num_seeds)

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

# Output file names
file_pre = '{}_components_'.format(num_components)
w_file = os.path.join(out_dir, '{}weight_matrix.tsv'.format(file_pre))
z_file = os.path.join(out_dir, '{}z_matrix.tsv'.format(file_pre))
recon_file = os.path.join(out_dir, '{}reconstruction.tsv'.format(file_pre))
w_det = os.path.join(out_dir,
                     '{}weight_matrix_corr_determinant.tsv'.format(file_pre))
z_det = os.path.join(out_dir,
                     '{}z_matrix_corr_determinant.tsv'.format(file_pre))
across_alg_det = os.path.join(out_dir,
                              'alg_corr_determinant.tsv'.format(file_pre))
tybalt_hist_file = os.path.join(out_dir,
                                '{}tybalt_training_hist.tsv'.format(file_pre))
adage_hist_file = os.path.join(out_dir,
                               '{}adage_training_hist.tsv'.format(file_pre))

# Load Data
rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv.gz')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)

# Initialize DataModel class with pancancer data
dm = DataModel(df=rnaseq_df)

# Build models
random_seeds = np.random.randint(0, high=1000000, size=num_seeds)
algorithms = ['pca', 'ica', 'nmf', 'adage', 'tybalt']

z_matrices = []
weight_matrices = []
reconstruction_results = []
tybalt_training_histories = []
adage_training_histories = []
for seed in random_seeds:
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
    adage_train_history = dm.adage_fit.history_df
    adage_train_history = adage_train_history.assign(seed=seed)

    # Store results for each seed
    z_matrices.append(full_z_matrix)
    weight_matrices.append(full_weight_matrix)
    reconstruction_results.append(full_reconstruction)
    tybalt_training_histories.append(tybalt_train_history)
    adage_training_histories.append(adage_train_history)

# Identify models with the lowest loss across seeds and chose these to save
reconstruction_df = pd.concat(reconstruction_results)
reconstruction_df.index = range(0, num_seeds)

# Process the final matrices that are to be stored
final_weight_matrix = get_lowest_loss(weight_matrices, reconstruction_df)
final_z_matrix = get_lowest_loss(z_matrices, reconstruction_df)

# Compile determinants of different correlation matrices
weight_corr_det = get_corr_determinants(weight_matrices)
weight_corr_det.index = [num_components]
z_corr_det = get_corr_determinants(z_matrices)
z_corr_det.index = [num_components]

best_weight_corr_det = full_correlation_determinant(final_weight_matrix)
best_z_corr_det = full_correlation_determinant(final_z_matrix)
best_corr_det_df = pd.DataFrame([best_weight_corr_det, best_z_corr_det],
                                index=['weight', 'z'])
best_corr_det_df.columns = [num_components]

# Save training histories
tybalt_history_df = pd.concat(tybalt_training_histories)
adage_history_df = pd.concat(adage_training_histories)

# Output files
final_weight_matrix.to_csv(w_file, sep='\t')
final_z_matrix.to_csv(z_file, sep='\t')
reconstruction_df.to_csv(recon_file, sep='\t')
weight_corr_det.to_csv(w_det, sep='\t')
z_corr_det.to_csv(z_det, sep='\t')
best_corr_det_df.to_csv(across_alg_det, sep='\t')
tybalt_history_df.to_csv(tybalt_hist_file, sep='\t')
adage_history_df.to_csv(adage_hist_file, sep='\t')
