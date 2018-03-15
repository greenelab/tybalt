"""
run_simulation.py
2018 Gregory Way

Compare latent space arithmetic of various compression models with different
simulated data

Usage: Run on the command line with required arguments:

    python run_simulation.py --data <DATA FILE>
                             --noise <AMOUNT OF NOISE ADDED>
                             --sample_size <NUMBER OF SAMPLES>
                             --num_genes <NUMBER OF GENES>
                             --out_file <SAVE LOCATION>

    And several optional parameters that have reasonable defaults:

        --seeds             How many times to compress input data
        --transform         How to transform the input data file
        --num_components    How many components to compress input down to
        --vae_epochs        How many times to loop input data to train VAE
        --adage_epochs      How many times to loop input data to train ADAGE
        --batch_size        How many samples to include after each weight
                            update in neural network training
        --vae_learning_rate         Variational Autoencoder learing rate
        --adage_learning_rate       ADAGE learning rate
        --loss              the loss function to minimize (str)
        --verbose           Whether or not to print training progress in neural
                            network models

    Note that this script does not generate the simulated data. The data is
    generated in scripts/util/simulate_expression.R.
"""

import argparse
import numpy as np
import pandas as pd
from tybalt.data_models import DataModel

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help='location of data file')
parser.add_argument('-o', '--noise', help='amount of noise injection')
parser.add_argument('-n', '--sample_size', help='number of samples')
parser.add_argument('-g', '--num_genes', help='number of genes')
parser.add_argument('-f', '--out_file', help='where to save output results')
parser.add_argument('-s', '--seeds', default=5,
                    help='number of different seeds to run on current data')
parser.add_argument('-t', '--transform', default='zeroone',
                    help='how to transform the data',
                    choices=['zeroone', 'zscore'])
parser.add_argument('-c', '--num_components', default=6,
                    help='length of the latent space features')
parser.add_argument('-ve', '--vae_epochs', default=60,
                    help='how many epochs for VAE models')
parser.add_argument('-ae', '--adage_epochs', default=60,
                    help='how many epochs for ADAGE models')
parser.add_argument('-b', '--batch_size', default=30,
                    help='How many samples to use to update weights')
parser.add_argument('-r', '--vae_learning_rate', default=0.0005,
                    help='learning rate for the VAE models')
parser.add_argument('-a', '--adage_learning_rate', default=0.0005,
                    help='learning Rate for the ADAGE models')
parser.add_argument('-l', '--loss', default='mse',
                    help='What loss function to optimize for the NN models',
                    choices=['mse', 'binary_crossentropy'])
parser.add_argument('-v', '--verbose', action='store_true',
                    help='output training history for NN models')
args = parser.parse_args()

# Load command arguments
data_file = args.data
noise = float(args.noise)
sample_size = int(args.sample_size)
num_genes = int(args.num_genes)
out_file = args.out_file
num_seeds = int(args.seeds)
how_transform = args.transform  # Note that NMF will not work with `zscore`
n_components = int(args.num_components)
vae_epochs = int(args.vae_epochs)
adage_epochs = int(args.adage_epochs)
batch_size = int(args.batch_size)
vae_learning_rate = float(args.vae_learning_rate)
adage_learning_rate = float(args.adage_learning_rate)
loss = args.loss
verbose = args.verbose

# Load and process data
sim_df = pd.read_table(data_file)
select_columns = range(0, sim_df.shape[1] - 1)
groups = sim_df['groups']
gene_modules = sim_df.iloc[0, range(0, sim_df.shape[1] - 1)]

sim_sub_df = sim_df.iloc[range(1, sim_df.shape[0]), :]

data_model = DataModel(df=sim_sub_df, select_columns=select_columns,
                       gene_modules=gene_modules)
data_model.transform(how=how_transform)

# Real target output
real_group_data = (
    pd.concat([data_model.df, data_model.other_df], axis=1)
    .query('groups == "C"').drop('groups', axis=1)
)

# Get random seeds
random_seeds = np.random.randint(0, high=1000000, size=num_seeds)

all_results = []
for seed in random_seeds:
    data_model.pca(n_components=n_components)
    data_model.ica(n_components=n_components)
    data_model.nmf(n_components=n_components)
    data_model.nn(n_components=n_components, model='tybalt', loss=loss,
                  epochs=vae_epochs, batch_size=batch_size,
                  learning_rate=vae_learning_rate, verbose=verbose)
    data_model.nn(n_components=n_components, model='adage', loss=loss,
                  epochs=adage_epochs, batch_size=batch_size,
                  learning_rate=adage_learning_rate, verbose=verbose)

    # Get eval results
    eval_results = data_model.subtraction_eval(noise_column=0,
                                               num_components=n_components,
                                               group_list=['A', 'B'],
                                               add_groups='D',
                                               expect_node=2,
                                               real_df=real_group_data)
    # Append seed and noise info to results
    eval_results = eval_results.assign(seed=seed)
    eval_results = eval_results.assign(noise=noise)
    eval_results = eval_results.assign(sample_size=sample_size)
    eval_results = eval_results.assign(num_genes=num_genes)
    eval_results = eval_results.assign(data_file=data_file)

    # Grow results
    all_results.append(eval_results)

# Write out file
all_results_df = pd.concat(all_results)
all_results_df.to_csv(out_file, sep='\t')
