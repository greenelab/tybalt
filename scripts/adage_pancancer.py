"""
Gregory Way 2017
Variational Autoencoder - Pan Cancer
scripts/adage_pancancer.py

Comparing a VAE learned features to ADAGE features. Use this script within
the context of a parameter sweep to compare performance across a grid of
hyper parameters.

Usage:

    Run in command line with required command arguments:

        python scripts/adage_pancancer.py --learning_rate
                                          --batch_size
                                          --epochs
                                          --sparsity
                                          --noise
                                          --output_filename

    Typically, arguments to this script are compiled automatically by:

        python scripts/vae_paramsweep.py --parameter_file <parameter-filepath>
                                         --config_file <configuration-filepath>

Output:
Loss and validation loss for the specific model trained
"""

import os
import argparse
import numpy as np
import pandas as pd

from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras import optimizers
from keras.regularizers import l1

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--learning_rate',
                    help='learning rate of the optimizer')
parser.add_argument('-b', '--batch_size',
                    help='Number of samples to include in each learning batch')
parser.add_argument('-e', '--epochs',
                    help='How many times to cycle through the full dataset')
parser.add_argument('-s', '--sparsity',
                    help='How much L1 regularization penalty to apply')
parser.add_argument('-n', '--noise',
                    help='How much Gaussian noise to add during training')
parser.add_argument('-f', '--output_filename',
                    help='The name of the file to store results')
args = parser.parse_args()

# Set hyper parameters
learning_rate = float(args.learning_rate)
batch_size = int(args.batch_size)
epochs = int(args.epochs)
sparsity = float(args.sparsity)
noise = float(args.noise)
output_filename = args.output_filename

# Random seed
seed = int(np.random.randint(low=0, high=10000, size=1))
np.random.seed(seed)

# Load Data
rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv.gz')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)

original_dim = rnaseq_df.shape[1]
latent_dim = 100

# Split 10% test set randomly
test_set_percent = 0.1
rnaseq_test_df = rnaseq_df.sample(frac=test_set_percent)
rnaseq_train_df = rnaseq_df.drop(rnaseq_test_df.index)

# Input place holder for RNAseq data with specific input size
input_rnaseq = Input(shape=(original_dim, ))
encoded_rnaseq = Dropout(noise)(input_rnaseq)
encoded_rnaseq_2 = Dense(latent_dim,
                         activity_regularizer=l1(sparsity))(encoded_rnaseq)
activation = Activation('relu')(encoded_rnaseq_2)
decoded_rnaseq = Dense(original_dim, activation='sigmoid')(activation)

autoencoder = Model(input_rnaseq, decoded_rnaseq)

adadelta = optimizers.Adadelta(lr=learning_rate)
autoencoder.compile(optimizer=adadelta, loss='mse')

hist = autoencoder.fit(np.array(rnaseq_train_df), np.array(rnaseq_train_df),
                       shuffle=True,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(np.array(rnaseq_test_df),
                                        np.array(rnaseq_test_df)))

# Save training performance
history_df = pd.DataFrame(hist.history)
history_df = history_df.assign(learning_rate=learning_rate)
history_df = history_df.assign(batch_size=batch_size)
history_df = history_df.assign(epochs=epochs)
history_df = history_df.assign(sparsity=sparsity)
history_df = history_df.assign(noise=noise)
history_df = history_df.assign(seed=seed)
history_df.to_csv(output_filename, sep='\t')
