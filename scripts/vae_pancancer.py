"""
Gregory Way 2017
Variational Autoencoder - Pan Cancer
scripts/vae_pancancer.py

Usage: Run in command line with required command arguments:

        python scripts/vae_pancancer.py --parameter_file <parameter-filepath>
                                         --config_file <configuration-filepath>

Output:
TBD
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.callbacks import Callback
import keras

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--learning_rate',
                    help='learning rate of the optimizer')
parser.add_argument('-b', '--batch_size',
                    help='Number of samples to include in each learning batch')
parser.add_argument('-e', '--epochs',
                    help='How many times to cycle through the full dataset')
parser.add_argument('-f', '--output_filename',
                    help='The name of the file to store results')
args = parser.parse_args()

learning_rate = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
output_filename = args.output_filename

test = [learning_rate, batch_size, epochs]
test = pd.DataFrame(test, columns=['param'],
                    index=['learning_rate', 'batch_size', 'epochs'])

test.to_csv(output_filename, sep='\t')
