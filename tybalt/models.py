"""
tybalt/models.py
2017 Gregory Way

Functions enabling the construction and usage of a Tybalt model
"""

import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K
from keras import optimizers
from keras.utils import plot_model
from keras.layers import Input, Dense, Lambda, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l1

from tybalt.utils.vae_utils import VariationalLayer, WarmUpCallback


class BaseModel():
    def __init__(self):
        pass

    def get_summary(self):
        self.full_model.summary()

    def visualize_architecture(self, output_file):
        # Visualize the connections of the custom VAE model
        plot_model(self.full_model, to_file=output_file)

    def visualize_training(self, output_file=None):
        # Visualize training performance
        history_df = pd.DataFrame(self.hist.history)
        ax = history_df.plot()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        fig = ax.get_figure()
        if output_file:
            fig.savefig(output_file)
        else:
            fig.show()

    def compress(self, df):
        # Encode rnaseq into the hidden/latent representation - and save output
        encoded_df = self.encoder(np.array(df))
        encoded_df = pd.DataFrame(encoded_df,
                                  columns=range(1, self.latent_dim + 1),
                                  index=df.index)
        return encoded_df

    def get_decoder_weights(self):
        # build a generator that can sample from the learned distribution
        # can generate from any sampled z vector
        weights = []
        for layer in self.decoder.layers:
            weights.append(layer.get_weights())
        return(weights)

    def predict(self, df):
        return self.decoder.predict(np.array(df))

    def save_models(self, encoder_file, decoder_file):
        self.encoder.save(encoder_file)
        self.decoder.save(decoder_file)


class Tybalt(BaseModel):
    """
    Training and evaluation of a tybalt model

    Usage: from tybalt.models import Tybalt
    """
    def __init__(self, original_dim, latent_dim, batch_size=50, epochs=50,
                 learning_rate=0.0005, kappa=1, epsilon_std=1.0,
                 beta=K.variable(0), loss='binary_crossentropy'):
        BaseModel.__init__(self)
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.kappa = kappa
        self.epsilon_std = epsilon_std
        self.beta = beta
        self.loss = loss

    def _sampling(self, args):
        """
        Function for reparameterization trick to make model differentiable
        """
        # Function with args required for Keras Lambda function
        z_mean, z_log_var = args

        # Draw epsilon of the same shape from a standard normal distribution
        epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0.,
                                  stddev=self.epsilon_std)

        # The latent vector is non-deterministic and differentiable
        # in respect to z_mean and z_log_var
        z = z_mean + K.exp(z_log_var / 2) * epsilon
        return z

    def initialize_model(self):
        """
        Helper function to run that builds and compiles Keras layers
        """
        self.build_encoder_layer()
        self.build_decoder_layer()
        self.compile_vae()

    def build_encoder_layer(self):
        """
        Function to build the encoder layer connections
        """
        # Input place holder for RNAseq data with specific input size
        self.rnaseq_input = Input(shape=(self.original_dim, ))

        # Input layer is compressed into a mean and log variance vector of
        # size `latent_dim`. Each layer is initialized with glorot uniform
        # weights and each step (dense connections, batch norm, and relu
        # activation) are funneled separately.
        # Each vector are connected to the rnaseq input tensor

        # input layer to latent mean layer
        z_mean = Dense(self.latent_dim,
                       kernel_initializer='glorot_uniform')(self.rnaseq_input)
        z_mean_batchnorm = BatchNormalization()(z_mean)
        self.z_mean_encoded = Activation('relu')(z_mean_batchnorm)

        # input layer to latent standard deviation layer
        z_var = Dense(self.latent_dim,
                      kernel_initializer='glorot_uniform')(self.rnaseq_input)
        z_var_batchnorm = BatchNormalization()(z_var)
        self.z_var_encoded = Activation('relu')(z_var_batchnorm)

        # return the encoded and randomly sampled z vector
        # Takes two keras layers as input to the custom sampling function layer
        self.z = Lambda(self._sampling,
                        output_shape=(self.latent_dim, ))([self.z_mean_encoded,
                                                           self.z_var_encoded])

    def build_decoder_layer(self):
        """
        Function to build the decoder layer connections
        """
        # The decoding layer is much simpler with a single layer glorot uniform
        # initialized and sigmoid activation
        self.decoder_model = Sequential()
        self.decoder_model.add(Dense(self.original_dim, activation='sigmoid',
                                     input_dim=self.latent_dim))
        self.rnaseq_reconstruct = self.decoder_model(self.z)

    def compile_vae(self):
        """
        Creates the vae layer and compiles all layer connections
        """
        adam = optimizers.Adam(lr=self.learning_rate)
        vae_layer = VariationalLayer(var_layer=self.z_var_encoded,
                                     mean_layer=self.z_mean_encoded,
                                     original_dim=self.original_dim,
                                     beta=self.beta, loss=self.loss)(
                                [self.rnaseq_input, self.rnaseq_reconstruct])
        self.full_model = Model(self.rnaseq_input, vae_layer)
        self.full_model.compile(optimizer=adam, loss=None,
                                loss_weights=[self.beta])

    def train_vae(self, train_df, test_df):
        self.hist = self.full_model.fit(np.array(train_df),
                                        shuffle=True,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        validation_data=(np.array(test_df),
                                                         np.array(test_df)),
                                        callbacks=[WarmUpCallback(self.beta,
                                                                  self.kappa)])

    def connect_layers(self):
        # Make connections between layers to build separate encoder and decoder
        self.encoder = Model(self.rnaseq_input, self.z_mean_encoded)

        decoder_input = Input(shape=(self.latent_dim, ))
        _x_decoded_mean = self.decoder_model(decoder_input)
        self.decoder = Model(decoder_input, _x_decoded_mean)

    def compress(self, df):
        # Encode rnaseq into the hidden/latent representation - and save output
        encoded_df = self.encoder.predict_on_batch(df)
        encoded_df = pd.DataFrame(encoded_df,
                                  columns=range(1, self.latent_dim + 1),
                                  index=df.index)
        return encoded_df


class Adage(BaseModel):
    """
    Training and evaluation of an ADAGE model

    Usage: from tybalt.models import Adage
    """
    def __init__(self, original_dim, latent_dim, noise=0.05, batch_size=50,
                 epochs=100, sparsity=0, learning_rate=1.1,
                 loss='mse'):
        BaseModel.__init__(self)
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.noise = noise
        self.batch_size = batch_size
        self.epochs = epochs
        self.sparsity = sparsity
        self.learning_rate = learning_rate
        self.loss = loss

    def initialize_model(self):
        """
        Helper function to run that builds and compiles Keras layers
        """
        self.build_graph()
        self.connect_layers()
        self.compile_adage()

    def build_graph(self):
        # Build the Keras graph for an ADAGE model
        self.input_rnaseq = Input(shape=(self.original_dim, ))
        drop = Dropout(self.noise)(self.input_rnaseq)
        self.encoded = Dense(self.latent_dim,
                             activity_regularizer=l1(self.sparsity))(drop)
        activation = Activation('relu')(self.encoded)
        decoded_rnaseq = Dense(self.original_dim,
                               activation='sigmoid')(activation)

        self.full_model = Model(self.input_rnaseq, decoded_rnaseq)

    def connect_layers(self):
        # Separate out the encoder and decoder model
        self.encoder = Model(self.input_rnaseq, self.encoded)

        encoded_input = Input(shape=(self.latent_dim, ))
        decoder_layer = self.full_model.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

    def compile_adage(self):
        # Compile the autoencoder to prepare for training
        adadelta = optimizers.Adadelta(lr=self.learning_rate)
        self.full_model.compile(optimizer=adadelta, loss=self.loss)

    def train_adage(self, train_df, test_df):
        self.hist = self.full_model.fit(np.array(train_df), np.array(train_df),
                                        shuffle=True,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        validation_data=(np.array(test_df),
                                                         np.array(test_df)))

    def compress(self, df):
        # Encode rnaseq into the hidden/latent representation - and save output
        encoded_df = self.encoder.predict(np.array(df))
        encoded_df = pd.DataFrame(encoded_df, index=df.index,
                                  columns=range(1, self.latent_dim + 1))
        return encoded_df
