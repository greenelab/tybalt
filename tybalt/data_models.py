"""
tybalt/data_models.py
2018 Gregory Way

Methods for loading, transforming, and compressing input gene expression data

Usage:

    from tybalt.data_models import DataModel

    dm = DataModel(filename='example_data.tsv')

    # Transform if necessary
    dm.transform(how='zscore')

    # Compress input data with various features
    n_comp = 10
    dm.pca(n_comp)
    dm.ica(n_comp)
    dm.nmf(n_comp)
    dm.nn(n_comp, model='adage')
    dm.nn(n_comp, model='tybalt')
    dm.nn(n_comp, model='ctybalt')

    # Extract compressed data from DataModel object
    # For example,
    pca_out = dm.pca_df

    # Combine all models into a single dataframe
    all_df = dm.combine_models()
"""

import pandas as pd

from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from keras import backend as K
from keras.utils import to_categorical

from tybalt.models import Tybalt, Adage, cTybalt


class DataModel():
    """
    Methods for loading and compressing input data

    Usage:
    from tybalt.data_models import DataModel

    data = DataModel(filename)
    """
    def __init__(self, filename=None, df=False, select_columns=False):
        self.filename = filename
        if filename is None:
            self.df = df
        else:
            self.df = pd.read_table(self.filename)

        if select_columns:
            subset_df = self.df.iloc[:, select_columns]
            other_columns = range(max(select_columns) + 1, self.df.shape[1])
            self.other_df = self.df.iloc[:, other_columns]
            self.df = subset_df

    def transform(self, how):
        self.transformation = how
        if how == 'zscore':
            self.transform_fit = StandardScaler().fit(self.df)
        elif how == 'zeroone':
            self.transform_fit = MinMaxScaler().fit(self.df)
        else:
            raise ValueError('how must be either "zscore" or "zeroone".')

        self.df = pd.DataFrame(self.transform_fit.transform(self.df),
                               index=self.df.index,
                               columns=self.df.columns)

    def pca(self, n_components, transform_df=False):
        self.pca_fit = decomposition.PCA(n_components=n_components)
        self.pca_df = self.pca_fit.fit_transform(self.df)
        colnames = ['pca_{}'.format(x) for x in range(0, n_components)]
        self.pca_df = pd.DataFrame(self.pca_df, index=self.df.index,
                                   columns=colnames)
        if transform_df:
            out_df = self.pca_fit.transform(transform_df)
            return out_df

    def ica(self, n_components, transform_df=False):
        self.ica_fit = decomposition.FastICA(n_components=n_components)
        self.ica_df = self.ica_fit.fit_transform(self.df)
        colnames = ['ica_{}'.format(x) for x in range(0, n_components)]
        self.ica_df = pd.DataFrame(self.ica_df, index=self.df.index,
                                   columns=colnames)
        if transform_df:
            out_df = self.ica_fit.transform(transform_df)
            return out_df

    def nmf(self, n_components, transform_df=False, init='nndsvdar', tol=5e-3):
        self.nmf_fit = decomposition.NMF(n_components=n_components, init=init,
                                         tol=tol)
        self.nmf_df = self.nmf_fit.fit_transform(self.df)
        colnames = ['nmf_{}'.format(x) for x in range(0, n_components)]
        self.nmf_df = pd.DataFrame(self.nmf_df, index=self.df.index,
                                   columns=colnames)
        if transform_df:
            out_df = self.nmf_fit.transform(transform_df)
            return out_df

    def nn(self, n_components, model='tybalt', transform_df=False, **kwargs):
        # unpack kwargs
        original_dim = kwargs.pop('original_dim', self.df.shape[1])
        latent_dim = kwargs.pop('latent_dim', n_components)
        batch_size = kwargs.pop('batch_size', 50)
        epochs = kwargs.pop('epochs', 50)
        learning_rate = kwargs.pop('learning_rate', 0.0005)
        noise = kwargs.pop('noise', 0)
        sparsity = kwargs.pop('sparsity', 0)
        kappa = kwargs.pop('kappa', 1)
        epsilon_std = kwargs.pop('epsilon_std', 1.0)
        beta = kwargs.pop('beta', 0)
        beta = K.variable(beta)
        loss = kwargs.pop('loss', 'binary_crossentropy')
        validation_ratio = kwargs.pop('validation_ratio', 0.1)

        # Extra processing for conditional vae
        if hasattr(self, 'other_df') and model == 'ctybalt':
            y_df = kwargs.pop('y_df', self.other_df)
            y_var = kwargs.pop('y_var', 'groups')
            label_dim = kwargs.pop('label_dim', len(set(y_df[y_var])))

            self.nn_train_y = y_df.drop(self.nn_test_df.index)
            self.nn_test_y = y_df.drop(self.nn_train_df.index)
            self.nn_train_y = self.nn_train_y.loc[self.nn_train_df.index, ]
            self.nn_test_y = self.nn_test_y.loc[self.nn_test_df.index, ]

            label_encoder = LabelEncoder().fit(self.other_df[y_var])

            self.nn_train_y = label_encoder.transform(self.nn_train_y[y_var])
            self.nn_test_y = label_encoder.transform(self.nn_test_y[y_var])
            self.other_onehot = label_encoder.transform(self.other_df[y_var])

            self.nn_train_y = to_categorical(self.nn_train_y)
            self.nn_test_y = to_categorical(self.nn_test_y)
            self.other_onehot = to_categorical(self.other_onehot)

        self.nn_test_df = self.df.sample(frac=validation_ratio)
        self.nn_train_df = self.df.drop(self.nn_test_df.index)

        if model == 'tybalt':
            self.tybalt_fit = Tybalt(original_dim=original_dim,
                                     latent_dim=latent_dim,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     learning_rate=learning_rate,
                                     kappa=kappa,
                                     epsilon_std=epsilon_std,
                                     beta=beta,
                                     loss=loss)
            self.tybalt_fit.initialize_model()
            self.tybalt_fit.train_vae(train_df=self.nn_train_df,
                                      test_df=self.nn_test_df)
            self.tybalt_weights = self.tybalt_fit.get_decoder_weights()

            self.tybalt_df = self.tybalt_fit.compress(self.df)
            colnames = ['vae_{}'.format(x) for x in range(0, latent_dim)]
            self.tybalt_df.columns = colnames

            if transform_df:
                out_df = self.tybalt_fit.compress(transform_df)
                return out_df

        if model == 'ctybalt':
            self.ctybalt_fit = cTybalt(original_dim=original_dim,
                                       latent_dim=latent_dim,
                                       label_dim=label_dim,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       learning_rate=learning_rate,
                                       kappa=kappa,
                                       epsilon_std=epsilon_std,
                                       beta=beta,
                                       loss=loss)
            self.ctybalt_fit.initialize_model()
            self.ctybalt_fit.train_cvae(train_df=self.nn_train_df,
                                        train_labels_df=self.nn_train_y,
                                        test_df=self.nn_test_df,
                                        test_labels_df=self.nn_test_y)
            self.ctybalt_weights = self.ctybalt_fit.get_decoder_weights()

            self.ctybalt_df = self.ctybalt_fit.compress([self.df,
                                                         self.other_onehot])
            colnames = ['cvae_{}'.format(x) for x in range(0, latent_dim)]
            self.ctybalt_df.columns = colnames

            if transform_df:
                # Note: transform_df must be a list of two dfs [x_df, y_df]
                out_df = self.ctybalt_fit.compress(transform_df)
                return out_df

        if model == 'adage':
            self.adage_fit = Adage(original_dim=original_dim,
                                   latent_dim=latent_dim,
                                   noise=noise,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   sparsity=sparsity,
                                   learning_rate=learning_rate,
                                   loss=loss)
            self.adage_fit.initialize_model()
            self.adage_fit.train_adage(train_df=self.nn_train_df,
                                       test_df=self.nn_test_df)
            self.adage_weights = self.adage_fit.get_decoder_weights()

            self.adage_df = self.adage_fit.compress(self.df)
            colnames = ['dae_{}'.format(x) for x in range(0, latent_dim)]
            self.adage_df.columns = colnames
            if transform_df:
                out_df = self.adage_fit.compress(transform_df)
                return out_df

    def combine_models(self, include_labels=False, include_raw=False):
        all_models = []
        if hasattr(self, 'pca_df'):
            all_models += [self.pca_df]
        if hasattr(self, 'ica_df'):
            all_models += [self.ica_df]
        if hasattr(self, 'nmf_df'):
            all_models += [self.nmf_df]
        if hasattr(self, 'tybalt_df'):
            all_models += [self.tybalt_df]
        if hasattr(self, 'ctybalt_df'):
            all_models += [self.ctybalt_df]
        if hasattr(self, 'adage_df'):
            all_models += [self.adage_df]

        if include_raw:
            all_models += [self.df]

        if include_labels:
            all_models += [self.other_df]

        all_df = pd.concat(all_models, axis=1)
        return all_df
