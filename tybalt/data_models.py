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
from scipy.stats.mstats import zscore

from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from keras import backend as K
from keras.utils import to_categorical

from tybalt.models import Tybalt, Adage, cTybalt
from tybalt.utils.vae_utils import approx_keras_binary_cross_entropy


class DataModel():
    """
    Methods for loading and compressing input data

    Usage:
    from tybalt.data_models import DataModel

    data = DataModel(filename)
    """
    def __init__(self, filename=None, df=False, select_columns=False,
                 gene_modules=None):
        """
        DataModel can be initialized with either a filename or a pandas
        dataframe and processes gene modules and sample labels if provided.

        Arguments:

        filename - if provided, load gene expression data into object
        df - dataframe of preloaded gene expression data
        select_columns - the columns of the dataframe to use
        gene_modules - a list of gene module assignments for each gene (for use
        with the simulated data or when ground truth gene modules are known)
        """
        self.filename = filename
        if filename is None:
            self.df = df
        else:
            self.df = pd.read_table(self.filename, index_col=0)

        if select_columns:
            subset_df = self.df.iloc[:, select_columns]
            other_columns = range(max(select_columns) + 1, self.df.shape[1])
            self.other_df = self.df.iloc[:, other_columns]
            self.df = subset_df

        if gene_modules is not None:
            self.gene_modules = pd.DataFrame(gene_modules).T
            self.gene_modules.index = ['modules']

        self.num_samples, self.num_genes = self.df.shape

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
        self.pca_weights = pd.DataFrame(self.pca_fit.components_,
                                        columns=self.df.columns,
                                        index=colnames)
        if transform_df:
            out_df = self.pca_fit.transform(transform_df)
            return out_df

    def ica(self, n_components, transform_df=False):
        self.ica_fit = decomposition.FastICA(n_components=n_components)
        self.ica_df = self.ica_fit.fit_transform(self.df)
        colnames = ['ica_{}'.format(x) for x in range(0, n_components)]
        self.ica_df = pd.DataFrame(self.ica_df, index=self.df.index,
                                   columns=colnames)
        self.ica_weights = pd.DataFrame(self.ica_fit.components_,
                                        columns=self.df.columns,
                                        index=colnames)

        if transform_df:
            out_df = self.ica_fit.transform(transform_df)
            return out_df

    def nmf(self, n_components, transform_df=False, init='nndsvdar', tol=5e-3):
        self.nmf_fit = decomposition.NMF(n_components=n_components, init=init,
                                         tol=tol)
        self.nmf_df = self.nmf_fit.fit_transform(self.df)
        colnames = ['nmf_{}'.format(x) for x in range(n_components)]

        self.nmf_df = pd.DataFrame(self.nmf_df, index=self.df.index,
                                   columns=colnames)
        self.nmf_weights = pd.DataFrame(self.nmf_fit.components_,
                                        columns=self.df.columns,
                                        index=colnames)
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
        tied_weights = kwargs.pop('tied_weights', True)
        if tied_weights and model == 'adage':
            use_decoder_weights = False
        else:
            use_decoder_weights = True
        verbose = kwargs.pop('verbose', True)
        tybalt_separate_loss = kwargs.pop('separate_loss', False)
        adage_comp_loss = kwargs.pop('multiply_adage_loss', False)
        adage_optimizer = kwargs.pop('adage_optimizer', 'adam')

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
                                     loss=loss,
                                     verbose=verbose)
            self.tybalt_fit.initialize_model()
            self.tybalt_fit.train_vae(train_df=self.nn_train_df,
                                      test_df=self.nn_test_df,
                                      separate_loss=tybalt_separate_loss)

            features = ['vae_{}'.format(x) for x in range(0, latent_dim)]
            self.tybalt_weights = (
                self.tybalt_fit.get_weights(decoder=use_decoder_weights)
                )
            self.tybalt_weights = pd.DataFrame(self.tybalt_weights[1][0],
                                               columns=self.df.columns,
                                               index=features)

            self.tybalt_df = self.tybalt_fit.compress(self.df)
            self.tybalt_df.columns = features
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
                                       loss=loss,
                                       verbose=verbose)
            self.ctybalt_fit.initialize_model()
            self.ctybalt_fit.train_cvae(train_df=self.nn_train_df,
                                        train_labels_df=self.nn_train_y,
                                        test_df=self.nn_test_df,
                                        test_labels_df=self.nn_test_y)
            self.ctybalt_decoder_w = (
                self.ctybalt_fit.get_weights(decoder=use_decoder_weights)
                )

            features = ['cvae_{}'.format(x) for x in range(0, latent_dim)]
            features_with_groups = features + ['group_{}'.format(x) for x in
                                               range(latent_dim,
                                                     latent_dim + label_dim)]

            w = pd.DataFrame(self.ctybalt_decoder_w[1][0])
            self.ctybalt_group_w = pd.DataFrame(w.iloc[:, -label_dim:])

            gene_range = range(0, w.shape[1] - label_dim)
            self.ctybalt_weights = pd.DataFrame(w.iloc[:, gene_range])
            self.ctybalt_weights.columns = self.df.columns
            self.ctybalt_weights.index = features_with_groups

            self.ctybalt_df = self.ctybalt_fit.compress([self.df,
                                                         self.other_onehot])
            self.ctybalt_df.columns = features
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
                                   loss=loss,
                                   verbose=verbose,
                                   tied_weights=tied_weights,
                                   optimizer=adage_optimizer)
            self.adage_fit.initialize_model()
            self.adage_fit.train_adage(train_df=self.nn_train_df,
                                       test_df=self.nn_test_df,
                                       adage_comparable_loss=adage_comp_loss)

            features = ['dae_{}'.format(x) for x in range(0, latent_dim)]
            self.adage_weights = (
                self.adage_fit.get_weights(decoder=use_decoder_weights)
                )
            self.adage_weights = pd.DataFrame(self.adage_weights[1][0],
                                              columns=self.df.columns,
                                              index=features)

            self.adage_df = self.adage_fit.compress(self.df)
            self.adage_df.columns = features
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

    def combine_weight_matrix(self):
        all_weight = []
        if hasattr(self, 'pca_df'):
            all_weight += [self.pca_weights]
        if hasattr(self, 'ica_df'):
            all_weight += [self.ica_weights]
        if hasattr(self, 'nmf_df'):
            all_weight += [self.nmf_weights]
        if hasattr(self, 'tybalt_df'):
            all_weight += [self.tybalt_weights]
        if hasattr(self, 'ctybalt_df'):
            all_weight += [self.ctybalt_weights]
        if hasattr(self, 'adage_df'):
            all_weight += [self.adage_weights]

        all_weight_df = pd.concat(all_weight, axis=0).T
        return all_weight_df

    def compile_reconstruction(self):
        all_reconstruction = {}
        if hasattr(self, 'pca_df'):
            pca_reconstruct = self.pca_fit.inverse_transform(self.pca_df)
            pca_recon = approx_keras_binary_cross_entropy(pca_reconstruct,
                                                          self.df,
                                                          self.num_genes)
            all_reconstruction['pca'] = [pca_recon]
        if hasattr(self, 'ica_df'):
            ica_reconstruct = self.ica_fit.inverse_transform(self.ica_df)
            ica_recon = approx_keras_binary_cross_entropy(ica_reconstruct,
                                                          self.df,
                                                          self.num_genes)
            all_reconstruction['ica'] = [ica_recon]
        if hasattr(self, 'nmf_df'):
            nmf_reconstruct = self.nmf_fit.inverse_transform(self.nmf_df)
            nmf_recon = approx_keras_binary_cross_entropy(nmf_reconstruct,
                                                          self.df,
                                                          self.num_genes)
            all_reconstruction['nmf'] = [nmf_recon]
        if hasattr(self, 'tybalt_df'):
            vae_reconstruct = self.tybalt_fit.decoder.predict_on_batch(
                self.tybalt_fit.encoder.predict_on_batch(self.df)
                )
            vae_recon = approx_keras_binary_cross_entropy(vae_reconstruct,
                                                          self.df,
                                                          self.num_genes)
            all_reconstruction['vae'] = [vae_recon]
        if hasattr(self, 'adage_df'):
            dae_reconstruct = self.adage_fit.decoder.predict_on_batch(
                self.adage_fit.encoder.predict_on_batch(self.df)
                )
            dae_recon = approx_keras_binary_cross_entropy(dae_reconstruct,
                                                          self.df,
                                                          self.num_genes)
            all_reconstruction['dae'] = [dae_recon]

        return pd.DataFrame(all_reconstruction)

    def get_modules_ranks(self, weight_df, num_components, noise_column=0):
        """
        Takes a compression algorithm's weight matrix (gene by latent feature),
        and reports two performance metrics:
            1) mean rank sum across modules:
               measures how well modules are separated into features
            2) min average rank across modules:
               measures how well modules of decreasing proportion are captured
        """
        # Rank absolute value compressed features for each gene
        weight_rank_df = weight_df.abs().rank(axis=0, ascending=False)

        # Add gene module membership to ranks
        module_w_df = pd.concat([weight_rank_df, self.gene_modules], axis=0)
        module_w_df = module_w_df.astype(int)

        # Get the total module by compressed feature mean rank
        module_meanrank_df = (module_w_df.T.groupby('modules').mean()).T

        # Drop the noise column and get the sum of the minimum mean rank.
        # This heuristic measures, on average, how well individual compressed
        # features capture ground truth gene modules. A lower number indicates
        # better separation performance for the algorithm of interest
        module_meanrank_minsum = (
            module_meanrank_df.drop(noise_column, axis=1).min(axis=0).sum()
            )

        # Process output data
        # Divide this by the total number of features in the model. Subtract by
        # one to account for the dropped noise column, if applicable
        module_meanrank_minavg = module_meanrank_minsum / (num_components - 1)

        # We are interested if the features encapsulate gene modules
        # A lower number across modules indicates a stronger ability to
        # aggregate genes into features
        module_min_rank = pd.DataFrame(module_meanrank_df.min(),
                                       columns=['min_rank'])

        return module_meanrank_df, module_min_rank, module_meanrank_minavg

    def get_group_means(self, df):
        """
        Get the mean latent space vector representation of input groups
        """
        return df.assign(groups=self.other_df).groupby('groups').mean()

    def get_subtraction(self, group_means, group_list):
        """
        Subtract two group means given by group list
        """
        a, b = group_list

        a_df = group_means.loc[a, :]
        b_df = group_means.loc[b, :]

        subtraction = pd.DataFrame(a_df - b_df).T

        return subtraction

    def subtraction_essense(self, group_subtract, mean_rank, node):
        """
        Obtain the difference between the subtraction and node of interest
        """
        # subset mean rank to node of the "dropped" feature in the simulation
        feature_essense = mean_rank.loc[:, node]

        # The node essense is the compressed feature with the lowest mean rank
        node_essense = feature_essense.idxmin()
        node_idx = int(node_essense.split('_')[1])

        # Ask how different this specific feature is from all others
        group_z = zscore(group_subtract.iloc[0, :].tolist())
        node_essense_zscore = group_z[node_idx]

        return node_essense_zscore

    def get_addition(self, group_means, subtraction, group):
        """
        Add node to subtraction
        """
        mean_feature = group_means.loc[group, :]
        return subtraction + mean_feature

    def reconstruct_group(self, lsa_result, algorithm=False):
        """
        Reconstruct the latent space arithmetic result back to input dim
        """
        if algorithm == 'tybalt':
            return self.tybalt_fit.decoder.predict_on_batch(lsa_result)
        elif algorithm == 'ctybalt':
            return self.ctybalt_fit.decoder.predict_on_batch(lsa_result)
        elif algorithm == 'adage':
            return self.adage_fit.decoder.predict_on_batch(lsa_result)
        elif algorithm == 'pca':
            return self.pca_fit.inverse_transform(lsa_result)
        elif algorithm == 'ica':
            return self.ica_fit.inverse_transform(lsa_result)
        elif algorithm == 'nmf':
            return self.nmf_fit.inverse_transform(lsa_result)
        else:
            raise ValueError('algorithm must be one of: "pca", "ica", "nmf",' +
                             ' "adage", "tybalt", or "ctybalt"')

    def get_average_distance(self, transform_df, real_df):
        """
        Obtain the average euclidean distance between the transformed vector
        and all samples as part of the real dataframe
        """
        return euclidean_distances(transform_df, real_df).mean()

    def _wrap_sub_eval(self, weight_df, compress_df, num_components,
                       noise_column, subtraction_groups, addition_group, node,
                       real_df, algorithm):
        """
        Helper function that wraps all evals
        """
        # Get the module mean rank and the min rank sum
        mean_rank_mod, mean_rank_min, min_rank_avg = (
            self.get_modules_ranks(weight_df, num_components, noise_column)
        )

        # Begin subtraction analysis - first, get group means
        group_means = self.get_group_means(compress_df)

        # Next, get the subtraction result
        sub_result = self.get_subtraction(group_means, subtraction_groups)

        # Then, get the relative minimum difference to determine if the
        # subtraction isolates the feature we expect is should - and how much
        relative_min_diff = self.subtraction_essense(sub_result,
                                                     mean_rank_mod, node)

        # Now reconstruct the subtraction back into original space
        lsa_result = self.get_addition(group_means, sub_result, addition_group)
        recon_lsa = self.reconstruct_group(lsa_result, algorithm)
        avg_dist = self.get_average_distance(recon_lsa, real_df)

        out_results = mean_rank_min.T
        out_results.index = algorithm.split()
        out_results = out_results.assign(minimum_rank_avg=min_rank_avg)
        out_results = out_results.assign(min_node_zscore=relative_min_diff)
        out_results = out_results.assign(avg_recon_dist=avg_dist)

        return out_results

    def subtraction_eval(self, num_components, noise_column, group_list,
                         add_groups, expect_node, real_df):
        tybalt_results = self._wrap_sub_eval(weight_df=self.tybalt_weights,
                                             compress_df=self.tybalt_df,
                                             num_components=num_components,
                                             noise_column=noise_column,
                                             subtraction_groups=group_list,
                                             addition_group=add_groups,
                                             node=expect_node,
                                             real_df=real_df,
                                             algorithm='tybalt')

        adage_results = self._wrap_sub_eval(weight_df=self.adage_weights,
                                            compress_df=self.adage_df,
                                            num_components=num_components,
                                            noise_column=noise_column,
                                            subtraction_groups=group_list,
                                            addition_group=add_groups,
                                            node=expect_node,
                                            real_df=real_df,
                                            algorithm='adage')

        pca_results = self._wrap_sub_eval(weight_df=self.pca_weights,
                                          compress_df=self.pca_df,
                                          num_components=num_components,
                                          noise_column=noise_column,
                                          subtraction_groups=group_list,
                                          addition_group=add_groups,
                                          node=expect_node,
                                          real_df=real_df,
                                          algorithm='pca')

        ica_results = self._wrap_sub_eval(weight_df=self.ica_weights,
                                          compress_df=self.ica_df,
                                          num_components=num_components,
                                          noise_column=noise_column,
                                          subtraction_groups=group_list,
                                          addition_group=add_groups,
                                          node=expect_node,
                                          real_df=real_df,
                                          algorithm='ica')

        nmf_results = self._wrap_sub_eval(weight_df=self.nmf_weights,
                                          compress_df=self.nmf_df,
                                          num_components=num_components,
                                          noise_column=noise_column,
                                          subtraction_groups=group_list,
                                          addition_group=add_groups,
                                          node=expect_node,
                                          real_df=real_df,
                                          algorithm='nmf')

        return pd.concat([tybalt_results, adage_results, pca_results,
                          ica_results, nmf_results])
