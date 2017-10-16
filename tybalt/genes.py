"""
tybalt/genes.py
2017 Gregory Way

Determination and diagnostic methods for extracting explanatory genes (high
weight genes) from compression models.

Usage:
from tybalt.genes import high_weight_genes

# Point to the weight matrix of interest (feature x gene)
model_file = os.path.join('results', 'FILENAME.tsv')

# Initialize the class
model_weights = high_weight_genes(model_file)

# Get high weight {0,1} matrix (feature x gene) for either tail
model_pos_tail = model_weights.get_high_weight_matrix(direction='positive')
model_neg_tail = model_weights.get_high_weight_matrix(direction='negative')

# Count high weight genes per feature and plot
model_count = model_weights.count_high_weight_genes(return_plot=True)

# Plot distribution of high weight genes per feature in positive/negative tail
model_weights.plot_weight_dist()

# Plot distribution of skewness and kurtosis for each node
model_weights.plot_skewkurtosis()

# Get the node categories for the model
node_type_df = model_weights.get_node_categories(melted=True)

# Adjust high weight gene discovery by node type classification
hw_pos_df = model_weights.get_high_weight_matrix(method='dynamic')
hw_pos_df = model_weights.get_high_weight_matrix(direction='negative',
                                                 method='dynamic')
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, skewtest
from scipy.stats import kurtosis, kurtosistest
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt


class high_weight_genes():
    """
    Loads and determines high weight genes given a weight matrix
    """
    def __init__(self, weight_file_name, metric='stddev', thresh=2.5,
                 algorithm=''):
        self.weight_file_name = weight_file_name
        self.weight_df = pd.read_table(weight_file_name, index_col=0)
        self.metric = metric
        self.thresh = thresh
        self.algorithm = algorithm

    def _get_cutoff(self, weight_array):
        """
        Private method for setting a cutoff for high weight gene determination
        """
        if self.metric == 'stddev':
            cutoff = weight_array.std() * self.thresh
        elif self.metric == 'mad':
            cutoff = weight_array.mad() * self.thresh
        elif self.metric == 'log_stddev':
            cutoff = np.log(weight_array)
            cutoff = weight_array.std() * self.thresh
        return cutoff

    def _get_node_type_cutoff(self, weight_array, node_type, direction):
        """
        Private method for setting a cutoff for high weight gene determination
        based on a dynamic method
        """
        if node_type == 'type_a':
            cutoff = weight_array.std() * (self.thresh + 0.5)

        if node_type == 'type_b':
            cutoff = weight_array.std() * self.thresh

        if node_type == 'type_c_neg':
            if direction == 'negative':
                cutoff = weight_array.std() * self.thresh
            elif direction == 'positive':
                cutoff = weight_array.std() * (self.thresh + 0.5)
        if node_type == 'type_c_pos':
            if direction == 'negative':
                cutoff = weight_array.std() * (self.thresh + 0.5)
            elif direction == 'positive':
                cutoff = weight_array.std() * self.thresh

        if node_type == 'type_d_neg':
            if direction == 'negative':
                cutoff = weight_array.std() * (self.thresh - 0.5)
            elif direction == 'positive':
                cutoff = weight_array.std() * self.thresh
        if node_type == 'type_d_pos':
            if direction == 'negative':
                cutoff = weight_array.std() * self.thresh
            elif direction == 'positive':
                cutoff = weight_array.std() * (self.thresh - 0.5)
        return cutoff

    def _get_skew(self):
        """
        Private method for testing skewness of feature weight distributions
        """
        return skew(self.weight_df, axis=1)

    def _get_kurtosis(self):
        """
        Private method for testing kurtosis of feature weight distributions
        """
        return kurtosis(self.weight_df, axis=1)

    def _get_skewtest(self):
        """
        Private method for calculating skewness statistics and p values of
        feature weight distributions
        """
        return skewtest(self.weight_df, axis=1)

    def _get_kurtosistest(self):
        """
        Private method for calculating kurtosis statistics and p values of
        feature weight distributions
        """
        return kurtosistest(self.weight_df, axis=1)

    def _skew_kurtosis(self):
        """
        Private method for combining results of skew and kurtosis tests
        """
        s = self._get_skew()
        k = self._get_kurtosis()

        return pd.DataFrame([s, k], index=['skew', 'kurtosis'],
                            columns=range(1, 101)).T

    def _skew_kurtosis_test(self):
        """
        Private method for combining results of skew and kurtosis tests
        """
        s_stat, s_pval = self._get_skewtest()
        k_stat, k_pval = self._get_kurtosistest()
        stat_result = [s_stat, s_pval, k_stat, k_pval]
        names = ['skew_stat', 'skew_p', 'kurtosis_stat', 'kurtosis_p']
        skew_kurtosis_df = pd.DataFrame(stat_result, index=names,
                                        columns=range(1, 101)).T
        self.skew_kurtosis_df = skew_kurtosis_df

    def _remove_outliers(self, df, z_thresh=3):
        """
        Remove outliers from dataframe

        Arguments:
        df - a given input pandas DataFrame
        z_thresh - numeric to indicate how outliers are defined (zscore)

        Output:
        A tuple (outlier removed DataFrame, outlier DataFrame)
        """
        outliers = (np.abs(zscore(df)) < z_thresh).all(axis=1)

        outlier_removed_df = df[outliers]
        outlier_df = df[~outliers]

        return (outlier_removed_df, outlier_df)

    def _high_weight(self, weight_array, direction):
        """
        Private method for finding high weight genes given an array of weights
        and a prespecified metric and threshold
        """
        cutoff = self._get_cutoff(weight_array)
        if direction == 'positive':
            weight_bool = weight_array > cutoff
        if direction == 'negative':
            weight_bool = weight_array < -cutoff
        return weight_bool.astype(int)

    def _dynamic_weight(self, weight_array, direction):
        """
        Private method for finding high weight genes given an array of weights
        and a prespecified metric, threshold, and node_type
        """
        index = weight_array.name
        node_type = self.node_type_df.loc[index, :]

        node_type = node_type[node_type == 1].index
        cutoff = self._get_node_type_cutoff(weight_array, node_type, direction)
        if direction == 'positive':
            weight_bool = weight_array > cutoff
        if direction == 'negative':
            weight_bool = weight_array < -cutoff
        return weight_bool.astype(int)

    def get_high_weight_matrix(self, direction='positive', method='standard'):
        """
        Method to obtain all high weight genes for each weight matrix feature

        Arguments:
        direction - string indicating which tail to get high weight genes for
                    options: {'positive', 'negative'}
        method - mechanism to define hw genes {'standard', 'dynamic'}

        Output:
        pandas DataFrame of high weight membership gene per feature {0,1}
        """

        if method == 'standard':
            df = self.weight_df.apply(
                lambda x: self._high_weight(x, direction=direction), axis=1)
        elif method == 'dynamic':
            self.get_node_categories()
            df = self.weight_df.apply(
                lambda x: self._dynamic_weight(x, direction=direction), axis=1)
        return df

    def count_high_weight_genes(self, return_plot=False, method='standard',
                                title='', color='black'):
        """
        Method to summarize high weight gene counts for postive and negative
        tails per feature

        Arguments:
        return_plot - boolean if scatter plot should be returned
        method - mechanism to define hw genes {'standard', 'dynamic'}
        title - string to name the title of the output plot
        color - the color to plot

        Output:
        pandas DataFrame of features summary counts of high weight genes and
        seaborn regplot if plot=True
        """
        pos = self.get_high_weight_matrix(direction='positive', method=method)
        neg = self.get_high_weight_matrix(direction='negative', method=method)

        pos_sum_df = pos.sum(axis=1).reset_index()
        neg_sum_df = neg.sum(axis=1).reset_index()

        gene_sum = pd.concat([pos_sum_df, neg_sum_df], axis=1)
        gene_sum.columns = ['pos_node', 'pos_genes', 'neg_node', 'neg_genes']

        if return_plot:
            g = sns.regplot(x='neg_genes', y='pos_genes', data=gene_sum,
                            fit_reg=False, color=color)
            g.set_title(title)
            return g
        else:
            return gene_sum

    def get_node_categories(self, adj_p='Bonferonni', melted=False):
        """
        Assign skewness and kurtosis category nodes:

        Type A - Low skewness and low kurtosis
        Type B - Low skewness and high kurtosis
        Type C - High skewness and low kurtosis
        Type D - High skewness and high kurtosis

        Additionally nodes with high skewness need to be defined as either
        positive or negative tails

        Arguments:
        adj_p - how the p value is adjusted
        melted - if the node_type dataframe is to be output in melted form
        """

        self._skew_kurtosis_test()

        if adj_p == 'Bonferonni':
            adj_p = 0.05 / self.weight_df.shape[0]

        skew_low = self.skew_kurtosis_df['skew_p'] > adj_p
        skew_high = self.skew_kurtosis_df['skew_p'] <= adj_p
        skew_neg = self.skew_kurtosis_df['skew_stat'] >= 0
        skew_pos = self.skew_kurtosis_df['skew_stat'] < 0

        kurtosis_low = self.skew_kurtosis_df['kurtosis_p'] > adj_p
        kurtosis_high = self.skew_kurtosis_df['kurtosis_p'] <= adj_p

        type_a = skew_low & kurtosis_low
        type_b = skew_low & kurtosis_high
        type_c = skew_high & kurtosis_low
        type_d = skew_high & kurtosis_high

        type_c_neg = type_c & skew_neg
        type_c_pos = type_c & skew_pos
        type_d_neg = type_d & skew_neg
        type_d_pos = type_d & skew_pos

        n = [type_a, type_b, type_c_neg, type_c_pos, type_d_neg, type_d_pos]
        self.node_type_df = pd.concat(n, axis=1).astype(int)
        self.node_type_df.columns = ['type_a', 'type_b', 'type_c_neg',
                                     'type_c_pos', 'type_d_neg', 'type_d_pos']

        if melted:
            melt_node_df = self.node_type_df.melt()
            melt_node_df.columns = ['node_type', 'count']
            melt_node_df = melt_node_df.assign(algorithm=self.algorithm)

            return melt_node_df

    def plot_weight_dist(self, features=list(), random=0, wrap=5, title='',
                         color='black'):
        """
        Plot weight activation distribution per feature/node

        Arguments:
        features - a list of specific features to plot
        random - integer of the number of random features to plot
        wrap - input to seaborn.FacetGrid of how many columns to wrap
        title - string to name the title of the output plot
        color - the color to plot

        Output:
        Seaborn FacetGrid distplot
        """
        if random > 0:
            features = np.random.randint(1, self.weight_df.shape[0], random)
        plt_ready = self.weight_df.loc[features, :].stack().reset_index()
        plt_ready.columns = ['encodings', 'genes', 'activation']
        g = sns.FacetGrid(plt_ready, col='encodings', col_wrap=wrap)
        g = g.map(sns.distplot, 'activation', color=color)
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(title)
        return g

    def plot_skewkurtosis(self, kind='hex', color='black', outliers=False):
        """
        Plot skew and kurtosis dimensions per node

        Arguments:
        kind - the type of seaborn jointplot to output
        color - the color to plot
        outliers - boolean to remove high skew/kurtosis outliers

        Output:
        tuple of seaborn jointplot and outlier DataFrame
        """
        # Obtain skew and kurtosis
        sk_df = self._skew_kurtosis()

        # Remove outliers
        if outliers:
            outlier_df = None
        else:
            sk_df, outlier_df = self._remove_outliers(sk_df)

        # Plot jointplot
        g = sns.jointplot(x='skew', y='kurtosis', data=sk_df, kind=kind,
                          color=color, stat_func=None)
        return (g, outlier_df)

    def plot_node_types(self):
        """
        Plot a bar chart distribution of node type features
        """
        node_type_df = self.get_node_categories(melted=True)
        g = sns.barplot("node_type", y="count", data=node_type_df, ci=None)
        g.set_title(self.algorithm)
        return g
