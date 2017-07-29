
# coding: utf-8

# # Subracting mean cancer-type away from individual tumors
# 
# I am performing this operation for each cancer type in raw gene expression and VAE space. The hypothesis is that by subtracting away the corresponding average cancer-type from each individual tumor, tumors from different anatomical sites can be better compared and specific activation patterns in common can be better isolated.

# In[1]:

import os
import pandas as pd
import numpy as np
from sklearn import manifold

pd.options.mode.chained_assignment = None 


# In[2]:

def subtract_cancertype_mean(encoded_row, mean_df, num_features, testing=False):
    """
    Subtract the corresponding cancer-type average from the given row of features
    Usage with pd.DataFrame().apply()
    """
    cancer_type = encoded_row['acronym']
    cancer_mean_vector = mean_df.loc[cancer_type, :]
    if not testing:
        subtracted_cancer = encoded_row.iloc[range(0, num_features)] - cancer_mean_vector
        encoded_row.iloc[range(0, num_features)] = subtracted_cancer
    else:
        added_back_cancer = encoded_row.iloc[range(0, num_features)] + cancer_mean_vector
        encoded_row.iloc[range(0, num_features)] = added_back_cancer
    return encoded_row


# In[3]:

encoded_file = os.path.join('data', 'vae_encoded_with_clinical.tsv')
encoded_df = pd.read_table(encoded_file, index_col=0)


# In[4]:

raw_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv')
raw_df = pd.read_table(raw_file, index_col=0)
# Add clinical information
raw_df = raw_df.join(encoded_df.iloc[:, range(101, encoded_df.shape[1])], how='inner')
raw_df.index.name = 'sample_id'


# In[5]:

# Get the mean of each cancer-type across all features
disease_vae_mean_df = encoded_df.groupby('acronym').mean().iloc[:, range(0, 100)]
disease_raw_mean_df = raw_df.groupby('acronym').mean().iloc[:, range(0, 5000)]


# In[6]:

subtracted_mean_vae_df = (
    encoded_df.apply(lambda x:
                     subtract_cancertype_mean(x, mean_df=disease_vae_mean_df,
                                              num_features=100), axis=1)
    )


# In[7]:

subtracted_mean_raw_df = (
    raw_df.apply(lambda x:
                 subtract_cancertype_mean(x, mean_df=disease_raw_mean_df,
                                          num_features=5000), axis=1)
    )


# In[8]:

# Test that the each vector is successfully subtracted, by adding them back and
# observing the difference from the original (should be close to zero)
test_subtract = (
    subtracted_mean_vae_df
    .apply(lambda x: subtract_cancertype_mean(x, mean_df=disease_vae_mean_df,
                                              num_features=100, testing=True), axis=1)
    )

test_raw_subtract = (
    subtracted_mean_raw_df
    .apply(lambda x: subtract_cancertype_mean(x, mean_df=disease_raw_mean_df,
                                              num_features=5000, testing=True), axis=1)
    )

print((encoded_df.iloc[:, range(0, 100)] - test_subtract.iloc[:, range(0, 100)]).sum().sum())
(raw_df.iloc[:, range(0, 5000)] - test_raw_subtract.iloc[:, range(0, 5000)]).sum().sum()


# In[9]:

# Save the files for future use
subtracted_mean_vae_df.to_csv(os.path.join('data', 'cancertype_subtraction_encoded.tsv'), sep='\t')
subtracted_mean_raw_df.to_csv(os.path.join('data', 'cancertype_subtraction_raw.tsv'), sep='\t')


# In[10]:

# Perform and output a t-sne dimensionality reduction on the subtracted VAE data
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=20,
                     learning_rate=300, n_iter=400)
tsne_out = tsne.fit_transform(subtracted_mean_vae_df.iloc[:, range(0, 100)])
tsne_out = pd.DataFrame(tsne_out, columns=['1', '2'])
tsne_out.index = subtracted_mean_vae_df.index
tsne_out.index.name = 'tcga_id'


# In[11]:

tsne_out_file = os.path.join('results', 'vae_subtracted_tsne_out.tsv')
tsne_out.to_csv(tsne_out_file, sep='\t')

