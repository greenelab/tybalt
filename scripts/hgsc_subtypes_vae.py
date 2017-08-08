
# coding: utf-8

# # Interpolating high grade serous ovarian cancer subtypes in VAE space
# 
# Recent applications of generative models (GANs and VAEs) in image processing has demonstrated the remarkable ability of the latent dimensions to capture a meaningful manifold representation of the input space. Here, we assess if the VAE learns a latent space that can be mathematically manipulated to reveal insight into the gene expression activation patterns of high grade serous ovarian cancer (HGSC) subtypes.
# 
# Several previous studies have reported the presence of four gene expression based HGSC subtypes. However, we recently [published a paper](https://doi.org/10.1534/g3.116.033514) that revealed the inconsistency of subtype assignments across populations. We observed repeatable structure in the data transitioning between setting clustering algorithms to find different solutions. For instance, when setting algorithms to find 2 subtypes, the mesenchymal and immunoreactive and the proliferative and differentiated subtype consistently collapsed together. These observations may suggest that the subtypes exist on a gene expression continuum of differential activation patterns, and may only be artificially associated with "subtypes". Here, we test if the VAE can help to identify some differential patterns of expression.

# In[1]:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras


# In[2]:

# Save models (full VAE, generator, encoder)
decoder_model_file = os.path.join('models', 'decoder_onehidden_vae.hdf5')
encoder_model_file = os.path.join('models', 'encoder_onehidden_vae.hdf5')

# This is how models will be loaded
# VAE loading is giving errors because of `sampling` function
# vae_loading_test = keras.models.load_model(vae_model_file)
decoder = keras.models.load_model(decoder_model_file)
encoder = keras.models.load_model(encoder_model_file)


# In[3]:

rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
rnaseq_df.shape


# In[4]:

ov_df = pd.read_table(os.path.join('data', 'ov_subtype_info.tsv'), index_col=0)
ov_df.head(2)


# In[5]:

encoded_df = pd.read_table(os.path.join('data', "encoded_rnaseq_onehidden_warmup_batchnorm.tsv"), index_col=0)
encoded_df.shape


# In[6]:

# Subset and merge the HGSC subtype info with the latent space feature activations
ov_samples = list(set(encoded_df.index) & (set(ov_df.index)))

ov_encoded = encoded_df.loc[ov_samples, ]
ov_encoded_subtype = pd.merge(ov_df.loc[:, ['SUBTYPE', 'SILHOUETTE WIDTH']], ov_encoded,
                              how='right', left_index=True, right_index=True)
ov_encoded_subtype = ov_encoded_subtype.assign(subtype_color = ov_encoded_subtype['SUBTYPE'])

ov_subtype_color_dict = {'Differentiated': 'purple',
                         'Immunoreactive': 'green',
                         'Mesenchymal': 'blue',
                         'Proliferative': 'red'}
ov_encoded_subtype = ov_encoded_subtype.replace({'subtype_color': ov_subtype_color_dict})

print(ov_encoded_subtype.shape)
ov_encoded_subtype.head(2)


# In[7]:

# Get the HGSC mean feature activation
ov_mean_subtypes = ov_encoded_subtype.groupby('SUBTYPE').mean()
ov_mean_subtypes


# ## HGSC Subtype Math
# 
# Because of the relationship observed in the consistent clustering solutions, perform the following subtractions
# 
# 1. Immunoreactive - Mesenchymal
# 2. Differentiated - Proliferative
# 
# The goal is to observe the features with the largest difference between the aformentioned comparisons. The differences should be in absolute directions

# In[8]:

mesenchymal_mean_vector = ov_mean_subtypes.loc['Mesenchymal', [str(x) for x in range(1, 101)]]
immuno_mean_vector = ov_mean_subtypes.loc['Immunoreactive', [str(x) for x in range(1, 101)]]
proliferative_mean_vector = ov_mean_subtypes.loc['Proliferative', [str(x) for x in range(1, 101)]]
differentiated_mean_vector = ov_mean_subtypes.loc['Differentiated', [str(x) for x in range(1, 101)]]


# In[9]:

high_immuno = (immuno_mean_vector - mesenchymal_mean_vector).sort_values(ascending=False).head(2)
high_mesenc = (immuno_mean_vector - mesenchymal_mean_vector).sort_values(ascending=False).tail(2)

print("Features with large differences: Immuno high, Mesenchymal low")
print(high_immuno)
print("Features with large differences: Mesenchymal high, Immuno low")
print(high_mesenc)


# In[10]:

# Select to visualize encoding 56 because it has high immuno and low everything else
ov_mean_subtypes.loc[:, ['87', '77', '56']]


# In[11]:

# Obtain the decoder weights
weights = []
for l in decoder.layers:
    weights.append(l.get_weights())
    
weight_layer = pd.DataFrame(weights[1][0], columns=rnaseq_df.columns)
weight_layer.head(2)


# In[12]:

# Get the high weight genes
immuno_genes = weight_layer.loc[87, :].sort_values(ascending = False)
immuno_v2_genes = weight_layer.loc[77, :].sort_values(ascending = False)
immuno_v3_genes = weight_layer.loc[56, :].sort_values(ascending = False)


# In[13]:

(immuno_genes[immuno_genes >
              immuno_genes.std() * 2]).to_csv(os.path.join('results',
                                                           'hgsc_node87genes_pos.tsv'))
(immuno_genes[immuno_genes <
              -1 * (immuno_genes.std() * 2)]).to_csv(os.path.join('results',
                                                                  'hgsc_node87genes_neg.tsv'))

(immuno_v2_genes[immuno_v2_genes >
                 immuno_v2_genes.std() * 2]).to_csv(os.path.join('results',
                                                                 'hgsc_node77genes_pos.tsv'))
(immuno_v2_genes[immuno_v2_genes <
                 -1 * (immuno_v2_genes.std() * 2)]).to_csv(os.path.join('results',
                                                                        'hgsc_node77genes_neg.tsv'))

(immuno_v3_genes[immuno_v3_genes >
                 immuno_v3_genes.std() * 2]).to_csv(os.path.join('results',
                                                                 'hgsc_node56genes_pos.tsv'))
(immuno_v3_genes[immuno_v3_genes <
                 -1 * (immuno_v3_genes.std() * 2)]).to_csv(os.path.join('results',
                                                                        'hgsc_node56genes_neg.tsv'))


# In[14]:

high_differ = (differentiated_mean_vector - proliferative_mean_vector).sort_values(ascending=False).head(2)
high_prolif = (differentiated_mean_vector - proliferative_mean_vector).sort_values(ascending=False).tail(2)

print("Features with large differences: Differentiated high, Proliferative low")
print(high_differ)
print("Features with large differences: Proliferative high, Differentiated low")
print(high_prolif)


# In[15]:

differ_genes = weight_layer.loc[79, :].sort_values(ascending = False)
differ_v2_genes = weight_layer.loc[38, :].sort_values(ascending = False)


# In[16]:

(differ_genes[differ_genes >
              differ_genes.std() * 2]).to_csv(os.path.join('results',
                                                            'hgsc_node79genes_diffpro_pos.tsv'))
(differ_genes[differ_genes <
              -1 * (differ_genes.std() * 2)]).to_csv(os.path.join('results',
                                                                  'hgsc_node79genes_diffpro_neg.tsv'))

(differ_v2_genes[differ_v2_genes >
                 differ_v2_genes.std() * 2]).to_csv(os.path.join('results',
                                                                 'hgsc_node38genes_diffpro_pos.tsv'))
(differ_v2_genes[differ_v2_genes <
                 -1 * (differ_v2_genes.std() * 2)]).to_csv(os.path.join('results',
                                                                        'hgsc_node38genes_diffpro_neg.tsv'))


# In[17]:

# get_ipython().magic('matplotlib inline')
# plt.style.use("seaborn-notebook")


# In[18]:

sns.set(style="white", color_codes=True)
sns.set_context("paper", rc={"font.size":12,"axes.titlesize":15,"axes.labelsize":20,
                             'xtick.labelsize':14, 'ytick.labelsize': 14})   


# In[19]:

# Node 87 has high mesenchymal, low immunoreactive
g = sns.swarmplot(y = '87', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 87')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(os.path.join('figures', 'node87_distribution_ovsubtype.pdf'))


# In[20]:

# Node 77 has high immunoreactive, low mesenchymal
g = sns.swarmplot(y = '77', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 77')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(os.path.join('figures', 'node77_distribution_ovsubtype.pdf'))


# In[21]:

# Node 56 has high immunoreactive, low mesenchymal (but also low proliferative)
g = sns.swarmplot(y = '56', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 56')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(os.path.join('figures', 'node56_distribution_ovsubtype.pdf'))


# In[22]:

# Node 79 has high proliferative, low differentiated
g = sns.swarmplot(y = '79', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 79')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(os.path.join('figures', 'node79_distribution_ovsubtype.pdf'))


# In[23]:

# Node 38 has high differentiated, low proliferative
g = sns.swarmplot(y = '38', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 38')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(os.path.join('figures', 'node38_distribution_ovsubtype.pdf'))

