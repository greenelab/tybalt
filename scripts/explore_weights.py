
# coding: utf-8

# # Exploring model weights
# 
# The weights the model learns indicates patterns of gene expression activations present across tumors. In an unsupervised model, these weights learned can point to known biology, unknown biology, or unrelated noise. One of the benefits of a variational autoencoder, is that the weights learned that represent each encoding are independent and nonlinear. Therefore, they can extract out signal representative of noise while still retaining relevant known and unknown biology.
# 
# Here we explore specific signals that should be present in the data - signals representing the sex of the patient, and metastatic tumor activation patterns.

# In[1]:

import os
import pandas as pd
from keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:

sns.set(style="white", color_codes=True)
sns.set_context("paper", rc={"font.size":8,"axes.titlesize":10,"axes.labelsize":15})   


# In[3]:

# get_ipython().magic('matplotlib inline')
# plt.style.use('seaborn-notebook')


# Because of the complex architecture involved in encoding the data, we will use the `decoded` weights to describe feature encoding specific activation patterns

# In[4]:

# Load the decoder model
decoder_model_file = os.path.join('models', 'decoder_onehidden_vae.hdf5')
decoder = load_model(decoder_model_file)


# In[5]:

# Extract the weights
decoded_weights = []
for layer in decoder.layers:
    decoded_weights.append(layer.get_weights())


# In[6]:

# For a future pathway analysis, the background genes to consider is important
# Also needed to set column names on weights
rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
print(rnaseq_df.shape)
rnaseq_df.head(2)

pd.DataFrame(rnaseq_df.columns).to_csv(os.path.join('results', 'background_genes.txt'), sep='\t')


# In[7]:

node_weights = pd.DataFrame(decoded_weights[1][0], columns=rnaseq_df.columns, index=range(1, 101))
node_weights.head()


# In[8]:

# We previously identified node 82 as robustly separating sex in the data set:
# Visualize the distribution of gene weights here
sex_node = node_weights.loc[82, :]
sex_node_plot = node_weights.loc[[82, 85], :]
sex_node_plot = sex_node_plot.T
sex_node_plot.columns = ['encoding 82', 'encoding 85']
sex_node_plot = sex_node_plot.reindex(sex_node_plot['encoding 82'].abs().sort_values(ascending=False).index)
g = sns.jointplot(x='encoding 82', y='encoding 85', data=sex_node_plot,
                  edgecolor="w", stat_func=None);


# In[9]:

sex_node_plot.head(17)


# By measuring expression of only *17 genes*, we can reliably predict the sex of the cancer patient. These are genes most expressed by sex chromosomes including x inactivating genes _XIST_ and _TSIX_.

# In[10]:

# Save outputs
sex_node_plot_file = os.path.join('figures', 'sex_node_gene_scatter.pdf')
sex_node_gene_file = os.path.join('results', 'sex_node_activated_genes.tsv')
g.savefig(sex_node_plot_file)
sex_node_plot.to_csv(sex_node_gene_file, sep='\t')


# In[11]:

# We previously observed metastasis samples being robustly separated by two features
# Visualize the feature scores here
met_node_plot = node_weights.loc[[53, 66], :]
met_node_plot = met_node_plot.T
met_node_plot.columns = ['encoding 53', 'encoding 66']
met_node_plot = met_node_plot.reindex(met_node_plot['encoding 53'].abs().sort_values(ascending=False).index)
g = sns.jointplot(x='encoding 53', y='encoding 66', data=met_node_plot,
                  color='green', edgecolor="w", stat_func=None);


# In[12]:

# Save outputs
met_node_plot_file = os.path.join('figures', 'skcm_metastasis_node_gene_scatter.pdf')
met_node_gene_file = os.path.join('results', 'skcm_metastasis_node_activated_genes.tsv')
g.savefig(met_node_plot_file)
met_node_plot.to_csv(met_node_gene_file, sep='\t')


# Because the genes involved in these nodes are not as cut and dry as the sex specific nodes, perform a pathway analysis on the high weight genes explaining each tail of node 53 and 66

# In[13]:

encoding66 = (
    met_node_plot
    .reindex(met_node_plot['encoding 66'].abs()
             .sort_values(ascending=False).index)['encoding 66']
    )

high_weight_pos = encoding66[encoding66 > encoding66.std() * 2.5]
high_weight_neg = encoding66[encoding66 < -encoding66.std() * 2.5]

high_weight_pos.to_csv(os.path.join('results', 'hightweightpos_genes_node66.tsv'), sep='\t')
high_weight_neg.to_csv(os.path.join('results', 'hightweightneg_genes_node66.tsv'), sep='\t')


# In[14]:

encoding53 = (
    met_node_plot
    .reindex(met_node_plot['encoding 53'].abs()
             .sort_values(ascending=False).index)['encoding 53']
    )

high_weight_pos = encoding53[encoding53 > encoding53.std() * 2.5]
high_weight_neg = encoding53[encoding53 < -encoding53.std() * 2.5]

high_weight_pos.to_csv(os.path.join('results', 'hightweightpos_genes_node53.tsv'), sep='\t')
high_weight_neg.to_csv(os.path.join('results', 'hightweightneg_genes_node53.tsv'), sep='\t')

