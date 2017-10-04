
# coding: utf-8

# # Extracting Tybalt Weights
# 
# The weights learned by the Tybalt model indicate patterns of gene expression variably activated across tumors. As Tybalt is an unsupervised model, these weights learned can point to known biology, unknown biology, or unrelated noise. One of the benefits of a variational autoencoder (VAE), is that the weights learned that represent each encoding are  nonlinear. Therefore, they can extract out signal representative of noise while still retaining relevant known and unknown biology.
# 
# Here, we extract the weights learned by the VAE and save them to file. We also explore specific signals that should be present in the data:
# 
# 1. Signals representing the sex of the patient
# 2. Melanoma activation patterns.

# In[1]:

import os
import pandas as pd
from keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:

sns.set(style='white', color_codes=True)
sns.set_context('paper', rc={'font.size':8, 'axes.titlesize':10, 'axes.labelsize':15})   


# In[3]:

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# Because of the complex architecture involved in encoding the data, we will use the `decoded` weights to describe feature encoding specific activation patterns

# In[4]:

# Load the decoder model
decoder_model_file = os.path.join('models', 'decoder_onehidden_vae.hdf5')
decoder = load_model(decoder_model_file)


# In[5]:

# Load RNAseq file
rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv.gz')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
rnaseq_df.head(2)


# In[6]:

# For a future pathway analysis, the background genes are important
# Also needed to set column names on weights
background_file = os.path.join('data', 'background_genes.txt')
background_genes = pd.DataFrame(rnaseq_df.columns)
background_genes.to_csv(background_file, index=False, header=False, sep='\t')


# ## Extract Tybalt weight matrix and write to file

# In[7]:

# Extract the weights from the decoder model
weights = []
for layer in decoder.layers:
    weights.append(layer.get_weights())
    
weight_layer_df = pd.DataFrame(weights[1][0], columns=rnaseq_df.columns, index=range(1, 101))
weight_layer_df.index.name = 'encodings'
weight_layer_df.head(2)


# In[8]:

# Write the genes to file
weight_file = os.path.join('results', 'tybalt_gene_weights.tsv')
weight_layer_df.to_csv(weight_file, sep='\t')


# ## Extracting example patterns learned by Tybalt
# 
# Focusing on two examples: Sex-specific and tissue-specific activation
# 
# ### Sex specific activation by node 82

# In[9]:

# We previously identified node 82 as robustly separating sex in the data set:
# Visualize the distribution of gene weights here
sex_node_plot = weight_layer_df.loc[[82, 85], :].T
sex_node_plot.columns = ['encoding 82', 'encoding 85']

sex_node_plot = (
    sex_node_plot.reindex(sex_node_plot['encoding 82'].abs()
                          .sort_values(ascending=False).index)
    )

g = sns.jointplot(x='encoding 82', y='encoding 85',
                  data=sex_node_plot, color='black',
                  edgecolor="w", stat_func=None);

# Save Figure
sex_node_plot_file = os.path.join('figures', 'sex_node_gene_scatter.pdf')
g.savefig(sex_node_plot_file)


# In[10]:

# There are 17 genes with high activation in node 82
# All genes are located on sex chromosomes
sex_node_plot.head(17)


# By measuring expression of only *17 genes*, we can reliably predict the sex of the cancer patient. These are genes most expressed by sex chromosomes including x inactivating genes _XIST_ and _TSIX_.

# ### Node separating melanoma samples

# In[11]:

# We previously observed metastasis samples being robustly separated by two features
# Visualize the feature scores here
met_node_plot = weight_layer_df.loc[[53, 66], :].T
met_node_plot.columns = ['encoding 53', 'encoding 66']

met_node_plot = (met_node_plot.reindex(met_node_plot['encoding 53'].abs()
                                       .sort_values(ascending=False).index)
                 )
g = sns.jointplot(x='encoding 53', y='encoding 66',
                  data=met_node_plot, color='black',
                  edgecolor="w", stat_func=None);

# Save outputs
met_node_plot_file = os.path.join('figures', 'skcm_metastasis_node_gene_scatter.pdf')
g.savefig(met_node_plot_file)


# #### Output high weight genes for two specific SKCM encodings
# 
# Because the genes involved in these nodes are not as cut and dry as the sex specific nodes, output the high weight genes explaining each tail of node 53 and 66. These will be processed though a pathway analysis downstream.

# In[12]:

def output_high_weight_genes(weight_df, encoding, filename, thresh=2.5):
    """
    Function to process and output high weight genes given specific feature encodings
    """
    
    # Sort initial encoding by absolute activation
    encoding_df = (
        weight_df
        .reindex(weight_df[encoding].abs()
                 .sort_values(ascending=False).index)[encoding]
    )
    
    hw_pos_df = pd.DataFrame(encoding_df[encoding_df > encoding_df.std() * thresh])
    hw_pos_df = hw_pos_df.assign(direction='positive')
    hw_neg_df = pd.DataFrame(encoding_df[encoding_df < -encoding_df.std() * thresh])
    hw_neg_df = hw_neg_df.assign(direction='negative')
    
    hw_df = pd.concat([hw_pos_df, hw_neg_df])
    hw_df.index.name = 'genes'
    hw_df.to_csv(filename, sep='\t')
    return hw_df


# In[13]:

# Encoding 66
hw_node66_file = os.path.join('results', 'high_weight_genes_node66_skcm.tsv')
node66_df = output_high_weight_genes(met_node_plot, 'encoding 66', hw_node66_file)
node66_df.head(5)


# In[14]:

# Encoding 53
hw_node53_file = os.path.join('results', 'high_weight_genes_node53_skcm.tsv')
node53_df = output_high_weight_genes(met_node_plot, 'encoding 53', hw_node53_file)
node53_df.head(5)

