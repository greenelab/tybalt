
# coding: utf-8

# # Tybalt latent space arithmetic with high grade serous ovarian cancer subtypes
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


# In[2]:

get_ipython().magic('matplotlib inline')
plt.style.use('seaborn-notebook')


# In[3]:

sns.set(style='white', color_codes=True)
sns.set_context('paper', rc={'font.size':12, 'axes.titlesize':15, 'axes.labelsize':20,
                             'xtick.labelsize':14, 'ytick.labelsize':14})   


# In[4]:

# Set seed for plotting
np.random.seed(123)


# In[5]:

rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv.gz')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
rnaseq_df.shape


# In[6]:

ov_file = os.path.join('data', 'ov_subtype_info.tsv')
ov_df = pd.read_table(ov_file, index_col=0)
ov_df.head(2)


# In[7]:

encoded_file = os.path.join('data', "encoded_rnaseq_onehidden_warmup_batchnorm.tsv")
encoded_df = pd.read_table(encoded_file, index_col=0)
print(encoded_df.shape)
encoded_df.head(2)


# In[8]:

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


# In[9]:

# Get the HGSC mean feature activation
ov_mean_subtypes = ov_encoded_subtype.groupby('SUBTYPE').mean()
ov_mean_subtypes


# ## HGSC Subtype Arithmetic
# 
# Because of the relationship observed in the consistent clustering solutions, perform the following subtractions
# 
# 1. Immunoreactive - Mesenchymal
# 2. Differentiated - Proliferative
# 
# The goal is to observe the features with the largest difference between the aformentioned comparisons. The differences should be in absolute directions

# ### 1) Immunoreactive - Mesenchmymal

# In[10]:

mes_mean_vector = ov_mean_subtypes.loc['Mesenchymal', [str(x) for x in range(1, 101)]]
imm_mean_vector = ov_mean_subtypes.loc['Immunoreactive', [str(x) for x in range(1, 101)]]


# In[11]:

high_immuno = (imm_mean_vector - mes_mean_vector).sort_values(ascending=False).head(2)
high_mesenc = (imm_mean_vector - mes_mean_vector).sort_values(ascending=False).tail(2)

print("Features with large differences: Immuno high, Mesenchymal low")
print(high_immuno)
print("Features with large differences: Mesenchymal high, Immuno low")
print(high_mesenc)


# In[12]:

# Select to visualize encoding 56 because it has high immuno and low everything else
ov_mean_subtypes.loc[:, ['87', '77', '56']]


# In[13]:

# Node 87 has high mesenchymal, low immunoreactive
node87_file = os.path.join('figures', 'node87_distribution_ovsubtype.pdf')
g = sns.swarmplot(y = '87', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 87')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(node87_file)


# In[14]:

# Node 77 has high immunoreactive, low mesenchymal
node77_file = os.path.join('figures', 'node77_distribution_ovsubtype.pdf')
g = sns.swarmplot(y = '77', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 77')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(node77_file)


# In[15]:

# Node 56 has high immunoreactive, low mesenchymal (and prolif/diff)
node56_file = os.path.join('figures', 'node56_distribution_ovsubtype.pdf')
g = sns.swarmplot(y = '56', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 56')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(node56_file)


# ### 2) Differentiated - Proliferative

# In[16]:

pro_mean_vector = ov_mean_subtypes.loc['Proliferative', [str(x) for x in range(1, 101)]]
dif_mean_vector = ov_mean_subtypes.loc['Differentiated', [str(x) for x in range(1, 101)]]


# In[17]:

high_differ = (dif_mean_vector - pro_mean_vector).sort_values(ascending=False).head(2)
high_prolif = (dif_mean_vector - pro_mean_vector).sort_values(ascending=False).tail(2)

print("Features with large differences: Differentiated high, Proliferative low")
print(high_differ)
print("Features with large differences: Proliferative high, Differentiated low")
print(high_prolif)


# In[18]:

# Select to visualize encoding 56 because it has high immuno and low everything else
ov_mean_subtypes.loc[:, ['38', '79']]


# In[19]:

# Node 38 has high differentiated, low proliferative
node38_file = os.path.join('figures', 'node38_distribution_ovsubtype.pdf')
g = sns.swarmplot(y = '38', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 38')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(node38_file)


# In[20]:

# Node 79 has high proliferative, low differentiated
node79_file = os.path.join('figures', 'node79_distribution_ovsubtype.pdf')
g = sns.swarmplot(y = '79', x = 'SUBTYPE', data=ov_encoded_subtype,
                  order=['Mesenchymal', 'Immunoreactive', 'Proliferative', 'Differentiated']);
g.set(xlabel='', ylabel='encoding 79')
plt.xticks(rotation=0);
plt.tight_layout()
plt.savefig(node79_file)


# ### Explore weights that explain the nodes

# In[21]:

def get_high_weight(weight_matrix, node, high_std=2.5, direction='positive'):
    """
    Determine high weight genes given a gene weight matrix and feature
    Output tab separated file
    """
    genes = weight_matrix.loc[node, :].sort_values(ascending=False)
    if direction == 'positive':
        node_df = (genes[genes > genes.std() * high_std])
        abbrev = 'pos'
    elif direction == 'negative':
        node_df = (genes[genes < -1 * (genes.std() * high_std)])
        abbrev = 'neg'

    node_df = pd.DataFrame(node_df).reset_index()
    node_df.columns = ['genes', 'weight']
    
    node_base_file = 'hgsc_node{}genes_{}.tsv'.format(node, abbrev)
    node_file = os.path.join('results', node_base_file)
    node_df.to_csv(node_file, index=False, sep='\t')
    
    return node_df


# In[22]:

# Obtain the decoder weights
weight_file = os.path.join('results', 'tybalt_gene_weights.tsv')

weight_df = pd.read_table(weight_file, index_col=0)
weight_df.head(2)


# In[23]:

# Output high weight genes for nodes representing mesenchymal vs immunoreactive
node87pos_df = get_high_weight(weight_df, node=87)
node87neg_df = get_high_weight(weight_df, node=87, direction='negative')

node77pos_df = get_high_weight(weight_df, node=77)
node77neg_df = get_high_weight(weight_df, node=77, direction='negative')

node56pos_df = get_high_weight(weight_df, node=56)
node56neg_df = get_high_weight(weight_df, node=56, direction='negative')


# In[24]:

# Output high weight genes for nodes representing proliferative vs differentiated
node79pos_df = get_high_weight(weight_df, node=79)
node79neg_df = get_high_weight(weight_df, node=79, direction='negative')

node38pos_df = get_high_weight(weight_df, node=38)
node38neg_df = get_high_weight(weight_df, node=38, direction='negative')

