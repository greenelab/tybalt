
# coding: utf-8

# # Encoded representation layer recapitulates signal identified in raw data
# 
# ## Visualized with t-SNE
# 
# Perform a t-sne on tybalt features to visualize if the latent layer recapitulates relationships observed through raw data t-sne.

# In[1]:


import os
import pandas as pd
from sklearn import manifold


# In[2]:


# Load VAE feature activations per sample
encoded_file = os.path.join('data', 'encoded_rnaseq_onehidden_warmup_batchnorm.tsv')
encoded_df = pd.read_table(encoded_file, index_col=0)
encoded_df.head(2)


# In[3]:


# Load ADAGE feature activations per sample
adage_file = os.path.join('data', 'encoded_adage_features.tsv')
adage_df = pd.read_table(adage_file, index_col=0)
adage_df.head(2)


# In[4]:


# Load Zero-One transformed (min-max scaled) RNAseq data
rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv.gz')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
print(rnaseq_df.shape)
rnaseq_df.head(2)


# In[5]:


# Perform t-SNE on VAE encoded_features
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=20,
                     learning_rate=300, n_iter=400)
tsne_out = tsne.fit_transform(encoded_df)
tsne_out = pd.DataFrame(tsne_out, columns=['1', '2'])
tsne_out.index = encoded_df.index
tsne_out.index.name = 'tcga_id'
tsne_out_file = os.path.join('results', 'tybalt_tsne_features.tsv')
tsne_out.to_csv(tsne_out_file, sep='\t')
tsne_out.head(2)


# In[6]:


# Perform t-SNE on ADAGE encoded_features
tsne_adage = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=20,
                           learning_rate=300, n_iter=400)
tsne_adage_out = tsne_adage.fit_transform(adage_df)
tsne_adage_out = pd.DataFrame(tsne_adage_out, columns=['1', '2'])
tsne_adage_out.index = adage_df.index
tsne_adage_out.index.name = 'tcga_id'
tsne_adage_out_file = os.path.join('results', 'adage_tsne_features.tsv')
tsne_adage_out.to_csv(tsne_adage_out_file, sep='\t')
tsne_adage_out.head(2)


# In[7]:


# Perform t-SNE on zero-one RNAseq features
tsne_rna = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity=20,
                         learning_rate=300, n_iter=400)
tsne_rna_out = tsne_rna.fit_transform(rnaseq_df)
tsne_rna_out = pd.DataFrame(tsne_rna_out, columns=['1', '2'])
tsne_rna_out.index = rnaseq_df.index
tsne_rna_out.index.name = 'tcga_id'
tsne_rna_out_file = os.path.join('results', 'rnaseq_tsne_features.tsv')
tsne_rna_out.to_csv(tsne_rna_out_file, sep='\t')
tsne_rna_out.head(2)

