
# coding: utf-8

# # Determining the distance of BLCA and LUAD tumors in different spaces
# 
# The script will output distance matrices for pairwise comparisons of all bladder cancer (BLCA) and lung adenocarcinoma (LUAD) tumors. The script will output distances for raw gene expression features and gene expression features encoded by a variational autoencoder. Additionally, distance matrices for the subtraction of cancer-type specific means is also output.

# In[1]:

import os
import pandas as pd
from scipy.spatial import distance


# In[2]:

# Load data
raw_data_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv')
raw_data_subtract_file = os.path.join('data', 'cancertype_subtraction_raw.tsv')
encoded_data_file = os.path.join('data', 'vae_encoded_with_clinical.tsv')
encoded_data_subtract_file = os.path.join('data', 'cancertype_subtraction_encoded.tsv')

raw_data_df = pd.read_table(raw_data_file, index_col=0)
raw_data_subtract_df = pd.read_table(raw_data_subtract_file, index_col=0)
encoded_df = pd.read_table(encoded_data_file, index_col=0)
encoded_subtract_df = pd.read_table(encoded_data_subtract_file, index_col=0)


# In[3]:

# Load and subset mutation data to TP53, KRAS, and NF1
mutation_data_file = os.path.join('data', 'pancan_mutation.tsv')
mutation_df = pd.read_table(mutation_data_file,
                            usecols = ['#sample', 'TP53', 'KRAS', 'NF1'],
                            index_col=0)


# In[4]:

# Subset data to BLCA and LUAD cancer-types
# BLCA and LUAD have 49.2% and 50.2% TP53 mutations, respecitively
subset_cancertypes = ['BLCA', 'LUAD']
encoded_subset_df = encoded_df[encoded_df['acronym'].isin(subset_cancertypes)]
encoded_subtract_subset_df = encoded_subtract_df[encoded_subtract_df['acronym'].isin(subset_cancertypes)]
raw_data_subset_df = raw_data_df.loc[encoded_subset_df.index, ]
raw_data_subtract_subset_df = raw_data_subtract_df.loc[encoded_subset_df.index, raw_data_df.columns]


# ## Get euclidean distances for each feature space

# In[5]:

raw_data_distance_file = os.path.join('data', 'distance', 'raw_distance.tsv')
raw_data_subtract_distance_file = os.path.join('data', 'distance', 'raw_subtract_distance.tsv')
encoded_distance_file = os.path.join('data', 'distance', 'encoded_distance.tsv')
encoded_subtract_distance_file = os.path.join('data', 'distance', 'encoded_subtraction_distance.tsv')


# In[6]:

data_array_df = distance.pdist(raw_data_subset_df)
data_dist_df = pd.DataFrame(distance.squareform(data_array_df),
                            index=raw_data_subset_df.index,
                            columns=raw_data_subset_df.index)
data_dist_df.to_csv(raw_data_distance_file, sep='\t')


# In[7]:

data_subtract_array_df = distance.pdist(raw_data_subtract_subset_df)
data_subtract_dist_df = pd.DataFrame(distance.squareform(data_subtract_array_df),
                            index=raw_data_subtract_subset_df.index,
                            columns=raw_data_subtract_subset_df.index)
data_subtract_dist_df.to_csv(raw_data_subtract_distance_file, sep='\t')


# In[8]:

encoded_array_df = distance.pdist(encoded_subset_df.iloc[:,range(0, 100)])
encoded_dist_df = pd.DataFrame(distance.squareform(encoded_array_df),
                               index=encoded_subset_df.index,
                               columns=encoded_subset_df.index)
encoded_dist_df.to_csv(encoded_distance_file, sep='\t')


# In[9]:

encoded_sub_array_df = distance.pdist(encoded_subtract_subset_df.iloc[:,range(0, 100)])
encoded_sub_dist_df = pd.DataFrame(distance.squareform(encoded_sub_array_df),
                                   index=encoded_subtract_subset_df.index,
                                   columns=encoded_subtract_subset_df.index)
encoded_sub_dist_df.to_csv(encoded_subtract_distance_file, sep='\t')


# ## Get TP53 mutated BLCA and LUAD

# In[10]:

blca_samples = encoded_df[encoded_df['acronym'] == 'BLCA'].index
luad_samples = encoded_df[encoded_df['acronym'] == 'LUAD'].index


# In[11]:

blca_df = mutation_df[mutation_df.index.isin(blca_samples) & mutation_df['TP53'] == 1]
blca_df = blca_df.assign(acronym = 'BLCA')
luad_df = mutation_df[mutation_df.index.isin(luad_samples) & mutation_df['TP53'] == 1]
luad_df = luad_df.assign(acronym = 'LUAD')
tp53_samples_df = blca_df.append(luad_df)


# In[12]:

tp53_samples_file = os.path.join('data', 'distance', 'blca_luad_tp53_mutations.tsv')
tp53_samples_df.to_csv(tp53_samples_file, sep='\t')

