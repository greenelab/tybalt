
# coding: utf-8

# # Processing all datasets to be used in downstream analyses
# 
# RNAseq, mutation, and copy number data were accessed from the UCSC Xena Data Browser. Clinical data was downloaded from the TCGA Snaptron curation effort. See `data_download.sh` for more details.

# In[1]:

import os
import requests
import numpy as np
import pandas as pd

from sklearn import preprocessing


# ## Define Input and Output Filenames

# In[2]:

# Input Files
rna_file = os.path.join('data', 'raw', 'HiSeqV2')
mut_file = os.path.join('data', 'raw', 'PANCAN_mutation')
copy_file = os.path.join('data', 'raw', 'Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes')
clinical_file = os.path.join('data', 'raw', 'samples.tsv')


# In[3]:

# Output Files
# Processing RNAseq data by z-score and zeroone norm
rna_out_file = os.path.join('data', 'pancan_scaled_rnaseq.tsv.gz')
rna_out_zeroone_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv.gz')

# Mutation Data
mut_out_file = os.path.join('data', 'pancan_mutation.tsv.gz')
mut_burden_file = os.path.join('data', 'pancan_mutation_burden.tsv')

# Two copy number matrices, for thresholded (2) gains and losses
copy_gain_out_file = os.path.join('data', 'copy_number_gain.tsv.gz')
copy_loss_out_file = os.path.join('data', 'copy_number_loss.tsv.gz')

# Clinical data
clinical_processed_out_file = os.path.join('data', 'clinical_data.tsv')

# OncoKB output file
oncokb_out_file = os.path.join('data', 'oncokb_genetypes.tsv')

# Status matirix integrating mutation and copy number events
known_status_file = os.path.join('data', 'status_matrix.tsv.gz')


# ## Load Data

# In[4]:

rnaseq_df = pd.read_table(rna_file, index_col=0)
mutation_df = pd.read_table(mut_file)
copy_df = pd.read_table(copy_file, index_col=0)
clinical_df = pd.read_table(clinical_file, index_col=0)


# ## Begin processing different data types
# 
# ### RNAseq
# 
# The RNAseq data was accessed through the UCSC Xena database.

# In[5]:

# Process RNAseq file
rnaseq_df.index = rnaseq_df.index.map(lambda x: x.split('|')[0])
rnaseq_df.columns = rnaseq_df.columns.str.slice(start=0, stop=15)
rnaseq_df = rnaseq_df.drop('?').fillna(0).sort_index(axis=1)

# Gene is listed twice in RNAseq data, drop both occurrences
rnaseq_df.drop('SLC35E2', axis=0, inplace=True)
rnaseq_df = rnaseq_df.T

# Determine most variably expressed genes and subset
num_mad_genes = 5000
mad_genes = rnaseq_df.mad(axis=0).sort_values(ascending=False)
top_mad_genes = mad_genes.iloc[0:num_mad_genes, ].index
rnaseq_subset_df = rnaseq_df.loc[:, top_mad_genes]


# In[6]:

# Scale RNAseq data using z-scores
rnaseq_scaled_df = preprocessing.StandardScaler().fit_transform(rnaseq_subset_df)
rnaseq_scaled_df = pd.DataFrame(rnaseq_scaled_df,
                                columns=rnaseq_subset_df.columns,
                                index=rnaseq_subset_df.index)
rnaseq_scaled_df.to_csv(rna_out_file, sep='\t', compression='gzip')

# Scale RNAseq data using zero-one normalization
rnaseq_scaled_zeroone_df = preprocessing.MinMaxScaler().fit_transform(rnaseq_subset_df)
rnaseq_scaled_zeroone_df = pd.DataFrame(rnaseq_scaled_zeroone_df,
                                        columns=rnaseq_subset_df.columns,
                                        index=rnaseq_subset_df.index)
rnaseq_scaled_zeroone_df.to_csv(rna_out_zeroone_file, sep='\t', compression='gzip')


# ### Mutation
# 
# Mutation data are stored in a long format. First, subset the data to only deleterious mutations listed below. Next, pivot the dataframe to have samples as the index, genes as the columns, and either a 1 or 0 to indicate a deleterious mutation or wild-type sample by gene.

# In[7]:

# Filter mutation types and generate binary matrix
mutations = {
    'Frame_Shift_Del',
    'Frame_Shift_Ins',
    'In_Frame_Del',
    'In_Frame_Ins',
    'Missense_Mutation',
    'Nonsense_Mutation',
    'Nonstop_Mutation',
    'RNA',
    'Splice_Site',
    'Translation_Start_Site',
}


# In[8]:

# Process mutation in long format to dataframe format
mut_pivot = (mutation_df.query("effect in @mutations")
                        .groupby(['#sample', 'chr',
                                  'gene'])
                        .apply(len).reset_index()
                        .rename(columns={0: 'mutation'}))

mut_pivot = (mut_pivot.pivot_table(index='#sample',
                                   columns='gene', values='mutation',
                                   fill_value=0)
                      .astype(bool).astype(int))
mut_pivot.index.name = 'SAMPLE_BARCODE'

mut_pivot.to_csv(mut_out_file, sep='\t', compression='gzip')


# In[9]:

# Process per sample mutation burden

# Mutation burden quantifies how many mutations are observed in a single tumor.
# It is a measure of how unstable a tumors' genome is, potentially reflecting
# deficiences in DNA repair and/or specific etiology. It is a major source of
# variation in expression data and is used by alternative downstream models or
# to explore relationships in tybalt features.
burden_df = mutation_df[mutation_df['effect'].isin(mutations)]
burden_df = burden_df.groupby('#sample').apply(len)
burden_df = np.log10(burden_df).reset_index()
burden_df.columns = ['SAMPLE_BARCODE', 'log10_mut']
burden_df.to_csv(mut_burden_file, index=False, sep='\t')


# ### Copy Number
# 
# Copy number data contains thresholded GISTIC2.0 calls where 0 equals wild-type copy number, 1 and -1 mean slight gain and slight loss, respectively, and 2 and -2 mean high gain and high loss, respectively.

# In[10]:

copy_df = copy_df.astype(int)
copy_df = copy_df.T
copy_df.columns.name = 'gene'
copy_df.index.name = 'Sample'


# In[11]:

# For our purposes, a copy loss status event (1 vs. 0) is conservatively defined only as a deep loss.
copy_loss_df = copy_df.replace(to_replace=[1, 2, -1], value=0)
copy_loss_df.replace(to_replace=-2, value=1, inplace=True)
copy_loss_df.to_csv(copy_loss_out_file, sep='\t', compression='gzip')

# A copy gain status event (1 vs. 0) is defined only as a high gain.
copy_gain_df = copy_df.replace(to_replace=[-1, -2, 1], value=0)
copy_gain_df.replace(to_replace=2, value=1, inplace=True)
copy_gain_df.to_csv(copy_gain_out_file, sep='\t', compression='gzip')


# ### Clinical Data
# 
# The clinical data consists of hundreds of parameters collected for the TCGA samples. Some columns are redundant, while others contain high amounts of missing data. Select and rename only a couple columns of interest.

# In[12]:

clinical_columns_dict = {
    'gdc_platform': 'platform',
    'gdc_center.short_name': 'analysis_center',
    'gdc_cases.submitter_id': 'sample_id',
    'gdc_cases.demographic.gender': 'gender',
    'gdc_cases.demographic.race': 'race',
    'gdc_cases.demographic.ethnicity': 'ethnicity',
    'gdc_cases.project.primary_site': 'organ',
    'gdc_cases.project.project_id': 'acronym',
    'gdc_cases.tissue_source_site.project': 'disease',
    'gdc_cases.diagnoses.vital_status': 'vital_status',
    'gdc_cases.samples.sample_type': 'sample_type',
    'cgc_case_age_at_diagnosis': 'age_at_diagnosis',
    'cgc_portion_id': 'portion_id',
    'cgc_slide_percent_tumor_nuclei': 'percent_tumor_nuclei',
    'cgc_drug_therapy_drug_name': 'drug',
    'xml_year_of_initial_pathologic_diagnosis': 'year_of_diagnosis',
    'xml_stage_event_pathologic_stage': 'stage' 
}


# In[13]:

clinical_sub_df = clinical_df.filter(items=clinical_columns_dict.keys())
clinical_sub_df = clinical_sub_df.rename(columns=clinical_columns_dict)
clinical_sub_df.index = clinical_sub_df['sample_id']
clinical_sub_df.drop('sample_id', axis=1, inplace=True)
clinical_sub_df['acronym'] = clinical_sub_df['acronym'].str[5:]
clinical_sub_df.to_csv(clinical_processed_out_file, sep='\t')


# ### OncoKB Gene-Types
# 
# Here, we use the [OncoKB API](http://oncokb.org/#/dataAccess) of [Chakravarty et al. 2017](http://ascopubs.org/doi/abs/10.1200/JCO.2016.34.15_suppl.11583) to download all oncogenes and tumor suppressor genes. These help to subset the copy number gain and copy loss data frames to identify a full status matrix.

# In[14]:

response = requests.get('http://oncokb.org/api/v1/genes')
oncokb_df = pd.read_json(response.content)
oncokb_df.to_csv(oncokb_out_file, sep='\t')


# ### Full Status Matrix
# 
# **We create the full status matrix with:** 
# 
# - Deleterious mutations = 1
# - High copy gains of oncogenes = 1
# - Deep copy losses of tumor suppressor genes = 1
# - All other gene by sample relationships = 0

# In[15]:

# Integrate copy number, oncokb gene-type, and mutation status to define status matrix
oncogenes_df = oncokb_df[oncokb_df['oncogene']]
tsg_df = oncokb_df[oncokb_df['tsg']]


# In[16]:

# Subset copy gains by oncogenes and copy losses by tumor suppressors (tsg)
status_gain = copy_gain_df.loc[:, oncogenes_df['hugoSymbol']]
status_loss = copy_loss_df.loc[:, tsg_df['hugoSymbol']]
copy_status = pd.concat([status_gain, status_loss], axis=1)


# In[17]:

# NOTCH1 and NOTCH2 are both tumor suppressors and oncogenes depending on tissue context
# drop them from copy number consideration for now
copy_status = copy_status.drop(['NOTCH1', 'NOTCH2'], axis=1)

# Subset each dataframe
status_samples = set(mut_pivot.index) & set(copy_status.index)

mutation_status = mut_pivot.loc[status_samples, :]
copy_status = copy_status.loc[status_samples, :]
copy_status = copy_status.loc[:, mutation_status.columns].fillna(0).astype(int)


# In[18]:

# Combine and write to file
full_status = mutation_status.add(copy_status, fill_value=0)
full_status.replace(to_replace=2, value=1, inplace=True)
full_status.to_csv(known_status_file, sep='\t', compression='gzip')

