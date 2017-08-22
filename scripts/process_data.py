
# coding: utf-8

# # Processing all datasets to be used in downstream analyses

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
rna_out_file = os.path.join('data', 'pancan_scaled_rnaseq.tsv')
rna_out_zeroone_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv')

# Mutation Data
mut_out_file = os.path.join('data', 'pancan_mutation.tsv')

# Two copy number matrices, for thresholded (2) gains and losses
copy_gain_out_file = os.path.join('data', 'copy_number_gain.tsv')
copy_loss_out_file = os.path.join('data', 'copy_number_loss.tsv')

# Clinical data
clinical_processed_out_file = os.path.join('data', 'clinical_data.tsv')

# OncoKB output file
oncokb_out_file = os.path.join('data', 'oncokb_genetypes.tsv')

# Status matirix integrating mutation and copy number events
known_status_file = os.path.join('data', 'status_matrix.tsv')


# ## Load Data

# In[4]:

rnaseq_df = pd.read_table(rna_file, index_col=0)
mutation_df = pd.read_table(mut_file)
copy_df = pd.read_table(copy_file, index_col=0)
clinical_df = pd.read_table(clinical_file, index_col=0, low_memory=False)


# ## Begin processing different data types
# 
# ### RNAseq

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
rnaseq_scaled_df.to_csv(rna_out_file, sep='\t')

# Scale RNAseq data using zero-one normalization
rnaseq_scaled_zeroone_df = preprocessing.MinMaxScaler().fit_transform(rnaseq_subset_df)
rnaseq_scaled_zeroone_df = pd.DataFrame(rnaseq_scaled_zeroone_df,
                                        columns=rnaseq_subset_df.columns,
                                        index=rnaseq_subset_df.index)
rnaseq_scaled_zeroone_df.to_csv(rna_out_zeroone_file, sep='\t')


# ### Mutation

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

mut_pivot.to_csv(mut_out_file, sep='\t')


# ### Copy Number

# In[9]:

copy_df = copy_df.fillna(0)
copy_df = copy_df.astype(int)
copy_df = copy_df.T
copy_df.columns.name = 'gene'
copy_df.index.name = 'Sample'


# In[10]:

copy_loss_df = copy_df.replace(to_replace=[1, 2, -1], value=0)
copy_loss_df.replace(to_replace=-2, value=1, inplace=True)
copy_loss_df.to_csv(copy_loss_out_file, sep='\t')

copy_gain_df = copy_df.replace(to_replace=[-1, -2, 1], value=0)
copy_gain_df.replace(to_replace=2, value=1, inplace=True)
copy_gain_df.to_csv(copy_gain_out_file, sep='\t')


# ### Clinical Data

# In[11]:

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
    'xml_stage_event_pathologic_stage': 'stage', 
}


# In[12]:

clinical_sub_df = clinical_df.filter(items=clinical_columns_dict.keys())
clinical_sub_df = clinical_sub_df.rename(columns=clinical_columns_dict)
clinical_sub_df.index = clinical_sub_df['sample_id']
clinical_sub_df.drop('sample_id', axis=1, inplace=True)
clinical_sub_df['acronym'] = clinical_sub_df['acronym'].str[5:]
clinical_sub_df.to_csv(clinical_processed_out_file, sep='\t')


# ### OncoKB Gene-Types

# In[13]:

response = requests.get('http://oncokb.org/api/v1/genes')
oncokb_df = pd.read_json(response.content)
oncokb_df.to_csv(oncokb_out_file, sep='\t')


# ### Full Status Matrix

# In[14]:

# Integrate copy number, oncokb gene-type, and mutation status to define status matrix
oncogenes_df = oncokb_df[oncokb_df['oncogene']]
tsg_df = oncokb_df[oncokb_df['tsg']]


# In[15]:

# Subset copy gains by oncogenes and copy losses by tumor suppressors (tsg)
status_gain = copy_gain_df.loc[:, oncogenes_df['hugoSymbol']]
status_loss = copy_loss_df.loc[:, tsg_df['hugoSymbol']]
copy_status = pd.concat([status_gain, status_loss], axis=1)


# In[16]:

# Subset each dataframe
status_samples = set(mut_pivot.index) & set(copy_status.index)

mutation_status = mut_pivot.loc[status_samples, :]
copy_status = copy_status.loc[status_samples, :]
copy_status = copy_status.loc[:, mutation_status.columns].fillna(0).astype(int)


# In[17]:

# Combine and write to file
full_status = copy_status + mutation_status
full_status.to_csv(known_status_file, sep='\t')

