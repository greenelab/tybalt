#!/bin/bash

# Download archived PanCancer RNAseq and Mutation data
url=https://zenodo.org/record/56735/files/gbm_classifier_data.tar.gz
wget --directory-prefix 'data/raw' $url

# Download archived Copy Number data
url=https://zenodo.org/record/827323/files/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz
wget --directory-prefix 'data/raw' $url

# Extract and move files
tar -zxvf data/raw/gbm_classifier_data.tar.gz
mv gbm_download/* data/raw
rm -rf gbm_download
rm data/raw/gbm_classifier_data.tar.gz

gunzip data/raw/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz

# Download clinical data files from JHU Snaptron Paper
# http://snaptron.cs.jhu.edu/
clinical_url=http://snaptron.cs.jhu.edu/data/tcga/samples.tsv
wget --directory-prefix 'data/raw' $clinical_url

