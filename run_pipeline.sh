#!/bin/bash

# 0) Download and process all data required in the pipeline

bash download_data.sh
python scripts/process_data.py

# 1) Run parameter sweep for variational autoencoder over: Learning rate, batch
# size, epochs, and kappa. 

# Parameter sweep was run on a cluster with 8 NVIDIA GEFORCE GTX 1080 Ti GPUs

bash param_sweep.sh
python scripts/summarize_paramsweep.py
Rscript scripts/viz/param_sweep_viz.R

# 2) Run optimized model and combine results with clinical data

python scripts/tybalt_vae.py
Rscript scripts/combine_clinical_encoded.R

# 3) Evaluate the model for sample activations

Rscript scripts/viz/sample_activation_distribution.R
python scripts/tsne_vae.py
Rscript scripts/viz/tsne_viz.R

# 4) Evaluate the model for learned weight (gene) coefficients

python scripts/explore_weights.py
Rscript scripts/viz/feature_activation_plots.R

# 5) Evaluate Tybalt manifold manipulations for HGSC subtypes
python scripts/hgsc_subtypes_vae.py

