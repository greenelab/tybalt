#!/bin/bash

# Note this was run at the PMACS cluster at University of Pennsvylvania on a cluster of
# 8 NVIDIA GEFORCE GTX Ti GPUs.
~/.conda/envs/vae_pancancer/bin/python scripts/vae_paramsweep.py --parameter_file 'config/parameter_sweep.tsv' --config_file 'config/pmacs_config.tsv' --python_path '~/.conda/envs/vae_pancancer/bin/python' --param_folder 'param_sweep'

