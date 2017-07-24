#!/bin/bash

~/.conda/envs/vae_pancancer/bin/python scripts/vae_paramsweep.py --parameter_file 'config/parameter_sweep.tsv' --config_file 'config/pmacs_config.tsv' --python_path '~/.conda/envs/vae_pancancer/bin/python' --param_folder 'param_sweep'

