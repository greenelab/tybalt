#!/bin/bash

# The purpose of this script is to determine a reasonable set of hyperparameters
# for different input z dimensionality. The ultimate goal is to perform an
# analysis on the latent space dimensionality, and we want to be sure that
# the comparisons being made are not a result of poor input hyperparameters

# The parameters were selected to represent a range of hyperparameters that
# we have previously observed to impact training. Learning rates appear to
# have the largest impact on performance, so we include a larger search space.

# Note this was run at the PMACS cluster at University of Pennsvylvania on a
# cluster of 8 NVIDIA GEFORCE GTX Ti GPUs.

PARAM_FILE_PREFIX='config/z_parameter_sweep_'
PMACS_FILE='config/pmacs_config.tsv'
PYTHON_PATH='python'
ALGORITHMS=( 'tybalt' 'adage' )

for alg in "${ALGORITHMS[@]}"
do
    PARAM_FILE=$PARAM_FILE_PREFIX$alg'.tsv'
    python scripts/num_components_paramsweep.py \
          --parameter_file $PARAM_FILE \
          --config_file $PMACS_FILE \
          --algorithm $alg \
          --python_path $PYTHON_PATH \
          --param_folder 'z_param_sweep_'$alg
done
