#!/bin/bash

# Get results for compression algorithms in simulated setting

SEEDS=20
NOISES=( 0 0.1 0.5 1 3 )
SAMPLE_SIZES=( 500 1000 2000 4000 8000 )
GENE_SIZES=( 500 1000 )
DATA_PREFIX='data/simulation/sim_data_samplesize_'
OUT_PREFIX='results/simulation/output_sim_data_samplesize_'

for noise in "${NOISES[@]}"
do
    for sample_size in "${SAMPLE_SIZES[@]}"
    do
        for num_genes in "${GENE_SIZES[@]}"
        do
            file_suffix=$sample_size'_noise_'$noise'_genes_'$num_genes'.tsv'
            echo $file_suffix
            data_file=$DATA_PREFIX$file_suffix
            out_file=$OUT_PREFIX$file_suffix
            python run_simulation.py --data $data_file \
                                     --noise $noise \
                                     --sample_size $sample_size \
                                     --num_genes $num_genes \
                                     --out_file $out_file \
                                     --seeds $SEEDS
        done
    done
done

