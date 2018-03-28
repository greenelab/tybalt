# Results of the Simulation Analyses

**Gregory Way 2018**

## Algorithms

Our goal was to evaluate the effectiveness of 5 different compression algorithms as applied to simulated gene expression data.
The algorithms included: PCA, ICA, NMF, [ADAGE](https://doi.org/10.1128/mSystems.00025-15), and [Tybalt](https://doi.org/10.1142/9789813235533_0008).

We simulated gene expression data and applied each algorithm to the simulated datasets.
We fit each model to each dataset 20 different times to measure the stability of solutions.
We performed several evaluations and report results after describing the data setup below.

## Setup

We simulated 50 datasets using the `simulateDatExpr` function from the R package [WGCNA](https://labs.genetics.ucla.edu/horvath/CoexpressionNetwork/Rpackages/WGCNA/).
Within each dataset we varied the sample size, number of genes, and amount of background noise added.
Specially, we used these parameters:

| Variable | Options |
| :------: | :------ |
| Sample Size | 500, 1000, 2000, 4000, 8000 |
| Number of Genes | 500, 1000 |
| Background Noise Added | 0, 0.1, 0.5, 1, 3 |

### Sample Groups

In each dataset, we simulated two different groups of samples.
For example, when `n = 500`, the groups of samples include the following correlation structure.
Note that this is _not_ the actual correlational structure, but is the underlying structure that is then sampled from stochastically given the gene modules described below.

![](figures/simulation/example_eigen_sample_plot.png?raw=true)

### Gene Modules

Additionally, we simulated 5 different "gene modules" that had differential activity across sample groups and included a decreasing proportion of genes.
Concretely, `module 1` included 25% of genes, `module 2` included 20%, `module 3` 15%, `module 4` 10% and `module 5` 5%.
The remaining 25% of genes are random noise independent of sample group (`module 0`).

A representative example of this correlation structure is included below:

![](figures/simulation/example_eigen_module_plot.png?raw=true)

We included correlations across modules to mimic a more realistic biological scenario.

### Module 3 - Testing Latent Space Arithmetic (LSA)

Lastly, `module 3` was included in _only_ 1/2 of all samples in each sample group.
Therefore, we can assign 4 distinct groups of samples, which include:

| Group | Description |
| :---: | :---------  |
| A | Sampled from group set 1 with `module 3` |
| B | Sampled from group set 2 with `module 3` |
| C | Sampled from group set 1 without `module 3` |
| D | Sampled from group set 2 without `module 3` |

The purpose of leaving out module 3 in some samples is to test the hypothesis that the latent space of compression algorithms can be manipulated mathematically.
We tested the following:

```
# Latent space arithmetic with the mean vector representations of each group
A - C + D = B_hat

# Test euclidean distance between the reconstructed B_hat and mean(B) vector
result = ||reconstruct(B_hat) - B||
```

### Example Data

An example of this data is provided below.
Note the four groups of samples (rows) and the separation of 5 modules (plus noise) of decreasing size (columns)

![](figures/simulation/example_simulated_data.png?raw=true)

## Run Analysis Pipeline

Perform the following to reproduce all results from the simulated data analysis.

```bash
# 1) Generate simulated data
Rscript scripts/util/simulate_expression.R

# 2) Compress and evaluate results in 50 simulated data sets
#    across 20 different initializations
./simulation_analysis.sh

# 3) Aggregate results of the parallelized simulation results extraction
python scripts/util/aggregate_simulation_results.py

# 4) Visualize all evaluations
Rscript scripts/util/plot_simulation.R
```

## Evaluate and Visualize Results

We evalute the models on the simulated data in the following ways:

1. Do the compressed features capture gene modules?
2. Are the algorithms' reconstruction of `B_hat` near `B`?
3. Does the latent space arithmetic capture the feature most representative of module 3?

All evaluations are compared across varying numbers of sample size and genes and varying degrees of noise injection.

### 1) Do compressed features capture gene modules?

The procedure is as follows.
Take the mean rank of all compressed features across all modules (we know ground truth gene to module membership).
For example, if module 1 genes were all ranked highly by the weight matrix transformation in PCA component 2, then assign PCA component 2 to module 1.
Do this for all components across all compression algorithms and take the mean of the ranks.
The result, across different sample sizes, number of genes, and noise injection is provided below:

![](figures/simulation/mean_module_rank.png?raw=true)

In this plot, a low value indicates an algorithm's ability to robustly assign latent space features to _unique_ gene modules.
All algorithms tend to perform worse with increasing noise and better with increasing sample size.
PCA (light blue) performed the best in nearly all scenarios with the exception to situations with high noise.
In high noise cases, NMF (green) performed the best.
Tybalt (VAE; purple) performed better than ADAGE, NMF, and ICA in almost all cases.
ADAGE (DAE, red) solutions tended to be unstable, particularly in simulations with a higher number of genes.
Many ADAGE solutions did not converge indicating a stronger need for finer hyperparameter tuning.

We also visualized results without taking the mean rank.
This visualization demonstrates how gene modules with decreasing size were captured by the algorithms.
An example with `noise = 0.5` and `sample size = 4000` is provided below:

![](figures/simulation/module_rank/mod_rank_noise_0.5_n_4000.png?raw=true)

Results for all combinations of `noise` and `sample size` is provided in `figures/simulation/module_rank/`.
Note that in the `mean rank` procedure, we did not include the noise module.

Based on this example, it is clear that PCA captures all modules uniquely quite well.
NMF also captures modules well, with stable solutions.
Both ICA and PCA capture the smaller gene modules (`module 4` and `module 5`) better than larger modules (`module 1` and `module 2`).
Both Tybalt and ADAGE did not capture the noise module (`module 0`).
ADAGE was also not able to capture modules for most initializations with high number of genes.
`module 3`, which is the module simulated to be absent in 1/2 of samples, was captured by all algorithms except for ADAGE.

### 2) Are the algorithms' reconstruction of `B_hat` near `B`?

We next asked if the LSA experiment resulted in reconstructions of `B_hat` that were "close" to `B`.

![](figures/simulation/lsa_reconstruction_distance.png?raw=true)

All algorithms performed worse with increasing noise injection.
ADAGE reconstructions were largely unstable.

### 3) Does the latent space arithmetic capture the feature most representative of module 3?

Taking this one step further, we next asked if the reconstruction worked because it isolated the known differences of the groups.
Essentially, [did the LSA subtraction capture the essence of module 3](https://arxiv.org/abs/1511.06434).

![](figures/simulation/lsa_module_3_capture.png?raw=true)

Here, the y axis is the z-score of the subtraction operation (`A - C`) (mean vector representations of both groups) of the compression feature with the lowest rank of module 3 genes (see evaluation 1 above).
Therefore, high and low scores indicate the subtraction was isolating the feature most representative of `module 3`.
This is a busy plot, but in most cases, Tybalt and PCA lead the pack.
Tybalt performed consistently strong, even in cases with high noise.
The solutions for Tybalt are also generally consistent.
The subtraction for PCA sometimes results in high or low z scores, but is always high for tybalt.

We show one representative combination of sample size (`n = 8000`) and genes (`p = 500`) to provide a clearer picture.

![](figures/simulation/lsa_module_3_capture_n8000_p500.png?raw=true)

## Summary

In summary, compression algorithms can extract gene modules and sample groups in simulated gene expression data.
PCA and ICA perform well in all tasks (particularly PCA).
This is likely because the simulation procedure assumes linear relationships among samples, and relatively low cross correlational structure across genes.
Therefore, this simulation may not adequately represent the complexity of real data.
NMF also performs well, particularly in solutions with high noise injection.
ADAGE solutions are generally unstable and may require additional hyperparameter tuning than other algorithms.
This instability may actually benefit biological knowledge identification when applied in an [ensemble approach](https://doi.org/10.1016/j.cels.2017.06.003).
Other algorithms can perform LSA, but Tybalt shines when asked to find the "essence" of the difference between sample groups.
