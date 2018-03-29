# Results for Initial Hyperparameter Sweep Considering Different Latent Dimensionality

**Gregory Way 2018**

## Latent Space Dimensionality

Compression algorithms reduce the dimesionality of input data by enforcing the number of dimensions to bottleneck.
A common problem is the decision of how many "useful" latent space features are present in data.
The solution is different for different problems or goals.
For example, when visualizing large differences between groups of data, a highly restrictive bottleneck, usually between 2 or 3 features, is required.
However, when the goal is to extract meaningful patterns in the data that may have more subtle relationships across samples, the recommendations are opaque.
As the bottleneck relaxes, the ability to explain the patterns decreases and the possibility of false positives increases.

In order to determine an optimal _range_ of compression dimensions, we propose the following.
We will first sweep over various different dimensions (results provided below) and perform several evals (to be described later).

Before sweeping over a large number of different dimensions, we perform a hyperparameter sweep of select dimensions.
In this sense, we want to minimize the effect of poor hyperparameter combinations across different dimensions contributing to performance differences.
In other words, we want to isolate the effect of changing dimensionality on the observed patterns and solutions.
Therefore, we perform a parameter sweep over several hyperparameters for both Tybalt and ADAGE models below.

### Number of dimensions

Previously, we used latent space dimensionality of `100` ([Way and Greene 2018](https://doi.org/10.1142/9789813235533_0008)).
Here, we sweep over dimensions: `5`, `25`, `50`, `75`, `100`, and `125`.

## Parameter Sweep

We sweep over the following parameter combinations for Tybalt and ADAGE models:

| Variable | Tybalt Values | ADAGE Values |
| :------- | :------------ | :----------- |
| Dimensionality | 5, 25, 50, 75, 100, 125 | 5, 25, 50, 75, 100, 125 |
| Learning Rate | 0.0005, 0.001, 0.0015, 0.002, 0.0025 | 0.0005, 0.001, 0.0015, 0.002, 0.0025 |
| Batch Size | 50, 100, 150 | 50, 100 |
| Epochs | 50, 100 | 50, 100 |
| Kappa | 0, 0.5, 1 | |
| Sparsity | | 0, 0.000001, 0.001 |
| Noise | | 0, 0.1, 0.5 |

This resulted in the training of 540 Tybalt models and 1,080 ADAGE models.

Our goal was to determine optimal hyperparameter combinations for both models across various bottleneck dimensionalities.

## Results

We report the results in a series of visualizations and tables for Tybalt and ADAGE separately below.

### Tybalt

Tybalt models had variable performance across models, but was generally stable across all hyperparameter combinations (**Figure 1**).

![](figures/param_sweep/z_param_tybalt/z_parameter_tybalt.png?raw=true)

**Figure 1.** The loss of validation sets at the end of training for all 540 Tybalt models.

Model performance (measured by observed validation loss) improved after increasing the capacity of the models from 5 to 125 dimensions.
However, the performance began to level off after around 50 dimensions.
All other hyperparameters had minor effects and varied with dimensionality, including `Kappa`.

Selecting a constant learning rate, batch size, and epochs for all models, we stratify the entire training process across all tested dimensionalities (**Figure 2**).

![](figures/param_sweep/z_param_tybalt/z_parameter_tybalt_training.png?raw=true)

**Figure 2.** Validation loss across all training epochs for a fixed combination of hyperparameters.
The `learning rate` was set to 0.0005, `batch size` 50, and `epochs` 100.
We also show performance differences across `kappa`.

This analysis allowed us to select optimal models based on tested hyperparameters.
For Tybalt, the optimal hyperparameters across dimensionality estimates are:

| Dimensions | Kappa | Epochs | Batch Size | Learning Rate | End Loss |
| :--------- | :---- | :----- | :--------- | :------------ | :------- |
| 5 | 1 | 50 | 50 | 0.002 | 2805.4 |
| 25 | 0 | 50 | 50 | 0.0015 | 2693.9 |
| 50 | 0 | 100 | 100 | 0.002 | 2670.5 |
| 75 | 0 | 100 | 150  | 0.002 | 2656.7 |
| 100 | 0 | 100 | 100 | 0.001 | 2651.7 |
| 125 | 0 | 100 | 150 | 0.0005 | 2650.3 |

Generally, it appears that the optimal `learning rate` and `kappa` decreases while the `batch size` and `epochs` increase as the dimensionality increases.
Training of models with optimal hyperparameters are shown in **figure 3**.

![](figures/param_sweep/z_param_tybalt/z_parameter_tybalt_best.png?raw=true)

**Figure 3.** Training optimal Tybalt models across different latent space dimensions.

### ADAGE

ADAGE models had variable performance across models and failed to converge with high levels of sparsity (**Figure 4**).
High levels of sparsity fail _worse_ with increasing dimensionality.

![](figures/param_sweep/z_param_adage/z_parameter_adage.png?raw=true)

**Figure 4.** The loss of validation sets at the end of training for all 1,080 ADAGE models.

After removing `sparsity = 0.001`, we see a clearer picture (**Figure 5**).

![](figures/param_sweep/z_param_adage/z_parameter_adage_remove_sparsity.png?raw=true)

**Figure 5.** The loss of validation sets at the end of training for 720 ADAGE models.

A similar pattern appears where lower dimensionality benefits from increased sparsity.
ADAGE models are also generally stable, particularly at high dimensions.

This analysis allowed us to select optimal models based on tested hyperparameters.
For ADAGE, the optimal hyperparameters across dimensionality estimates are:

| Dimensions | Sparsity | Noise | Epochs | Batch Size | Learning Rate | End Loss |
| :--------- | :------- | :---- | :----- | :--------- | :------------ | :------- |
| 5 | 0 | 0.5 | 100 | 100 | 0.0005 | 0.021 |
| 25 | 0 | 0.5 | 100 | 100 | 0.0005 | 0.012 |
| 50 | 0.000001 | 0 | 100 | 50 | 0.0005 | 0.011 |
| 75 | 0 | 0.1 | 50 | 100  | 0.0005 | 0.010 |
| 100 | 0 | 0.1 | 100 | 100 | 0.0005 | 0.010 |
| 125 | 0 | 0.1 | 100 | 50 | 0.0005 | 0.009 |

It appears that `learning rate` is globally optimal at 0.0005.
`batch_size` and `epochs` are also generally consistent at 100.
Also, the lower the dimensionality, the more regularization required, whether it is `noise` or `sparsity` added.

![](figures/param_sweep/z_param_adage/z_parameter_adage_best.png?raw=true)

**Figure 6.** Training optimal ADAGE models across different latent space dimensions.

## Summary

Selection of hyperparameters across different latent space dimensionality operated as expected.
Loss was higher for lower dimensions and lower dimensions benefitted the most from increased regularization.
Nevertheless, we have obtained a broad set of optimal hyperparameters for use in a larger and more specific sweep of dimensionality.


