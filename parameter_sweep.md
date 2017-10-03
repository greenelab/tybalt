# Tybalt :smirk_cat: Pan-Cancer Variational Autoencoder

*Gregory Way and Casey Greene 2017*

## Parameter Sweep

We performed a parameter sweep for three distinct architectures:

1. Compression with one hidden layer into 100 features
  - 5000 -> 100 -> 5000
2. Compression with two hidden layers into 100 hidden units and 100 features
  - 5000 -> 100 -> 100 -> 100 -> 5000
3. Compression with two hidden layers into 300 hidden units and 100 features
  - 5000 -> 300 -> 100 -> 300 -> 500

### One Hidden Layer

![parameter sweep](figures/param_sweep/full_param_final_param_val_loss.png?raw=true)

Based on optimal validation loss, we chose optimal hyperparameters to be
`learning rate` = 0.0005, `batch size` = 50, and `epochs` = 100.

We performed a parameter sweep under a small grid of
[possible values](config/parameter_sweep.tsv).

parameter      |  sweep
---------------|----------------------------------
learning_rate  |  0.0005,0.001,0.0015,0.002,0.0025
batch_size     |  50,100,128,200
epochs         |  10,25,50,100
kappa          |  0.01,0.05,0.1,1

In general, we observed little to no difference across many parameters
indicating training the VAE on this data with this architecture is relatively
robust across parameter settings. This is particularly true for different
settings of `kappa`. 

`kappa` controls the warmup period from transitioning from a deterministic
autoencoder to a variational model. `kappa` linearly increases the KL divergence
loss penalty until weighted evenly with reconstruction cost. We do not observe
this parameter influencing training time or optimal loss.

### Two Hidden Layers

![param sweep two](figures/param_sweep/twohidden/full_param_final_param_val_loss.png?raw=true)

Based on optimal validation loss, the two hidden layer model has optimal
hyperparameters at `learning_rate` = 0.001, `batch size` = 100, and `epochs` = 100.

Again, training was relatively stable with comparable performance over a large grid.
With two layers, `kappa` made a larger difference. The burn in `kappa` period actually
penalized model performance, with `kappa` < 1 having consistently worse performance.

We also trained a model with an alternative two layer architecture with 300 hidden features.

### Comparison

Two hidden layers does not improve performance as much as initially thought. There is also
not much benefit in 2 compression layers. Observed below are the three optimal models
described above.

![compare](figures/param_sweep/best_model_comparisons.png?raw=true)

We have yet to perform comparisons in regards to the biology learned by each model.

