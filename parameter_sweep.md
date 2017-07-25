# Variational Autoencoder - Pan Cancer

*Gregory Way and Casey Greene 2017*

## Parameter Sweep

![parameter sweep](figures/param_sweep/final_param_val_loss.png?raw=true)

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
indicating training the VAE in this data with this architecture is relatively
robust across parameter settings. This is particularly true for different
settings of `kappa`. 

![sweep kappa 0.01](figures/param_sweep/full_param_0.01_kappa.png?raw=true)

![sweep kappa 0.05](figures/param_sweep/full_param_0.05_kappa.png?raw=true)

![sweep kappa 0.1](figures/param_sweep/full_param_0.1_kappa.png?raw=true)

![sweep kappa 1](figures/param_sweep/full_param_1.0_kappa.png?raw=true)

`kappa` controls the warmup period from transitioning from a deterministic
autoencoder to a variational model. `kappa` linearly increases the KL divergence
loss penalty until weighted evenly with reconstruction cost. We do not observe
this parameter influencing training time or optimal loss.