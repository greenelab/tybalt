# Pan-Cancer Variational Autoencoder
# Gregory Way 2018
#
# Visualizing results of tybalt and adage parameter sweeps over z dimensions
#
# First, to summarize the parameter sweep into a single file run:
#
#     python scripts/summarize_paramsweep.py
#               --results_directory "z_param_sweep_tybalt"
#               --output_filename "z_parameter_sweep_tybalt_full_results.tsv"
#
# Also perform the same for the ADAGE sweep
#
# Output:
# Several figures describing the results of a parameter sweep over latent space
# dimensionality for both Tybalt and ADAGE models

library(ggplot2)
library(colorblindr)

set.seed(123)

`%>%` <- dplyr::`%>%`

# Store base theme for plotting
base_theme <-  theme(axis.text = element_text(size = rel(0.5)),
                     axis.title = element_text(size = rel(0.7)),
                     axis.text.x = element_text(angle = 45),
                     strip.text = element_text(size = rel(0.5)),
                     legend.text = element_text(size = rel(0.6)),
                     legend.title = element_text(size = rel(0.8)),
                     legend.key.height = unit(0.5, "line"))

# Set input and output file names
tyb_file <- file.path("results", "z_parameter_sweep_tybalt_full_results.tsv")
adage_file <- file.path("results", "z_parameter_sweep_adage_full_results.tsv")

tyb_fig <- file.path("figures", "param_sweep", "z_param_tybalt")
adage_fig <- file.path("figures", "param_sweep", "z_param_adage")

# Read and process the tybalt latent space parameter sweep
tybalt_df <- readr::read_tsv(tyb_file,
                             col_types = readr::cols(
                               .default = readr::col_character(),
                               train_epoch = readr::col_integer(),
                               loss = readr::col_double(),
                               val_loss = readr::col_double()))

# Melt the data frame with loss and val loss as measurement variables
tybalt_melt_df <- reshape2::melt(tybalt_df,
                                 id.vars = c("learning_rate", "batch_size",
                                             "epochs", "kappa", "train_epoch",
                                             "num_components"),
                                 measure.vars = c("loss", "val_loss"),
                                 variable.name = "loss_type",
                                 value.name = "loss")

# Get the validation loss at epoch end for each hyperparameter combination
tybalt_select_df <- tybalt_melt_df %>%
  dplyr::filter(loss_type == "val_loss") %>%
  dplyr::group_by(learning_rate, batch_size, epochs, kappa, num_components) %>%
  dplyr::summarize(end_loss = tail(loss, 1))

# Order batch size and epoch variables
tybalt_select_df$batch_size <- factor(tybalt_select_df$batch_size,
                                      levels = c(50, 100, 150))
tybalt_select_df$epochs <- factor(tybalt_select_df$epochs ,
                                  levels = c(50, 100))

# Recode batch size and learning rate variables for plotting
tybalt_select_df$batch_size <-
  dplyr::recode_factor(tybalt_select_df$batch_size, 
                       `50` = "Batch: 50",
                       `100` = "Batch: 100",
                       `150` = "Batch: 150")

tybalt_select_df$learning_rate <-
  dplyr::recode_factor(tybalt_select_df$learning_rate, 
                       `0.0005` = "Learning: 0.0005",
                       `0.001` = "Learning: 0.001", 
                       `0.0015` = "Learning: 0.0015",
                       `0.002` = "Learning: 0.002",
                       `0.0025` = "Learning: 0.0025")

# 1) Describe final validation loss for all hyperparameter combinations
# across number of latent space dimensionality (x axis)
tybalt_param_z_png <- file.path(tyb_fig, "z_parameter_tybalt.png")
tybalt_param_z_pdf <- file.path(tyb_fig, "z_parameter_tybalt.pdf")

p <- ggplot(tybalt_select_df, aes(x = as.numeric(paste(num_components)),
                             y = end_loss)) +
  geom_point(aes(shape = epochs, color = kappa), size = 0.8, alpha = 0.7,
             position = position_jitter(w = 5, h = 0)) +
  scale_x_continuous(breaks = c(5, 25, 50, 75, 100, 125)) +
  scale_color_OkabeIto() +
  facet_grid(batch_size ~ learning_rate) +
  ylab("Final Validation Loss") +
  xlab("Latent Space Dimensionality") +
  theme_bw() + base_theme

ggsave(tybalt_param_z_png, plot = p, height = 3, width = 5.5)
ggsave(tybalt_param_z_pdf, plot = p, height = 3, width = 5.5)

# 2) Plot more detail of a single learning rate and batch size combination
tybalt_one_model <- tybalt_df %>%
  dplyr::filter(learning_rate == "0.0005", batch_size == "50", epochs == 100)

# Recode number of components for plotting
tybalt_one_model$num_components <-
  dplyr::recode_factor(tybalt_one_model$num_components, 
                       `5` = "Latent Dim: 5",
                       `25` = "Latent Dim: 25", 
                       `50` = "Latent Dim: 50",
                       `75` = "Latent Dim: 75",
                       `100` = "Latent Dim: 100",
                       `125` = "Latent Dim: 125")

tybalt_one_model_png <- file.path(tyb_fig, "z_parameter_tybalt_training.png")
tybalt_one_model_pdf <- file.path(tyb_fig, "z_parameter_tybalt_training.pdf")

p <- ggplot(tybalt_one_model, aes(x = as.numeric(paste(train_epoch)),
                             y = val_loss)) +
  geom_line(aes(color = kappa), size = 0.2, alpha = 0.7) +
  scale_color_OkabeIto() +
  facet_wrap(~ num_components) +
  ylab("Final Validation Loss") +
  xlab("Training Epochs") +
  theme_bw() +
  base_theme +
  theme(legend.key.height = unit(1, "line"))

ggsave(tybalt_one_model_png, plot = p, height = 2.5, width = 5)
ggsave(tybalt_one_model_pdf, plot = p, height = 2.5, width = 5)

# 3) Plot the optimal hyperparameters for each dimensionality
tybalt_best_params <- tybalt_select_df %>%
  dplyr::group_by(num_components) %>%
  dplyr::top_n(1, -end_loss) %>%
  dplyr::arrange(as.numeric(paste(num_components)))

best_param_file <- file.path("results" , "z_latent_dim_best_tybalt_params.tsv")
readr::write_tsv(tybalt_best_params, best_param_file)

# Subset tybalt to single model with stable hyperparamters
# These parameters will be used in downstream applications sweeping over
# latent space dimensionality.

# We observed that increasing the number of components resulted in changes to
# optimal hyperparameters. As the dimensionality increased, the learning rate
# decreased, batch size and epochs increased, and kappa decreased.
tybalt_good_training_df <- tybalt_melt_df %>%
  dplyr::filter(
    (learning_rate == "0.002" & num_components == 5 & epochs == 50 &
       batch_size == 50 & kappa == "1.0") |
      (learning_rate == "0.002" & num_components == 25 & epochs == 50 &
         batch_size == 50 & kappa == "0.0") |
      (learning_rate == "0.002" & num_components == 50 & epochs == 100 &
         batch_size == 100 & kappa == "0.0") |
      (learning_rate == "0.002" & num_components == 75 & epochs == 100 &
         batch_size == 150 & kappa == "0.0") |
      (learning_rate == "0.001" & num_components == 100 & epochs == 100 &
         batch_size == 150 & kappa == "0.0") |
      (learning_rate == "0.001" & num_components == 125 & epochs == 100 &
         batch_size == 150 & kappa == "0.0")
  )

# Reorder the latent space dimensionality for plotting
num_com <- tybalt_good_training_df$num_components
num_com <- factor(num_com, levels = sort(as.numeric(paste(unique(num_com)))))
tybalt_good_training_df$num_components <- num_com

tybalt_best_model_png <- file.path(tyb_fig, "z_parameter_tybalt_best.png")
tybalt_best_model_pdf <- file.path(tyb_fig, "z_parameter_tybalt_best.pdf")

p <- ggplot(tybalt_good_training_df, aes(x = train_epoch, y = loss)) +
  geom_line(aes(color = num_components, linetype = loss_type), size = 0.2) +
  xlab("Training Epoch") +
  ylab("Loss") +
  scale_color_OkabeIto(name = "Latent Dimensions") +
  scale_linetype_manual(name = "Loss Type", values = c("solid", "dotted"),
                        labels = c("Train", "Validation")) +
  theme_bw() +
  base_theme +
  theme(axis.text.x = element_text(angle = 0))

ggsave(tybalt_best_model_png, plot = p, height = 2.5, width = 4)
ggsave(tybalt_best_model_pdf, plot = p, height = 2.5, width = 4)

# ADAGE visualizations
# Load ADAGE results, process and recode variables for plotting
adage_df <- readr::read_tsv(adage_file,
                            col_types = readr::cols(
                              .default = readr::col_character(),
                              train_epoch = readr::col_integer(),
                              loss = readr::col_double(),
                              val_loss = readr::col_double()))

adage_melt_df <- reshape2::melt(adage_df,
                                id.vars = c("learning_rate", "batch_size",
                                            "epochs", "noise", "train_epoch",
                                            "num_components", "sparsity",
                                            "seed"),
                                measure.vars = c("loss", "val_loss"),
                                variable.name = "loss_type",
                                value.name = "loss")

adage_select_df <- adage_melt_df %>%
  dplyr::filter(loss_type == "val_loss") %>%
  dplyr::group_by(learning_rate, batch_size, epochs, noise, sparsity,
                  num_components) %>%
  dplyr::summarize(end_loss = tail(loss, 1))

adage_select_df$noise <-
  dplyr::recode_factor(adage_select_df$noise, 
                       `0.0` = "Noise: 0",
                       `0.1` = "Noise: 0.1",
                       `0.5` = "Noise: 0.5")

adage_select_df$learning_rate <-
  dplyr::recode_factor(adage_select_df$learning_rate, 
                       `0.0005` = "Learning: 0.0005",
                       `0.001` = "Learning: 0.001", 
                       `0.0015` = "Learning: 0.0015",
                       `0.002` = "Learning: 0.002",
                       `0.0025` = "Learning: 0.0025")

spars <- adage_select_df$sparsity
spars <- factor(spars, levels = c("0.0", "1e-06", "0.001"))
adage_select_df$sparsity <- spars

# 1) Describe final validation loss for all hyperparameter combinations
# across number of latent space dimensionality (x axis) for ADAGE
adage_param_z_png <- file.path(adage_fig, "z_parameter_adage.png")
adage_param_z_pdf <- file.path(adage_fig, "z_parameter_adage.pdf")

p <- ggplot(adage_select_df,
            aes(x = as.numeric(paste(num_components)), y = end_loss)) +
  geom_point(aes(shape = epochs, size = batch_size, color = sparsity),
             alpha = 0.7, position = position_jitter(w = 5, h = 0)) + 
  facet_grid(noise ~ learning_rate) +
  scale_size_manual(values = c(0.8, 0.4), name = "Batch Size") +
  scale_color_OkabeIto(name = "Sparsity") +
  scale_shape_discrete(name = "Epochs") +
  scale_x_continuous(breaks = c(5, 25, 50, 75, 100, 125)) +
  xlab("Latent Space Dimensionality") +
  ylab("Final Validation Loss") +
  theme_bw() + base_theme

ggsave(adage_param_z_png, plot = p, height = 2.5, width = 5.5)
ggsave(adage_param_z_pdf, plot = p, height = 2.5, width = 5.5)

# 2) Sparsity 0.001 did not converge, remove it and replot
adage_select_subset_df <- adage_select_df %>% dplyr::filter(sparsity != 0.001)

adage_sub_png <- file.path(adage_fig, "z_parameter_adage_remove_sparsity.png")
adage_sub_pdf <- file.path(adage_fig, "z_parameter_adage_remove_sparsity.pdf")

p <- ggplot(adage_select_subset_df,
            aes(x = as.numeric(paste(num_components)), y = end_loss)) +
  geom_point(aes(shape = epochs, size = batch_size, color = sparsity), 
             position = position_jitter(w = 5, h = 0),
             alpha = 0.7) + 
  facet_grid(noise ~ learning_rate) +
  scale_size_manual(values = c(0.8, 0.4), name = "Batch Size") +
  scale_color_OkabeIto(name = "Sparsity") +
  scale_shape_discrete(name = "Epochs") +
  scale_x_continuous(breaks = c(5, 25, 50, 75, 100, 125)) +
  xlab("Latent Space Dimensionality") +
  ylab("Final Validation Loss") +
  theme_bw() +
  base_theme

ggsave(adage_sub_png, plot = p, height = 2.5, width = 5.5)
ggsave(adage_sub_pdf, plot = p, height = 2.5, width = 5.5)

# 3) Plot the optimal hyperparameters for each dimensionality for ADAGE
adage_best_params <- adage_select_df %>%
  dplyr::group_by(num_components) %>%
  dplyr::top_n(1, -end_loss) %>%
  dplyr::arrange(as.numeric(paste(num_components)))

best_param_file <- file.path("results" , "z_latent_dim_best_adage_params.tsv")
readr::write_tsv(adage_best_params, best_param_file)

# Subset ADAGE to single model with stable hyperparamters
# These parameters will be used in downstream applications sweeping over
# latent space dimensionality.

# We observed that increasing the number of components resulted in changes to
# optimal hyperparameters. As the dimensionaly increased, the learning rate,
# batch size, epochs and sparsity remained relatively consistent. However,
# The noise added was reduced with higher dimensions

adage_good_training_df <- adage_melt_df %>%
  dplyr::filter(learning_rate == "0.0005") %>%
  dplyr::filter(sparsity == "0.0") %>%
  dplyr::filter(epochs == 100) %>%
  dplyr::filter(batch_size == 100) %>%
  dplyr::filter(
    (num_components == 5 & noise == 0.5) |
      (num_components == 25 & noise == 0.5) |
      (num_components == 50 & noise == 0.1) |
      (num_components == 75 & noise == 0.1) |
      (num_components == 100 & noise == 0.1) |
      (num_components == 125 & noise == 0.1)
  )

# Reorder dimensionality for plotting
num_com <- adage_good_training_df$num_components
num_com <- factor(num_com, levels = sort(as.numeric(paste(unique(num_com)))))
adage_good_training_df$num_components <- num_com

adage_best_model_png <- file.path(adage_fig, "z_parameter_adage_best.png")
adage_best_model_pdf <- file.path(adage_fig, "z_parameter_adage_best.pdf")

p <- ggplot(adage_good_training_df, aes(x = train_epoch, y = loss)) +
  geom_line(aes(color = num_components, linetype = loss_type), size = 0.2) +
  xlab("Training Epoch") + ylab("Loss") +
  scale_color_OkabeIto(name = "Latent Dimensions") +
  scale_linetype_manual(name = "Loss Type", values = c("solid", "dotted"),
                        labels = c("Train", "Validation")) +
  theme_bw() +
  base_theme

ggsave(adage_best_model_png, plot = p, height = 2.5, width = 4)
ggsave(adage_best_model_pdf, plot = p, height = 2.5, width = 4)
