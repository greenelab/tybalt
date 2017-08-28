# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
#
# Visualizing results of ADAGE model parameter sweep

library(ggplot2)

`%>%` <- dplyr::`%>%`

param_file <- file.path("results", "parameter_sweep_adage_results.tsv")
param_df <- readr::read_tsv(param_file,
                            col_types = readr::cols(
                              .default = readr::col_character(),
                              train_epoch = readr::col_integer(),
                              loss = readr::col_double(),
                              val_loss = readr::col_double()))

param_melt_df <- reshape2::melt(param_df,
                                id.vars = c("learning_rate", "batch_size",
                                            "epochs", "sparsity", "noise",
                                            "train_epoch", "zero_nodes"),
                                measure.vars = c("loss", "val_loss"),
                                variable.name = "loss_type",
                                value.name = "loss")

# Order and recode batch size variables
param_melt_df$batch_size <- factor(param_melt_df$batch_size,
                                   levels = c(50, 100, 200))
param_melt_df$batch_size <- 
  dplyr::recode_factor(param_melt_df$batch_size, 
                       `50` = "batch: 50", 
                       `100` = "batch: 100",
                       `200` = "batch: 200") 

# Order and recode epoch variables
param_melt_df$epochs <- factor(param_melt_df$epochs,
                               levels = c(40, 80, 100))
param_melt_df$epochs <- 
  dplyr::recode_factor(param_melt_df$epochs, 
                       `40` = "epochs: 40", 
                       `80` = "epochs: 80",
                       `100` = "epochs: 100")

# Order and recode learning rate variables
param_melt_df$learning_rate <- factor(param_melt_df$learning_rate,
                                      levels = c("0.9", "1.0", "1.1"))
param_melt_df$learning_rate <- 
  dplyr::recode_factor(param_melt_df$learning_rate, 
                       `0.9` = "lr: 0.9", 
                       `1.0` = "lr: 1.0",
                       `1.1` = "lr: 1.1")


# Order and recode kappa variables
param_melt_df$sparsity <- factor(param_melt_df$sparsity,
                                 levels = c("0.0", "0.001", "1e-06"))
param_melt_df$sparsity <-
  dplyr::recode_factor(param_melt_df$sparsity,
                       `0.0` = "sparsity: 0",
                       `1e-06` = "sparsity: 1e-6",
                       `0.001` = "sparsity: 0.001")

# Order and recode kappa variables
param_melt_df$noise <- factor(param_melt_df$noise,
                                 levels = c("0.0", "0.05", "0.1", "0.5"))
param_melt_df$noise <-
  dplyr::recode_factor(param_melt_df$noise,
                       `0.0` = "noise: 0",
                       `0.05` = "noise: 0.05",
                       `0.1` = "noise: 0.1",
                       `0.5` = "noise: 0.5")

# Sweep over measurements of kappa with batch_size and epochs as facets
for (n in unique(param_melt_df$noise)) {
  subset_param <- param_melt_df %>% dplyr::filter(noise == n)
  
  for (s in unique(param_melt_df$sparsity)) {
    
    subset_subset_param <- subset_param %>% dplyr::filter(sparsity == s)
    # Remove label in `n` for plotting and filename
    n <- gsub("noise: ", "", n)
    s <- gsub("sparsity: ", "", s)
    
    p <- ggplot(subset_subset_param, aes(x = train_epoch, y = loss)) +
      geom_line(aes(color = learning_rate, linetype = loss_type), size = 0.5) + 
      facet_grid(batch_size ~ epochs, scales = "free") + theme_bw() + 
      ggtitle(paste("noise =", n, ", sparsity =", s))
    
    output_fig <- file.path("figures", "param_sweep", "adage", "full_param_")
    output_fig <- paste0(output_fig, n, "_noise_", s, "_sparsity.png")
    ggsave(output_fig, plot = p, height = 5, width = 6)
  }
}

# Plot the final loss at training end
final_df <- param_melt_df %>% dplyr::filter(loss_type == "val_loss")
final_df <- final_df %>%
  dplyr::group_by(learning_rate, batch_size, epochs, noise, sparsity,
                  zero_nodes) %>%
  dplyr::summarize(end_loss = tail(loss, 1))
final_df <- final_df[order(as.numeric(final_df$zero_nodes)), ]

p <- ggplot(final_df, aes(x = learning_rate, y = end_loss)) +
  geom_point(aes(color = sparsity, shape = epochs), size = 1) +
  facet_grid(batch_size ~ noise, scales = "free") +
  theme_bw() +
  ylab("Validation Loss at Training End") + xlab("") +
  theme(axis.text = element_text(size = rel(1.2)),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.text = element_text(size = rel(1.1)))

output_sweep <- file.path("figures", "param_sweep", "adage",
                          "final_param_val_loss.png")
ggsave(output_sweep, plot = p, height = 7, width = 8)

# Incorporate the number of non-zero nodes into selection
p <- ggplot(final_df, aes(x = end_loss, y = as.numeric(zero_nodes))) +
  facet_grid(batch_size~sparsity) + 
  xlab("Validation Loss") +
  ylab("Number of Zero Nodes") + 
  geom_point(aes(color = learning_rate, shape = epochs, size = noise)) +
  theme_bw()

output_sweep <- file.path("figures", "param_sweep", "adage",
                          "final_param_val_loss_zero_nodes.png")
ggsave(output_sweep, plot = p, height = 7, width = 8)

# What about number of non-zero nodes for all models with < 0.025 val loss?
# Replot this condensed figure
good_models <- final_df %>% dplyr::filter(end_loss < 0.025)
good_models <- good_models[order(good_models$zero_nodes), ]

p <- ggplot(good_models, aes(x = end_loss, y = as.numeric(zero_nodes))) +
  facet_grid(batch_size~sparsity) + 
  xlab("Validation Loss") +
  ylab("Number of Zero Nodes") + 
  geom_point(aes(color = learning_rate, shape = epochs, size = noise)) +
  theme_bw()

output_sweep <- file.path("figures", "param_sweep", "adage",
                          "final_param_goodmodels_loss_zero_nodes.png")
ggsave(output_sweep, plot = p, height = 7, width = 8)
