# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
#
# Visualizing results of parameter sweep

library(ggplot2)

`%>%` <- dplyr::`%>%`

param_sweep_file <- file.path("results", "parameter_sweep_full_results.tsv")
param_df <- readr::read_tsv(param_sweep_file,
                            col_types = readr::cols(
                              .default = readr::col_character(),
                              train_epoch = readr::col_integer(),
                              loss = readr::col_double(),
                              val_loss = readr::col_double()))

param_melt_df <- reshape2::melt(param_df,
                                id.vars = c("learning_rate", "batch_size",
                                            "epochs", "kappa", "train_epoch"),
                                measure.vars = c("loss", "val_loss"),
                                variable.name = "loss_type",
                                value.name = "loss")

# Order and recode batch size variables
param_melt_df$batch_size <- factor(param_melt_df$batch_size,
                                   levels = c(50, 100, 128, 200))
param_melt_df$batch_size <- 
  dplyr::recode_factor(param_melt_df$batch_size, 
                       `50` = "batch: 50", 
                       `100` = "batch: 100",
                       `128` = "batch: 128",
                       `200` = "batch: 200") 

# Order and recode epoch variables
param_melt_df$epochs <- factor(param_melt_df$epochs,
                               levels = c(10, 25, 50, 100))
param_melt_df$epochs <- 
  dplyr::recode_factor(param_melt_df$epochs, 
                       `10` = "epochs: 10", 
                       `25` = "epochs: 25",
                       `50` = "epochs: 50",
                       `100` = "epochs: 100") 

# Order and recode kappa variables
param_melt_df$kappa <- factor(param_melt_df$kappa,
                               levels = c("0.01", "0.05", "0.1", "1.0"))
param_melt_df$kappa <- 
  dplyr::recode_factor(param_melt_df$kappa, 
                       `0.01` = "kappa: 0.01", 
                       `0.05` = "kappa: 0.05",
                       `0.1` = "kappa: 0.1",
                       `1.0` = "kappa: 1.0") 

# Sweep over measurements of kappa with batch_size and epochs as facets
for (k in unique(param_melt_df$kappa)) {
  subset_param <- param_melt_df %>% dplyr::filter(kappa == k)
  
  # Remove label in `e` for plotting and filename
  k <- gsub("kappa: ", "", k)
  
  p <- ggplot(subset_param, aes(x = train_epoch, y = loss)) +
    geom_line(aes(color = learning_rate, linetype = loss_type), size = 0.5) + 
    facet_grid(batch_size ~ epochs, scales = "free") + theme_bw() + 
    ggtitle(paste("kappa =", k))
  
  output_fig <- file.path("figures", "param_sweep", "full_param_")
  output_fig <- paste0(output_fig, k, "_kappa.png")
  ggsave(output_fig, plot = p, height = 5, width = 6)
}

# Sweep over epochs with batch_size and kappa as facets
for (e in unique(param_melt_df$epochs)) {
  subset_param <- param_melt_df %>% dplyr::filter(epochs == e)
  
  # Remove label in `k` for plotting and filename
  e <- gsub("epochs: ", "", e)

  p <- ggplot(subset_param, aes(x = train_epoch, y = loss)) +
    geom_line(aes(color = learning_rate, linetype = loss_type), size = 0.5) + 
    facet_grid(batch_size ~ kappa, scales = "free") + theme_bw() + 
    ggtitle(paste("epochs =", e))
  
  output_fig <- file.path("figures", "param_sweep", "full_param_")
  output_fig <- paste0(output_fig, e, "_epochs.png")
  ggsave(output_fig, plot = p, height = 5, width = 6)
}

final_select_df <- param_melt_df %>% dplyr::filter(loss_type == "val_loss")
final_select_df <- final_select_df %>%
  dplyr::group_by(learning_rate, batch_size, epochs) %>%
  dplyr::summarize(end_loss = tail(loss, 1))

p <- ggplot(final_select_df, aes(x = learning_rate, y = end_loss)) +
  geom_point(aes(color = batch_size, shape = epochs), size = 2) + theme_bw() +
  ylab("Validation Loss at Training End") + xlab("Learning Rate") +
  theme(axis.text = element_text(size = rel(1.2)),
        legend.text = element_text(size = rel(1.1)))

output_sweep <- file.path("figures", "param_sweep", "final_param_val_loss.png")
ggsave(output_sweep, plot = p, height = 5, width = 6)
