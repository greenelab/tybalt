# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
#
# Visualizing results of parameter sweep

library(readr)
library(ggplot2)
library(dplyr)

param_sweep_file <- file.path("results", "parameter_sweep_full_results.tsv")
param_df <- readr::read_tsv(param_sweep_file,
                            col_types = cols(train_epoch = col_integer(),
                                             loss = col_double(),
                                             val_loss = col_double(),
                                             learning_rate = col_character(),
                                             batch_size = col_character(),
                                             epochs = col_character(),
                                             kappa = col_character()))

param_melt_df <- reshape2::melt(param_df,
                                id.vars = c("learning_rate", "batch_size",
                                            "epochs", "kappa", "train_epoch"),
                                measure.vars = c("loss", "val_loss"),
                                variable.name = "loss_type",
                                value.name = "loss")

param_melt_df$batch_size <- factor(param_melt_df$batch_size,
                                   levels = c(50, 100, 128, 200))
param_melt_df$epochs <- factor(param_melt_df$epochs,
                               levels = c(10, 25, 50, 100))

# Sweep over measurements of kappa with batch_size and epochs as facets
for (k in unique(param_melt_df$kappa)) {
  subset_param <- param_melt_df %>% dplyr::filter(kappa == k)
  
  p <- ggplot(subset_param, aes(x = train_epoch, y = loss)) +
    geom_line(aes(color = learning_rate, linetype = loss_type), size = 0.5) + 
    facet_grid(batch_size ~ epochs, scales = "free") + theme_bw() + 
    ggtitle(paste("kappa =", k))
  
  output_fig <- file.path("figures", "param_sweep", "full_param_")
  output_fig <- paste0(output_fig, k, "_kappa.png")
  ggsave(output_fig, height = 5, width = 6)
  print(p)
}

# Sweep over epochs with batch_size and kappa as facets
for (e in unique(param_melt_df$epochs)) {
  subset_param <- param_melt_df %>% dplyr::filter(epochs == e)
  
  p <- ggplot(subset_param, aes(x = train_epoch, y = loss)) +
    geom_line(aes(color = learning_rate, linetype = loss_type), size = 0.5) + 
    facet_grid(batch_size ~ kappa, scales = "free") + theme_bw() + 
    ggtitle(paste("epochs =", e))
  
  output_fig <- file.path("figures", "param_sweep", "full_param_")
  output_fig <- paste0(output_fig, e, "_epochs.png")
  ggsave(output_fig, height = 5, width = 6)
  print(p)
}

final_select_df <- param_melt_df %>% filter(loss_type == "val_loss")
final_select_df <- final_select_df %>%
  group_by(learning_rate, batch_size, epochs) %>%
  summarize(min_loss = min(loss))

ggplot(final_select_df, aes(x = learning_rate, y = min_loss)) +
  geom_point(aes(color = batch_size, shape = epochs), size = 2) + theme_bw() +
  ylab("Validation Loss") + xlab("Learning Rate") +
  theme(axis.text = element_text(size = rel(1.2)),
        legend.text = element_text(size = rel(1.1)))

output_sweep <- file.path("figures", "param_sweep", "final_param_val_loss.png")
ggsave(output_sweep, height = 5, width = 6)
