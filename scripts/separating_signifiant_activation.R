# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
#
# Visualize the distribution of activation per sample (representative nodes)

library(ggplot2)
library(ggjoy)

encoding_file <- file.path("data",
                           "encoded_rnaseq_onehidden_warmup_batchnorm.tsv")
encoding_df <- readr::read_tsv(encoding_file)
encoding_melt <- reshape2::melt(encoding_df)
colnames(encoding_melt) <- c("sample", "node", "activation")

ggplot(encoding_melt[encoding_melt$node %in% 1:10, ],
       aes(x = activation, y = node)) +
  geom_joy(scale = 3) +
  theme_joy(font_size = 5, line_size = 0.5, grid = TRUE)

node_dist_fig <- file.path("figures", "node_activation_distribution.png")
ggsave(node_dist_fig, height = 3, width = 2)
