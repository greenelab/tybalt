# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
#
# Visualize t-SNE plots of VAE and raw RNAseq data features

library(dplyr)
library(ggplot2)

# Load encodings file with matched clinical data to subset
encoded_file <- file.path("data", "vae_encoded_with_clinical.tsv")
encoded_df <- readr::read_tsv(encoded_file)
encoded_df <- encoded_df %>% dplyr::select(sample_id, acronym)

# Load tsne data
tsne_vae_file <- file.path("results", "vae_tsne_out.tsv")
tsne_rnaseq_file <- file.path("results", "rnaseq_tsne_out.tsv")

tsne_vae_df <- readr::read_tsv(tsne_vae_file)
tsne_rnaseq_df <- readr::read_tsv(tsne_rnaseq_file)

tsne_vae_df <- dplyr::inner_join(tsne_vae_df, encoded_df,
                                 by = c("tcga_id" = "sample_id"))
tsne_rnaseq_df <- dplyr::inner_join(tsne_rnaseq_df, encoded_df,
                                    by = c("tcga_id" = "sample_id"))

# Load color data
tcga_colors <- readr::read_tsv(file.path("data", "tcga_colors.tsv"))
tcga_colors <- tcga_colors[order(tcga_colors$`Study Abbreviation`), ]

palette_order <- c("BRCA", "PRAD", "TGCT", "KICH", "KIRP", "KIRC", "BLCA",
                   "OV", "UCS", "CESC", "UCEC", "THCA", "PCPG", "ACC", "SKCM",
                   "UVM", "HNSC", "SARC", "ESCA", "STAD", "COAD", "READ",
                   "CHOL", "PAAD", "LIHC", "DLBC", "MESO", "LUSC", "LUAD",
                   "GBM", "LGG", "LAML", "THYM", NA)

# Plot and save VAE tsne
ggplot(tsne_vae_df, aes(x = `1`, y = `2`, color = acronym)) +
  geom_point(size = 0.001) +
  scale_colour_manual(limits = tcga_colors$`Study Abbreviation`,
                      values = tcga_colors$`Hex Colors`,
                      na.value = "black",
                      breaks = palette_order) +
  theme_classic() +
  theme(legend.title = element_text(size = 8),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.5, "line")) +
  guides(colour = guide_legend(override.aes = list(size = 0.1),
                               title = 'Cancer-Type'))

vae_tsne_pdf_out_file <- file.path("figures", "tsne_vae.pdf")
vae_tsne_png_out_file <- file.path("figures", "tsne_vae.png")
ggsave(vae_tsne_pdf_out_file, width = 6, height = 4.5)
ggsave(vae_tsne_png_out_file, width = 4, height = 3)  # PNG for repo

# Plot and save RNAseq tsne
ggplot(tsne_rnaseq_df, aes(x = `1`, y = `2`, color = acronym)) +
  geom_point(size = 0.001) +
  scale_colour_manual(limits = tcga_colors$`Study Abbreviation`,
                      values = tcga_colors$`Hex Colors`,
                      na.value = "black",
                      breaks = palette_order) +
  theme_classic() +
  theme(legend.title = element_text(size = 8),
        legend.text = element_text(size = 5),
        legend.key.size = unit(0.5, "line")) +
  guides(colour = guide_legend(override.aes = list(size = 0.1),
                               title = 'Cancer-Type'))

vae_tsne_out_file <- file.path("figures", "tsne_rnaseq.pdf")
ggsave(vae_tsne_out_file, width = 6, height = 4.5)
