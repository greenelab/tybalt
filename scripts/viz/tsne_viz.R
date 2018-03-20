# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
#
# Visualize t-SNE plots of VAE, ADAGE, and raw RNAseq data features

library(dplyr)
library(ggplot2)
library(readr)

plot_tsne <- function(tsne_df, color_df) {
  # Function to output tsne visualizations colored by cancer-type for the given
  # input data (compressed by either Tybalt or ADAGE or raw RNAseq data)
  p <- ggplot(tsne_df, aes(x = `1`, y = `2`, color = acronym)) +
    geom_point(size = 0.001) +
    scale_colour_manual(limits = color_df$`Study Abbreviation`,
                        values = color_df$`Hex Colors`,
                        na.value = "black",
                        breaks = palette_order) +
    theme_classic() +
    theme(legend.title = element_text(size = 8),
          legend.text = element_text(size = 5),
          legend.key.size = unit(0.5, "line")) +
    guides(colour = guide_legend(override.aes = list(size = 0.1),
                                 title = 'Cancer-Type'))
  return(p)
}

# Load encodings file with matched clinical data to subset
clinical_file <- file.path("data", "tybalt_features_with_clinical.tsv")
clinical_df <- readr::read_tsv(clinical_file)
clinical_df <- clinical_df %>% dplyr::select(sample_id, acronym)

# Load tsne data
tsne_vae_file <- file.path("results", "tybalt_tsne_features.tsv")
tsne_adage_file <- file.path("results", "adage_tsne_features.tsv")
tsne_rnaseq_file <- file.path("results", "rnaseq_tsne_features.tsv")

tsne_vae_df <- readr::read_tsv(tsne_vae_file)
tsne_adage_df <- readr::read_tsv(tsne_adage_file)
tsne_rnaseq_df <- readr::read_tsv(tsne_rnaseq_file)

tsne_vae_df <- dplyr::inner_join(tsne_vae_df, clinical_df,
                                 by = c("tcga_id" = "sample_id"))
tsne_adage_df <- dplyr::inner_join(tsne_adage_df, clinical_df,
                                 by = c("tcga_id" = "sample_id"))
tsne_rnaseq_df <- dplyr::inner_join(tsne_rnaseq_df, clinical_df,
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
vae_tsne_pdf_out_file <- file.path("figures", "tsne_vae.pdf")
vae_tsne_png_out_file <- file.path("figures", "tsne_vae.png")
p <- plot_tsne(tsne_vae_df, tcga_colors)
ggsave(vae_tsne_pdf_out_file, plot = p, width = 6, height = 4.5)
ggsave(vae_tsne_png_out_file, plot = p, width = 4, height = 3)  # PNG for repo

# Plot and save ADAGE tsne
adage_tsne_pdf_out_file <- file.path("figures", "tsne_adage.pdf")
p <- plot_tsne(tsne_adage_df, tcga_colors)
ggsave(adage_tsne_pdf_out_file, plot = p, width = 6, height = 4.5)

# Plot and save RNAseq tsne
rnaseq_tsne_out_file <- file.path("figures", "tsne_rnaseq.pdf")
p <- plot_tsne(tsne_rnaseq_df, tcga_colors)
ggsave(rnaseq_tsne_out_file, plot = p, width = 6, height = 4.5)
