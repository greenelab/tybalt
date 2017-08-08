# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
#
# Visualize VAE feature activation
#
# 1) Scatter plots of specific covariate information by encodings
# 2) Heatmap of full encoded features by all samples

library(readr)
library(dplyr)
library(heatmap.plus)
library(ggplot2)

# 1) Scatter plots of encoding specific sample activation
# Load the combined data frame where columns are different data and algorithms
vae_clinical_file <- file.path("data", "vae_encoded_with_clinical.tsv")
combined_df <- readr::read_tsv(vae_clinical_file)

# Load official colors
tcga_colors <- readr::read_tsv(file.path("data", "tcga_colors.tsv"))

tcga_colors <- tcga_colors[order(tcga_colors$`Study Abbreviation`), ]
match_colors <- match(combined_df$acronym, tcga_colors$`Study Abbreviation`)
combined_df$colors <- tcga_colors$`Hex Colors`[match_colors]

palette_order <- c("BRCA", "PRAD", "TGCT", "KICH", "KIRP", "KIRC", "BLCA",
                   "OV", "UCS", "CESC", "UCEC", "THCA", "PCPG", "ACC", "SKCM",
                   "UVM", "HNSC", "SARC", "ESCA", "STAD", "COAD", "READ",
                   "CHOL", "PAAD", "LIHC", "DLBC", "MESO", "LUSC", "LUAD",
                   "GBM", "LGG", "LAML", "THYM", NA)

plot_vae <- function(x, y, covariate, legend_show=TRUE) {
  # Helper function to plot given coordinates and color by specific covariates
  x_coord <- paste(x)
  y_coord <- paste(y)
  color_ <- covariate

  p <- ggplot(combined_df,
              aes_string(x = combined_df[[x_coord]],
                         y = combined_df[[y_coord]],
                         color = color_)) + 
    geom_point() +
    xlab(paste("encoding", x_coord)) +
    ylab(paste("encoding", y_coord)) +
    theme_bw() +
    theme(text = element_text(size = 20))
  
  if (color_ == 'acronym') {
    p <- p + scale_colour_manual(limits = tcga_colors$`Study Abbreviation`,
                                 values = tcga_colors$`Hex Colors`,
                                 na.value = 'black',
                                 breaks = palette_order)
  }

  if (!legend_show) {
    p <- p + theme(legend.position = 'none')
  }
  print(p)
}

gender_encodings <- file.path("figures", "gender_encodings.pdf")
gender_encodings_legend <- file.path("figures", "gender_encodings_legend.pdf")
sample_type_file <- file.path("figures", "sample_type_encodings.pdf")
sample_legend_file <- file.path("figures", "sample_type_encodings_legend.pdf")

plot_vae(x = 82, y = 85, covariate = "gender", legend_show = FALSE)
ggsave(gender_encodings, width = 6, height = 5)

plot_vae(x = 82, y = 85, covariate = "gender", legend_show = TRUE)
ggsave(gender_encodings_legend, width = 6, height = 5)

# Change the legend to also capture melanoma vs. non-melanoma
met_df <- combined_df %>%
  dplyr::select("53", "66", "sample_type", "acronym") %>%
  dplyr::mutate(label = paste0(sample_type, acronym))

met_df$label[(met_df$acronym == 'SKCM') &
               (met_df$sample_type != 'Metastatic')] <- 'Non-metastatic SKCM'
met_df$label[(met_df$acronym != 'SKCM') &
               (met_df$sample_type != 'Metastatic')] <- 'Non-metastatic Other'
met_df$label[(met_df$acronym != 'SKCM') &
               (met_df$sample_type == 'Metastatic')] <- 'Metastatic Other'
met_df$label[(met_df$label == 'MetastaticSKCM')] <- 'Metastatic SKCM'

p <- ggplot(met_df, aes(x = met_df$`53`, y = met_df$`66`, color = label)) + 
  geom_point() +
  xlab(paste("encoding 53")) +
  ylab(paste("encoding 66")) +
  theme_bw() +
  theme(text = element_text(size = 20))
p + theme(legend.position = 'none')
ggsave(sample_type_file, width = 6, height = 5)

p
ggsave(sample_legend_file, width = 6, height = 5)

# 2) Full heatmap
mat <- as.matrix(combined_df[, 2:101])
sample_type_vector <- combined_df$sample_type
sample_type_vector <- sample_type_vector %>%
  dplyr::recode("Primary Tumor" = "green",
                "Additional - New Primary" = "green",
                "Recurrent Tumor" = "green",
                "Metastatic" = "red",
                "Additional Metastatic" = "red",
                "Solid Tissue Normal" = "blue",
                "Primary Blood Derived Cancer - Peripheral Blood" = "purple")
sex_vector <- combined_df$gender
sex_vector <- sex_vector %>%
  dplyr::recode("female" = "orange",
                "male" = "black")
row_color_matrix <- as.matrix(cbind(sample_type_vector, sex_vector))
colnames(row_color_matrix) <- c('Sample', 'Sex')

heatmap_file <- file.path("figures", "encoding_heatmap.pdf")
pdf(heatmap_file, width = 8, height = 9)
heatmap.plus(mat, RowSideColors = row_color_matrix, scale = "row",
             labRow = FALSE, labCol = FALSE,
             ylab = "Samples", xlab = "VAE Encodings")
legend(x = -0.08, y = 1.08, xpd = TRUE,
       legend = c("", "Tumor", "Metastasis", "Normal", "Blood Tumor"),
       fill = c("white", "green", "red", "blue", "purple"), border = FALSE,
       bty = "n", cex = 0.7)
legend(x = 0.05, y = 1.08, xpd = TRUE,
       legend = c("", "Male", "Female"),
       fill = c("white", "black", "orange"), border = FALSE,
       bty = "n", cex = 0.7)
dev.off()
