# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
#
# Combine VAE features with clinical data: for use in downstream analyses

library(readr)
library(dplyr)

vae_file <- file.path("data", "encoded_rnaseq_onehidden_warmup_batchnorm.tsv")
clinical_file <- file.path("data", "clinical_data.tsv")
vae_df <- readr::read_tsv(vae_file)
colnames(vae_df)[1] <- "sample_id"
clinical_df <- readr::read_tsv(clinical_file)

vae_df <- vae_df %>%
  dplyr::rowwise() %>%
  dplyr::mutate(sample_base = substring(sample_id, 1, 12))

combined_df <- dplyr::inner_join(vae_df, clinical_df,
                                 by = c("sample_base" = "sample_id"))
combined_df <- combined_df[!duplicated(combined_df$sample_id), ]

# Process improper drug names that cause parsing failures
combined_df$drug <- tolower(combined_df$drug )
combined_df$drug <- gsub("\t", "", combined_df$drug)
combined_df$drug <- gsub('"', "", combined_df$drug)
combined_df$drug <- gsub("\\\\", "", combined_df$drug)

comb_out_file <- file.path("data", "tybalt_features_with_clinical.tsv")
write.table(combined_df, file = comb_out_file, sep = "\t", row.names = FALSE)
