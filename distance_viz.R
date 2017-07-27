# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
#
# visualizing transformed space distances between cancer-types

library(dplyr)
library(ggplot2)
library(ggjoy)

get_density <- function(distance_df, label, ras='all') {
  # Take a distance matrix, subset data according to `ras` variable and add
  # label for plotting.
  #
  # Output: list of length two
  #         First element is a distance summary in long data format
  #         Second element is a ggplot2 object
  
  if (ras == 'mutated') {
    distance_df <- distance_df %>%
      filter(sample_id %in% ras_df$`#sample`) %>%
      select(sample_id, ras_df$`#sample`)
  
    subset_sample_df <- sample_acronym_df %>%
      filter(sample_id %in% ras_df$`#sample`)
    ras_label <- 'ras/nf1 mutated'
  } 
  
  if (ras == 'wild-type') {
    distance_df <- distance_df %>% filter(!(sample_id %in% ras_df$`#sample`))
    distance_df <- distance_df %>% select(sample_id, distance_df$sample_id)
    subset_sample_df <- sample_acronym_df %>%
      filter(!(sample_id %in% ras_df$`#sample`))
    ras_label <- 'wild-type'
  }
  
  if (ras == 'all') {
    subset_sample_df <- sample_acronym_df
    ras_label <- 'all'
  }
  
  # Only take the upper triangle to avoid duplicate observations
  upper_mat <- reshape2::melt(upper.tri(distance_df))
  dist_long <- reshape2::melt(distance_df)[upper_mat$value, ]
  colnames(dist_long) <- c('sample_id', 'other_sample_id', 'distance')
  dist_long$column_a_acronym <- 
    subset_sample_df$acronym[match(dist_long$sample_id,
                                   subset_sample_df$sample_id)]
  dist_long$column_b_acronym <-
    subset_sample_df$acronym[match(dist_long$other_sample_id,
                                   subset_sample_df$sample_id)]
  
  dist_long <- dist_long[complete.cases(dist_long), ]
  dist_long$comparison <- paste(dist_long$column_b_acronym,
                                dist_long$column_a_acronym,
                                sep = '_')
  dist_long$comparison <- dist_long$comparison  %>%
    dplyr::recode('GBM_COAD' = 'COAD_GBM')
  dist_long$label <- label
  dist_long$ras_label <- ras_label
  p <- ggplot(dist_long, aes(x = distance, fill = comparison)) +
    geom_density(alpha = 0.3)
  return(list(dist_long, p))
  
}

# Get sample ID by acronym dataframe
encoded_file <- file.path("data", "vae_encoded_with_clinical.tsv")
encoded_df <- readr::read_tsv(encoded_file)
sample_acronym_df <- encoded_df %>%
  filter(acronym %in% c("GBM", "COAD")) %>%
  select(sample_id, acronym)

base_folder <- file.path("data", "distance")
raw_dist_file <- file.path(base_folder, "raw_distance.tsv")
raw_dist_sub_file <- file.path(base_folder, "raw_subtract_distance.tsv")
enc_dist_file <- file.path(base_folder, "encoded_distance.tsv")
enc_dist_sub_file <- file.path(base_folder, "encoded_subtraction_distance.tsv")
ras_status_file <- file.path(base_folder, "gbm_coad_rasmutations.tsv")

raw_dist_df <- readr::read_tsv(raw_dist_file)
raw_dist_subtract_df <- readr::read_tsv(raw_dist_sub_file)
encoded_dist_df <- readr::read_tsv(enc_dist_file)
encoded_sub_dist_df <- readr::read_tsv(enc_dist_sub_file)
ras_df <- readr::read_tsv(ras_status_file)

# Observe density plots of distances across GBM and COAD
raw_data_list <- get_density(raw_dist_df, label = 'untransformed')
raw_sub_data_list <- get_density(raw_dist_subtract_df,
                                 label = 'untransformed sub.')
encoded_data_list <- get_density(encoded_dist_df, label = 'vae encoded')
encoded_sub_data_list <- get_density(encoded_sub_dist_df,
                                     label = 'vae subtracted')

ras_raw_data_list <- get_density(raw_dist_df, ras = "mutated",
                                 label = 'untransformed')
ras_raw_sub_data_list <- get_density(raw_dist_subtract_df, ras = "mutated",
                                     label = 'untransformed sub.')
ras_encoded_data_list <- get_density(encoded_dist_df, ras = "mutated",
                                     label = 'vae encoded')
ras_encoded_sub_data_list <- get_density(encoded_sub_dist_df, ras = "mutated",
                                         label = 'vae subtracted')

wt_raw_data_list <- get_density(raw_dist_df, ras = "wild-type",
                                label = 'untransformed')
wt_raw_sub_data_list <- get_density(raw_dist_subtract_df, ras = "wild-type",
                                    label = 'untransformed sub.')
wt_encoded_data_list <- get_density(encoded_dist_df, ras = "wild-type",
                                    label = 'vae encoded')
wt_encoded_sub_data_list <- get_density(encoded_sub_dist_df, ras = "wild-type",
                                        label = 'vae subtracted')

full_df <- dplyr::bind_rows(raw_data_list[[1]], raw_sub_data_list[[1]],
                            encoded_data_list[[1]], encoded_sub_data_list[[1]],
                            ras_raw_data_list[[1]], ras_raw_sub_data_list[[1]],
                            ras_encoded_data_list[[1]],
                            ras_encoded_sub_data_list[[1]],
                            wt_raw_data_list[[1]], wt_raw_sub_data_list[[1]],
                            wt_encoded_data_list[[1]],
                            wt_encoded_sub_data_list[[1]])

full_df$ras_label <- factor(full_df$ras_label, levels = c('all', 'wild-type',
                                                          'ras/nf1 mutated'))

distance_plot_file <- file.path("figures", "distance_analysis.png")
ggplot(full_df, aes(x = distance, y = comparison, fill = comparison)) +
  geom_joy(scale = 2, alpha = 0.6) + facet_grid(ras_label ~ label) +
  theme_bw() + theme(legend.position = "none")
ggsave(distance_plot_file, height = 4, width = 6)
 



