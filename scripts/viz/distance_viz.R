# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
#
# Visualizing transformed space distances between cancer-types

library(dplyr)
library(ggplot2)
library(ggjoy)

get_density <- function(distance_df, label, gene = "all") {
  # Take a distance matrix, subset data according to `gene` variable and add
  # label for plotting.
  #
  # Output: list of length two
  #         First element is a distance summary in long data format
  #         Second element is a ggplot2 object

  if (gene == "mutated") {
    distance_df <- distance_df %>%
      filter(sample_id %in% tp53_df$`#sample`) %>%
      select(sample_id, tp53_df$`#sample`)
    subset_sample_df <- sample_acronym_df %>%
      filter(sample_id %in% tp53_df$`#sample`)

    gene_label <- "TP53 mutated"
  } 

  if (gene == "wild-type") {
    distance_df <- distance_df %>% filter(!(sample_id %in% tp53_df$`#sample`))
    distance_df <- distance_df %>% select(sample_id, distance_df$sample_id)
    subset_sample_df <- sample_acronym_df %>%
      filter(!(sample_id %in% tp53_df$`#sample`))

    gene_label <- "wild-type"
  }

  if (gene %in% c("all", "wt_mut")) {
    subset_sample_df <- sample_acronym_df

    gene_label <- "all"
  }

  # Only take the upper triangle to avoid duplicate observations
  upper_mat <- reshape2::melt(upper.tri(distance_df))
  dist_long <- reshape2::melt(distance_df)[upper_mat$value, ]
  colnames(dist_long) <- c("sample_id", "other_sample_id", "distance")

  # Get tissue type comparison
  dist_long$column_a_acronym <- 
    subset_sample_df$acronym[match(dist_long$sample_id,
                                   subset_sample_df$sample_id)]
  dist_long$column_b_acronym <-
    subset_sample_df$acronym[match(dist_long$other_sample_id,
                                   subset_sample_df$sample_id)]
  
  dist_long <- dist_long[complete.cases(dist_long), ]
  dist_long$comparison <- paste(dist_long$column_b_acronym,
                                dist_long$column_a_acronym,
                                sep = "_")
  dist_long$comparison <- dist_long$comparison  %>%
    dplyr::recode("LUAD_BLCA" = "BLCA_LUAD")
  
  # Get TP53 mutation comparison
  dist_long$column_a_status <-
    subset_sample_df$mutate[match(dist_long$sample_id,
                                  subset_sample_df$sample_id)]
  dist_long$column_b_status <-
    subset_sample_df$mutate[match(dist_long$other_sample_id,
                                  subset_sample_df$sample_id)]
  dist_long$mutation <- paste(dist_long$column_a_status,
                              dist_long$column_b_status,
                              sep = "_")
  dist_long$mutation <- dist_long$mutation  %>%
    dplyr::recode("MUT_WT" = "WT_MUT")
  
  if (gene == "wt_mut") {
    dist_long <- dist_long %>% dplyr::filter(mutation == "WT_MUT")
    gene_label = "wild-type vs. mut"
  }
  
  dist_long$label <- label
  dist_long$gene_label <- gene_label
  
  p <- ggplot(dist_long, aes(x = distance, fill = comparison)) +
    geom_density(alpha = 0.3)
  return(list(dist_long, p))
  
}

# Get sample ID by acronym dataframe
encoded_file <- file.path("data", "vae_encoded_with_clinical.tsv")
encoded_df <- readr::read_tsv(encoded_file)

# Process encoded data to obtain long acronym and mutation status data
sample_acronym_df <- encoded_df %>%
  filter(acronym %in% c("BLCA", "LUAD")) %>%
  select(sample_id, acronym)

base_folder <- file.path("data", "distance")
raw_dist_file <- file.path(base_folder, "raw_distance.tsv")
raw_dist_sub_file <- file.path(base_folder, "raw_subtract_distance.tsv")
enc_dist_file <- file.path(base_folder, "encoded_distance.tsv")
enc_dist_sub_file <- file.path(base_folder, "encoded_subtraction_distance.tsv")
tp53_status_file <- file.path(base_folder, "blca_luad_tp53_mutations.tsv")

raw_dist_df <- readr::read_tsv(raw_dist_file)
raw_dist_subtract_df <- readr::read_tsv(raw_dist_sub_file)
encoded_dist_df <- readr::read_tsv(enc_dist_file)
encoded_sub_dist_df <- readr::read_tsv(enc_dist_sub_file)
tp53_df <- readr::read_tsv(tp53_status_file)

sample_acronym_df$mutate <- "WT"
sample_acronym_df$mutate[match(tp53_df$`#sample`,
                               sample_acronym_df$sample_id)] <- "MUT"

####################################
# Observe density plots of distances across BLCA and LUAD
####################################
# 1) All comparison
raw_data_list <- get_density(raw_dist_df, label = "untransformed")
raw_sub_data_list <- get_density(raw_dist_subtract_df,
                                 label = "untransformed sub.")
encoded_data_list <- get_density(encoded_dist_df, label = "vae encoded")
encoded_sub_data_list <- get_density(encoded_sub_dist_df,
                                     label = "vae subtracted")

# 2) All TP53 mutations
p53_raw_data_list <- get_density(raw_dist_df, gene = "mutated",
                                 label = "untransformed")
p53_raw_sub_data_list <- get_density(raw_dist_subtract_df, gene = "mutated",
                                     label = "untransformed sub.")
p53_encoded_data_list <- get_density(encoded_dist_df, gene = "mutated",
                                     label = "vae encoded")
p53_encoded_sub_data_list <- get_density(encoded_sub_dist_df, gene = "mutated",
                                         label = "vae subtracted")

# 3) All wild-type samples
wt_raw_data_list <- get_density(raw_dist_df, gene = "wild-type",
                                label = "untransformed")
wt_raw_sub_data_list <- get_density(raw_dist_subtract_df, gene = "wild-type",
                                    label = "untransformed sub.")
wt_encoded_data_list <- get_density(encoded_dist_df, gene = "wild-type",
                                    label = "vae encoded")
wt_encoded_sub_data_list <- get_density(encoded_sub_dist_df, gene = "wild-type",
                                        label = "vae subtracted")

# 4) Mutated vs. wild-type samples
wt_mut_raw_data_list <- get_density(raw_dist_df, gene = "wt_mut",
                                    label = "untransformed")
wt_mut_raw_sub_data_list <- get_density(raw_dist_subtract_df, gene = "wt_mut",
                                        label = "untransformed sub.")
wt_mut_enc_data_list <- get_density(encoded_dist_df, gene = "wt_mut",
                                    label = "vae encoded")
wt_mut_enc_sub_data_list <- get_density(encoded_sub_dist_df, gene = "wt_mut",
                                        label = "vae subtracted")

full_df <- dplyr::bind_rows(raw_data_list[[1]],
                            raw_sub_data_list[[1]],
                            encoded_data_list[[1]],
                            encoded_sub_data_list[[1]],
                            p53_raw_data_list[[1]],
                            p53_raw_sub_data_list[[1]],
                            p53_encoded_data_list[[1]],
                            p53_encoded_sub_data_list[[1]],
                            wt_raw_data_list[[1]],
                            wt_raw_sub_data_list[[1]],
                            wt_encoded_data_list[[1]],
                            wt_encoded_sub_data_list[[1]],
                            wt_mut_raw_data_list[[1]],
                            wt_mut_raw_sub_data_list[[1]],
                            wt_mut_enc_data_list[[1]],
                            wt_mut_enc_sub_data_list[[1]])

full_df$gene_label <- factor(full_df$gene_label,
                             levels = c("all", "wild-type", "TP53 mutated",
                                        "wild-type vs. mut"))

distance_plot_file <- file.path("figures", "tp53_distance_analysis.png")
ggplot(full_df, aes(x = distance, y = comparison, fill = comparison)) +
  geom_joy(scale = 2, alpha = 0.6) + facet_grid(gene_label ~ label) +
  theme_bw() +
  theme(legend.position = "none")
ggsave(distance_plot_file, height = 5, width = 6)
 
blca_luad_vaesub_mut <- p53_encoded_sub_data_list[[1]] %>%
  dplyr::filter(comparison == "BLCA_LUAD") %>% dplyr::select(distance)
blca_luad_vaesub_wtmut <- wt_mut_enc_sub_data_list[[1]] %>%
  dplyr::filter(comparison == "BLCA_LUAD") %>% dplyr::select(distance)

blca_luad_rawsub_mut <- p53_raw_sub_data_list[[1]] %>%
  dplyr::filter(comparison == "BLCA_LUAD") %>% dplyr::select(distance)
blca_luad_rawsub_wtmut <- wt_mut_raw_sub_data_list[[1]] %>%
  dplyr::filter(comparison == "BLCA_LUAD") %>% dplyr::select(distance)
  

t.test(blca_luad_vaesub_wtmut, blca_luad_vaesub_mut)
t.test(blca_luad_rawsub_wtmut, blca_luad_rawsub_mut)

