# Pan-Cancer Variational Autoencoder
# Gregory Way 2018
# scripts/viz/plot_simulation.R
#
# Visualize the results of the simulation analyses
#
# Note that all simulation results were compiled by running:
#
#     python scripts/util/aggregate_simulation_results.py
#
# With the following flags set:
#
#     --simulation_directory 'results/simulation_tiedweights'
#     --output_filename 'results/all_simulation_results_tiedweights.tsv'
#
# Usage: run once output all simulated results
#
#         Rscript scripts/viz/plot_simulation.R

library(dplyr)
library(reshape2)
library(ggplot2)
library(ggbeeswarm)

# Base location where all figures are to be saved
fig_base <- file.path("figures", "simulation")

sim_file <- file.path("results", "all_simulation_results_tiedweights.tsv")
sim_df <- readr::read_tsv(sim_file)

sim_df$sample_size <-
  dplyr::recode_factor(sim_df$sample_size, 
                       "500" = "Samples: 500",
                       "1000" = "Samples: 1000",
                       "2000" = "Samples: 2000",
                       "4000" = "Samples: 4000",
                       "8000" = "Samples: 8000")

sim_df$num_genes <-
  dplyr::recode_factor(sim_df$num_genes, 
                       "500" = "Genes: 500",
                       "1000" = "Genes: 1000")

# 1) Feature ranks of module genes aggregating into single feature
module_rank_png <- file.path(fig_base, "mean_module_rank.png")
module_rank_pdf <- file.path(fig_base, "mean_module_rank.pdf")

p <- ggplot(sim_df, aes(x = as.factor(noise), y = minimum_rank_avg)) +
  geom_boxplot(aes(color = algorithm), outlier.size = 0.1, lwd = 0.2) +
  facet_grid(num_genes ~ sample_size, scales = "free") +
  xlab("Noise Added") +
  ylab("Mean Rank of Modules") +
  theme_bw() + 
  theme(axis.text = element_text(size = rel(0.5)),
        axis.title = element_text(size = rel(0.5)),
        strip.text = element_text(size = rel(0.6)))

ggsave(module_rank_png, plot = p, height = 3, width = 6)
ggsave(module_rank_pdf, plot = p, height = 3, width = 6)

# 2) Plot Latent Space Arithmetic Reconstruction Distance
reconstruction_png <- file.path(fig_base, "lsa_reconstruction_distance.png")
reconstruction_pdf <- file.path(fig_base, "lsa_reconstruction_distance.pdf")

p <- ggplot(sim_df, aes(x = as.factor(noise), y = avg_recon_dist)) +
  geom_boxplot(aes(color = algorithm), outlier.size = 0.1, lwd = 0.2) +
  facet_grid(num_genes ~ sample_size, scales = "free") +
  xlab("Noise Added") +
  ylab("Distance Between B_hat and B") +
  theme_bw() + 
  theme(axis.text = element_text(size = rel(0.5)),
        axis.title = element_text(size = rel(0.5)),
        strip.text = element_text(size = rel(0.6)))

ggsave(reconstruction_png, plot = p, height = 3, width = 6)
ggsave(reconstruction_pdf, plot = p, height = 3, width = 6)

# 3) Node essence of module 3 - Does LSA capture module 3?
module_z_png <- file.path(fig_base, "lsa_module_3_capture.png")
module_z_pdf <- file.path(fig_base, "lsa_module_3_capture.pdf")

p <- ggplot(sim_df, aes(x = as.factor(noise), y = min_node_zscore)) +
  geom_boxplot(aes(color = algorithm), outlier.size = 0.1, lwd = 0.2) +
  facet_grid(num_genes ~ sample_size, scales = "free") +
  xlab("Noise Added") +
  ylab("Module 3 Feature Essence Z-Score") +
  theme_bw() + 
  theme(axis.text = element_text(size = rel(0.5)),
        axis.title = element_text(size = rel(0.5)),
        strip.text = element_text(size = rel(0.6)))

ggsave(module_z_png, plot = p, height = 3, width = 6)
ggsave(module_z_pdf, plot = p, height = 3, width = 6)

# These are kind of tough to see, plot a representative example:
# sample_size = 8000, genes = 500
z_essence_df <- sim_df %>% dplyr::filter(sample_size == "Samples: 8000",
                                         num_genes == "Genes: 500")

# Recode variables for plotting
z_essence_df$noise <-
  dplyr::recode_factor(z_essence_df$noise, 
                       "0" = "Noise: 0",
                       "0.1" = "Noise: 0.1",
                       "0.5" = "Noise: 0.5",
                       "1" = "Noise: 1",
                       "3" = "Noise: 3")

module_z_eg_png <- file.path(fig_base, "lsa_module_3_capture_n8000_p500.png")
module_z_eg_pdf <- file.path(fig_base, "lsa_module_3_capture_n8000_p500.pdf")

p <- ggplot(z_essence_df,
       aes(x = algorithm, y = min_node_zscore, color = algorithm)) +
  geom_beeswarm(dodge.width = 1, size = 0.1) +
  facet_grid(~ as.factor(noise)) +
  xlab("Algorithm") +
  ylab("Module 3 Feature Essence Z-Score") +
  theme_bw() + 
  theme(axis.text = element_text(size = rel(0.6)),
        axis.text.x = element_text(angle = 45),
        axis.title = element_text(size = rel(0.6)),
        strip.text = element_text(size = rel(0.6)),
        legend.position = "none")

ggsave(module_z_eg_png, plot = p, height = 2.5, width = 6)
ggsave(module_z_eg_pdf, plot = p, height = 2.5, width = 6)

# 4) Each module includes a different number of genes. Module 0 is the noise
# module, but modules 1-5 include a decreasing proportion of genes.

module_rank_df <- sim_df %>%
  dplyr::select(c(algorithm, `0`, `1`, `2`, `3`, `4`, `5`))

module_rank_df <-
  reshape2::melt(sim_df,
                 id.vars = c("algorithm", "noise", "sample_size", "num_genes"),
                 measure.vars = c("0", "1", "2", "3", "4", "5"),
                 value.name = "avg_rank")

# Loop over noise parameters and sample sizes to plot individual figures
for (sep_noise in unique(module_rank_df$noise)) {
  for (n in unique(module_rank_df$sample_size)) {
    module_rank_sub_df <- module_rank_df %>% dplyr::filter(noise == sep_noise,
                                                           sample_size == n)

    samp_size <- unlist(strsplit(n, ": "))[2]
    fig <-  paste0("mod_rank_noise_", sep_noise, "_n_", samp_size, ".pdf")
    fig_png <-  paste0("mod_rank_noise_", sep_noise, "_n_", samp_size, ".png")
    module_rank_pdf <- file.path(fig_base, "module_rank", fig)
    module_rank_png <- file.path(fig_base, "module_rank", fig_png)

    p <- ggplot(module_rank_sub_df, aes(x = as.numeric(paste(variable)),
                                        y = avg_rank,
                                        color = algorithm)) +
      geom_beeswarm(size = 0.1, groupOnX = TRUE) + 
      facet_grid(num_genes ~ algorithm) +
      theme_bw() +
      ggtitle(paste0("Module Detection with Noise: ", sep_noise, ", ", n)) +
      xlab("Gene Modules") +
      ylab("Min Avg Rank of Module Genes in Feature") +
      scale_x_continuous(breaks = c(0, 1, 2, 3, 4 , 5)) +
      theme(axis.text = element_text(size = rel(0.6)),
            axis.title = element_text(size = rel(0.7)),
            strip.text = element_text(size = rel(0.6)),
            legend.position = "none")

    ggsave(module_rank_pdf, plot = p, height = 3, width = 5)
    ggsave(module_rank_png, plot = p, height = 3, width = 5)
  }
}
