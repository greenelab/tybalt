# Pan-Cancer Variational Autoencoder
# Gregory Way 2018
# scripts/util/simulate_expression.R
#
# Functions to facilitate gene expression data simulation
#
# Usage: run once to simulate 25 datasets with different noise and sample size
#   Rscript scripts/util/simulate_expression.R

library(WGCNA)
library(ggcorrplot)
library(gplots)

allowWGCNAThreads()

set.seed(1234)

simulateExpression <- function(n, num_sets, num_genes, num_modules,
                               background_noise, min_cor, max_cor,
                               sample_set_A, sample_set_B, mod_prop,
                               leave_out_A, leave_out_B, return_eigen=FALSE) {
  # Output a simulated gene expression matrix using WGCNA
  #
  # Arguments:
  # n - The number of samples to simulate
  # num_sets - The number of simulated "groups" of samples
  # num_genes - The number of genes to simulate
  # num_modules - The number of distinct simulated "groups" of genes
  # background_noise - N(0, sd = background_noise) added to all values
  # min_cor - The minimum correlation of genes in modules to core eigengene
  # max_cor - The maximum correlation of genes in modules to core eigengene
  # sample_set_A - character vector of sample group A centroids
  # sample_set_B - character vector of sample group B centroids
  # mod_prop - character vector (must sum to 1) of gene module proportions
  # leave_out_A - character vector of booleans indicating gene module presence
  # leave_out_B - adds option to leave out modules for different group
  #
  # Output:
  # An n x num_genes simulated gene expression matrix with predefined groups
  # of samples and gene modules

  # Sample eigen gene matrix
  eigen_gene_samples <- rbind(
    matrix(rnorm(num_modules * n / 2, mean = sample_set_A, sd = 1),
           n / 2, num_modules, byrow = TRUE),
    matrix(rnorm(num_modules * n / 2, mean = sample_set_B, sd = 1),
           n / 2, num_modules, byrow = TRUE)
  )
  
  # Simulate Expression
  # the two matrices differ based on the leave out argument. This is to enable
  # an assessment of latent space arithmetic to test if an algorithm can isolate
  # the "essence" of the left out gene module through subtraction.
  x_1 <- WGCNA::simulateDatExpr(eigengenes = eigen_gene_samples,
                                nGenes = num_genes,
                                modProportions = mod_prop,
                                minCor = min_cor,
                                maxCor = max_cor,
                                leaveOut = leave_out_A,
                                propNegativeCor = 0.1,
                                backgroundNoise = background_noise)

  x_2 <- WGCNA::simulateDatExpr(eigengenes = eigen_gene_samples,
                                nGenes = num_genes,
                                modProportions = mod_prop,
                                minCor = min_cor,
                                maxCor = max_cor,
                                leaveOut = leave_out_B,
                                propNegativeCor = 0.1,
                                backgroundNoise = background_noise)

  # Group labels
  sample_labels <- sort(rep(rep(LETTERS[1:num_sets], n / num_sets), 2))
  num_remainder <- (n * 2) - length(sample_labels)

  if (num_remainder > 0) {
    sample_labels <- sort(c(sample_labels, LETTERS[1:num_remainder]))
  }

  gene_labels <- x_1$allLabels

  # Combine Matrix
  x_matrix <- tibble::as_data_frame(rbind(x_1$datExpr,
                                          x_2$datExpr))
  colnames(x_matrix) <- gsub('[.]', '_', colnames(x_matrix))
  x_matrix$groups <- sample_labels
  x_matrix <- rbind(gene_labels, x_matrix)
  
  if (return_eigen) {
    return(list(x_matrix, eigen_gene_samples))
  }
  
  return(x_matrix)
}

# Vary sample size and amount of background noise
ns <- c(250, 500, 1000, 2000, 4000)
background_noises <- c(0, 0.1, 0.5, 1, 3)
genes <- c(500, 1000)

# Other constants
num_sets <- 4
num_modules <- 5

min_cor <- 0.4
max_cor <- 0.9

sample_set_A <- c(1, -1, 3, 0, -3)
sample_set_B <- c(0, 6, 3, -3, 1)

mod_prop <- c(0.25, 0.2, 0.15, 0.1, 0.05, 0.25)

leave_out_A <- rep(FALSE, num_modules)
leave_out_B <- c(rep(FALSE, num_modules / 2), TRUE,
                 rep(FALSE, (num_modules / 2)))

for (n in ns) {
  for (noise in background_noises) {
    for (g in genes) {
      out_file <- paste0("sim_data_samplesize_", n * 2, "_noise_", noise,
                         "_genes_", g, ".tsv")
      out_file <- file.path("data", "simulation", out_file)
      x <- simulateExpression(n = n,
                              num_genes = g,
                              background_noise = noise,
                              num_sets = num_sets,
                              num_modules = num_modules,
                              min_cor = min_cor,
                              max_cor = max_cor,
                              sample_set_A = sample_set_A,
                              sample_set_B = sample_set_B,
                              mod_prop = mod_prop,
                              leave_out_A = leave_out_A,
                              leave_out_B = leave_out_B)

      readr::write_tsv(x, out_file)
    }
  }
}

# Plot an example of simulated data
example_sim <- simulateExpression(n = 500,
                                  num_genes = 1000,
                                  background_noise = 0.1,
                                  num_sets = num_sets,
                                  num_modules = num_modules,
                                  min_cor = min_cor,
                                  max_cor = max_cor,
                                  sample_set_A = sample_set_A,
                                  sample_set_B = sample_set_B,
                                  mod_prop = mod_prop,
                                  leave_out_A = leave_out_A,
                                  leave_out_B = leave_out_B,
                                  return_eigen = TRUE)

# The eigen samples matrix forms the basis of the simulated patterns. It
# contains both sample group and gene module information
eigen_samples <- example_sim[[2]] 

# Plot the module correlations (there are 5 modules) Module 3 is made
# uncoupled from other modules, since it will be the focus of the latent space
# arithmetic test. Some modules are correlated to mimic real data.
eigen_module_file <- file.path("figures", "example_eigen_module_plot.png")
module_corr <- ggcorrplot::ggcorrplot(cor(eigen_samples), hc.order = FALSE,
                            outline.color = 'white') +
  ggtitle("Correlation across artificial gene modules \n(n = 5)")
ggsave(eigen_module_file, module_corr, height = 5, width = 6)

# Plot the sample correlations for this example. There are 1000 simulated
# samples. Each block of samples are distinct units - but it is important
# to note that this is before adding noise and dropping out module 3.
eigen_sample_file <- file.path("figures", "example_eigen_sample_plot.png")
sample_corr <- ggcorrplot::ggcorrplot(cor(t(eigen_samples)), hc.order = FALSE,
                                      outline.color = 'white') +
  ggtitle("Correlation across artificial sample groups\n(n = 500)")
ggsave(eigen_sample_file, sample_corr, height = 5, width = 6)

# Now, extract the actual sampled data and visualize a sample by gene heatmap.
# The red and orange samples are sampled from the same sample set module origin
# as are the blue and green. The green and orange samples have random noise
# expression of module 3.
simulated_example_file <- file.path("figures", "example_simulated_data.pdf")
sampled_data <- example_sim[[1]][2:nrow(example_sim[[1]]),
                                 1:ncol(example_sim[[1]]) - 1]

pdf(simulated_example_file)
heatmap.2(as.matrix(sampled_data), trace = "none",
          RowSideColors =  c(rep('blue', 250),
                            rep('red', 250),
                            rep('green', 250),
                            rep('orange', 250)),
          labRow = "", labCol = "", scale = 'none',
          hclustfun = function(x) hclust(x, method = 'average'),
          distfun = function(x) dist(x, method = 'euclidean'),
          dendrogram = "row", Rowv = TRUE, Colv = FALSE)
dev.off()

simulated_example_png <- file.path("figures", "example_simulated_data.png")
png(simulated_example_png, height = 600, width = 800)
heatmap.2(as.matrix(sampled_data), trace = "none",
          RowSideColors =  c(rep('blue', 250),
                             rep('red', 250),
                             rep('green', 250),
                             rep('orange', 250)),
          labRow = "", labCol = "", scale = 'none',
          hclustfun = function(x) hclust(x, method = 'average'),
          distfun = function(x) dist(x, method = 'euclidean'),
          dendrogram = "row", Rowv = TRUE, Colv = FALSE)
dev.off()
