# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
# scripts/encoding_pathway_analysis.R
#
# Performs Overrepresentation Pathway Analyses (ORA)
#
# Usage: Run in command line
#
#     Rscript --vanilla scripts/gestalt_pathway_analysis.R
#
# Output:
# ORA results for:
# 1) HGSC-specific feature encodings identified by latent space arithmetic
# 2) SKCM distinguishing encodings
# 3) Dimensionality reduction algorithm comparison

library(WebGestaltR)
library(dplyr)

RunWebGestalt <- function(genes, output_name, out_dir, background_file) {
  # Function to run a WebGestalt Pathway Analysis
  #
  # Arguments:
  #    genes: A character of genes to perform ORA test on
  #    output_name: The name of the sub folder to save results in
  #    out_dir: The directory to save all pathway analysis results in
  #    background_file: A filepath pointing to a .txt file with a single column
  #                     indicating the background to perform the ORA against
  #
  # Output:
  #    Return overrepresented pathways as a dataframe and write full results
  #    to the specified folder (`gestalt/output_name`)

  webgestalt_output <-
    WebGestaltR::WebGestaltR(enrichMethod = "ORA",
                             organism = "hsapiens",
                             interestGene = genes,
                             interestGeneType = "genesymbol",
                             minNum = 4,
                             fdrMethod = "BH",
                             is.output = TRUE,
                             outputDirectory = out_dir,
                             referenceGeneFile = background_file,
                             referenceGeneType = "genesymbol",
                             projectName = output_name)
  return(webgestalt_output)
}

# 1) HGSC Analysis

# Specify filenames
hgsc_out_dir <- file.path("results", "pathway", "hgsc")
bg_file <- file.path("data", "background_genes.txt")
hgsc_files <- list.files("results", pattern = "hgsc_node", full.names = TRUE)

# Perform pathway analysis
for (hgsc_file in hgsc_files) {
  # Process the base name for the genes being processed
  hgsc_info <- unlist(strsplit(hgsc_file, "_"))
  node <- hgsc_info[2]
  direction <- hgsc_info[3]
  direction <- unlist(strsplit(direction, "[.]"))[1]
  base_name <- paste(node, direction, sep = "_")
  
  # Read in hgsc genes file
  hgsc_node_df <- readr::read_tsv(hgsc_file)
  hgsc_genes <- hgsc_node_df$genes

  # Perform pathway analysis with WebGestaltR
  g <- RunWebGestalt(hgsc_genes, base_name, hgsc_out_dir, bg_file)
}

# 2) SKCM Analysis

# Specify filenames and load data
skcm_out_dir <- file.path("results", "pathway", "skcm")
node53_file <- file.path("results", "high_weight_genes_node53_skcm.tsv")
node66_file <- file.path("results", "high_weight_genes_node66_skcm.tsv")

node53_df <- readr::read_tsv(node53_file)
node66_df <- readr::read_tsv(node66_file)

# Filter high weight genes
node_53_pos_df <- node53_df %>% dplyr::filter(direction == "positive")
node_53_neg_df <- node53_df %>% dplyr::filter(direction == "negative")
node_66_pos_df <- node66_df %>% dplyr::filter(direction == "positive")
node_66_neg_df <- node66_df %>% dplyr::filter(direction == "negative")

# Perform pathway analysis and save results to file
g <- RunWebGestalt(node_53_pos_df$genes, "node53_pos", skcm_out_dir, bg_file)
g <- RunWebGestalt(node_53_neg_df$genes, "node53_neg", skcm_out_dir, bg_file)
g <- RunWebGestalt(node_66_pos_df$genes, "node66_pos", skcm_out_dir, bg_file)
g <- RunWebGestalt(node_66_neg_df$genes, "node66_neg", skcm_out_dir, bg_file)

# 3) Dimensionality Reduction Analysis

# Specify filenames
dimred_out_dir <- file.path("results", "pathway", "dim_reduction")
dimred_files <- list.files(file.path("results", "feature_comparison"),
                           pattern = "mesenchymal_genes.tsv", full.names = TRUE)

# Perform pathway analysis for each dimensionality reduction algorithm
for (dimred_file in dimred_files) {
  # Process the base name for the algorithm being processed
  dimred_info <- unlist(strsplit(basename(dimred_file), "_"))
  algorithm <- dimred_info[1]

  # Read in hgsc genes file
  dimred_node_df <- readr::read_tsv(dimred_file)
  dimred_genes <- dimred_node_df$genes
  
  # Perform pathway analysis with WebGestaltR
  g <- RunWebGestalt(dimred_genes, algorithm, dimred_out_dir, bg_file)
}
