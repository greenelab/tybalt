# Pan-Cancer Variational Autoencoder
# Gregory Way 2019
# scripts/encoding_pathway_analysis.R
#
# Performs Overrepresentation Pathway Analyses (ORA)
#
# Usage: Run in command line
#
#     Rscript --vanilla scripts/encoding_pathway_analysis.R
#
# Output:
# ORA results for:
# 1) HGSC-specific feature encodings identified by latent space arithmetic
# 2) SKCM distinguishing encodings
# 3) Dimensionality reduction algorithm comparison

library(WebGestaltR)
library(dplyr)

RunWebGestalt <- function(genes, output_name, out_dir, background_file,
                          sigMethod = "fdr") {
  # Function to run a WebGestalt Pathway Analysis
  #
  # Arguments:
  #    genes: A character of genes to perform ORA test on
  #    output_name: The name of the sub folder to save results in
  #    out_dir: The directory to save all pathway analysis results in
  #    background_file: A filepath pointing to a .txt file with a single column
  #                     indicating the background to perform the ORA against
  #    sigMethod: a string indicating how to consider top genes
  #               (either "fdr" or "top")
  #
  # Output:
  #    Return overrepresented pathways as a dataframe and write full results
  #    to the specified folder (`gestalt/output_name`)

  webgestalt_df <- WebGestaltR::WebGestaltR(
    enrichMethod = "ORA",
    organism = "hsapiens",
    interestGene = genes,
    interestGeneType = "genesymbol",
    minNum = 4,
    sigMethod = sigMethod,
    fdrMethod = "BH",
    isOutput = TRUE,
    outputDirectory = out_dir,
    referenceGeneFile = background_file,
    referenceGeneType = "genesymbol",
    projectName = output_name
    )

  # Load ID mapping
  id_map_file <- file.path(out_dir,
                           paste0("Project_", output_name),
                           paste0("interestingID_mappingTable_",
                                  output_name, ".txt")
                           )

  id_df <- readr::read_tsv(id_map_file,
                           col_types =
                             readr::cols(.default = readr::col_character(),
                                         entrezgene = readr::col_character())
                           )
  
  # Perform ID Mapping
  webgestalt_df <- webgestalt_df %>%
    dplyr::left_join(id_df, by = c("overlapId" = "entrezgene")) %>%
    dplyr::select(-userId)

  return(webgestalt_df)
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
  hgsc_node_df <- readr::read_tsv(
    hgsc_file,
    col_types = readr::cols(genes = readr::col_character(),
                            weight = readr::col_double())
    )

  hgsc_genes <- hgsc_node_df$genes

  # Perform pathway analysis with WebGestaltR
  g <- RunWebGestalt(genes = hgsc_genes,
                     output_name = base_name,
                     out_dir = hgsc_out_dir,
                     sigMethod = "top",
                     background_file = bg_file)
}

# 2) SKCM Analysis

# Specify filenames and constants
skcm_out_dir <- file.path("results", "pathway", "skcm")
nodes <- c("node53", "node66")
directions <- c("positive", "negative")

# Perform analysis
for (node in nodes) {

  skcm_file <- file.path("results",
                         paste0("high_weight_genes_", node, "_skcm.tsv"))

  node_df <- readr::read_tsv(
    skcm_file,
    col_types = readr::cols(.default = readr::col_double(),
                            genes = readr::col_character(),
                            direction = readr::col_character())
    )

  for (hw_direction in directions) {

    node_hw_df <- node_df %>% dplyr::filter(direction == !!hw_direction)
    skcm_genes <- node_hw_df$genes
    output_name <- paste0(node, "_", substr(hw_direction, 1, 3))

    # Perform pathway analysis and save results to file
    g <- RunWebGestalt(genes = skcm_genes,
                       output_name = output_name,
                       out_dir = skcm_out_dir,
                       sigMethod = "top",
                       background_file = bg_file)
  }
}

# 3) Dimensionality Reduction Analysis

# Specify filenames
dimred_out_dir <- file.path("results", "pathway", "dim_reduction")
dimred_files <- list.files(file.path("results", "feature_comparison"),
                           pattern = "mesenchymal_genes.tsv",
                           full.names = TRUE)

# Perform pathway analysis for each dimensionality reduction algorithm
for (dimred_file in dimred_files) {
  # Process the base name for the algorithm being processed
  algorithm_info <- gsub("_genes.tsv", "", basename(dimred_file))

  # Read in hgsc genes file
  dimred_node_df <- readr::read_tsv(
    dimred_file,
    col_types = readr::cols(genes = readr::col_character(),
                            weight = readr::col_double())
    )
  dimred_genes <- dimred_node_df$genes
  
  # Perform pathway analysis with WebGestaltR
  g <- RunWebGestalt(genes = dimred_genes,
                     output_name = algorithm_info,
                     out_dir = dimred_out_dir,
                     sigMethod = "top",
                     background_file = bg_file)
}
