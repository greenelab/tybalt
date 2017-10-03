# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
#
# Visualizing results of variational autoencoder parameter sweeps
#
# Output:
# Figures describing the results of a parameter sweep over preselected
# hyperparameters. Included in the analysis are sweeps for a single hidden
# layer and a two hidden layer model.

library(ggplot2)

`%>%` <- dplyr::`%>%`

recodeParameters <- function(filename, depth = 1) {
  # Read in parameter sweep file in long format and output processed dataframe
  # with recoded factors
  # 
  # Arguments:
  # filename - the file path pointing to the parameter sweep results
  # depth - how many hidden layers are included in the model
  #
  # Output:
  # A list with the following two elements:
  # 1) A processed dataframe ready for plotting
  # 2) Dataframe with the final output results at training end
  
  param_df <- readr::read_tsv(filename,
                              col_types = readr::cols(
                                .default = readr::col_character(),
                                train_epoch = readr::col_integer(),
                                loss = readr::col_double(),
                                val_loss = readr::col_double()))
  
  param_melt_df <- reshape2::melt(param_df,
                                  id.vars = c("learning_rate", "batch_size",
                                              "epochs", "kappa", "train_epoch"),
                                  measure.vars = c("loss", "val_loss"),
                                  variable.name = "loss_type",
                                  value.name = "loss")

  # Assign the number of hidden layers to its own variable
  param_melt_df$depth <- paste("depth =", depth)
  
  # Order and recode batch size variables
  param_melt_df$batch_size <- factor(param_melt_df$batch_size,
                                     levels = c(50, 100, 128, 200))
  param_melt_df$batch_size <- 
    dplyr::recode_factor(param_melt_df$batch_size, 
                         `50` = "batch: 50", 
                         `100` = "batch: 100",
                         `128` = "batch: 128",
                         `200` = "batch: 200") 

  # Order and recode epoch variables
  param_melt_df$epochs <- factor(param_melt_df$epochs,
                                 levels = c(10, 25, 50, 100))
  param_melt_df$epochs <- 
    dplyr::recode_factor(param_melt_df$epochs, 
                         `10` = "epochs: 10", 
                         `25` = "epochs: 25",
                         `50` = "epochs: 50",
                         `100` = "epochs: 100") 

  # Order and recode kappa variables
  param_melt_df$kappa <- factor(param_melt_df$kappa,
                                levels = c("0.01", "0.05", "0.1", "1.0"))
  param_melt_df$kappa <- 
    dplyr::recode_factor(param_melt_df$kappa, 
                         `0.01` = "kappa: 0.01", 
                         `0.05` = "kappa: 0.05",
                         `0.1` = "kappa: 0.1",
                         `1.0` = "kappa: 1.0") 

  # Also output the final results after training completion
  final_select_df <- param_melt_df %>% dplyr::filter(loss_type == "val_loss")
  final_select_df <- final_select_df %>%
    dplyr::group_by(learning_rate, batch_size, epochs, kappa, depth) %>%
    dplyr::summarize(end_loss = tail(loss, 1))
  
  return_list <- list(param_melt_df, as.data.frame(final_select_df))
  return(return_list)
} 

plotSweep <- function(df, param, base_file) {
  # Function to plot parameter sweep results across the given measurement
  #
  # Arguments:
  # df - melted dataframe storing results of a parameter sweep
  # param - the variable of interest to stratify results over
  # base_file - file path location of where plots will be saved
  #
  # Output:
  # Several plots stratifying validation loss across parameter combinations

  for (v in unique(df[, param])) {
    subset_param <- df %>%
      dplyr::filter_(paste(param, "==", paste0('"', v, '"')))
    
    if (param == "kappa") {
      facet_var <- "epochs"
    } else if (param == "epochs") {
      facet_var <- "kappa"
    }
    
    # Remove label in `k` for plotting and filename
    v <- gsub(paste0(param, ": "), "", v)
    
    p <- ggplot(subset_param, aes(x = train_epoch, y = loss)) +
      geom_line(aes(color = learning_rate, linetype = loss_type), size = 0.5) + 
      facet_grid(as.formula(paste("batch_size ~", facet_var)),
                 scales = "free") +
      theme_bw() + 
      ggtitle(paste(param, "=", v))
    
    output_fig <- paste0(base_file, v, "_", param, ".png")
    ggsave(output_fig, plot = p, height = 5, width = 6)
  }
}

plotFinalParams <- function(df, base_file, twohidden = FALSE) {
  # Plot the final validation loss at the end of training
  #
  # Arguments:
  # df - dataframe that stores validation loss at training end across params
  # base_file - directory and basename of where file will be saved
  # twohidden - boolean to determine how variables are displayed
  #
  # Output:
  # Saves the resulting plot to file

  if (twohidden) {
    # Recode epochs variable for plotting
    df$epochs <-  dplyr::recode_factor(df$epochs, 
                           "epochs: 10" = 10, 
                           "epochs: 25" = 25, 
                           "epochs: 50" = 50, 
                           "epochs: 100" = 100) 
    p <- ggplot(df, aes(x = epochs, y = end_loss)) +
      geom_point(aes(shape = batch_size, color = learning_rate),
                 size = 2) + facet_grid(~ kappa) +
      xlab("Epochs")
    height <- 5
    width <- 10
  } else {
    df <- df %>% dplyr::filter(kappa == "kappa: 1.0")
    p <- ggplot(df, aes(x = learning_rate, y = end_loss)) +
      geom_point(aes(color = batch_size, shape = epochs), size = 2) +
      xlab("Learning Rate")
    height <- 5
    width <- 6
  }

  p <- p + theme_bw() +
    ylab("Validation Loss at Training End") +
    theme(axis.text = element_text(size = rel(1.2)),
          legend.text = element_text(size = rel(1.1)))

  output_fig <- paste0(base_file, "final_param_val_loss.png")
  ggsave(output_fig, plot = p, height = height, width = width)
}

getBestModel <- function(param_obj) {
  # Function to output training across epochs of the most optimal model
  # based on validation loss at training end
  #
  # Arguments:
  # param_obj - a list object output from `recodeParameters` stores parameter
  #             sweep results and final validation loss for all hyper params
  #
  # Output:
  # Dataframe of training and validation loss at the end of each training epoch
  # for the optimal model

  best_model <- param_obj[[2]] %>% dplyr::top_n(1, -end_loss)

  # Subset best model training
  best_model_training <- param_obj[[1]] %>%
    dplyr::filter(epochs == best_model$epochs,
                  batch_size == best_model$batch_size,
                  learning_rate == best_model$learning_rate,
                  kappa == best_model$kappa)
  
  return(best_model_training)
}

# Set File names
param_sweep_file <- file.path("results", "parameter_sweep_full_results.tsv")
param_sweep_two_file <- file.path("results",
                                  "parameter_sweep_twohidden_full_results.tsv")
param_two300_file <- file.path("results",
                               "parameter_sweep_twohidden300_full_results.tsv")
out_fig <- file.path("figures", "param_sweep", "full_param_")
out_two_fig <- file.path("figures", "param_sweep", "twohidden", "full_param_")
out_two300_fig <- file.path("figures", "param_sweep", "twohidden300",
                            "full_param_")

# Process files
tybalt <- recodeParameters(param_sweep_file, depth = 1)
tybalt_twohidden <- recodeParameters(param_sweep_two_file, depth = 2)
tybalt_twohidden300 <- recodeParameters(param_two300_file, depth = "2 (300)")

# Plot parameter sweep results
plotSweep(tybalt[[1]], "kappa", out_fig)
plotSweep(tybalt_twohidden[[1]], "kappa", out_two_fig)
plotSweep(tybalt_twohidden300[[1]], "kappa", out_two300_fig)

plotSweep(tybalt[[1]], "epochs", out_fig)
plotSweep(tybalt_twohidden[[1]], "epochs", out_two_fig)
plotSweep(tybalt_twohidden300[[1]], "epochs", out_two300_fig)

plotFinalParams(tybalt[[2]], out_fig)
plotFinalParams(tybalt_twohidden[[2]], twohidden = TRUE, out_two_fig)
plotFinalParams(tybalt_twohidden300[[2]], twohidden = TRUE, out_two300_fig)

# Identify the best models and compare
best_one_layer <- getBestModel(tybalt)
best_two_layer <- getBestModel(tybalt_twohidden)
best_two300_layer <- getBestModel(tybalt_twohidden300)

combined_param_df <- dplyr::bind_rows(best_one_layer, best_two_layer,
                                      best_two300_layer)

# Plot best models across training epochs
p <- ggplot(combined_param_df, aes(x = train_epoch, y = loss)) +
  geom_line(aes(color = depth, linetype = loss_type), size = 0.5) +
  theme_bw() + xlab("Training Epoch") + ylab("Loss") +
  scale_color_manual(name = "Depth",
                     values = c("goldenrod", "darkgreen", "red"),
                     labels = c("1 (100)", "2 (100 -> 100)",
                                "2 (300 -> 100)")) +
  scale_linetype_manual(name = "Loss Type", values = c("solid", "dotted"),
                        labels = c("Train", "Validation")) +
  theme(axis.text = element_text(size = rel(1.4)),
        legend.text = element_text(size = rel(1.0)),
        legend.title = element_text(size = rel(1.3)),
        axis.title = element_text(size = rel(1.6)))

output_fig <- file.path("figures", "param_sweep", "best_model_comparisons.png")
ggsave(output_fig, plot = p, height = 4, width = 6)
