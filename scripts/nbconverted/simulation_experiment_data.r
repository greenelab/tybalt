
library(tibble)
library(dplyr)
library(ggplot2)

# Load custom simulation functions
source(file.path("util", "simulate_expression.R"))

# Store seeds for each dataset
seeds <- c(1, 2, 3, 4)

# Set Variables
num_samples <- 12000

group_mean <- as.matrix(cbind(c(1, 5, 3), c(4, 0, 2)))
group_sd <- as.matrix(cbind(c(1, 1, 1), c(1, 1, 1)))

num_boolean_features <- 1

cell_mean <- as.matrix(cbind(c(1, 4), c(5, 2)))
cell_sd <- as.matrix(cbind(c(1, 1), c(1, 1)))

x_squared <- function(x) { x ** 2 + 1}
two_x_third <- function(x) {2 * x ** 3 + 1}
func_list <- list(x_squared, two_x_third)

random_noise <- c(0, 5, 20, 100)

# Simulate data and save to folder
output_folder <- file.path("..", "data", "simulation")
noise_idx <- 1
for (num_noise_features in random_noise) {
  sim_df <- getSimulatedExpression(n = num_samples,
                                   mean_df = group_mean,
                                   sd_df = group_sd,
                                   b = num_boolean_features,
                                   func_list = func_list,
                                   seed = seeds[noise_idx],
                                   cell_type_mean_df = cell_mean,
                                   cell_type_sd_df = cell_sd,
                                   r = num_noise_features,
                                   concat = TRUE)

  sim_data_file <- paste0("sim_data_", num_noise_features, "_noise.tsv")
  sim_data_file <- file.path(output_folder, sim_data_file)
  write.table(sim_df, sim_data_file, sep = "\t")

  noise_idx <- noise_idx + 1
}

# Example with 100 noise features
head(sim_df)

ggplot(sim_df, aes(x=group_1, y=group_2)) + geom_point(aes(color=groups), alpha=0.5) +
  theme_bw()

ggplot(sim_df, aes(x=continuous_domain_1, y=continuous_1)) +
  geom_point(aes(color=groups), alpha=0.5) + theme_bw()

ggplot(sim_df, aes(x=continuous_domain_2, y=continuous_2)) +
  geom_point(aes(color=groups), alpha=0.5) + theme_bw()

ggplot(sim_df, aes(x=group_1, y=presence_1)) + geom_point(aes(color=groups), alpha=0.5) +
  theme_bw()

ggplot(sim_df, aes(x=cell_type_1, y=cell_type_2)) +
  geom_point(aes(color=groups, size=cell_type_prop), alpha=0.5) +
  theme_bw()

ggplot(sim_df, aes(x=random_1, y=random_2)) + geom_point(aes(color=groups), alpha=0.5) +
  theme_bw()
