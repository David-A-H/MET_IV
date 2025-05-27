
# MAIN TASK:
# Your task is to build a model to predict if a respondent reports any plans to emigrate (Variable emig).

### COMMENCE SETUP ##########################################################################################

# Load Packages
## Import Packages here:
library(tidyverse)
library(tidymodels)
library(caret)
library(cluster)
library(yardstick)

# Clear Global Environment
rm(list=ls())

# Import Data
data <-  load("final_project/AB4x_train.Rdata")
data <-  train
rm(train)



### COMMENCE PRE-PROCESSING #################################################################################

# Locate NA values in the dataset
na_count <- sapply(data, function(y) sum(length(which(is.na(y)))))
na_count[na_count > 0]

# Create backup for comparing dropping NAs and imputing NAs
data_dropped_na <-  na.omit(data)



# Median imputation for 'age'
median_age <- median(data$age, na.rm = TRUE)
data$age[is.na(data$age)] <- median_age

# Mode function for factors
get_mode <- function(x) {
  ux <- unique(x[!is.na(x)])
  ux[which.max(tabulate(match(x, ux)))]
}

# Mode imputation for 'q1002'
mode_q1002 <- get_mode(data$q1002)
data$q1002[is.na(data$q1002)] <- mode_q1002

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Overall Summary
summary(data)
## NOTE: Main issue is that variables have very unintuitive names. Main objective of this step: Rename variables and check for major class imbalance.
# 
# # Current names of variables
# names(data)
# 
# #Import rename-frame 
# var_rename <- read_csv("var_rename.csv")
# 
# # Make new_label the new column names in data
# data <- data %>%
#   rename_with(~ var_rename$new_label[match(., var_rename$original_label)], everything())

# NOTE: Renaming code works. However, the recoding file "var_rename.csv" is not entirely correct because it was created by AI. If worth the effort, maybe re-check. Otherwise proceed without renaming variables
  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Check for class imbalance
table(data$emig)
## NOTE: Class imbalance is present. 1 = emigration, 0 = no emigration

# Check for class imbalance in other variables
hist(data$age)
hist(as.numeric(data$income), breaks = 17)
hist(as.numeric(data$edu_combined))

# NOTE: Might want to think about rescaling some values like age. Slight skewness towards younger respondents

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Split Data into Train and Test and Stratify for emig
set.seed(123)
train_index <- createDataPartition(data$emig, p = .8, list = FALSE)
train <- data[train_index, ]
test <- data[-train_index, ]

# Check for dependent Variable proportion in train and test
prop.table(table(train$emig))
prop.table(table(test$emig))

# Create backup data file
backup <-  data
backup_dropped_na <- data_dropped_na

# # Load evaluation data (Used for evaluating the model at the very end)
# eval <- load("final_project/AB4x_eval_mock.Rdata")

# Clear global environment except for "train", "test", "backup" and "eval"
rm(list=ls()[!ls() %in% c("train", "test", "backup", "backup_dropped_na", "eval")])


### COMMENCE PREDICTION #####################################################################################

# Set seed for reproducibility
set.seed(123)

# Create a 10-fold cross-validation split on the training dataset
folds <- vfold_cv(train, v = 10)

# Set up a recipe for preprocessing
# The model will predict 'emig' using all other variables in 'train'
svm_recipe <- recipe(emig ~ ., data = train)

# Specify a Support Vector Machine model with a radial basis function (RBF) kernel
# Set the engine to 'kernlab', which is used for fitting the model
svm_rbf_spec <- svm_rbf(mode = "classification") %>%
  set_engine("kernlab")

# Create a modeling workflow
# Add the SVM model specification and the preprocessing recipe
wf_rbf <- workflow() %>%
  add_model(svm_rbf_spec) %>%
  add_recipe(svm_recipe)

# Define a set of performance metrics to evaluate the model
# Includes accuracy, precision, recall, and F1 score (f_meas)
my_metrics <- metric_set(
  accuracy,
  yardstick::precision,
  yardstick::recall,
  f_meas
)

# Perform cross-validated resampling
# Fit the model on each fold and evaluate using the specified metrics
# Save predictions for each resample
res_rbf <- fit_resamples(
  wf_rbf, 
  resamples = folds, 
  metrics = my_metrics,
  control = control_resamples(save_pred = TRUE)
)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FIT ON THE TEST DATA

# Fit the final model on the full training data
final_rbf_fit <- fit(wf_rbf, data = train)

# Set a custom cutoff threshold (for some reason, the default value was 0.75 earlier)
custom_cutoff <- 0.5

# Get predictions with class probabilities
test_preds <- predict(final_rbf_fit, new_data = test, type = "prob") %>%
  bind_cols(test)

# Create predicted class based on custom cutoff
test_preds <- test_preds %>%
  mutate(.pred_class_custom = if_else(.pred_Yes >= custom_cutoff, "Yes", "No"),
         .pred_class_custom = factor(.pred_class_custom, levels = levels(as.factor(emig))),
         emig = as.factor(emig))

# Make sure Yes Class is correctly specified
test_preds <- test_preds %>%
  mutate(
    emig = factor(emig, levels = c("Yes", "No")),
    .pred_class_custom = factor(.pred_class_custom, levels = c("Yes", "No"))
  )


# Compute confusion matrix using the custom predictions
conf_mat(data = test_preds, 
         truth = emig, 
         estimate = .pred_class_custom)


# Ensure variables are factors with correct levels
test_preds <- test_preds %>%
  mutate(emig = as.factor(emig),
         .pred_class_custom = factor(.pred_class_custom, levels = levels(emig)))

# Apply metrics
my_metrics(data = test_preds,
               truth = emig,
               estimate = .pred_class_custom)

# ROC AUC requires class probabilities
roc_auc(data = test_preds,
        truth = emig,
        .pred_Yes)

########
# 
# 
# collect_metrics(res_rbf)
# 
# # collect confusion matrix from res_rbf
# confusion_rbf <- collect_predictions(res_rbf) %>%
#   conf_mat(truth = emig, estimate = .pred_class)
# # Display confusion matrix
# confusion_rbf
# 
# 
# 
# 
# ##############
# 
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # Calculate and display ROC curve and AUC value
# 
# # Get ROC data for each model
# get_roc_data <- function(results, model_name) {
#   collect_predictions(results, summarize = FALSE) %>%
#     roc_curve(truth = emig, .pred_1) %>%  # assuming "1" is the positive class
#     mutate(Model = model_name)
# }
# 
# roc_linear <- get_roc_data(res_linear, "Linear")
# roc_poly <- get_roc_data(res_poly, "Polynomial")
# roc_rbf <- get_roc_data(res_rbf, "RBF")
# 
# # Combine and plot
# roc_data <- bind_rows(roc_linear, roc_poly, roc_rbf)
# 
# ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity, color = Model)) +
#   geom_path(size = 1.2) +
#   geom_abline(lty = 2, color = "gray") +
#   coord_equal() +
#   labs(title = "ROC Curve", x = "1 - Specificity", y = "Sensitivity")
# 
# 
# #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
# # Calculate AUC for each model
# get_auc <- function(results) {
#   collect_predictions(results, summarize = FALSE) %>%
#     roc_auc(truth = emig, .pred_1)
# }
# 
# auc_linear <- get_auc(res_linear)
# auc_poly <- get_auc(res_poly)
# auc_rbf <- get_auc(res_rbf)
# 
# bind_rows(
#   Linear = auc_linear,
#   Polynomial = auc_poly,
#   RBF = auc_rbf,
#   .id = "Model"
# )







### COMMENCE MODEL COMPARISON AND EXTRACTION #################################################################





