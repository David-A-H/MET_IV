
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
# NOTE: BEFORE EVERY PREDICTION MODEL, RUN set.seed(123) to reset the RNG.
set.seed(123)


# Create a 10-fold cross-validation split
folds <- vfold_cv(train, v = 10)

# Set up a recipe
svm_recipe <- recipe(emig ~ ., data = train)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Define models
svm_linear_spec <- svm_linear(mode = "classification") %>% 
  set_engine("kernlab")

svm_poly_spec <- svm_poly(mode = "classification") %>%
  set_engine("kernlab")

svm_rbf_spec <- svm_rbf(mode = "classification") %>%
  set_engine("kernlab")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Wrap each into a workflow
wf_linear <- workflow() %>%
  add_model(svm_linear_spec) %>%
  add_recipe(svm_recipe)

wf_poly <- workflow() %>%
  add_model(svm_poly_spec) %>%
  add_recipe(svm_recipe)

wf_rbf <- workflow() %>%
  add_model(svm_rbf_spec) %>%
  add_recipe(svm_recipe)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define wanted performance metrics
my_metrics <- metric_set(
  accuracy,
  yardstick::precision,
  yardstick::recall,
  f_meas,
  roc_auc
)


# Evaluate each with cross-validation
res_linear <- fit_resamples(
  wf_linear, 
  resamples = folds, 
  metrics = my_metrics,
  control = control_resamples(save_pred = TRUE)
)

res_poly <- fit_resamples(
  wf_poly, 
  resamples = folds, 
  metrics = my_metrics,
  control = control_resamples(save_pred = TRUE)
)

res_rbf <- fit_resamples(
  wf_rbf, 
  resamples = folds, 
  metrics = my_metrics,
  control = control_resamples(save_pred = TRUE)
)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Compare results
collect_metrics(res_linear)
collect_metrics(res_poly)
collect_metrics(res_rbf)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Calculate and display ROC curve and AUC value

# Get ROC data for each model
get_roc_data <- function(results, model_name) {
  collect_predictions(results, summarize = FALSE) %>%
    roc_curve(truth = emig, .pred_1) %>%  # assuming "1" is the positive class
    mutate(Model = model_name)
}

roc_linear <- get_roc_data(res_linear, "Linear")
roc_poly <- get_roc_data(res_poly, "Polynomial")
roc_rbf <- get_roc_data(res_rbf, "RBF")

# Combine and plot
roc_data <- bind_rows(roc_linear, roc_poly, roc_rbf)

ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity, color = Model)) +
  geom_path(size = 1.2) +
  geom_abline(lty = 2, color = "gray") +
  coord_equal() +
  labs(title = "ROC Curve", x = "1 - Specificity", y = "Sensitivity")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Calculate AUC for each model
get_auc <- function(results) {
  collect_predictions(results, summarize = FALSE) %>%
    roc_auc(truth = emig, .pred_1)
}

auc_linear <- get_auc(res_linear)
auc_poly <- get_auc(res_poly)
auc_rbf <- get_auc(res_rbf)

bind_rows(
  Linear = auc_linear,
  Polynomial = auc_poly,
  RBF = auc_rbf,
  .id = "Model"
)


# Current Results:
# > collect_metrics(res_linear)
# # A tibble: 5 × 6
# .metric   .estimator  mean     n std_err .config             
# <chr>     <chr>      <dbl> <int>   <dbl> <chr>               
#   1 accuracy  binary     0.716    10 0.00536 Preprocessor1_Model1
# 2 f_meas    binary     0.818    10 0.00382 Preprocessor1_Model1
# 3 precision binary     0.790    10 0.00822 Preprocessor1_Model1
# 4 recall    binary     0.848    10 0.00667 Preprocessor1_Model1
# 5 roc_auc   binary     0.657    10 0.00739 Preprocessor1_Model1

# > collect_metrics(res_poly)
# # A tibble: 5 × 6
# .metric   .estimator  mean     n std_err .config             
# <chr>     <chr>      <dbl> <int>   <dbl> <chr>               
#   1 accuracy  binary     0.716    10 0.00538 Preprocessor1_Model1
# 2 f_meas    binary     0.818    10 0.00382 Preprocessor1_Model1
# 3 precision binary     0.790    10 0.00823 Preprocessor1_Model1
# 4 recall    binary     0.848    10 0.00666 Preprocessor1_Model1
# 5 roc_auc   binary     0.657    10 0.00739 Preprocessor1_Model1

# > collect_metrics(res_rbf)
# # A tibble: 5 × 6
# .metric   .estimator  mean     n  std_err .config             
# <chr>     <chr>      <dbl> <int>    <dbl> <chr>               
#   1 accuracy  binary     0.752    10 0.00567  Preprocessor1_Model1
# 2 f_meas    binary     0.858    10 0.00367  Preprocessor1_Model1
# 3 precision binary     0.752    10 0.00572  Preprocessor1_Model1
# 4 recall    binary     0.999    10 0.000376 Preprocessor1_Model1
# 5 roc_auc   binary     0.711    10 0.00707  Preprocessor1_Model1
# > 








### COMMENCE MODEL COMPARISON AND EXTRACTION #################################################################





