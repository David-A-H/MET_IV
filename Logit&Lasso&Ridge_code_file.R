
# MAIN TASK:
# Your task is to build a model to predict if a respondent reports any plans to emigrate (Variable emig).

### COMMENCE SETUP ##########################################################################################

# Load Packages
## Import Packages here:
library(tidyverse)
library(tidymodels)
library(caret)
library(cluster)
library(glmnet)
library(mice)
library(pROC)

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



# NA Imputation
mice_init <- mice(data %>% select(age, q1002),
                  m = 5, # creates 5 different imputed datasets (i.e., 5 multiple imputations).
                  method = c(age = "pmm", q1002 = "polyreg"), #Uses Predictive Mean Matching to impute age (numeric) /Uses polytomous logistic regression to impute
                  seed = 123,
                  parallel = "multicore",
                  ncore = parallel::detectCores()-1)
# Extract one complete dataset (first imputed dataset of the 5 generated).
data_imp <- complete(mice_init, action = 1)
# Replace into original
data <- data %>%
  select(-age, -q1002) %>%
  bind_cols(data_imp)


# Convert emig to factor
data$emig <- as.factor(ifelse(data$emig == "Yes", 1, 0))  # Convert to binary factor (1 = emigration, 0 = no emigration)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Overall Summary
summary(data)
# NOTE: Main issue is that variables have very unintuitive names. Main objective of this step: Rename variables and check for major class imbalance.

# Current names of variables
names(data)

#Import rename-frame
var_rename <- read_csv("var_rename.csv")

# Make new_label the new column names in data
data <- data %>%
  rename_with(~ var_rename$new_label[match(., var_rename$original_label)], everything())

# NOTE: Renaming code works. However, the recoding file "var_rename.csv" is not entirely correct because it was created by AI. If worth the effort, maybe re-check. Otherwise proceed without renaming variables
  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Check for class imbalance
table(data$emig)
## NOTE: Class imbalance is present. 1 = emigration, 0 = no emigration

# Check for class imbalance in other variables
hist(data$age)
hist(as.numeric(data$income), breaks = 17)
hist(as.numeric(data$edu_combined))

# NOTE: Might want to think about rescaling some values like age. Slight skewness towards younger respondents

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

# 
# # 1 Class duplicating (because reweighing didn't work)
# ## show current imbalance
# table(train$emig) # 0 = 0.85, 1 = 0.15. Aim is to upsample 3368 1 observations
# 
# ### Set seed again
# set.seed(123)
# 
# ## Upsample the minority class (1) to match the majority class (0)
# train <- train %>%
#   group_by(emig) %>%
#   slice_sample(n = max(table(train$emig)), replace = TRUE) %>%
#   ungroup()
# 
# ## Check the new class distribution
# table(train$emig) # 0 = 0.5, 1 = 0.5. Now balanced
# table(test$emig)


# # Load evaluation data (Used for evaluating the model at the very end)
# eval <- load("final_project/AB4x_eval_mock.Rdata")

# Clear global environment except for "train", "test", "backup" and "eval"
rm(list=ls()[!ls() %in% c("train", "test", "backup", "backup_dropped_na", "eval")])


### COMMENCE PREDICTION #####################################################################################
# Logistic Regression on all features as a baseline
set.seed(123)


# Define a recipe to handle preprocessing
logistic_recipe <- recipe(emig ~ ., data = train) %>%
  step_naomit(all_predictors(), all_outcomes()) %>%   # Remove NAs
  step_nzv(all_predictors()) %>%                      # Remove near-zero variance predictors
  step_unknown(all_nominal_predictors()) %>%          # Handle unseen levels in test data
  step_other(all_nominal_predictors(), threshold = 0.01) %>%  # Combine infrequent levels
  step_dummy(all_nominal_predictors())               # Convert factors to dummies

# Define the logistic regression model
logistic_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Create the workflow
logistic_wf <- workflow() %>%
  add_model(logistic_model) %>%
  add_recipe(logistic_recipe)

# Fit the model to training data
logistic_fit <- logistic_wf %>%
  fit(data = train)
# NOTE: The case_weights argument is used to apply the weights calculated earlier

# Generate predictions on the test set
logistic_pred      <- predict(logistic_fit, new_data = test, type = "class")
logistic_pred_prob <- predict(logistic_fit, new_data = test, type = "prob")

# Evaluate with confusion matrix
logistic_result <- confusionMatrix(
  data = logistic_pred$.pred_class,
  reference = test$emig,
  positive = "1"
)
print(logistic_result)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LASSO REGRESSION
set.seed(123)

# Define recipe
lasso_recipe <- recipe(emig ~ ., data = train) %>%
  step_naomit(all_predictors(), all_outcomes()) %>%
  step_nzv(all_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())  %>% 
  step_normalize(all_numeric_predictors()) 

# Define Lasso model (glmnet with alpha = 1)
lasso_model <- logistic_reg(
  penalty = tune(),       # We'll tune lambda
  mixture = 1             # 1 = Lasso (L1)
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# Create workflow
lasso_wf <- workflow() %>%
  add_model(lasso_model) %>%
  add_recipe(lasso_recipe)

# Cross-validation setup
cv_folds <- vfold_cv(train, v = 5, strata = emig)

# Tune penalty parameter (lambda)
lasso_res <- tune_grid(
  lasso_wf,
  resamples = cv_folds,
  grid = 30,
  metrics = metric_set(roc_auc)
)

# Get best lambda (penalty) based on AUC
best_lasso <- select_best(lasso_res, metric = "roc_auc")
best_lasso

# Finalize workflow with best penalty
final_lasso_wf <- finalize_workflow(lasso_wf, best_lasso)

# Fit to training data
final_lasso_fit <- fit(final_lasso_wf, data = train)

# Predict on test set
#lasso_pred_class <- predict(final_lasso_fit, new_data = test, type = "class")
lasso_pred_prob  <- predict(final_lasso_fit, new_data = test, type = "prob")

# Attempt to improve performance by modifying classification threshold
threshold <- 0.25
lasso_pred_class <- ifelse(lasso_pred_prob$.pred_1 > threshold, "1", "0") %>%
  factor(levels = c("0", "1"))

# Confusion matrix
lasso_result <- confusionMatrix(
  data = lasso_pred_class,
  reference = test$emig,
  positive = "1"
)
print(lasso_result)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RIDGE REGRESSION
set.seed(123)

# Define recipe
ridge_recipe <- recipe(emig ~ ., data = train) %>%
  step_naomit(all_predictors(), all_outcomes()) %>%
  step_nzv(all_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) 

# Define ridge model (glmnet with alpha = 1)
ridge_model <- logistic_reg(
  penalty = tune(),       # We'll tune lambda
  mixture = 0             
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# Create workflow
ridge_wf <- workflow() %>%
  add_model(ridge_model) %>%
  add_recipe(ridge_recipe)

# Cross-validation setup
cv_folds <- vfold_cv(train, v = 5, strata = emig)

# Tune penalty parameter (lambda)
ridge_res <- tune_grid(
  ridge_wf,
  resamples = cv_folds,
  grid = 30,
  metrics = metric_set(roc_auc)
)

# Get best lambda (penalty) based on AUC
best_ridge <- select_best(ridge_res, metric = "roc_auc")
best_ridge

# Finalize workflow with best penalty
final_ridge_wf <- finalize_workflow(ridge_wf, best_ridge)

# Fit to training data
final_ridge_fit <- fit(final_ridge_wf, data = train)

# Predict on test set
#ridge_pred_class <- predict(final_ridge_fit, new_data = test, type = "class")
ridge_pred_prob  <- predict(final_ridge_fit, new_data = test, type = "prob")

# Attempt to improve performance by modifying classification threshold
threshold <- 0.25
ridge_pred_class <- ifelse(ridge_pred_prob$.pred_1 > threshold, "1", "0") %>%
  factor(levels = c("0", "1"))

# Confusion matrix
ridge_result <- confusionMatrix(
  data = ridge_pred_class,
  reference = test$emig,
  positive = "1"
)
print(ridge_result)



### COMMENCE MODEL COMPARISON AND EXTRACTION #################################################################


# Calculate AUC for each model

# Logistic Regression AUC
logistic_roc <- roc(test$emig, as.numeric(logistic_pred_prob$.pred_1))
logistic_auc <- auc(logistic_roc)

# Lasso Regression AUC
lasso_roc <- roc(test$emig, as.numeric(lasso_pred_prob$.pred_1))
lasso_auc <- auc(lasso_roc)

# Ridge Regression AUC
ridge_roc <- roc(test$emig, as.numeric(ridge_pred_prob$.pred_1))
ridge_auc <- auc(ridge_roc)




# Based on logistic_result, calculate accuracy, precision, recall and F1
logistic_accuracy <- logistic_result$overall["Accuracy"]
logistic_precision <- logistic_result$byClass["Precision"]
logistic_recall <- logistic_result$byClass["Recall"]
logistic_f1 <- logistic_result$byClass["F1"]

# Based on lasso_result, calculate accuracy, precision, recall and F1
lasso_accuracy <- lasso_result$overall["Accuracy"]
lasso_precision <- lasso_result$byClass["Precision"]
lasso_recall <- lasso_result$byClass["Recall"]
lasso_f1 <- lasso_result$byClass["F1"]

# Based on ridge_result, calculate accuracy, precision, recall and F1
ridge_accuracy <- ridge_result$overall["Accuracy"]
ridge_precision <- ridge_result$byClass["Precision"]
ridge_recall <- ridge_result$byClass["Recall"]
ridge_f1 <- ridge_result$byClass["F1"]


# Summarise all performance metrics in a table
performance_metrics <- tibble(
  Model = c("Logistic Regression", "Lasso Regression", "Ridge Regression"),
  Accuracy = c(logistic_accuracy, lasso_accuracy, ridge_accuracy),
  Precision = c(logistic_precision, lasso_precision, ridge_precision),
  Recall = c(logistic_recall, lasso_recall, ridge_recall),
  F1_Score = c(logistic_f1, lasso_f1, ridge_f1)
)
performance_metrics

logistic_auc
lasso_auc
ridge_auc




# 
# > performance_metrics
# # A tibble: 3 × 5
# Model               Accuracy Precision Recall F1_Score
# <chr>                  <dbl>     <dbl>  <dbl>    <dbl>
# 1 Logistic Regression    0.636     0.358  0.582    0.444
# 2 Lasso Regression       0.506     0.309  0.794    0.445
# 3 Ridge Regression       0.500     0.308  0.806    0.446
# > logistic_auc
# Area under the curve: 0.6558
# > lasso_auc
# Area under the curve: 0.6719
# > ridge_auc
# Area under the curve: 0.674
