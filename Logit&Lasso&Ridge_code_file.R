
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


# Convert emig to factor
data$emig <- as.factor(ifelse(data$emig == "Yes", 1, 0))  # Convert to binary factor (1 = emigration, 0 = no emigration)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
  step_dummy(all_nominal_predictors())                # Convert factors to dummies

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

# Combine predictions with actuals for ROC analysis
logistic_results <- bind_cols(test, logistic_pred_prob) %>%
  mutate(emig = factor(emig, levels = c("1", "0")))  # Ensure positive class is first

# ROC and AUC
roc_data <- roc_curve(logistic_results, truth = emig, .pred_1)
auc_value <- roc_auc(logistic_results, truth = emig, .pred_1)
print(auc_value)

# ROC Curve
ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "blue", size = 1.2) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(
    title = "ROC Curve - Logistic Regression",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  ) +
  theme_minimal()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LASSO REGRESSION
set.seed(123)

# Define recipe
lasso_recipe <- recipe(emig ~ ., data = train) %>%
  step_naomit(all_predictors(), all_outcomes()) %>%
  step_nzv(all_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

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

# Prepare data for ROC
lasso_results <- bind_cols(test, lasso_pred_prob) %>%
  mutate(emig = factor(emig, levels = c("1", "0")))  # Ensure correct order

# ROC and AUC
lasso_roc_data <- roc_curve(lasso_results, truth = emig, .pred_1)
lasso_auc_value <- roc_auc(lasso_results, truth = emig, .pred_1)
print(lasso_auc_value)

# Plot ROC Curve
ggplot(lasso_roc_data, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "blue", size = 1.2) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(
    title = "ROC Curve - Lasso Logistic Regression",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  ) +
  theme_minimal()




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# RIDGE REGRESSION
set.seed(123)

# Define recipe
ridge_recipe <- recipe(emig ~ ., data = train) %>%
  step_naomit(all_predictors(), all_outcomes()) %>%
  step_nzv(all_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

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

# Prepare data for ROC
ridge_results <- bind_cols(test, ridge_pred_prob) %>%
  mutate(emig = factor(emig, levels = c("1", "0")))  # Ensure correct order

# ROC and AUC
ridge_roc_data <- roc_curve(ridge_results, truth = emig, .pred_1)
ridge_auc_value <- roc_auc(ridge_results, truth = emig, .pred_1)
print(ridge_auc_value)

# Plot ROC Curve
ggplot(ridge_roc_data, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "blue", size = 1.2) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(
    title = "ROC Curve - ridge Logistic Regression",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  ) +
  theme_minimal()






### COMMENCE MODEL COMPARISON AND EXTRACTION #################################################################


# Compare models based on AUC
model_comparison <- tibble(
  Model = c("Logistic Regression", "Lasso Regression", "Ridge Regression"),
  AUC = c(auc_value$.estimate, lasso_auc_value$.estimate, ridge_auc_value$.estimate)
)
print(model_comparison)



# Results of this
# Model                 AUC
# 1 Logistic Regression 0.677
# 2 Lasso Regression    0.727
# 3 Ridge Regression    0.715