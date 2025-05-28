
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
library(mice)

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


# NA imputation
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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




# 1 Class duplicating (because reweighing didn't work)
## show current imbalance
table(train$emig) # 0 = 0.85, 1 = 0.15. Aim is to upsample 3368 1 observations

### Set seed again
set.seed(123)

## Upsample the minority class (1) to match the majority class (0)
train <- train %>%
  group_by(emig) %>%
  slice_sample(n = max(table(train$emig)), replace = TRUE) %>%
  ungroup()

## Check the new class distribution
table(train$emig) # 0 = 0.5, 1 = 0.5. Now balanced
table(test$emig)




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





### COMMENCE MODEL COMPARISON AND EXTRACTION #################################################################





