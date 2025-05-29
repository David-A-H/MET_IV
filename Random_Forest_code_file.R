
# MAIN TASK:
# Your task is to build a model to predict if a respondent reports any plans to emigrate (Variable emig).

### COMMENCE SETUP ##########################################################################################

# Load Packages
## Import Packages here:
library(tidymodels)
library(themis)  # for sampling steps
library(recipes) # for pre-processing
library(dplyr)
library(doFuture) # for Parallel processing
library(mice) # for imputation
library(caret)


# Clear Global Environment
rm(list=ls())

# Import Data
data <-  load("final_project/AB4x_train.Rdata")
data <-  train
rm(train)

typeof(data$q261a1)

class(data$q261a1)
### COMMENCE PRE-PROCESSING #################################################################################

# Locate NA values in the dataset
#na_count <- sapply(data, function(y) sum(length(which(is.na(y)))))
#na_count[na_count > 0]

# Create backup for comparing dropping NAs and imputing NAs
#data_dropped_na <-  na.omit(data)




# Median imputation for 'age'
#median_age <- median(data$age, na.rm = TRUE)
#data$age[is.na(data$age)] <- median_age

# Mode function for factors
#get_mode <- function(x) {
#  ux <- unique(x[!is.na(x)])
#  ux[which.max(tabulate(match(x, ux)))]
#}

# Mode imputation for 'q1002'
#mode_q1002 <- get_mode(data$q1002)
#data$q1002[is.na(data$q1002)] <- mode_q1002


# MICE to fill 'age' and 'q1002' simultaneously
# Generate different 5 imputations
mice_init <- mice(data %>% select(age, q1002),
                  m = 5, # Number of imputed datasets to create
                  method = c(age = "pmm", q1002 = "logreg"),  # Use Predictive Mean Matching (PMM) for'age' and logistic regression (logreg) for 'q1002'
                  seed = 123,
                  parallel = "multicore",  # Enable parallel processing using multiple cores
                  ncore = parallel::detectCores()-1) # Use all but one of the available CPU cores for parallel processing

# Extract one of the five complete datasets
data_imp <- complete(mice_init, action = 1)

# Replace into original
data <- data %>%
  select(-age, -q1002) %>%
  bind_cols(data_imp)

table(data$q1002)

#################################################################################

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
  
#################################################################################

# Check for class imbalance
table(data$emig)
## NOTE: Class imbalance is present. 1 = emigration, 0 = no emigration

# Check for class imbalance in other variables
hist(data$age)
hist(as.numeric(data$income), breaks = 17)
hist(as.numeric(data$edu_combined))

# NOTE: Might want to think about rescaling some values like age. Slight skewness towards younger respondents

#################################################################################


# Turn 'emig' from character into factor variable
data$emig <- factor(data$emig, levels = c ("No", "Yes")) 

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


# Create Folds
set.seed(123)
folds <- vfold_cv(train, v = 10)

# Specify recipe for tidymodels
rec <- recipe(emig ~ ., data = train) %>% 
  step_novel(all_nominal_predictors()) %>% # Handle novel levels in categorical predictors during prediction
  step_zv(all_predictors()) %>% # Remove predictors that have zero variance (does not seem to be entirely working as intended)
  step_dummy(all_nominal_predictors()) %>% # Convert categorical predictors into dummy/one-hot
  step_smote(emig) %>%  # Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes of the outcome variable
  step_normalize(all_numeric_predictors())  # Normalize (center and scale) all numeric predictors to have mean 0 and standard deviation 1 (important when applying Cross-validation)


# Define a random forest model specification with tunable hyperparameters
rf_spec <- rand_forest(
  mtry  = tune(),  # number of predictors randomly sampled at each split is set to be tuned
  trees = tune(),  # total number of trees in the forest is set to be tuned
  min_n = tune()   # minimum number of data points in a node for a split to be attempted is to be tuned
) %>%
  set_engine("ranger") %>%           # Use the 'ranger' engine for efficient random forest implementation
  set_mode("classification")         # Set the model's mode to classification (for predicting categorical outcomes)


# Create a modeling workflow by combining preprocessing and model specification
wf <- workflow() %>%
  add_recipe(rec) %>%  # Add the preprocessing recipe  to the workflow
  add_model(rf_spec)# Add the random forest model specification  to the workflow


rf_grid <- grid_space_filling(
  mtry(range = c(20, 50)),  # tune 'mtry' between range of 20 and 50
  trees(range = c(600, 1800)),  # tune 'trees' between range of 600 and 1800
  min_n(range = c(3, 9)),  # tune 'min_n' between rang 3 and 9
  size = 50)  # Generate 50 hyperparameter combinations


# Parallel processing (leaving 1 core free)
plan(multisession, workers = parallel::detectCores() - 1)
registerDoFuture()


# Perform hyperparameter tuning using grid search
set.seed(123)
rf_tune <- tune_grid(
  wf,                                    
  resamples = folds,                     
  grid      = rf_grid,                   # A predefined grid of hyperparameter values to search over
  metrics   = metric_set(                # Set of performance metrics to evaluate for each combination
    roc_auc,                             # Area Under the ROC Curve 
    accuracy,                            # Overall classification accuracy
    f_meas,                              # F1-score 
    pr_auc                               # Area Under the Precision-Recall Curve 
  ),
  control = control_grid(               # Additional control options for the tuning process
    save_pred = TRUE,                   # Save the predictions for each resample/fold 
    verbose = FALSE                     # Suppress verbose output 
  )
)


plan(sequential) # Undo parallel processing 


###########################################

# Show top ten performing models by specified metrics
rf_tune |> 
  show_best(metric="roc_auc",n=10)
rf_tune |> 
  show_best(metric="f_meas",n=10)
rf_tune |> 
  show_best(metric="accuracy",n=10)
rf_tune |> 
  show_best(metric="pr_auc",n=10)

# Choose preferred model
best_roc <- rf_tune %>%
  select_best(metric="roc_auc")
best_pr_auc <- rf_tune %>%
  select_best(metric="pr_auc")

# Finalize workflow
final_wf_rpc <- finalize_workflow(wf, best_roc)
final_wf_f <- finalize_workflow(wf, pr_auc)

# Fit on (test) data
final_fit1 <- final_wf_rpc %>%
  fit(data = train)

final_fit2 <- final_wf_f %>%
  fit(data = train)

# Extract all saved predictions from the tuning results
er <-rf_tune %>% collect_predictions()

# Generate data for plotting and analyzing the ROC curve (alternatively, apply pr_curve() for PR curve)
roc_data <- er %>%
  roc_curve(truth = emig, .pred_Yes) %>%# Compute ROC curve data: true labels vs. predicted probabilities for "Yes" class
  filter(is.finite(.threshold)) %>%     # Remove rows with infinite threshold values (can appear at extremes)
  mutate(
    youden = sensitivity + specificity - 1 # Calculate Youden's J statistic for each threshold
  )

# Find the threshold where sensitivity and specificity are most balanced
th_equal <- roc_data %>%
  mutate(diff = abs(sensitivity - specificity)) %>% # Calculate absolute difference between sensitivity and specificity
  slice_min(diff, n = 1) %>% # Select the threshold with the smallest difference (i.e., most equal)
  pull(.threshold) # Extract the corresponding threshold value



###########################################
# Predict on Hold out

preds <- predict(final_fit1, test, type = "prob") %>%
  bind_cols(predict(final_fit1, test)) %>%
  bind_cols(test %>% select(emig))

preds2 <- predict(final_fit2, test, type = "prob") %>%
  bind_cols(predict(final_fit2, test)) %>%
  bind_cols(test %>% select(emig))


preds_r <- preds %>%
  mutate(
    emig = factor(emig, levels = c("No", "Yes")),      
    .pred_class = factor(.pred_class, levels = c("No", "Yes"))  # estimate
  )

preds_r <- preds %>%
  mutate(
    emig = factor(emig, levels = c("No", "Yes")),      
    .pred_class = if_else(.pred_Yes >= th_equal, "Yes", "No"),
    .pred_class = factor(.pred_class, levels = c("No", "Yes"))  # estimate
  )

preds_f <- preds2 %>%
  mutate(
    emig = factor(emig, levels = c("No", "Yes")),         # truth
    .pred_class = factor(.pred_class, levels = c("No", "Yes"))  # estimate
  )

preds_f <- preds %>%
  mutate(
    emig = factor(emig, levels = c("No", "Yes")),      
    .pred_class = if_else(.pred_Yes >= th_equal, "Yes", "No"),
    .pred_class = factor(.pred_class, levels = c("No", "Yes"))  # estimate
  )



roc_auc(preds_r,
        truth       = emig,
        .pred_Yes,               # <-- numeric probability
        event_level = "second")

accuracy(preds_r, truth = emig, .pred_class)
yardstick::precision(preds_r, truth = emig, .pred_class, event_level = "second")
yardstick::recall(preds_r, truth = emig, .pred_class, event_level = "second")
f_meas(preds_r, truth = emig, .pred_class, event_level = "second")

roc_auc(preds_f,
        truth       = emig,
        .pred_Yes,               # <-- numeric probability
        event_level = "second")

accuracy(preds_f, truth = emig, .pred_class)
yardstick::precision(preds_f, truth = emig, .pred_class, event_level = "second")
yardstick::recall(preds_f, truth = emig, .pred_class, event_level = "second")
f_meas(preds_f, truth = emig, .pred_class, event_level = "second")


cm1<- preds_r %>%
  conf_mat(truth = emig, estimate = .pred_class)

cm2<- preds_f %>%
  conf_mat(truth = emig, estimate = .pred_class)

cm1
cm2

### COMMENCE MODEL COMPARISON AND EXTRACTION #################################################################




