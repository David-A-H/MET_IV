
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

#data <- data %>%
# mutate(
#  any_imputed = as.integer(is.na(age) | is.na(q1002))
# )




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

#################################################################################

# Overall Summary
#summary(data)
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
#table(data$emig)
## NOTE: Class imbalance is present. 1 = emigration, 0 = no emigration

# Check for class imbalance in other variables
#hist(data$age)
#hist(as.numeric(data$income), breaks = 17)
#hist(as.numeric(data$edu_combined))

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
#prop.table(table(train$emig))
#prop.table(table(test$emig))

# Create backup data file
#backup <-  data
#backup_dropped_na <- data_dropped_na

# # Load evaluation data (Used for evaluating the model at the very end)
# eval <- load("final_project/AB4x_eval_mock.Rdata")

# Clear global environment except for "train", "test", "backup" and "eval"
# rm(list=ls()[!ls() %in% c("train", "test", "backup", "backup_dropped_na", "eval")])

# Create Folds
set.seed(123)
folds <- vfold_cv(data, v = 10)


### Stage 1) Model Comparison #####################################################################################
# NOTE: BEFORE EVERY PREDICTION MODEL, RUN set.seed(123) to reset the RNG.


# Specify recipe for tidymodels
#rec <- recipe(emig ~ ., data = train) %>% 
# step_novel(all_nominal_predictors()) %>% # Handle novel levels in categorical predictors during prediction
# step_zv(all_predictors()) %>% # Remove predictors that have zero variance 
# step_dummy(all_nominal_predictors()) %>% # Convert categorical predictors into dummy/one-hot
# step_smote(emig) %>%  # Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes of the outcome variable
# step_normalize(all_numeric_predictors())  # Normalize (center and scale) all numeric predictors to have mean 0 and standard deviation 1 (important when applying Cross-validation)


# Define a random forest model specification with tunable hyperparameters
#rf_spec <- rand_forest(
# mtry  = tune(),  # number of predictors randomly sampled at each split is set to be tuned
# trees = tune(),  # total number of trees in the forest is set to be tuned
# min_n = tune()   # minimum number of data points in a node for a split to be attempted is to be tuned
#) %>%
#  set_engine("ranger") %>%           # Use the 'ranger' engine for efficient random forest implementation
#  set_mode("classification")         # Set the model's mode to classification (for predicting categorical outcomes)


# Create a modeling workflow by combining preprocessing and model specification
#wf <- workflow() %>%
#  add_recipe(rec) %>%  # Add the preprocessing recipe  to the workflow
#  add_model(rf_spec)# Add the random forest model specification  to the workflow


#rf_grid <- grid_space_filling(
# mtry(range = c(20, 100)),  # tune 'mtry' between range of 20 and 50
# trees(range = c(500, 2000)),  # tune 'trees' between range of 600 and 1800
# min_n(range = c(1, 10)),  # tune 'min_n' between rang 3 and 9
# size = 200)  # Generate 50 hyperparameter combinations


#rf_grid <- grid_space_filling(
#  mtry(range = c(20, 50)),  # tune 'mtry' between range of 20 and 50
#  trees(range = c(600, 1800)),  # tune 'trees' between range of 600 and 1800
#  min_n(range = c(3, 9)),  # tune 'min_n' between rang 3 and 9
#  size = 200)  # Generate 50 hyperparameter combinations



# Parallel processing (leaving 1 core free)
#plan(multisession, workers = parallel::detectCores() - 1)
#registerDoFuture()


# Perform hyperparameter tuning using grid search
#set.seed(123)
#rf_tune <- tune_grid(
#  wf,                                    
#  resamples = folds,                     
#  grid      = rf_grid,                   # A predefined grid of hyperparameter values to search over
#  metrics   = metric_set(                # Set of performance metrics to evaluate for each combination
#   roc_auc,                             # Area Under the ROC Curve 
#    accuracy,                            # Overall classification accuracy
#    f_meas,                              # F1-score 
#    pr_auc                               # Area Under the Precision-Recall Curve 
#  ),
#  control = control_grid(               # Additional control options for the tuning process
#    save_pred = TRUE,                   # Save the predictions for each resample/fold 
#    verbose = FALSE                     # Suppress verbose output 
#  )
#)



#plan(sequential) # Undo parallel processing 



###########################################

# Show top ten performing models by specified metrics
#rf_tune |> 
# show_best(metric="roc_auc",n=10)
#rf_tune |> 
# show_best(metric="f_meas",n=10)
#rf_tune |> 
# show_best(metric="accuracy",n=10)
  #rf_tune |> 
 # show_best(metric="pr_auc",n=10)

# Choosing preferred model
#best_roc <- rf_tune %>%
 # select_best(metric="roc_auc")

#final_wf_rpc <- finalize_workflow(wf, best_roc) 


# Fit on (test) data
#final_fit1 <- final_wf_rpc %>%
 # fit(data = train)


###########################################
# Predict on Hold out

#preds <- predict(final_fit1, test, type = "prob") %>%
 # bind_cols(predict(final_fit1, test)) %>%
  #bind_cols(test %>% select(emig))


#preds_r <- preds %>%
 # mutate(
  #  emig = factor(emig, levels = c("No", "Yes")),      
   # .pred_class = factor(.pred_class, levels = c("No", "Yes"))  # estimate
  #)
# Checking alternate thresholds 
#preds_f <- preds %>%
#  mutate(
#    emig = factor(emig, levels = c("No", "Yes")),      
#    .pred_class = if_else(.pred_Yes >= 0.35, "Yes", "No"),
#    .pred_class = factor(.pred_class, levels = c("No", "Yes"))  # estimate
#  )


# Checking metrics
#roc_auc(preds_r,
 #       truth       = emig,
  #      .pred_Yes,               # <-- numeric probability
   #     event_level = "second")

#accuracy(preds_r, truth = emig, .pred_class)
#yardstick::precision(preds_r, truth = emig, .pred_class, event_level = "second")
#yardstick::recall(preds_r, truth = emig, .pred_class, event_level = "second")
#f_meas(preds_r, truth = emig, .pred_class, event_level = "second")

#roc_auc(preds_f,
#        truth       = emig,
#        .pred_Yes,               # <-- numeric probability
#        event_level = "second")

#accuracy(preds_f, truth = emig, .pred_class)
#yardstick::precision(preds_f, truth = emig, .pred_class, event_level = "second")
#yardstick::recall(preds_f, truth = emig, .pred_class, event_level = "second")
#f_meas(preds_f, truth = emig, .pred_class, event_level = "second")


# Comfusion Matrix
#cm1<- preds_r %>%
#  conf_mat(truth = emig, estimate = .pred_class)

#cm2<- preds_f %>%
# conf_mat(truth = emig, estimate = .pred_class)

#cm1
#cm2

### Stage 2) Random Forest Tuning #####################################################################################


# Specify recipe for tidymodels
#rec <- recipe(emig ~ ., data = data) %>% 
 # step_novel(all_nominal_predictors()) %>% # Handle novel levels in categorical predictors during prediction
# step_zv(all_predictors()) %>% # Remove predictors that have zero variance 
# step_dummy(all_nominal_predictors()) %>% # Convert categorical predictors into dummy/one-hot
# step_smote(emig) %>%  # Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes of the outcome variable
# step_normalize(all_numeric_predictors())  # Normalize (center and scale) all numeric predictors to have mean 0 and standard deviation 1 (important when applying Cross-validation)


# Define a random forest model specification with tunable hyperparameters
#rf_spec <- rand_forest(
# mtry  = tune(),  # number of predictors randomly sampled at each split is set to be tuned
 # trees = tune(),  # total number of trees in the forest is set to be tuned
# min_n = tune()   # minimum number of data points in a node for a split to be attempted is to be tuned
#) %>%
# set_engine("ranger") %>%           # Use the 'ranger' engine for efficient random forest implementation
# set_mode("classification")         # Set the model's mode to classification (for predicting categorical outcomes)


# Create a modeling workflow by combining preprocessing and model specification
#wf <- workflow() %>%
# add_recipe(rec) %>%  # Add the preprocessing recipe  to the workflow
# add_model(rf_spec)# Add the random forest model specification  to the workflow


#rf_grid <- grid_space_filling(
# mtry(range = c(20, 100)),  # tune 'mtry' between range of 20 and 100
# trees(range = c(500, 2000)),  # tune 'trees' between range of 500 and 2000
# min_n(range = c(1, 10)),  # tune 'min_n' between rang 1 and 10
# size = 200)  # Generate 200 hyperparameter combinations


# Parallel processing (leaving 1 core free)
#plan(multisession, workers = parallel::detectCores() - 1)
#registerDoFuture()


# Perform hyperparameter tuning using grid search
# set.seed(123)
# rf_tune <- tune_grid(
#   wf,                                    
#   resamples = folds,                     
#   grid      = rf_grid,                   # A predefined grid of hyperparameter values to search over
#   metrics   = metric_set(                # Set of performance metrics to evaluate for each combination
#     roc_auc,                             # Area Under the ROC Curve 
#     accuracy,                            # Overall classification accuracy
#     f_meas,                              # F1-score 
#     pr_auc                               # Area Under the Precision-Recall Curve 
#   ),
#   control = control_grid(               # Additional control options for the tuning process
#     save_pred = TRUE,                   # Save the predictions for each resample/fold 
#     verbose = FALSE                     # Suppress verbose output 
#   )
# )

# plan(sequential) # Undo parallel processing 
# 
# rf_tune |> 
#   show_best(metric="roc_auc",n=10)
# rf_tune |> 
#   show_best(metric="f_meas",n=10)

### Stage 3) Narrowing Range based on tuning and specifying final model #####################################################################################
# Specify recipe for tidymodels

rec <- recipe(emig ~ ., data = data) %>% 
  step_novel(all_nominal_predictors()) %>% # Handle novel levels in categorical predictors during prediction
  step_zv(all_predictors()) %>% # Remove predictors that have zero variance 
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
  mtry(range = c(80, 110)),  # tune 'mtry' between range of 80 and 110
  trees(range = c(800, 2000)),  # tune 'trees' between range of 800 and 2000
  min_n(range = c(2, 8)),  # tune 'min_n' between rang 2 and 8
  size = 100)  # Generate 100 hyperparameter combinations

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

rf_tune |> 
  show_best(metric="roc_auc",n=10)

# Choose preferred model
best_roc <- rf_tune %>%
  select_best(metric="roc_auc")

# Finalize workflow
final_wf_roc <- finalize_workflow(wf, best_roc)

# Fit on data
final_fit1 <- final_wf_roc %>%
  fit(data = data)

# Save Model
saveRDS(final_fit1, file = "finished model.rds")

### Stage 4) Re-run Model specifications on 80/20 split to check models predictive abiltiy on hold-out#####################################################################################
# 
# rec2 <- recipe(emig ~ ., data = train) %>% 
#   step_novel(all_nominal_predictors()) %>% # Handle novel levels in categorical predictors during prediction
#   step_zv(all_predictors()) %>% # Remove predictors that have zero variance 
#   step_dummy(all_nominal_predictors()) %>% # Convert categorical predictors into dummy/one-hot
#   step_smote(emig) %>%  # Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes of the outcome variable
#   step_normalize(all_numeric_predictors())  # Normalize (center and scale) all numeric predictors to have mean 0 and standard deviation 1 (important when applying Cross-validation)
# 
# 
# # Define a random forest model specification with tunable hyperparameters
# rf_spec <- rand_forest(
#   mtry  = tune(),  # number of predictors randomly sampled at each split is set to be tuned
#   trees = tune(),  # total number of trees in the forest is set to be tuned
#   min_n = tune()   # minimum number of data points in a node for a split to be attempted is to be tuned
# ) %>%
#   set_engine("ranger") %>%           # Use the 'ranger' engine for efficient random forest implementation
#   set_mode("classification")         # Set the model's mode to classification (for predicting categorical outcomes)
# 
# 
# # Create a modeling workflow by combining preprocessing and model specification
# wf2 <- workflow() %>%
#   add_recipe(rec2) %>%  # Add the preprocessing recipe  to the workflow
#   add_model(rf_spec)# Add the random forest model specification  to the workflow
# 
# # Select Grid using hyperparameters of top two performing models in previous step
# test_grid <- expand_grid(
#   mtry = c(95, 110),
#   trees = c(836, 1333),
#   min_n = c(3, 7)
# )
# 
# set.seed(123)
# zu <- tune_grid(
#   wf2,                                    
#   resamples = folds,                     
#   grid      = test_grid,                  
#   metrics   = metric_set(               
#     roc_auc,                           
#     accuracy,                            
#     f_meas,                              
#     pr_auc                               
#   ),
#   control = control_grid(            
#     save_pred = TRUE,                   
#     verbose = FALSE                     
#   )
# )
# 
# zu |> 
#   show_best(metric="roc_auc",n=10)
# 
# # Choose best model from Stage 3
# x <- zu %>%
#   collect_metrics()%>%
#   filter(mtry == 95, trees == 836, min_n == 3, .metric == "roc_auc")
# 
# tt <- finalize_workflow(wf2, x)
# tt2 <- fgfg %>%
#   fit(data = train)
# 
# preds2 <- predict(tt2, test, type = "prob") %>%
#   bind_cols(predict(tt2, test)) %>%
#   bind_cols(test %>% select(emig))
# 
# preds_f <- preds2 %>%
#   mutate(
#     emig = factor(emig, levels = c("No", "Yes")),         # truth
#     .pred_class = factor(.pred_class, levels = c("No", "Yes"))  # estimate
#   )
# 
# 
# roc_auc(preds_f,
#         truth       = emig,
#         .pred_Yes,              
#         event_level = "second")
# 
# accuracy(preds_f, truth = emig, .pred_class)
# yardstick::precision(preds_f, truth = emig, .pred_class, event_level = "second")
# yardstick::recall(preds_f, truth = emig, .pred_class, event_level = "second")
# f_meas(preds_f, truth = emig, .pred_class, event_level = "second")


### Stage A) Images #####################################################################################

# preds_tbl <- collect_predictions(rf_tune)
# 
# residuals_tbl <- preds_tbl %>%
#   mutate(
#     # assuming “1” is the positive class
#     pred_prob = .pred_Yes,
#     truth_num = as.numeric(emig) - 1,   # convert factor to 0/1
#     residual  = truth_num - pred_prob
#   )
# 
# residuals_tbl %>% 
#   ggplot(aes(residual, fill = id)) +
#   geom_density(alpha = 0.3) +
#   labs(title = "Residuals by CV Fold")


# # Compute ROC curve data
# roc_data <- roc_curve(preds2, truth = emig, .pred_Yes, event_level = "second")
# # Plot
# autoplot(roc_data) +
#   ggtitle("ROC Curve: Random Forest on Test Set") +
#   theme_minimal()

