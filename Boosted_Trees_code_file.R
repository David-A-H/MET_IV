
# MAIN TASK:
# Your task is to build a model to predict if a respondent reports any plans to emigrate (Variable emig).

### COMMENCE SETUP ##########################################################################################

# Load Packages
## Import Packages here:
library(tidyverse)
library(tidymodels)
library(caret)
library(cluster)
library(VIM)
library(themis)

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




#################################################################################



md.pattern(data)

init <- mice(train, maxit = 0, printFlag = FALSE)

typeof(data)
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

# Split Data into Train and Test and Stratify for emig

data$emig <- factor(data$emig, levels = c("No", "Yes"))


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


train_imp <- kNN(train,
                 variable = c("age", "q1002"),
                 k        = 5,
                 weightDist = TRUE,
                 dist_var   = setdiff(names(train), c("age","q1002")),
                 imp_var    = FALSE)



um_vars <- names(train)[vapply(train, is.numeric, logical(1))]

### COMMENCE PREDICTION #####################################################################################
# NOTE: BEFORE EVERY PREDICTION MODEL, RUN set.seed(123) to reset the RNG.

library(doFuture)


plan(multisession, workers = parallel::detectCores() - 1)
registerDoFuture()

set.seed(123)
folds <- vfold_cv(train, v = 10)

knn_rec <- recipe(emig ~ ., data = train) %>%
  step_novel(all_nominal_predictors(), new_level = "__new__") %>%
  step_impute_knn(all_predictors(), neighbors = 5) %>%
  step_nzv(all_predictors()) %>% 
  step_corr(all_numeric(), threshold = 0.9) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_zv() %>% 
  step_smote(all_outcomes(), over_ratio = 1) %>%
  step_normalize(all_numeric_predictors())

knn_rec <- recipe(emig ~ ., data = train) %>%
  step_novel(all_nominal_predictors(), new_level = "__new__") %>%
  step_nzv(all_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv() %>% 
  step_smote(all_outcomes(), over_ratio = 1) %>%
  step_normalize(all_numeric_predictors())


xgb_model <- boost_tree(
  trees = tune(),         
  tree_depth = tune(),    
  learn_rate = tune()     
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")


xgb_model2 <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost"
  ) %>%
  set_mode("classification")

xgb_grid <- expand_grid(
  trees = c(200, 500, 800),
  tree_depth = c(2, 3, 4),
  learn_rate = c(0.005, 0.01, 0.015)
)

xgb_grid2 <- expand_grid(
  trees = c(200, 500, 800),
  tree_depth = c(2, 3, 4),
  learn_rate = c(0.005, 0.01, 0.015),
  subsample           = c(0.5, 0.8, 1.0),
  colsample_bytree    = c(0.5, 0.8, 1.0),
  min_child_weight    = c(1, 5, 10),
  gamma               = c(0, 1, 5),
  size                = 30
)

xgb_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(xgb_model)

xgb_wf2 <- workflow() %>%
  add_recipe(knn_rec) %>%
  add_model(xgb_model2)



xgb_tuned <- xgb_wf %>%
  tune_grid(
    resamples = folds,
    grid = xgb_grid,
    metrics = metric_set(roc_auc, accuracy, f_meas),
    control   = control_grid(
      verbose      = FALSE,
      save_pred    = TRUE
      # with future, parallelism is automatic under the hood
    )
  )


xgb_tuned2 <- xgb_wf2 %>%
  tune_grid(
    resamples = folds,
    grid = xgb_grid,
    metrics = metric_set(roc_auc, accuracy, f_meas),
    control   = control_grid(
      verbose      = FALSE,
      save_pred    = TRUE
      # with future, parallelism is automatic under the hood
    )
  )


plan(sequential)


xgb_tuned %>% 
  show_best(metric= "roc_auc", n = 10)
xgb_tuned %>% 
  show_best(metric= "accuracy", n = 10)
xgb_tuned %>% 
  show_best(metric= "f_meas", n = 10)


xgb_tuned2 %>% 
  show_best(metric= "roc_auc", n = 10)
xgb_tuned2 %>% 
  show_best(metric= "accuracy", n = 10)
xgb_tuned2 %>% 
  show_best(metric= "f_meas", n = 10)


xgb_best_roc <- xgb_tuned %>% 
  select_best(metric= "roc_auc")
xgb_best_a<- xgb_tuned %>% 
  select_best(metric= "accuracy")
xgb_best<- xgb_tuned %>% 
  select_best(metric= "f_meas")

xgb_best_roc <- xgb_tuned2 %>% 
  select_best(metric= "f_meas")


#–– 10. Finalize workflow and fit on full training set
final_wf_xg <- finalize_workflow(xgb_wf, xgb_best)
final_wf_xg2 <- finalize_workflow(xgb_wf2, xgb_best_roc)

final_fit_xg <- final_wf_xg %>%
  fit(data = train)

final_fit_xg2 <- final_wf_xg2 %>%
  fit(data = train)

final_fit_xg <- final_wf_xg %>%
  fit(data = train)

er <- xgb_tuned2 %>% 
  collect_predictions()

roc_data <- er %>%
  roc_curve(truth = emig, .pred_Yes) %>%
  filter(is.finite(.threshold)) %>%     # drop Inf / -Inf
  mutate(
    youden = sensitivity + specificity - 1
  )

th_equal <- roc_data %>%
  mutate(diff = abs(sensitivity - specificity)) %>%
  slice_min(diff, n = 1) %>%
  pull(.threshold)


#–– 11. Evaluate on your held-out future test set
preds_x <- predict(final_fit_xg, test, type = "prob") %>%
  bind_cols(predict(final_fit_xg, test)) %>%
  bind_cols(test %>% select(emig))

preds2 <- predict(final_fit_xg2, new_data = test, type = "prob") %>% 
  collect_predictions()

preds2 <- predict(final_fit_xg2, test, type = "prob") %>%
  bind_cols(predict(final_fit_xg2, test)) %>%
  bind_cols(test %>% select(emig)) 



preds_xg <- preds_x %>%
  mutate(
    emig = factor(emig, levels = c("No", "Yes")),         # truth
    .pred_class = factor(.pred_class, levels = c("No", "Yes"))  # estimate
  )

preds2x <- preds2 %>%
  mutate(
    emig = factor(emig, levels = c("No", "Yes")),  
    .pred_class = if_else(.pred_Yes >= th_equal, "Yes", "No"),# truth
    .pred_class = factor(.pred_class, levels = c("No", "Yes")),
    # estimate
  )

preds_xg$emig <- factor(preds_xg$emig)
preds2$emig <- factor(preds2$emig)
preds2$.pred_class <- factor(preds2$.pred_class)

cm <- conf_mat(preds_xg, truth = emig, estimate = .pred_class, 
               mode = "everything") 

cm2 <- conf_mat(preds2x, truth = emig, estimate = .pred_class, 
               mode = "everything") 

cm3 <- conf_mat(preds2x, truth = emig, estimate = .pred_class, 
                mode = "everything") 

cm2
cm3

# Compute metrics
roc_auc(preds_xg, truth = emig, .pred_class, event_level = "second")
accuracy(preds_xg, truth = emig, .pred_class)
yardstick::precision(preds_xg, truth = emig, estimate = .pred_class, event_level = "second")
yardstick::recall(preds_xg, truth = emig, .pred_class, event_level = "second")
f_meas(preds_xg, truth = emig, .pred_class, event_level = "second")


roc_auc(preds2x, truth = emig, .pred_class, event_level = "second")
accuracy(preds2x, truth = emig, .pred_class)
yardstick::precision(preds2x, truth = emig, estimate = .pred_class, event_level = "second")
yardstick::recall(preds2x, truth = emig, .pred_class, event_level = "second")
f_meas(preds2x, truth = emig, .pred_class, event_level = "second")

?recall()

conf_mat_result <- preds_xg %>%
  conf_mat(truth = emig, estimate = .pred_class)


table(preds_xg$emig)
table(preds_xg$.pred_class)

collect_metrics()


### COMMENCE MODEL COMPARISON AND EXTRACTION #################################################################





