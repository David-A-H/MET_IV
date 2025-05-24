
# MAIN TASK:
# Your task is to build a model to predict if a respondent reports any plans to emigrate (Variable emig).

### COMMENCE SETUP ##########################################################################################

# Load Packages
## Import Packages here:
library(tidyverse)
library(tidymodels)
library(caret)
library(cluster)

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

library(doFuture)

set.seed(123)

plan(multisession, workers = parallel::detectCores() - 1)
registerDoFuture()



folds <- vfold_cv(train, v = 10)

rec <- recipe(emig ~ ., data = train) %>%
  step_novel(all_nominal_predictors(), new_level = "__new__") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_nzv(all_predictors()) 


step_normalize(all_numeric_predictors()) %>%


xgb_model <- boost_tree(
  trees = tune(),         
  tree_depth = tune(),    
  learn_rate = tune()     
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(xgb_model)

xgb_grid <- expand_grid(
  trees = c(200, 500, 800),
  tree_depth = c(2, 3, 4),
  learn_rate = c(0.005, 0.01, 0.015)
)


xgb_tuned <- xgb_wf %>%
  tune_grid(
    resamples = folds,
    grid = xgb_grid,
    metrics = metric_set(roc_auc, accuracy, f_meas),
    control   = control_grid(
      verbose      = TRUE,
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


xgb_best_roc <- xgb_tuned %>% 
  select_best(metric= "roc_auc")
xgb_best_a<- xgb_tuned %>% 
  select_best(metric= "accuracy")
xgb_best<- xgb_tuned %>% 
  select_best(metric= "f_meas")


#–– 10. Finalize workflow and fit on full training set
final_wf_xg <- finalize_workflow(xgb_wf, xgb_best)

final_fit_xg <- final_wf_xg %>%
  fit(data = train)

#–– 11. Evaluate on your held-out future test set
preds_x <- predict(final_fit_xg, test, type = "prob") %>%
  bind_cols(predict(final_fit_xg, test)) %>%
  bind_cols(test %>% select(emig))


preds_xg <- preds_x %>%
  mutate(
    emig = factor(emig, levels = c("No", "Yes")),         # truth
    .pred_class = factor(.pred_class, levels = c("No", "Yes"))  # estimate
  )

preds_xg$emig <- factor(preds_xg$emig)

cm <- conf_mat(preds_xg, truth = emig, estimate = .pred_class, 
               mode = "everything") 

# Compute metrics
roc_auc(preds_xg, truth = emig, .pred_class, event_level = "second")
accuracy(preds_xg, truth = emig, .pred_class)
yardstick::precision(preds_xg, truth = emig, estimate = .pred_class, event_level = "second")
yardstick::recall(preds_xg, truth = emig, .pred_class, event_level = "second")
f_meas(preds_xg, truth = emig, .pred_class, event_level = "second")

?recall()

conf_mat_result <- preds_xg %>%
  conf_mat(truth = emig, estimate = .pred_class)


table(preds_xg$emig)
table(preds_xg$.pred_class)


### COMMENCE MODEL COMPARISON AND EXTRACTION #################################################################





