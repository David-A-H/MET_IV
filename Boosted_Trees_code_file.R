
# MAIN TASK:
# Your task is to build a model to predict if a respondent reports any plans to emigrate (Variable emig).

### COMMENCE SETUP ##########################################################################################

# Load Packages
## Import Packages here:
library(tidymodels)
library(themis)        # for sampling steps
library(recipes)       # for pre-processing
library(dplyr)
library(doFuture)
library(mice)   
library(tidyverse)
library(caret)

# Clear Global Environment
rm(list=ls())

# Import Data
data <-  load("final_project/AB4x_train.Rdata")
data <-  train
rm(train)

table(data$emig)

### COMMENCE PRE-PROCESSING #################################################################################

# Locate NA values in the dataset
na_count <- sapply(data, function(y) sum(length(which(is.na(y)))))
na_count[na_count > 0]

# Create backup for comparing dropping NAs and imputing NAs
data_dropped_na <-  na.omit(data)


# Median imputation for 'age'
#median_age <- median(data$age, na.rm = TRUE)
#data$age[is.na(data$age)] <- median_age

# Mode function for factors
#get_mode <- function(x) {
 # ux <- unique(x[!is.na(x)])
 # ux[which.max(tabulate(match(x, ux)))]
#}

# Mode imputation for 'q1002'
#mode_q1002 <- get_mode(data$q1002)
#data$q1002[is.na(data$q1002)] <- mode_q1002


# MICE to fill 'age' and 'q1002' simultaneously
# Use 5 imputations, then pool later

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


# Factorization of emig
data$emig <- factor(data$emig, levels = c("No", "Yes"))

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
# rm(list=ls()[!ls() %in% c("train", "test", "backup", "backup_dropped_na", "eval")])



um_vars <- names(train)[vapply(train, is.numeric, logical(1))]

### COMMENCE PREDICTION #####################################################################################
# NOTE: BEFORE EVERY PREDICTION MODEL, RUN set.seed(123) to reset the RNG.


set.seed(123)
folds <- vfold_cv(train, v = 10)

rec <- recipe(emig ~ ., data = train) %>%
  step_novel(all_nominal_predictors()) %>% # handle novel (or unseen) factor levels in categorical variables. 
  step_zv(all_predictors()) %>% #  removes any predictors that have zero variance
  step_dummy(all_nominal_predictors()) %>%
 # This step performs oversampling (up-sampling) of the minority class in the target variable emig. 
  # It balances the classes by duplicating instances from the minority class to match the size of the majority class.
  # The ratio of the minority class to the majority class is set to 1, 
  #meaning the minority class will be upsampled until it has the same number of instances as the majority class.
 step_upsample(emig, over_ratio = 1) %>% 
  step_downsample(emig, under_ratio = 1) %>%
  step_normalize(all_numeric_predictors())


xgb_spec <- expand_grid(
  trees = c(200, 500, 800),
  tree_depth = c(2, 3, 4),
  learn_rate = c(0.005, 0.01, 0.015)
)

xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  mtry = tune(),               # number of predictors randomly sampled
  min_n = tune(),              # min node size
  loss_reduction = tune(),     # gamma
  sample_size = tune(),        # subsample
  stop_iter = 10               # early stopping rounds
) %>%
  set_engine("xgboost",
             objective = "binary:logistic",
             scale_pos_weight = tune()  # handle imbalance in tree
  ) %>%
  set_mode("classification")


xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost"
  ) %>%
  set_mode("classification")

#xgb_grid <- expand_grid(
  #trees = c(200, 500, 800),
  #tree_depth = c(2, 3, 4),
 # learn_rate = c(0.005, 0.01, 0.015)
#)

#xgb_grid2 <- expand_grid(
  #tree_depth = c(2, 3, 4),
 # trees = c(200, 500, 800),
 # learn_rate = c(0.005, 0.01, 0.015),
 # subsample           = c(0.5, 0.8, 1.0),
 # colsample_bytree    = c(0.5, 0.8, 1.0),
 # min_child_weight    = c(1, 5, 10),
 # gamma               = c(0, 1, 5),
 # size                = 30
#)

sample_prop <- sample_prop(range = c(0.5, 1)) %>% finalize(train)




xgb_grid<- expand_grid(
  trees = c(200, 500, 800),
  tree_depth = c(2, 3, 4),
  learn_rate = c(0.005, 0.01, 0.015)
)

xgb_grid <- grid_latin_hypercube(
  trees(), tree_depth(), learn_rate(), mtry(range = c(5, 20)),
  min_n(), loss_reduction(), sample_prop, scale_pos_weight(range = c(1, 4)),
  size = 30
)

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(xgb_spec)


plan(multisession, workers = parallel::detectCores() - 1)
registerDoFuture()

set.seed(123)
tune_res <- tune_grid(
  wf,
  resamples = folds,
  grid = xgb_grid,
  metrics = metric_set(roc_auc, pr_auc, accuracy, f_meas),
  control   = control_grid(
    verbose      = FALSE,
    save_pred    = TRUE
    # with future, parallelism is automatic under the hood
  )
)

plan(sequential)


tune_res %>% 
  show_best(metric= "roc_auc", n = 10)
tune_res %>% 
  show_best(metric= "accuracy", n = 10)
tune_res %>% 
  show_best(metric= "f_meas", n = 10)
tune_res %>% 
  show_best(metric= "pr_auc", n = 10)

best <- tune_res %>% 
  select_best(metric= "pr_auc")

#–– 10. Finalize workflow and fit on full training set
final_wf <- finalize_workflow(wf, best)
final_fit <- final_wf %>%
  fit(data = train)



er <- tune_res %>% 
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
preds_x <- predict(final_fit, test, type = "prob") %>%
  bind_cols(predict(final_fit, test)) %>%
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

preds_xg  <- preds_x %>%
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

cm2 <- conf_mat(preds_xg, truth = emig, estimate = .pred_class, 
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





