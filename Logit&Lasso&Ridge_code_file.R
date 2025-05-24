
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
# Logistic Regression on all features as a baseline

set.seed(123)


# Define the formula for the model
formula <- emig ~ .

# Define the logistic regression model
logistic_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")  # Set to classification mode

# Create the workflow and add the model and formula
logistic_wf <- workflow() %>%
  add_model(logistic_model) %>%
  add_formula(formula)

# Fit the model to the training data
logistic_fit <- logistic_wf %>%
  fit(data = train)

# Generate predictions on the test set
logistic_pred      <- predict(logistic_fit, new_data = test, type = "class")
logistic_pred_prob <- predict(logistic_fit, new_data = test, type = "prob")

# Evaluate model performance using a confusion matrix
logistic_result <- confusionMatrix(logistic_pred$.pred_class, test$emig, positive = "1")
logistic_result


# VISUALISATION
# Combine predicted probabilities with true labels
logistic_results <- bind_cols(test, logistic_pred_prob) %>%
  mutate(emig = factor(emig, levels = c("1", "0")))  # Ensure positive class is first

# Calculate ROC and AUC
roc_data <- roc_curve(logistic_results, truth = emig, .pred_1)
auc_value <- roc_auc(logistic_results, truth = emig, .pred_1)
print(auc_value)

# Plot ROC curve
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


# create x_train, x_test, y_train, y_test from train and test dataset
x_train <- model.matrix(emig ~ ., data = train)[, -1]  # Remove intercept
x_test <- model.matrix(emig ~ ., data = test)[, -1]  # Remove intercept
y_train <- train$emig
y_test <- test$emig


# Perform cross-validated lasso logistic regression
cv.lasso <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")

# Plot the cross-validation curve
plot(cv.lasso)

# Get Lambda 1SE to select a more interpretable model
best_lambda <- cv.lasso$lambda.min
best_lambda


# Fit the final model
lasso_model <- glmnet(x_train, y_train, alpha = 1, family = "binomial", lambda = best_lambda)

# Extract coefficients
coef(lasso_model)

# See which variables are selected (non-zero coefficients)
lasso_coefs <- coef(lasso_model)
selected_variables <- rownames(lasso_coefs)[lasso_coefs[,1] != 0]
selected_variables


# Predict on test set
predictions_prob <- predict(lasso_model, newx = x_test, type = "response")
predictions_class <- ifelse(predictions_prob > 0.5, 1, 0)

# Confusion matrix
lasso_result <-  confusionMatrix(as.factor(predictions_class), y_test, positive = "1")
lasso_result


# VISUALISATION
# Ensure labels are factors in the correct order
y_test_factor <- factor(y_test, levels = c("1", "0"))

# Convert predictions to data frame and bind with true labels
lasso_results <- data.frame(
  truth = y_test_factor,
  .pred_1 = as.vector(predictions_prob)
)

# Calculate ROC curve
lasso_roc_data <- roc_curve(lasso_results, truth = truth, .pred_1)
lasso_auc_value <- roc_auc(lasso_results, truth = truth, .pred_1)
print(lasso_auc_value)

# Plot ROC curve
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

# create x_train, x_test, y_train, y_test from train and test dataset
x_train <- model.matrix(emig ~ ., data = train)[, -1]  # Remove intercept
x_test <- model.matrix(emig ~ ., data = test)[, -1]  # Remove intercept
y_train <- train$emig
y_test <- test$emig


# Perform cross-validated lasso logistic regression
cv.lasso <- cv.glmnet(x_train, y_train, alpha = 0, family = "binomial")

# Plot the cross-validation curve
plot(cv.lasso)

# Get Lambda 1SE to select a more interpretable model
best_lambda <- cv.lasso$lambda.min
best_lambda


# Fit the final model
lasso_model <- glmnet(x_train, y_train, alpha = 0, family = "binomial", lambda = best_lambda)

# Extract coefficients
coef(lasso_model)

# See which variables are selected (non-zero coefficients)
lasso_coefs <- coef(lasso_model)
selected_variables <- rownames(lasso_coefs)[lasso_coefs[,1] != 0]
selected_variables


# Predict on test set
predictions_prob <- predict(lasso_model, newx = x_test, type = "response")
predictions_class <- ifelse(predictions_prob > 0.5, 1, 0)

# Confusion matrix
lasso_result <-  confusionMatrix(as.factor(predictions_class), y_test, positive = "1")
lasso_result


# VISUALISATION
# Ensure labels are factors in the correct order
y_test_factor <- factor(y_test, levels = c("1", "0"))

# Convert predictions to data frame and bind with true labels
lasso_results <- data.frame(
  truth = y_test_factor,
  .pred_1 = as.vector(predictions_prob)
)

# Calculate ROC curve
lasso_roc_data <- roc_curve(lasso_results, truth = truth, .pred_1)
lasso_auc_value <- roc_auc(lasso_results, truth = truth, .pred_1)
print(lasso_auc_value)

# Plot ROC curve
ggplot(lasso_roc_data, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(color = "blue", size = 1.2) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(
    title = "ROC Curve - Lasso Logistic Regression",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)"
  ) +
  theme_minimal()






### COMMENCE MODEL COMPARISON AND EXTRACTION #################################################################





