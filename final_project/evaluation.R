# version 25 May 2021

############################################################
# Template for final project                               #
############################################################

rm(list=ls())

# Set your working directory here. 
setwd("...")

# Load necessary packages here. All packages must
# be available for install via install.packages()
# or remotes::install_github().

library("ROCR")
library("dplyr")

# define function to assess classification performance

auc <- function(phat,y){
	pred <- ROCR::prediction(phat, y)
	perf <- ROCR::performance(pred,"auc")
	auc <- perf@y.values[[1]]
	return(auc)
}

# load train and test data

load("AB4x_train.Rdata")
load("AB4x_eval_mock.Rdata")
# The mock data will be replaced by the 
# hold-out data for final evaluation.

# Include all necessary preprocessing steps
# that should be applied to the test data.

# ~~~ example code start ~~~

# Standardize variables
age_mu <- mean(test$age,na.rm=TRUE)
age_sd <- sd(test$age,na.rm=TRUE)
test <- test |> mutate(age  = (age-age_mu)/age_sd)

# define outcome as binary
test <- test |> mutate(emig = as.integer(emig=="Yes"))

# simple mean imputation
test <- test |> mutate(age = if_else(is.na(age),age_mu,age))

# ~~~ example code end ~~~

# Load trained model 
load("final_model.Rdata")

# Obtain predictions
# Note that `pred` should be a vector of predicted probabilities;
# not a vector of predicted classes
# ~~~ example code start ~~~
pred <- predict(m,newdata=test,type="response")
# ~~~ example code end ~~~

# Model performance:
truth <- test$emig

cat("The final score:","\n") 
auc(pred,truth)
