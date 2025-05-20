rm(list=ls())

############################################################
# Template for final project                               #
############################################################

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

# load training data

load("AB4x_train.Rdata")

# Build your model; example code follows:

# ~~~ example code start ~~~

# Standardize variables
age_mu <- mean(train$age,na.rm=TRUE)
age_sd <- sd(train$age,na.rm=TRUE)
train <- train |> mutate(age  = (age-age_mu)/age_sd)

# define outcome as binary
train <- train|> mutate(emig = as.integer(emig=="Yes"))

# simple mean imputation
train <- train |> mutate(age = if_else(is.na(age),age_mu,age))

# Train models
m <- glm(emig ~ age, data=train, family=binomial(link="logit"))
summary(m)

# ~~~ example code end ~~~

# Save final model 
save(m, file="final_model.Rdata")