# Try different validation methods, models here will simply be random forest
library(caret)
library(rpart)
library(plyr)
library(data.table)
library(AUC)

data("iris")
head(iris)


# Mehtod 1 - K-fold cross validation
cv_control <- trainControl(method = "cv", number = 10)  # 10 fold
model <- train(Species~., data=iris, trControl=cv_control, method="rf")
print(model)


# Method 2 - Stratified k-fold
## assign 1 to 10 fold to each row
folds <- createFolds(factor(iris$Species), k = 10, list = FALSE) 
## check the label value distribution in each fold
unique(iris$Species)
iris$folds <- folds
head(iris)
dist_df <- count(iris, c("folds", "Species"))
dist_df  # in fact you stop at this step is totaly fine
dist_dt <- data.table(dist_df)
sum_dt <- dist_dt[, sum(freq), by = folds]
setnames(sum_dt, "V1", "fold_ct")
sum_dt
merge_df <- merge(dist_dt, sum_dt)
merge_df$perct <- merge_df$freq/merge_df$fold_ct
merge_df
## cross validation with given folds
### reference: https://cran.r-project.org/web/packages/groupdata2/vignettes/cross-validation_with_groupdata2.html
k=3
performances <- c()
for (fold in 1:k){
  # Create training set for this iteration
  # Subset all the datapoints where .folds does not match the current fold
  training_set <- iris[iris$folds != fold,]
  
  # Create test set for this iteration
  # Subset all the datapoints where .folds matches the current fold
  val_set <- iris[iris$folds == fold,]
  
  ## Train model
  model <- train(Species~., data=training_set, method="rf")
  ## Validate model
  predicted <- predict(model, val_set)
  dcm <- confusionMatrix(val_set$Species, predicted)
  balanced_auc <- (auc(sensitivity(predicted, val_set$Species)) + auc(specificity(predicted, val_set$Species)))/2
  
  performances[fold] <- balanced_auc
}

cv_auc <- mean(performances)
cv_auc


# Method 3 - Leave One Out Cross Validation
library(tidyverse)
iris <- tibble::rowid_to_column(iris, "ID")
head(iris)

k=dim(iris)[1]
performances <- c()
for (fold in 1:k){
  # Create training set for this iteration
  # Subset all the datapoints where .folds does not match the current fold
  training_set <- iris[iris$ID != fold,]
  
  # Create test set for this iteration
  # Subset all the datapoints where .folds matches the current fold
  val_set <- iris[iris$ID == fold,]
  
  ## Train model
  model <- train(Species~., data=training_set, method="rf")
  ## Validate model
  predicted <- predict(model, val_set)
  dcm <- confusionMatrix(val_set$Species, predicted)
  balanced_auc <- (auc(sensitivity(predicted, val_set$Species)) + auc(specificity(predicted, val_set$Species)))/2
  
  performances[fold] <- balanced_auc
}

cv_auc <- mean(performances)
cv_auc
