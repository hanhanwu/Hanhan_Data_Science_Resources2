# Try different validation methods, models here will simply be random forest
library(caret)
library(rpart)
library(plyr)
library(data.table)

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
