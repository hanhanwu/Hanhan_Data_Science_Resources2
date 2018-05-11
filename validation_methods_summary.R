# Try different validation methods, models here will simply be random forest
library(caret)
library(rpart)

data("iris")
head(iris)


# Mehtod 1 - K-fold cross validation
cv_control <- trainControl(method = "cv", number = 10)  # 10 fold
model <- train(Species~., data=iris, trControl=cv_control, method="rf")
print(model)
