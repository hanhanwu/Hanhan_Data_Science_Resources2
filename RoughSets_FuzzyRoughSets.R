# reference: https://cran.r-project.org/web/packages/RoughSets/RoughSets.pdf

library(RoughSets)
library('caret')
library('e1071')

data("RoughSetData")
summary(RoughSetData)  # it has 7 DecisionTable
head(RoughSetData)

################################Rough Set Theory Basic Concepts#################################
hiring_dt <- RoughSetData$hiring.dt
head(hiring_dt)  # Diploma, Experience, French, Reference, Decision

# compute indiscernibility relation
rst_indiscernibility_relation <- BC.IND.relation.RST(hiring_dt)
rst_indiscernibility_relation
 
# compute lower and upper approximation
## it indicates to what extent objects can be classified with certainty or not
rst_lu_approx <- BC.LU.approximation.RST(hiring_dt, rst_indiscernibility_relation)
rst_lu_approx

# determine regions
regin_rst <- BC.positive.reg.RST(hiring_dt, rst_lu_approx)
regin_rst

# decision related discernibility matrix and reduct
discernibility_max <- BC.discernibility.mat.RST(hiring_dt, range.object = NULL)
discernibility_max
################################Rough Set Theory Basic Concepts#################################

################################Fuzzy Rough Set Theory Basic Concepts#######################
hiring_dt <- RoughSetData$hiring.dt
head(hiring_dt)  # Diploma, Experience, French, Reference, Decision

decision_Attr = c(5)

# compute indiscernibility relation
control.ind <- list(type.aggregation = c("t.tnorm", "lukasiewicz"),
                    type.relation = c("tolerance", "eq.1"))
frst_indiscernibility_relation <- BC.IND.relation.FRST(hiring_dt, control = control.ind)
frst_indiscernibility_relation

# Compute fuzzy indiscernibility relation of decision attribute
## it indicates to what extent objects can be classified with certainty or not
control.dec <- list(type.aggregation = c("crisp"), type.relation = "crisp")
IND.decAttr <- BC.IND.relation.FRST(hiring_dt, attributes = decision_Attr,
                                    control = control.dec)

# Define control parameter containing type of implicator and t-norm
control <- list(t.implicator = "lukasiewicz", t.tnorm = "lukasiewicz")

# compute lower and upper approximation
frst_lu_approx <- BC.LU.approximation.FRST(hiring_dt, frst_indiscernibility_relation,
                                           IND.decAttr,
                                           type.LU = "implicator.tnorm", control = control)
frst_lu_approx

# determine regions
regin_frst <- BC.positive.reg.FRST(hiring_dt, frst_lu_approx)
regin_frst
################################Fuzzy Rough Set Theory Basic Concepts#######################

# Data Analysis with RST
mydata <- RoughSetData$wine.dt
head(mydata)  # lots of attributes

## shuffle the data with random seed
set.seed(410)
shuffled_dt <- mydata[sample(nrow(mydata)),]

## split data into training and testing
idx <- round(0.77*nrow(shuffled_dt))
idx
training_dt <- SF.asDecisionTable(shuffled_dt[1:idx,], decision.attr = 14, indx.nominal = 14)
testing_dt <- SF.asDecisionTable(shuffled_dt[(idx+1):nrow(shuffled_dt),])
head(training_dt)
real_label <- testing_dt$class
testing_dt <- SF.asDecisionTable(shuffled_dt[(idx+1):nrow(shuffled_dt), -ncol(shuffled_dt)])
head(testing_dt)

## discretization - transfer continuous variables into discrete counterparts
cut_values <- D.discretization.RST(training_dt, type.method = "global.discernibility")
dis_train <- SF.applyDecTable(training_dt, cut_values)
head(dis_train)
dis_test <- SF.applyDecTable(testing_dt, cut_values)
head(dis_test)

## feature selection
red_rst <- FS.feature.subset.computation(dis_train, method="quickreduct.rst")
fs_train <- SF.applyDecTable(dis_train, red_rst)
head(fs_train)

## rule induction
rst_rules <- RI.indiscernibilityBasedRules.RST(dis_train, red_rst)

##  prediction
pred_vals <- predict(rst_rules, dis_test)
head(pred_vals)
dCM <- confusionMatrix(real_label, unlist(pred_vals))
dCM
summary(as.factor(mydata$class))


# Data Analysis with FRST, still to predict wine classes
## feature selection
reduct <- FS.feature.subset.computation(training_dt, method = "quickreduct.frst")
reduct

## generate new decision tables
frst_training <- SF.applyDecTable(training_dt, reduct)
head(frst_training)
frst_testing <- SF.applyDecTable(testing_dt, reduct)
head(frst_testing)

## instance selection 
### remove noisy, unnecessary and inconsistant instances from the training data
indx <- IS.FRIS.FRST(frst_training, control = list(thresholder.tau=0.2, alpha=1))
indx
is_frst_training <- SF.applyDecTable(frst_training, indx)
head(is_frst_training)

## rule induction
control.ri <- list(type.aggregation=c("t.tnorm", "lukasiewicz"), 
                   type.relation=c("tolerance", "eq.3"),
                   t.implicator="kleene_dienes")
decRules_hybrid <- RI.hybridFS.FRST(is_frst_training, control.ri)
decRules_hybrid

## prediction
pred_vals <- predict(decRules_hybrid, frst_testing)
pred_vals
dCM <- confusionMatrix(real_label, unlist(pred_vals))
dCM  # 0.9167   0.8524   0.9375
summary(as.factor(mydata$class))  # 59 71 48
