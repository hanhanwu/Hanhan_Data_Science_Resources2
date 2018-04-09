library(RoughSets)

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

# shuffle the data with random seed
set.seed(410)
shuffled_dt <- mydata[sample(nrow(mydata)),]

# split data into training and testing
idx <- round(0.77*nrow(shuffled_dt))
idx
training_dt <- SF.asDecisionTable(shuffled_dt[1:idx,], decision.attr = 14, indx.nominal = 14)
testing_dt <- SF.asDecisionTable(shuffled_dt[(idx+1):nrow(shuffled_dt),])
head(training_dt)
real_label <- testing_dt$class
testing_dt <- SF.asDecisionTable(shuffled_dt[(idx+1):nrow(shuffled_dt), -ncol(shuffled_dt)])
head(testing_dt)

# discretization - transfer continuous variables into discrete counterparts
cut_values <- D.discretization.RST(training_dt, type.method = "global.discernibility")
dis_train <- SF.applyDecTable(training_dt, cut_values)
head(dis_train)
dis_test <- SF.applyDecTable(testing_dt, cut_values)
head(dis_test)

# feature selection
red_rst <- FS.feature.subset.computation(dis_train, method="quickreduct.rst")
fs_train <- SF.applyDecTable(dis_train, red_rst)
head(fs_train)

# rule induction
rst_rules <- RI.indiscernibilityBasedRules.RST(dis_train, red_rst)

# prediction
pred_vals <- predict(rst_rules, dis_test)
head(pred_vals)

library('caret')
library('e1071')
dCM <- confusionMatrix(real_label, unlist(pred_vals))
dCM  # Balanced Accuracy      0.8846   0.9423   1.0000

summary(as.factor(mydata$class))  # 59 71 48 
