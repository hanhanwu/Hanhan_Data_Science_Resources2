# Hanhan_Data_Science_Resources2
More data science resources
It seems that my Data Science Resources cannot be updated, create a new one here for more resources

************************************************************************

SUMMARIZED RESOURCES

* Hanhan_Data_Science_Resource 1: https://github.com/hanhanwu/Hanhan_Data_Science_Resources
* <b>Check Awesome Big Data when looking for new ways to solve data science problems</b>: https://github.com/onurakpolat/awesome-bigdata
* Categorized Resources for Machine Learning: https://www.analyticsvidhya.com/resources-machine-learning-deep-learning-neural-networks/
* Summarized Tableau Learning Resources: https://www.analyticsvidhya.com/learning-paths-data-science-business-analytics-business-intelligence-big-data/tableau-learning-path/
* Summarized Big Data Learning Resources: https://www.analyticsvidhya.com/resources-big-data/
* Data Science Media Resources: https://www.analyticsvidhya.com/data-science-blogs-communities-books-podcasts-newsletters-follow/
* This is a new UC Berkeley data science cousre, it servers for undergraduate and therefore everything is introductory, however it covers relative statistics, math, data visualization, I think it will be helpful, since sometimes if we only study statistics may still have difficulty to apply the knowledge in data science. This program has slides and video for each class online, available to the public immeddiately: http://www.ds100.org/sp17/


************************************************************************

TREE BASED MODELS & ENSEMBLING

* For more ensembling, check `ENSEMBLE` sections and Experiences.md here: https://github.com/hanhanwu/Hanhan_Data_Science_Resources
* Tree based models in detail with R & Python example: https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/?utm_content=bufferade26&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* [R Implementation] Choose models for emsemling: https://www.analyticsvidhya.com/blog/2015/10/trick-right-model-ensemble/?utm_content=buffer6b42d&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer
  * The models are les correlated to each other
  * The code in this tutorial is trying to test the results made by multiple models and choose the model combination that gets the best result (I'm thinking how do they deal with random seed issues)


************************************************************************

DATA PREPROCESSING

* For more data preprocessing, check `DATA PREPROCESSING` section: https://github.com/hanhanwu/Hanhan_Data_Science_Resources

* Entity Resolution
  * Basics of Entity Resolution with Python and Dedup: http://blog.districtdatalabs.com/basics-of-entity-resolution?imm_mid=0f0aec&cmp=em-data-na-na-newsltr_20170412
  * Three primary tasks
    * Deduplication: eliminating duplicate (exact) copies of repeated data.
    * Record linkage: identifying records that reference the same entity across different sources.
    * Canonicalization: converting data with more than one possible representation into a standard form.
  * In the url above, they have done some experiments with Python and Dedup
  
* Dimension Reduction
  * t-SNE, non-linear dimensional reduction
    * Reference (a pretty good one!): https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
    * (t-SNE) t-Distributed Stochastic Neighbor Embedding is a non-linear dimensionality reduction algorithm used for exploring high-dimensional data, it considers nearest neighbours when reduce the data. It is a non-parametric mapping.
    * The problem with linear dimensional reduction, is that they concentrate on placing dissimilar data points far apart in a lower dimension representation. However, it is also important to put similar data close together, linear dimensional reduction does not do this
    * In t-SNE, there are local approaches and global approaches. Local approaches seek to map nearby points on the manifold to nearby points in the low-dimensional representation. Global approaches on the other hand attempt to preserve geometry at all scales, i.e mapping nearby points to nearby points and far away points to far away points  
    * It is important to know that most of the nonlinear techniques other than t-SNE are not capable of retaining both the local and global structure of the data at the same time.
    * The algorithm computes pairwise conditional probabilities and tries to minimize the sum of the difference of the probabilities in higher and lower dimensions. This involves a lot of calculations and computations. So the algorithm is quite heavy on the system resources. t-SNE has a quadratic O(n2) time and space complexity in the number of data points. This makes it particularly slow and resource draining while applying it to data sets comprising of more than 10,000 observations. Another drawback is, it doesn’t always provide a similar output on successive runs.
    * How it works: it clusters similar data reocrds together, but it's not clustering because once the data has been mapped to lower dimensional, the original features are no longer recognizable. 
    * NOTE: t-SNE could also help to make semanticly similar words close to each other, which could help create text summary, text comparison
    * R practice code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/t-SNE_practice.R
    * Python practice code: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/t-SNE_Practice.ipynb
  * [For more Dimensional Reduction, check "DATA PREPROCESSING" section][9] 
  * [Some dimentional reduction methods][10]
  * [A little more about Factor Analysis][11]
    * Fator Analysis is a variable reduction technique. It is used to determine <b>factor structure or model</b>. It also explains the maximum amount of variance in the model
    * EFA (Exploratory Factor Analysis) – Identifies and summarizes the <b>underlying correlation structure</b> in a data set
    * CFA (Confirmatory Factor Analysis) – Attempts to <b>confirm hypothesis</b> using the correlation structure and <b>rate ‘goodness of fit’</b>.
  * Dimension Reduction Must Know
    * Reference: https://www.analyticsvidhya.com/blog/2017/03/questions-dimensionality-reduction-data-scientist/?utm_content=bufferc792d&utm_medium=social&utm_source=linkedin.com&utm_campaign=buffer
    * Besides different algorithms to help reduce number of features, we can also use existing features to form less features as a dimensional reduction method. For example, we have features A, B, C, D, then we form E = 2*A+B, F = 3*C-D, then only choose E, F as the features for analysis
    * Cost function of SNE is asymmetric in nature. Which makes it difficult to converge using gradient decent. A symmetric cost function is one of the major differences between SNE and t-SNE.
    * For the perfect representations of higher dimensions to lower dimensions, the conditional probabilities for similarity of two points must remain unchanged in both higher and lower dimension, which means the similarity is unchanged
    * LDA aims to maximize the distance between class and minimize the within class distance. If the discriminatory information is not in the mean but in the variance of the data, LDA will fail.
    * Both LDA and PCA are linear transformation techniques. LDA is supervised whereas PCA is unsupervised. PCA maximize the variance of the data, whereas LDA maximize the separation between different classes.
    * When eigenvalues are roughly equal, PCA will perform badly, because when all eigen vectors are same in such case you won’t be able to select the principal components because in that case all principal components are equal. <b> When using PCA</b>, it is better to scale data in the same unit
    * When using PCA, features will lose interpretability and they may not carry all the info of the data. <b>You don’t need to initialize parameters in PCA, and PCA can’t be trapped into local minima problem</b>. PCA is a <b>deterministic algorithm</b> which doesn’t have parameters to initialize. PCA can be used for lossy image compression, and it is not invariant to shadows.
    * A deterministic algorithm has no param to initialize, and it gives the same result if we run again.
    * Logistic Regression vs LDA: If the classes are well separated, the parameter estimates for logistic regression can be unstable. If the sample size is small and distribution of features are normal for each class. In such case, linear discriminant analysis (LDA) is more stable than logistic regression.


************************************************************************

MODEL EVALUATION

* 7 important model evaluation metrics and cross validation: https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/
  * Confusion Matrix
  * <b>Lift / Gain charts</b> are widely used in campaign targeting problems. This tells us till which decile can we target customers for an specific campaign. Also, it tells you how much response do you expect from the new target base.
  * Kolmogorov-Smirnov (K-S) chart is a measure of the degree of separation between the positive and negative distributions. The K-S is 100, the higher the value the better the model is at separating the positive from negative cases.
  * The ROC curve is the plot between sensitivity and (1- specificity). (1- specificity) is also known as false positive rate and sensitivity is also known as True Positive rate. To bring ROC curve down to a single number, AUC, which is  the ratio under the curve and the total area. .90-1 = excellent (A) ; .80-.90 = good (B) ; .70-.80 = fair (C) ; .60-.70 = poor (D) ; .50-.60 = fail (F). But this might simply be over-fitting. In such cases it becomes very important to to in-time and out-of-time validations. For a model which gives class as output, will be represented as a single point in ROC plot. In case of probabilistic model, we were fortunate enough to get a single number which was AUC-ROC. But still, we need to look at the entire curve to make conclusive decisions.
  * Lift is dependent on total response rate of the population. ROC curve on the other hand is almost independent of the response rate, because the numerator and denominator of both x and y axis will change on similar scale in case of response rate shift.
  * Gini = 2*AUC – 1. Gini Coefficient is nothing but ratio between area between the ROC curve and the diagnol line & the area of the above triangle
  * <b>The concordant pair</b> is where the probability of responder was higher than non-responder. Whereas <b>discordant pair</b> is where the vice-versa holds true. Concordant ratio of more than 60% is considered to be a good model. It is <b>primarily used to access the model’s predictive power</b>. For decisions like how many to target are again taken by KS / Lift charts.
  * RMSE: The power of ‘square root’  empowers this metric to show large number deviations. The ‘squared’ nature of this metric helps to deliver more robust results which prevents cancelling the positive and negative error values. When we have more samples, reconstructing the error distribution using RMSE is considered to be more reliable. RMSE is highly affected by outlier values. Hence, make sure you’ve removed outliers from your data set prior to using this metric. As compared to mean absolute error, RMSE gives higher weightage and punishes large errors.
  * k-fold cross validation is widely used to check whether a model is an overfit or not. <b>If the performance metrics at each of the k times modelling are close to each other and the mean of metric is highest.</b> For a small k, we have a higher selection bias but low variance in the performances. For a large k, we have a small selection bias but high variance in the performances. <b>Generally a value of k = 10 is recommended for most purpose.</b>
  * <b>Tolerance</b> (1 / VIF) is used as an indicator of multicollinearity. It is an indicator of percent of variance in a predictor which cannot be accounted by other predictors. Large values of tolerance is desirable.
 
* To measure linear regression, we could use Adjusted R² or F value.
* To measure logistic regression:
  * AUC-ROC curve along with confusion matrix to determine its performance.
  * The analogous metric of adjusted R² in logistic regression is AIC. AIC is the measure of fit which penalizes model for the number of model coefficients. Therefore, we always prefer model with minimum AIC value.
  * Null Deviance indicates the response predicted by a model with nothing but an intercept. Lower the value, better the model. 
  * Residual deviance indicates the response predicted by a model on adding independent variables. Lower the value, better the model.
* Regularization becomes necessary when the model begins to ovefit / underfit. This technique introduces a cost term for bringing in more features with the objective function. Hence, <b>it tries to push the coefficients for many variables to zero and hence reduce cost term.</b> This helps to reduce model complexity so that the model can become better at predicting (generalizing).


************************************************************************

Applied Data Science in Python/R/Java

* [R] Caret package for data imputing, feature selection, model training (I will show my experience of using caret with detailed code in Hanhan_Data_Science_Practice): https://www.analyticsvidhya.com/blog/2016/12/practical-guide-to-implement-machine-learning-with-caret-package-in-r-with-practice-problem/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* [Python & R] A brief classified summary of Python Scikit-Learn and R Caret: https://www.analyticsvidhya.com/blog/2016/12/cheatsheet-scikit-learn-caret-package-for-python-r-respectively/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* [Python] What to pay attention to when you are using Naive Bayesian with Scikit-Learn: https://www.analyticsvidhya.com/blog/2015/09/naive-bayes-explained/?utm_content=bufferaa6aa&utm_medium=social&utm_source=linkedin.com&utm_campaign=buffer
  * 3 types of Naive Bayesian: Gaussian (if you assume that features follow a normal distribution); Multinomial (used for discrete counts, you can think of it as “number of times outcome number x_i is observed over the n trials”.); Bernoulli(useful if your <b>feature vectors are binary</b>); 
  * Tips to improve the power of Naive Bayes Model: If test data set has zero frequency issue, apply <b>smoothing techniques</b> “Laplace Correction” to predict the class of test data set. Focus on your  pre-processing of data and the feature selection, because of thelimited paramter choices. “ensembling, boosting, bagging” won’t help since their purpose is to reduce variance. <b>Naive Bayes has no variance to minimize</b>
* [R] Clsuter Analysis: https://rstudio-pubs-static.s3.amazonaws.com/33876_1d7794d9a86647ca90c4f182df93f0e8.html
* [Java] SPMF, it contains many algorithms that cannot be found in R/Scikit-Learn/Spark, especailly algorithms about Pattern Mining: http://www.philippe-fournier-viger.com/spmf/index.php?link=algorithms.php
  * SPMF examples: http://www.philippe-fournier-viger.com/spmf/index.php?link=documentation.php
  * Algorithm mapping: http://www.philippe-fournier-viger.com/spmf/map_algorithms_spmf_data_mining097.png
  * Download: http://www.philippe-fournier-viger.com/spmf/index.php?link=download.php
* [Python] Scikit-Learn algorithms map and Estimators
  * map: http://scikit-learn.org/stable/tutorial/machine_learning_map/
  * I'm sure that I have copied this before, but today I have learned something new about this map! Those green squares are clickable, and they are the estimators
  * An estimator is used to help tune parameters and estimate the model
  * This map also helps you find some algorithms in Scikit-Learn based on data size, data type


************************************************************************

Statistics in Data Science

* Termology glossary for statistics in machine learning: https://www.analyticsvidhya.com/glossary-of-common-statistics-and-machine-learning-terms/
* Statistics behind Boruta feature selection: https://github.com/hanhanwu/Hanhan_Data_Science_Resources2/blob/master/boruta_statistics.pdf
* How the laws of group theory provide a useful codification of the practical lessons of building efficient distributed and real-time aggregation systems (from 22:00, he started to talk about HyperLogLog and other approximation data structures): https://www.infoq.com/presentations/abstract-algebra-analytics
* Confusing Concepts
  * Errors and Residuals: https://en.wikipedia.org/wiki/Errors_and_residuals
  * Heteroskedasticity: led by non-constant variance in error terms. Usually, non-constant variance is caused by outliers or extreme values
  * Coefficient and p-value/t-statistics: coefficient measures the strength of the relationship of 2 variables, while p-value/t-statistics measures how strong the evidence that there is non-zero association
  * Anscombe's quartet comprises four datasets that have nearly identical simple statistical properties, yet appear very different when graphed: https://en.wikipedia.org/wiki/Anscombe's_quartet
  * Difference between gradient descent and stochastic gradient descent: https://www.quora.com/Whats-the-difference-between-gradient-descent-and-stochastic-gradient-descent
  * Correlation & Covariance: In probability theory and statistics, correlation and covariance are two similar measures for assessing how much two attributes change together. The mean values of A and B, respectively, are also known as the <b>expected values on A and B</b>, E(A), E(B). <b>Covariance, Cov(A,B)=E(A·B) - E(A)*E(B)</b>
  * Rate vs Proportion: A rate differs from a proportion in that the numerator and the denominator need not be of the same kind and that the numerator may exceed the denominator. For example, the rate of pressure ulcers may be expressed as the number of pressure ulcers per 1000 patient days.
 * <b>Bias</b> is useful to quantify how much on an average are the predicted values different from the actual value. A high bias error means we have a <b>under-performing</b> model which keeps on missing important trends. <b>Varianc</b> on the other side quantifies how are the prediction made on same observation different from each other. A high variance model will <b>over-fit</b> on your training population and perform badly on any observation beyond training.
  * <b>OLS</b> and <b>Maximum likelihood</b> are the methods used by the respective regression methods to approximate the unknown parameter (coefficient) value. OLS is to linear regression. Maximum likelihood is to logistic regression. Ordinary least square(OLS) is a method used in linear regression which approximates the parameters resulting in <b>minimum distance between actual and predicted values.</b> Maximum Likelihood helps in choosing the the values of parameters which <b>maximizes the likelihood that the parameters are most likely to produce observed data.</b>
  * <b>Standard Deviation</b> – It is the amount of variation in the <b>population data</b>. It is given by σ. <b>Standard Error</b> – It is the amount of variation in the <b>sample data</b>. It is related to Standard Deviation as σ/√n, where, n is the sample size, σ is the standandard deviation of the <b>population</b>
  * 95% <b> confidence interval does not mean</b> the probability of a population mean to lie in an interval is 95%. Instead, 95% C.I <b>means that 95% of the Interval estimates will contain the population statistic</b>.
  * If a sample mean lies in the margin of error range then, it might be possible that its actual value is equal to the population mean and the difference is occurring by chance.
  * <b>Difference between z-scores and t-values</b> are that t-values are dependent on Degree of Freedom of a sample, and t-values use sample standard deviation while z-scores use population standard deviation.
  * <b>The Degree of Freedom</b> – It is the number of variables that have the choice of having more than one arbitrary value. For example, in a sample of size 10 with mean 10, 9 values can be arbitrary but the 1oth value is forced by the sample mean.
  * <b>Residual Sum of Squares (RSS)</b> - It can be interpreted as the amount by which the predicted values deviated from the actual values. Large deviation would indicate that the model failed at predicting the correct values for the dependent variable. <b>Regression (Explained) Sum of Squares (ESS)</b> – It can be interpreted as the amount by which the predicted values deviated from the the mean of actual values.
  * Residuals is also known as the prediction error, they are vertical distance of points from the regression line
  * <b>Co-efficient of Determination = ESS/(ESS + RSS)</b>. It represents the strength of correlation between two variables. <b>Correlation Coefficient = sqrt(Co-efficient of Determination)</b>, also represents the strength of correlation between two variables, ranges between [-1,1]. 0 means no correlation, 1 means strong positive correlation, -1 means strong neagtive correlation.
  

* About Data Sampling: http://psc.dss.ucdavis.edu/sommerb/sommerdemo/sampling/types.htm
  * Probability sampling can be representative, non-probability sampling may not
  * probability Sampling
    * Random sample. (I guess R `sample()` is random sampling by default, so that each feature has the same weight)
    * Stratified sample
  * Nonprobability Sampling
    * Quota sample
    * Purposive sample
    * Convenience sample
 
 
* [Comprehensive and Practical Statistics Guide for Data Science][1] - A real good one!
  * Sample Distribution and Population Distribution, Central Limit Theorem, Confidence Interval
  * Hypothesis Testing
  * [t-test calculator][2]
  * ANOVA (Analysis of Variance), continuous and categorical variables, ANOVA also requires data from approximately normally distributed populations with equal variances between factor levels.
  * [F-ratio calculator][3]
  * Chi-square test, categorical variables
  * [p value (chi-square) calculator][4]
  * Regression and ANOVA, it is important is knowing the degree to which your model is successful in explaining the trend (variance) in dependent variable. ANOVA helps finding the effectiveness of regression models.
 

* Probability cheat sheet: http://www.cs.elte.hu/~mesti/valszam/kepletek
* [Probability basics with examples][5]
  * binonial distribution: a binomial distribution is the discrete probability distribution of the number of success in a sequence of n independent Bernoulli trials (having only yes/no or true/false outcomes).
  * The normal distribution is perfectly symmetrical about the mean. The probabilities move similarly in both directions around the mean. The total area under the curve is 1, since summing up all the possible probabilities would give 1.
  * Area Under the Normal Distribution
  * Z score: The distance in terms of number of standard deviations, the observed value is away from the mean, is the standard score or the Z score. <b>Observed value = µ+zσ</b> [µ is the mean and σ is the standard deviation]
  * [Z table !!!][6]
* [Very Basic Conditional Probability and Bayes Theorem][7]
  * Independent, Exclusive, Exhaustive events
  * Each time, when it's something about statistics pr probability, I will still read all the content to guarantee that I won't miss anything useful. This one is basic but I like the way it starts from simple concepts, using real life examples and finally leads to how does Bayes Theorem work. Although, there is an error in formula `P (no cancer and +) = P (no cancer) * P(+) = 0.99852*0.99`, it should be `0.99852*0.01`
  * There are some major formulas here are important to Bayes Theorem: 
    * `P(A|B) = P(A AND B)/P(B)`
    * `P(A|B) = P(B|A)*P(A)/P(B)`
    * `P(A AND B) = P(B|A)*P(A) = P(A|B)*P(B)`
    * `P(b1|A) + P(b2|A) + .... + P(bn|A) = P(A)`


* [Dispersion][8] - In statistics, dispersion (also called variability, scatter, or spread) is the extent to which a distribution is stretched or squeezed. Common examples of measures of statistical dispersion are the variance, standard deviation, and interquartile range.

* Common Formulas
  * Examples: https://www.analyticsvidhya.com/blog/2017/05/41-questions-on-statisitics-data-scientists-analysts/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * <b>About z-score</b>
    * When looking for probability, calculate z score first and check the value in the Z table, that value is the probability
    * <b>Observed value = µ+zσ</b> [µ is the mean and σ is the standard deviation]
    * <b>Standard Error = σ/√N</b>, [N is the number of Sample]; <b>z = (Sample Mean - Population Mean)/Standard Error</b>
    * [Z table][6]
    * [Z table for different distributions, and z critical values][12]
  * <b>About t-score</b>
    * When compare 2 groups, calculate t-score
    * <b>t-statistic = (group1 Mean - group2 Mean)/Standard Error</b>
    * <b>degree of freedom (df)</b>, if there are n sample, <b>df=n-1</b>
    * [t table with df, 1-tail, 2-tails and confidence level][13], compare your t-statistic with the relative value in this table, and decide whether to reject null hypothesis
    * <b>percentage of variability = t-statistic^2/(t-statistic^2 + degree of freedom) = correlation_coefficient ^2</b>, so <b>coefficient of determination</b> equals to "percentage of variability"
  * <b>About F-statistic</b>
    * F-statistic is the value we receive when we run an ANOVA test on different groups to <b>understand the differences between them</b>.
    * <b>F-statistic  = (sum of squared error for between group/degree of freedom for between group)/(sum of squared error for within group/degree of freedom for within group)</b>, as you can see from this formula, it cannot be negative
  * <b>Correlation</b>
    * Methods to calculate correlations between different data types: https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/
    * Formula to calculate correlation between 2 numerical variables (Question 28): https://www.analyticsvidhya.com/blog/2017/05/41-questions-on-statisitics-data-scientists-analysts/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
    * Correlation between the features won’t change if you add or subtract a constant value in the features.
  * <b>Significance level = 1- Confidence level</b>
  * <b>Mean Absolute Error</b> = the mean of absolute errors
  
* Linear regression line attempts to minimize the squared distance between the points and the regression line. By definition the ordinary least squares (OLS) regression tries to have the minimum sum of squared errors. This means that the sum of squared residuals should be minimized. This may or may not be achieved by passing through the maximum points in the data. The most common case of not passing through all points and reducing the error is when the data has a lot of outliers or is not very strongly linear.
* Person vs Spearman: Pearson correlation evaluated the <b>linear relationship</b> between two continuous variables. <b>A relationship is linear when a change in one variable is associated with a proportional change in the other variable.</b> Spearman evaluates a <b>monotonic relationship</b>. <b>A monotonic relationship is one where the variables change together but not necessarily at a constant rate.</b>


************************************************************************

Machine Learning Algorithms

* KNN with R example: https://www.analyticsvidhya.com/blog/2015/08/learning-concept-knn-algorithms-programming/
  * KNN unbiased and no prior assumption, fast
  * It needs good data preprocessing such as missing data imputing, categorical to numerical
  * k normally choose the square root of total data observations
  * It is also known as lazy learner because it involves minimal training of model. Hence, it doesn’t use training data to make generalization on unseen data set.

* SVM with Python example: https://www.analyticsvidhya.com/blog/2015/10/understaing-support-vector-machine-example-code/?utm_content=buffer02b8d&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 
* Basic Essentials of Some Popular Machine Learning Algorithms with R & Python Examples: https://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/?utm_content=buffer00918&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
  * Linear Regression: Y = aX + b, a is slope, b is intercept. The intercept term shows model prediction without any independent variable. When there is only 1 independent variable, it is Simple Linear Regression, when there are multiple independent variables, it is Multiple Linear Regression. For Multiple Linear Regression, we can fit Polynomial Courvilinear Regression.
  * When to use Ridge or Lasso: In presence of few variables with medium / large sized effect, use lasso regression. In presence of many variables with small / medium sized effect, use ridge regression. Lasso regression does both variable selection and parameter shrinkage, whereas Ridge regression only does parameter shrinkage and end up including all the coefficients in the model. In presence of correlated variables, ridge regression might be the preferred choice. Also, ridge regression works best in situations where the least square estimates have higher variance. 
  * Logistic Regression: it is classification, predicting the probability of discrete values. It chooses parameters that maximize the likelihood of observing the sample values rather than that minimize the sum of squared errors (like in ordinary regression).
  * Decision Tree: serves for both categorical and numerical data. Split with the most significant variable each time to make as distinct groups as possible, using various techniques like Gini, Information Gain = (1- entropy), Chi-square. A decision tree algorithm is known to work best to detect non – linear interactions. The reason why decision tree failed to provide robust predictions because it couldn’t map the linear relationship as good as a regression model did. 
  * SVM: seperate groups with a line and maximize the margin distance. Good for small dataset, especially those with large number of features
  * Naive Bayes: the assumption of equally importance and the independence between predictors. Very simple and good for large dataset, also majorly used in text classification and multi-class classification. <b>Likelihood</b> is the probability of classifying a given observation as 1 in presence of some other variable. For example: The probability that the word ‘FREE’ is used in previous spam message is likelihood. <b>Marginal likelihood</b> is, the probability that the word ‘FREE’ is used in any message. 
  * KNN: can be used for both classification and regression. Computationally expensive since it stores all the cases. Variables should be normalized else higher range variables can bias it. Data preprocessing before using KNN, such as dealing with outliers, missing data, noise
  * K-Means
  * Random Forest: bagging, which means if the number of cases in the training set is N, then sample of N cases is taken at random but with replacement. This sample will be the training set for growing the tree. If there are M input variables, a number m<<M is specified such that at each node, m variables are selected at random out of the M and the best split on these m is used to split the node. The value of m is held constant during the forest growing. Each tree is grown to the largest extent possible. <b>There is no pruning</b>. Random Forest has to go with cross validation, otherwise overfitting could happen.
  * PCA: Dimensional Reduction, it selects fewer components (than features) which can explain the maximum variance in the data set, using Rotation. Personally, I like Boruta Feature Selection. Filter Methods for feature selection are my second choice. <b>Remove highly correlated variables before using PCA</b>
  * GBM (try C50, XgBoost at the same time in practice)
  * Difference between Random Forest and GBM: Random Forest is bagging while GBM is boosting. In bagging technique, a data set is divided into n samples using randomized sampling. Then, using a <b>single learning algorithm</b> a model is build on all samples. Later, the resultant predictions are combined using voting or averaging. <b>Bagging is done is parallel.</b> In boosting, after the first round of predictions, the algorithm weighs misclassified predictions higher, such that they can be corrected in the succeeding round. This sequential process of giving higher weights to misclassified predictions continue until a stopping criterion is reached.Random forest improves model accuracy by reducing variance (mainly). The trees grown are uncorrelated to maximize the decrease in variance. On the other hand, <b>GBM improves accuracy my reducing both bias and variance in a model.</b>
 
* Online Learning vs Batch Learning: https://www.analyticsvidhya.com/blog/2015/01/introduction-online-machine-learning-simplified-2/
 
* Optimization - Genetic Algorithm
  * More about Crossover and Mutation: https://www.researchgate.net/post/What_is_the_role_of_mutation_and_crossover_probability_in_Genetic_algorithms
  
* Survey of Optimization
  * Page 10, strength and weakness of each optimization method: https://github.com/hanhanwu/readings/blob/master/SurveyOfOptimization.pdf
  
* Optimization - Gradient Descent
  * Reference: https://www.analyticsvidhya.com/blog/2017/03/introduction-to-gradient-descent-algorithm-along-its-variants/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Challenges for gradient descent
    * data challenge: cannot be used on non-convex optimization problem; may end up at local optimum instead of global optimum; may even not an optimal point when gradient is 0 (saddle point)
    * gradient challenge: when gradient is too small or too large, vanishing gradient or exploding gradient could happen
    * implementation chanllenge: memory, hardware/software limitations
  * Type 1 - Vanilla Gradient Descent
    * "Vanilla" means pure here
    * `update = learning_rate * gradient_of_parameters`
    * `parameters = parameters - update`
  * Type 2 - Gradient Descent with Momentum
    * `update = learning_rate * gradient`
    * `velocity = previous_update * momentum`
    * `parameter = parameter + velocity – update`
    * With `velocity`, it considers the previous update
  * Type 3 - ADAGRAD
    * ADAGRAD uses adaptive technique for learning rate updation. 
    * `grad_component = previous_grad_component + (gradient * gradient)`
    * `rate_change = square_root(grad_component) + epsilon`
    * `adapted_learning_rate = learning_rate * rate_change`
    * `update = adapted_learning_rate * gradient`
    * `parameter = parameter – update`
    * `epsilon` is a constant which is used to keep rate of change of learning rate
  * Type 4 - ADAM
    * ADAM is one more adaptive technique which builds on adagrad and further reduces it downside. In other words, you can consider this as momentum + ADAGRAD.
    * `adapted_gradient = previous_gradient + ((gradient – previous_gradient) * (1 – beta1))`
    * `gradient_component = (gradient_change – previous_learning_rate)`
    * `adapted_learning_rate =  previous_learning_rate + (gradient_component * (1 – beta2))`
    * `update = adapted_learning_rate * adapted_gradient`
    * `parameter = parameter – update`
  * Tips for choose models
    * For rapid prototyping, use adaptive techniques like Adam/Adagrad. These help in getting quicker results with much less efforts. As here, you don’t require much hyper-parameter tuning.
    * To get the best results, you should use vanilla gradient descent or momentum. gradient descent is slow to get the  desired results, but these results are mostly better than adaptive techniques.
    * If your data is small and can be fit in a single iteration, you can use 2nd order techniques like l-BFGS. This is because 2nd order techniques are extremely fast and accurate, but are only feasible when data is small enough
 
 
************************************************************************

Data Visualization

* Previous Visualization collections, check "VISUALIZATION" section: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/README.md

* Readings
  * Psychology of Intelligence Analysis: https://www.cia.gov/library/center-for-the-study-of-intelligence/csi-publications/books-and-monographs/psychology-of-intelligence-analysis/PsychofIntelNew.pdf
  * How to lie with Statistics: http://www.horace.org/blog/wp-content/uploads/2012/05/How-to-Lie-With-Statistics-1954-Huff.pdf
  * The Signal and the Noise (to respect the author): https://www.amazon.com/Signal-Noise-Many-Predictions-Fail-but/dp/0143125087/ref=sr_1_1?ie=UTF8&qid=1488403387&sr=8-1&keywords=signal+and+the+noise
  * Non-designer's design book: https://diegopiovesan.files.wordpress.com/2010/07/livro_-_the_non-designers_desi.pdf

* NLP Visualization
  * On Jan 20, 2017, SFU Linguistics Lab invited an UBC researcher to show NLP data visualization, which is very interesting. By doing topic modeling, graph base clustering, they are able to categorize large amount of comments and opinions into groups, by using interactive visualization, the tools they developed will help readers read online comments in a more efficient way.
  * ConVis: https://www.cs.ubc.ca/cs-research/lci/research-groups/natural-language-processing/ConVis.html
  * MultiConVis: https://www.cs.ubc.ca/cs-research/lci/research-groups/natural-language-processing/MultiConVis.html

* Python Bokeh, an open source data visualization tool (its presentation target is web browser, but I think Tableau and d3 all could do this): https://www.analyticsvidhya.com/blog/2015/08/interactive-data-visualization-library-python-bokeh/?utm_content=buffer58668&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer

* Tableau Resources
  * Reference Guide: http://www.dataplusscience.com/TableauReferenceGuide/
  * Advanced Highlight: http://onlinehelp.tableau.com/current/pro/desktop/en-us/actions_highlight_advanced.html
  * Having 1+ pills in Color: http://drawingwithnumbers.artisart.org/measure-names-on-the-color-shel/
  * Mark Labels: http://onlinehelp.tableau.com/current/pro/desktop/en-us/annotations_marklabels_showhideindividual.html
  * Show or hide labels: http://paintbynumbersblog.blogspot.ca/2013/08/a-quick-tableau-tip-showing-and-hiding.html
  * Filter and parameter: https://www.quora.com/What-is-the-difference-between-filters-and-parameters-in-a-tableau-What-is-an-explanation-of-this-scenario
  * Tableau detailed user guide on parameter: http://onlinehelp.tableau.com/current/pro/desktop/en-us/help.html#parameters_swap.html
  * Logincal function: http://onlinehelp.tableau.com/current/pro/desktop/en-us/functions_functions_logical.html
  * Tableau functions: http://onlinehelp.tableau.com/current/pro/desktop/en-us/functions.html
 
* Jigsaw - Data Visualization for text data
  * Tutorial Videos (each video is very short, good): http://www.cc.gatech.edu/gvu/ii/jigsaw/tutorial/
  * Jigsaw manual: http://www.cc.gatech.edu/gvu/ii/jigsaw/tutorial/manual/#initiating-session
 
* Gephi
  * It create interactive graph-network visualization
  * Tutorials: https://gephi.org/users/
  * This is a sample data input, showing what kind of data structure you need for visualizing network in Grphi, normally, 2 files, one for Nodes one for Edges: https://github.com/hanhanwu/Hanhan_Data_Science_Resources2/blob/master/gephi%20data%20set.zip
  * With the dataset I give you here, you will be able to find girls dinning group at school and make assumptions about how rumor spreaded (network is so scary, right?)
  * Here is what I listed about the advantages of using Gephi for network visualization, compared with writing python visualization: https://github.com/hanhanwu/Hanhan_Data_Science_Resources2/blob/master/gephi.png
  * But something is wrong with the Gephi on my machine.. I don't have partition or ranking settings and cannot choose dirested graph and so on...
  * However, seeing the changes of those vosualization with different graph algorithm is interesting, for example, Force Atlas 2: https://github.com/gephi/gephi/wiki/Force-Atlas-2
 
* MicrosoStrategy - Visual Analytics Course List: https://www.microstrategy.com/us/services/education/course-list#filter-role:path=default|filter-platform:path=default|filter-version:path=._10_5|filter-certification:path=default|sort-course-number:path~type~order=.card-corner~text~asc

* Python Elastic Search & Kibana for data visualization: https://www.analyticsvidhya.com/blog/2017/05/beginners-guide-to-data-exploration-using-elastic-search-and-kibana/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * I still prefer R or Spark for visualized data exploration, if you care about the speed. otherwise, tableau
  * "Elastic Search is an open source, RESTful distributed and scalable search engine. Elastic search is extremely fast in fetching results for simple or complex queries on large amounts of data (Petabytes) because of it’s simple design and distributed nature. It is also much easier to work with than a conventional database constrained by schemas, tables."
  * The tutorial prevides the method to indexing the data for Elastic Search

* My d3 Practice
 * code: https://github.com/hanhanwu/Hanhan_d3


************************************************************************

Big Data

* Pig vs Hive: https://upxacademy.com/pig-vs-hive/
* Spark 2.0, SparkSession (can be used for both Spark SQL Context and Hive Context, Spark SQL is based on Haive but has its own strength): http://cdn2.hubspot.net/hubfs/438089/notebooks/spark2.0/SparkSession.html
* Mastering Spark 2.0: https://github.com/hanhanwu/Hanhan_Data_Science_Resources2/blob/master/Mastering-Apache-Spark-2.0.pdf
  * Spark Session - supports both SQL Context and Hive Context
  * Structured Streaming - just write batch computation and let Spark deal with streaming with you. I’m waiting to see its better integration with MLLib and other machine learning libraries
  * HyperLogLog - the story is interesting
  * `val ds = spark.read.json("/databricks-public-datasets/data/iot/iot_devices.json").as[DeviceIoTData]`
  * Choice of Spark DataFrame, DataSets and RDD, P35
  * Structured Streaming, P57-59
* Lessons from large scale machine learning deployment on Spark, 2.0: https://github.com/hanhanwu/Hanhan_Data_Science_Resources2/blob/master/Lessons_from_Large-Scale_Machine_Learning_Deployments_on_Spark.pdf
* Hadoop 10 years: https://upxacademy.com/hadoop-10-years/
* Hive function cheat sheet: https://www.qubole.com/resources/cheatsheet/hive-function-cheat-sheet/
* Distributing System
  * Spack New Physical Plan: https://databricks.com/blog/2017/04/01/next-generation-physical-planning-in-apache-spark.html?utm_campaign=Databricks&utm_content=51844317&utm_medium=social&utm_source=linkedin
  * CAP Theorem: https://en.wikipedia.org/wiki/CAP_theorem
  * CAPP: https://drive.google.com/file/d/0B3Um1hpy8q7gVjhVT3dGUWFxRm8/view
* S survey summary on NoSQL: https://www.linkedin.com/pulse/survey-nosql-key-value-pair-databases-uzma-ali-pmp?trk=v-feed&lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BspSXDtIY3PN6lbk4C%2FTwTg%3D%3D
  * I like the summary table in this poster
  * They did survey on Cassandra, Redis, DynamoDB, Riak, HBASE, Voldemort (among them, I have only used HBASE so far...)
* A book - I Heart Logs
  * The book: https://github.com/hanhanwu/readings/blob/master/I_Heart_Logs.pdf
  * It talks about how powerful logs can be in distributing systems, streaming processing, etc, each section has a real-life example. I like the streaming processing part in this book most.
  * It is a great book that has broaden and deepen the concepts we have heard many time times but may has misunderstanding, such as "logs", "streaming processing" and even "ETL".
  * Google Caffeine: https://googleblog.blogspot.ca/2010/06/our-new-search-index-caffeine.html
    * It rebuilt its web crawling, processing, and indexing pipeline—what has to be one of the most complex, largest scale data processing systems on the planet —on top of a stream processing system.
* NoSQL
  * CAP theorem (could only choose 2)
    * Consistency: all clients always have the same view of data
    * Avability: client can always read and write
    * Partition tolerance means that the system works well across physical network partitions
  * There are different types of NoSql databases
    * Column oriented
    * Documented oriented
    * Graph database
    * Key-value oriented
  * NoSql visual guide: http://blog.nahurst.com/visual-guide-to-nosql-systems
  * Some other description: http://stackoverflow.com/questions/2798251/whats-the-difference-been-nosql-and-a-column-oriented-database
  * Amazon DynamoDB vs. Amazon Redshift vs. Oracle NoSQL: https://db-engines.com/en/system/Amazon+DynamoDB%3BAmazon+Redshift
    * It seems that DynamoDB is NoSql while Redshift is a data warehouse based on psql   

************************************************************************

Cloud

* For Cloud Machine Learning in Spark, AWS and Azure Machine Learning, check my previous summary here: https://github.com/hanhanwu/Hanhan_Data_Science_Resources
* My AWS practice: https://github.com/hanhanwu/Hanhan_AWS

* Compute Canada & West Grid
  * How to apply for an account: https://www.computecanada.ca/research-portal/account-management/apply-for-an-account/
  * login West Grid cloud through terminal: https://www.westgrid.ca//support/quickstart/new_users#about
  * Note: the GUI the above guidance mentioned means the graphical tool, not interactive user interface...
  * Run X11 in Mac OS X with XQuartz: http://osxdaily.com/2012/12/02/x11-mac-os-x-xquartz/
  * When you are using XQuartz, once xterm appeared, loggin into Canada Computer with SSH X forwarding through xterm: https://www.westgrid.ca/support/visualization/remote_visualization/x11
  * Creat Computer Canada Cloud (it seems that Computer Canada Cloud and West Grid are different cloud, but you apply for West Grid through Compute Canada account): https://www.computecanada.ca/research-portal/national-services/compute-canada-cloud/create-a-cloud-account/
  * Tutorial for creating Compute Canada cloud (the video at the bottom): https://www.computecanada.ca/research-portal/national-services/compute-canada-cloud/ 
  * Cloud Canada Quick Start: https://docs.computecanada.ca/wiki/Cloud_Quick_Start
  * Command lines: https://github.com/hanhanwu/Hanhan_Data_Science_Resources2/blob/master/WestGrid_commands.md


************************************************************************

TEXT ANALYSIS

* Text data preprocessing basic steps: https://www.analyticsvidhya.com/blog/2015/06/quick-guide-text-data-cleaning-python/
* My NLP practice: https://github.com/hanhanwu/Hanhan_NLP
* Jigsaw - Data Visualization for text data
  * Tutorial Videos (each video is very short, good): http://www.cc.gatech.edu/gvu/ii/jigsaw/tutorial/
  * Jigsaw manual: http://www.cc.gatech.edu/gvu/ii/jigsaw/tutorial/manual/#initiating-session


************************************************************************

Non-Machine Learning Data Analysis Examples亖？云

* Analysis with data visualization: https://www.analyticsvidhya.com/blog/2016/12/who-is-the-superhero-of-cricket-battlefield-an-in-depth-analysis/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


************************************************************************

AI

* Deep Learning basic concepts (it's a real good one!): https://www.analyticsvidhya.com/blog/2017/05/25-must-know-terms-concepts-for-beginners-in-deep-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * how does `activation function` work:
    * `u = ∑W*X+b`, X is the input vector, W is the weights vector, b is bias
    * apply activation function `f()` to u, `f(u)` is the output of the layer
  * Different activation functions
    * The <b>sigmoid</b> transformation generates a more smooth range of values between 0 and 1. We might need to observe the changes in the output with slight changes in the input values. Smooth curves allow us to do that and are hence preferred over step functions.
    * The major benefit of using <b>ReLU</b> is that it has a constant derivative value for all inputs greater than 0. The constant derivative value helps the network to train faster.
    * <b>softmax</b>, normally used in the output layer for classification. It is similar to sigmoid but its output is normalized to make the sum as 1. Meanwhile, sigmoid is used for binary classification while softmax can be used on multi-class classification.
  * Gradient Descent & Optimization & Cost function: Gradient descent is an optimization method, aiming at minimizing the cost/loss
  * Backpropagation is used to update weights
  * Batches: While training a neural network, instead of sending the entire input in one go, we divide in input into <b>several chunks of equal size randomly</b>. Training the data on batches makes the model more generalized as compared to the model built when the entire data set is fed to the network in one go.
  * Epochs: An epoch is defined as <b>a single training iteration of all batches</b> in both forward and back propagation. This means 1 epoch is a single forward and backward pass of the entire input data. Higher epochs could lead to higher accuracy but maybe also overfitting, the higher one also takes longer time.
  * Dropout: Dropout is a regularization technique which prevents over-fitting of the network. When training a certain number of neurons in the hidden layer is randomly dropped. This means that the training happens on several architectures of the neural network on different combinations of the neurons. You can think of drop out as an ensemble technique, where the output of multiple networks is then used to produce the final output.
  * Batch Normalization, it is used to ensure the data distribution will be the same as the next layer expected. Because after backpropagation, the weights changed and the data distribution may also changed while te next layer expects to see similar data distribution it has seen before.
  * CNN (Convolutional Neural Network)
    * Filter: is a smaller window of data, it used to filter the entire data into multiples windows, each window generates a convoluted value. All the convoluted values form a new set of data. Using this method, an image can be convoluted into less parameters. CNN is often used on image
    * Pooling: a pooling layer is often added between convolutional layers to reduce parameters in order to reduce overfitting. For example, in practice, MAX pooling works better
    * Padding: Padding refers to adding extra layer of zeros across the images so that the output image has the same size as the input
    * Data Augmentation: the addition of new data derived from the given data, which might prove to be beneficial for prediction. Such as brightening/rotating the image
  * RNN (Recurrent Neural Network)
    * Recurrent Neuron: A recurrent neuron is one in which the output of the neuron is sent back to it for t time stamps.
    * RNN is often used for sequential data, such as time series
  * Vanishing Gradient Problem – Vanishing gradient problem arises in cases where the gradient of the activation function is very small. During back propagation when the weights are multiplied with these low gradients, they tend to become very small and “vanish” as they go further deep in the network. This makes the neural network to forget the long range dependency. This generally becomes a problem in cases of recurrent neural networks where long term dependencies are very important for the network to remember. <b>This can be solved by using activation functions like ReLu which do not have small gradients.</b>
  * Exploding Gradient Problem – This is the exact opposite of the vanishing gradient problem, where the gradient of the activation function is too large. During back propagation, it makes the weight of a particular node very high with respect to the others rendering them insignificant. This can be easily solved by clipping the gradient so that it doesn’t exceed a certain value.
* Deep Learning talks from PyData (it also has some other data science talks): https://www.analyticsvidhya.com/blog/2017/05/pydata-amsterdam-2017-machine-learning-deep-learning-data-science/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Neural Network vs Deep Learning: <b>When there are more hidden layers and increase depth of neural network a neural network becomes deep learning.</b>
* Something about AI (those brief explaination about real life applications are useful and intresting): https://www.analyticsvidhya.com/blog/2016/12/artificial-intelligence-demystified/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Deep Learning Videos: https://www.analyticsvidhya.com/blog/2016/12/21-deep-learning-videos-tutorials-courses-on-youtube-from-2016/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* <b>Deep Learning Learning Resources</b>: https://www.analyticsvidhya.com/blog/2016/08/deep-learning-path/
* Reinforcement Learning Open Sources: https://www.analyticsvidhya.com/blog/2016/12/getting-ready-for-ai-based-gaming-agents-overview-of-open-source-reinforcement-learning-platforms/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Reinforcement Learning with Python Example: https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Deeping Learning APIs, help you build simple apps (it's interesting): https://www.analyticsvidhya.com/blog/2017/02/6-deep-learning-applications-beginner-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* 5 More Deep Learning APIs [Python]: https://www.analyticsvidhya.com/blog/2017/02/5-deep-learning-applications-beginner-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Deep Learning Skillset Test1: https://www.analyticsvidhya.com/blog/2017/01/must-know-questions-deep-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Neural Networks cannot do data preprocessing themselves
  * <b>Neural Network Dropout</b> can be seen as an extreme form of <b>Bagging</b> in which each model is trained on a single case and each parameter of the model is very strongly regularized by sharing it with the corresponding parameter in all the other models.
  * Neural Network uses hidden layers to reduce dimensionality, it is based on  <b>predictive capability</b> of the features. By comparison, PCA does dimensional reduction based on <b>feature correlation</b>
  * People set a metric called <b>bayes error</b> which is the error they hope to achieve, this is because: Input variables may not contain complete information about the output variable; System (that creates input-output mapping) may be stochastic; Limited training data
  * The number of neurons in the output layer dose NOT have to match the number of classes. If your outputis using one-hot encoding, they have to match. Otherwise, you can use other methods. For example, 2 neurons represent 4 classes using binary bits (00, 01, 10, 11)
  * Without knowing what are the weights and biases of a neural network, we cannot comment on what output it would give.
  * Convolutional Neural Network would be better suited for image related problems because of its inherent nature for taking into account changes in nearby locations of an image
  * Recurrent neural network works best for sequential data. Recurrent neuron can be thought of as a neuron sequence of infinite length of time steps. Dropout does not work well with recurrent layer.
  * A neural network is said to be a universal function approximator, so it can theoretically represent any decision boundary.
  * To decrease the “ups and downs” when visualizing errors, you can try to increase the batch size. But the "ups and downs" are no need to worry as long as there is a cumulative decrease in both training and validation error.
  * When you want to re-use a pre-trained NN model on similar problems, you can keep the previous layers but only re-train the last layer, since all the previous layers work as feature extractors
  * To deal with overfitting in NN, you can use Dropout, Regularization and Batch Normalization. Using Batch Normalization is possbile to reach higher level accuracy. Dropout is designed as a regulazer in order to reduce the gap between the tester and the trainer; Batch Normalization is designed to make optimization easier, so it does less regularization. So Batch Normalization is not as strong as dropout. <b>When the dataset is very small, Dropout should be better than Batch Normalization.</b> Batch Normalization video: https://www.youtube.com/watch?v=Xogn6veSyxA
* Deep Learning Skillset Test2: https://www.analyticsvidhya.com/blog/2017/04/40-questions-test-data-scientist-deep-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * NN is a representative algorithm, which means: it converts data to a form that better solve the problem
  * <b>Perplexity</b> is a commonly used evaluation technique when applying deep learning for NLP tasks. Lower the perplexity the better.
  * Sigmoid was the most commonly used activation function in neural network, until an issue was identified. The issue is that when the gradients are too large in positive or negative direction, the resulting gradients coming out of the activation function get squashed. This is called <b>saturation of the neuron</b>. That is why ReLU function was proposed, which kept the gradients same as before in the positive direction. ReLU also gets saturated, but it's on the negative side of x-axis.
  * Dropout Rate is the probability of keeping a neuron active. Higher the dropout rate, lower is the regularization
  * l-BFGS is a second order gradient descent technique whereas SGD is a first order gradient descent technique. When `Data is sparse` or `Number of parameters of neural network are small`, l-BFGS is better
  * For non-continuous objective during optimization in deep neural net, Subgradient method is better
* Some ideas about GPU: https://www.analyticsvidhya.com/blog/2017/05/gpus-necessary-for-deep-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


************************************************************************

Experiences/Suggestions from Others

* From a data scientist (I agree with many points he said here, especially the one to get enough sleep, I also think we have to have enough good food before focusing on data science work, this is an area really needs strong focus and cost energy): https://www.analyticsvidhya.com/blog/2016/12/exclusive-ama-with-data-scientist-sebastian-raschka/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Suggestons for analytics learning (I agree with some, although I don't think they should call them "rules"): https://www.analyticsvidhya.com/blog/2014/04/8-rules-age-analytics-learning/?utm_content=buffer9e51f&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


************************************************************************

Data Science Skillset Tests

* Regression skillset test: https://www.analyticsvidhya.com/blog/2016/12/45-questions-to-test-a-data-scientist-on-regression-skill-test-regression-solution/?utm_content=buffer5229b&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Tree based skillset test: https://www.analyticsvidhya.com/blog/2016/12/detailed-solutions-for-skilltest-tree-based-algorithms/?utm_content=bufferde46d&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Clustering Skillset test: https://www.analyticsvidhya.com/blog/2017/02/test-data-scientist-clustering/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Machine Learning Skillset test: https://www.analyticsvidhya.com/blog/2017/04/40-questions-test-data-scientist-machine-learning-solution-skillpower-machine-learning-datafest-2017/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * 2 variables can relate to each other but with 0 pearson correlation
  * SGD vs GD: In SGD for each iteration you choose the batch which is generally contain the random sample of data But in case of GD each iteration contain the all of the training observations.
* Statistics Skillset test: https://www.analyticsvidhya.com/blog/2017/05/41-questions-on-statisitics-data-scientists-analysts/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Probability Skillset test: https://www.analyticsvidhya.com/blog/2017/04/40-questions-on-probability-for-all-aspiring-data-scientists/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Some answers may have problems. For example, I think if question 33 has the right answer then question 28 has the wrong answer
  * Majorly uses the Byesian Theorem taught in conditional probability theorem: https://www.analyticsvidhya.com/blog/2017/03/conditional-probability-bayes-theorem/
  * An interesting take-away is about Monty Hall’s problem (I don't fullt understand, especially after seeing the debat after the post), the problem looks interesting: https://www.analyticsvidhya.com/blog/2014/04/probability-action-monty-halls-money-show/
  * When applying Onehot Encoding, make sure the frequency distribution is the same in training and testing data
  * Output value range:
    * tanh function: [-1, 1]
    * SIGMOID function: [0, 1]
    * ReLU function: [0, infinite]
  * When there are multicollinear features (highly correlated features), solutions can be:
    * remove one of the correlated variables
    * use penalized regression models like ridge or lasso regression
  * Ensembling is using weak learners, these learners are less likely to have overfit since each of them are sure about part of the problems, and therefore they may have low variance but high bias (how much the predicted value is different from the real value)
  * If a classifier is confident about an incorrect classification, then log-loss will penalise it heavily. For a particular observation, the classifier assigns a very small probability for the correct class then the corresponding contribution to the log-loss will be very large. Lower the log-loss, the better is the model.
* Ensembling Skillset test: https://www.analyticsvidhya.com/blog/2017/02/40-questions-to-ask-a-data-scientist-on-ensemble-modeling-techniques-skilltest-solution/?
  * Creating an ensemble of diverse models is a very important factor to achieve better results. Generally, an ensemble method works better, if the individual base models have less correlation among predictions
  * If you have m base models in stacking, that will generate m features for second stage models
  * Dropout in a neural network can be considered as an ensemble technique, where multiple sub-networks are trained together by “dropping” out certain connections between neurons.
  * !! Bagging of unstable classifiers is a good idea. [Based on this paper][14], "If perturbing the learning set can cause signicant
changes in the predictor constructed, then bagging can improve accuracy."
* Time series skillset test: https://www.analyticsvidhya.com/blog/2017/04/40-questions-on-time-series-solution-skillpower-time-series-datafest-2017/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Dimensional Reduction skillset test: https://www.analyticsvidhya.com/blog/2017/03/questions-dimensionality-reduction-data-scientist/?utm_content=bufferc792d&utm_medium=social&utm_source=linkedin.com&utm_campaign=buffer
* Deep Learning Skillset Test1: https://www.analyticsvidhya.com/blog/2017/01/must-know-questions-deep-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Deep Learning Skillset Test2: https://www.analyticsvidhya.com/blog/2017/04/40-questions-test-data-scientist-deep-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* R Skillset test: https://www.analyticsvidhya.com/blog/2017/05/40-questions-r-for-data-science/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* SQL Skillset test1: https://www.analyticsvidhya.com/blog/2017/01/46-questions-on-sql-to-test-a-data-science-professional-skilltest-solution/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * TRUNCATE vs DELETE: `truncate` is ddl (data definition language) command, it does not has rollback info and will release the memory; `delete`  is dml (data manipulation language) command, it contains rollback info and will not release the memory.
  * If a relation is satisfying higher normal forms, it automatically satisfies lower normal forms. Such as if it satisfies 3NF, it should automatically satisfies 1NF.
  * Minimal super key is a candidate key. Only one Candidate Key can be Primary Key.
  * PROJECT vs SELECT: In relational algebra ‘PROJECT’ operation gives the unique record but in case of ‘SELECT’ operation in SQL you need to use distinct keyword for getting unique records.
  * SQL Index doesn’t help for the `like` clause. The addition of the index didn’t change the query execution plan.  for example, the index on rating will not work for the query (Salary * 100 > 5000). But you can create an index on (Salary * 100) which will help.
  * `CREATE TABLE avian ( emp_id SERIAL PRIMARY KEY, name varchar);` creates index as primary key
  * My disagrees: Q10, the answer should be (B)
* SQL Skillset test2: https://www.analyticsvidhya.com/blog/2017/05/questions-sql-for-all-aspiring-data-scientists/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * When you have created a view based on the table, using `drop` cannot drop the table. Try `drop .. cascade` will drop the table and its dependent objects too
  * <b>cartesian product</b>: https://en.wikipedia.org/wiki/Cartesian_product
  * My disagree: Q1, the answer should be B
  * Some of the answers in this test make me doubt, such as Q2, Q4. Meanwhile, Q8, Q35 has confusing question
  * In Q27, the answer description should be, "A" to 1, "N" to 2 and "K" to 3. Did the author fall asleep when wrinting this article?
  * In Q28, the column names should start with an upper case, otherwise it will be an error
  * I didn't think about questions from 39 to 42, but only quickly went through them, it seems that the question of Q40 is still not logically strict to me. I really cannot stand this post. Howcome it has so many problems.
* Python Skillset test: https://www.analyticsvidhya.com/blog/2017/05/questions-python-for-data-science/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * I didn't check this one
* Interview questions: https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/


************************************************************************

Interview Tips

* Tips for Estimate Questions: https://www.analyticsvidhya.com/blog/2014/01/tips-crack-guess-estimate-case-study/?utm_content=buffer5f90d&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* 4 R interview questions (the last 2): https://www.analyticsvidhya.com/blog/2014/05/tricky-interview-questions/
* Data Science Hiring Guidance (I especially like the last part, which questions to ask employers): https://github.com/hanhanwu/Hanhan_Data_Science_Resources2/blob/master/Data%20Science%20Hiring%20Guide.pdf


************************************************************************

TRAIN YOUR BRAIN

* Interview puzzles I: https://www.analyticsvidhya.com/blog/2014/09/commonly-asked-puzzles-analytics-interviews/
* Interview puzzles II: https://www.analyticsvidhya.com/blog/2014/10/commonly-asked-interview-puzzles-part-ii/
* Train mind analytical thinking: https://www.analyticsvidhya.com/blog/2014/01/train-mind-analytical-thinking/
* Brain training for analytical thinking: https://www.analyticsvidhya.com/blog/2015/07/brain-training-analytical-thinking/


************************************************************************

OTHER

* Make a wish to Data Science Santa! (I really like this idea, you just need to choose your 2017 data science learning goals and finally, they will give you a Christmas gift which is full of  relative learning resurces cater for your goals!): https://www.analyticsvidhya.com/blog/2016/12/launching-analytics-vidhya-secret-santa-kick-start-2017-with-this-gift/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Interesting Data Science videos (I mean some of them looks interesting): https://www.analyticsvidhya.com/blog/2016/12/30-top-videos-tutorials-courses-on-machine-learning-artificial-intelligence-from-2016/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Some ideas about social media analysis: https://www.analyticsvidhya.com/blog/2017/02/social-media-analytics-business/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Linear Programming: https://www.analyticsvidhya.com/blog/2017/02/lintroductory-guide-on-linear-programming-explained-in-simple-english/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* How to create an R package, and publish on CRAN, GitHub: https://www.analyticsvidhya.com/blog/2017/03/create-packages-r-cran-github/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Behavioral Science: https://www.analyticsvidhya.com/blog/2017/04/behavioral-analytics-when-psychology-collides-analytics/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Most of the fundamentals in behavioral science apply on any type of population.
  * compromise effect - Human has a tendency to find the middle option.
  * decoy effect - Human mind is trained to make choices between similar objects
  * anchoring effect - The price of any item is based on perception rather than the actual cost of the raw materials used.  We start with a price and start bidding higher. The starting price is an anchor and the final price at which the item is sold is highly correlated to this anchor.
  * steep temporal discounting effect - We have a tendency to value money in near future with a strong discounting factor but such discounting factor becomes small when we talk about longer time frames.
  * “unknown unknowns” effect - Human has a tendency to underestimate probabilities when they face ambiguity.
  * extreme probability bias effect - We tend to underestimate probability between 0 and 0.5 if event is favorable and exactly opposite happens when event is unfavorable. With underestimated perceived probability, we underestimate the value of Risky transactions.
* Winnning Strategy in Casino (blackjack): https://www.analyticsvidhya.com/blog/2017/04/flawless-winning-strategy-casino-blackjack-data-science/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Looks calculation intensive, I dind't read them throughly, but only checked the insights :)



[1]:https://www.analyticsvidhya.com/blog/2017/01/comprehensive-practical-guide-inferential-statistics-data-science/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[2]:http://www.danielsoper.com/statcalc/calculator.aspx?id=98
[3]:http://stattrek.com/online-calculator/f-distribution.aspx
[4]:http://stattrek.com/online-calculator/chi-square.aspx
[5]:https://www.analyticsvidhya.com/blog/2017/02/basic-probability-data-science-with-examples/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[6]:https://s3.amazonaws.com/udacity-hosted-downloads/ZTable.jpg
[7]:https://www.analyticsvidhya.com/blog/2017/03/conditional-probability-bayes-theorem/?utm_content=buffer7afce&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer
[8]:https://en.wikipedia.org/wiki/Statistical_dispersion
[9]:https://github.com/hanhanwu/Hanhan_Data_Science_Resources
[10]:https://www.analyticsvidhya.com/blog/2015/07/dimension-reduction-methods/
[11]:https://www.analyticsvidhya.com/blog/2015/11/8-ways-deal-continuous-variables-predictive-modeling/?utm_content=bufferfb56f&utm_medium=social&utm_source=linkedin.com&utm_campaign=buffer
[12]:http://pegasus.cc.ucf.edu/~pepe/Tables
[13]:http://www.sjsu.edu/faculty/gerstman/StatPrimer/t-table.pdf
[14]:http://statistics.berkeley.edu/sites/default/files/tech-reports/421.pdf
