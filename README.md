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


************************************************************************

TREE BASED MODELS

* Tree based models in detail with R & Python example: https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/?utm_content=bufferade26&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


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

Applied Data Science in Python/R

* [R] Caret package for data imputing, feature selection, model training (I will show my experience of using caret with detailed code in Hanhan_Data_Science_Practice): https://www.analyticsvidhya.com/blog/2016/12/practical-guide-to-implement-machine-learning-with-caret-package-in-r-with-practice-problem/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* [Python & R] A brief classified summary of Python Scikit-Learn and R Caret: https://www.analyticsvidhya.com/blog/2016/12/cheatsheet-scikit-learn-caret-package-for-python-r-respectively/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* [Python] What to pay attention to when you are using Naive Bayesian with Scikit-Learn: https://www.analyticsvidhya.com/blog/2015/09/naive-bayes-explained/?utm_content=bufferaa6aa&utm_medium=social&utm_source=linkedin.com&utm_campaign=buffer
  * 3 types of Naive Bayesian: Gaussian (if you assume that features follow a normal distribution); Multinomial (used for discrete counts, you can think of it as “number of times outcome number x_i is observed over the n trials”.); Bernoulli(useful if your <b>feature vectors are binary</b>); 
  * Tips to improve the power of Naive Bayes Model: If test data set has zero frequency issue, apply <b>smoothing techniques</b> “Laplace Correction” to predict the class of test data set. Focus on your  pre-processing of data and the feature selection, because of thelimited paramter choices. “ensembling, boosting, bagging” won’t help since their purpose is to reduce variance. <b>Naive Bayes has no variance to minimize</b>


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
 * <b>Bias</b> is useful to quantify how much on an average are the predicted values different from the actual value. A high bias error means we have a <b>under-performing</b> model which keeps on missing important trends. <b>Varianc</b> on the other side quantifies how are the prediction made on same observation different from each other. A high variance model will <b>over-fit</b> on your training population and perform badly on any observation beyond training.
  * <b>OLS</b> and <b>Maximum likelihood</b> are the methods used by the respective regression methods to approximate the unknown parameter (coefficient) value. OLS is to linear regression. Maximum likelihood is to logistic regression. Ordinary least square(OLS) is a method used in linear regression which approximates the parameters resulting in <b>minimum distance between actual and predicted values.</b> Maximum Likelihood helps in choosing the the values of parameters which <b>maximizes the likelihood that the parameters are most likely to produce observed data.</b>
  * <b>Standard Deviation</b> – It is the amount of variation in the <b>population data</b>. It is given by σ. <b>Standard Error</b> – It is the amount of variation in the <b>sample data</b>. It is related to Standard Deviation as σ/√n, where, n is the sample size, σ is the standandard deviation of the <b>population</b>
  * 95% <b> confidence interval does not mean</b> the probability of a population mean to lie in an interval is 95%. Instead, 95% C.I <b>means that 95% of the Interval estimates will contain the population statistic</b>.
  * If a sample mean lies in the margin of error range then, it might be possible that its actual value is equal to the population mean and the difference is occurring by chance.
  * <b>Difference between z-scores and t-values</b> are that t-values are dependent on Degree of Freedom of a sample, and t-values use sample standard deviation while z-scores use population standard deviation.
  * <b>The Degree of Freedom</b> – It is the number of variables that have the choice of having more than one arbitrary value. For example, in a sample of size 10 with mean 10, 9 values can be arbitrary but the 1oth value is forced by the sample mean.
  * <b>Residual Sum of Squares (RSS)</b> - It can be interpreted as the amount by which the predicted values deviated from the actual values. Large deviation would indicate that the model failed at predicting the correct values for the dependent variable. <b>Regression (Explained) Sum of Squares (ESS)</b> – It can be interpreted as the amount by which the predicted values deviated from the the mean of actual values.
  * <b>Co-efficient of Determination = ESS/(ESS + RSS)</b>. It represents the strength of correlation between two variables. <b>Correlation Coefficient = sqrt(Co-efficient of Determination)</b>, also represents the strength of correlation between two variables, ranges between [-1,1]. 0 means no correlation, 1 means strong positive correlation, -1 means strong neagtive correlation.
 
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
  * There are 2 major formulas here are important to Bayes Theorem: `P(A/B) = P(A AND B)/P(B)`, `P(b1|A) + P(b2|A) + .... + P(bn|A) = P(A)`

[1]:https://www.analyticsvidhya.com/blog/2017/01/comprehensive-practical-guide-inferential-statistics-data-science/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[2]:http://www.danielsoper.com/statcalc/calculator.aspx?id=98
[3]:http://stattrek.com/online-calculator/f-distribution.aspx
[4]:http://stattrek.com/online-calculator/chi-square.aspx
[5]:https://www.analyticsvidhya.com/blog/2017/02/basic-probability-data-science-with-examples/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
[6]:https://s3.amazonaws.com/udacity-hosted-downloads/ZTable.jpg
[7]:https://www.analyticsvidhya.com/blog/2017/03/conditional-probability-bayes-theorem/?utm_content=buffer7afce&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer


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
 
* t-SNE, non-linear dimensional reduction
  * (t-SNE) t-Distributed Stochastic Neighbor Embedding is a non-linear dimensionality reduction algorithm used for exploring high-dimensional data
  * The problem with linear dimensional reduction, is that they concentrate on placing dissimilar data points far apart in a lower dimension representation. However, it is also important to put similar data close together, linear dimensional reduction does not do this
  * In t-SNE, there are local approaches and global approaches. Local approaches seek to map nearby points on the manifold to nearby points in the low-dimensional representation. Global approaches on the other hand attempt to preserve geometry at all scales, i.e mapping nearby points to nearby points and far away points to far away points  
  * It is important to know that most of the nonlinear techniques other than t-SNE are not capable of retaining both the local and global structure of the data at the same time.
  * The algorithm computes pairwise conditional probabilities and tries to minimize the sum of the difference of the probabilities in higher and lower dimensions. This involves a lot of calculations and computations. So the algorithm is quite heavy on the system resources. t-SNE has a quadratic time and space complexity in the number of data points. This makes it particularly slow and resource draining while applying it to data sets comprising of more than 10,000 observations. 
 
* Optimization - Genetic Algorithm
  * More about Crossover and Mutation: https://www.researchgate.net/post/What_is_the_role_of_mutation_and_crossover_probability_in_Genetic_algorithms
 
 
************************************************************************

Data Visualization

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


************************************************************************

Cloud

* For Cloud Machine Learning in Spark, AWS and Azure Machine Learning, check my previous summary here: https://github.com/hanhanwu/Hanhan_Data_Science_Resources

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

Non-Machine Learning Data Analysis Examples

* Analysis with data visualization: https://www.analyticsvidhya.com/blog/2016/12/who-is-the-superhero-of-cricket-battlefield-an-in-depth-analysis/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


************************************************************************

AI

* Something about AI (those brief explaination about real life applications are useful and intresting): https://www.analyticsvidhya.com/blog/2016/12/artificial-intelligence-demystified/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Deep Learning Videos: https://www.analyticsvidhya.com/blog/2016/12/21-deep-learning-videos-tutorials-courses-on-youtube-from-2016/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Reinforcement Learning Open Sources: https://www.analyticsvidhya.com/blog/2016/12/getting-ready-for-ai-based-gaming-agents-overview-of-open-source-reinforcement-learning-platforms/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Reinforcement Learning with Python Example: https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Deeping Learning APIs, help you build simple apps (it's interesting): https://www.analyticsvidhya.com/blog/2017/02/6-deep-learning-applications-beginner-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* 5 More Deep Learning APIs [Python]: https://www.analyticsvidhya.com/blog/2017/02/5-deep-learning-applications-beginner-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


************************************************************************

Experiences/Suggestions from Others

* From a data scientist (I agree with many points he said here, especially the one to get enough sleep, I also think we have to have enough good food before focusing on data science work, this is an area really needs strong focus and cost energy): https://www.analyticsvidhya.com/blog/2016/12/exclusive-ama-with-data-scientist-sebastian-raschka/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Suggestons for analytics learning (I agree with some, although I don't think they should call them "rules"): https://www.analyticsvidhya.com/blog/2014/04/8-rules-age-analytics-learning/?utm_content=buffer9e51f&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


************************************************************************

Data Science Skillset Tests

* Regression: https://www.analyticsvidhya.com/blog/2016/12/45-questions-to-test-a-data-scientist-on-regression-skill-test-regression-solution/?utm_content=buffer5229b&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Tree based skillset test: https://www.analyticsvidhya.com/blog/2016/12/detailed-solutions-for-skilltest-tree-based-algorithms/?utm_content=bufferde46d&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
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
