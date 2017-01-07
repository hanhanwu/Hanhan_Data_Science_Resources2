# Hanhan_Data_Science_Resources2
More data science resources
It seems that my Data Science Resources cannot be updated, create a new one here for more resources


Hanhan_Data_Science_Resource 1: https://github.com/hanhanwu/Hanhan_Data_Science_Resources


************************************************************************

TREE BASED MODELS

* Tree based models in detail with R & Python example: https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/?utm_content=bufferade26&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


************************************************************************

Applied Data Science in Python/R

* [R] Caret package for data imputing, feature selection, model training (I will show my experience of using caret with detailed code in Hanhan_Data_Science_Practice): https://www.analyticsvidhya.com/blog/2016/12/practical-guide-to-implement-machine-learning-with-caret-package-in-r-with-practice-problem/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* [Python & R] A brief classified summary of Python Scikit-Learn and R Caret: https://www.analyticsvidhya.com/blog/2016/12/cheatsheet-scikit-learn-caret-package-for-python-r-respectively/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


************************************************************************

Statistics in Data Science

* Termology glossary for statistics in machine learning: https://www.analyticsvidhya.com/glossary-of-common-statistics-and-machine-learning-terms/
* Statistics behind Boruta feature selection: https://github.com/hanhanwu/Hanhan_Data_Science_Resources2/blob/master/boruta_statistics.pdf
* How the laws of group theory provide a useful codification of the practical lessons of building efficient distributed and real-time aggregation systems: https://www.infoq.com/presentations/abstract-algebra-analytics
* Confusing Concepts
 * Errors and Residuals: https://en.wikipedia.org/wiki/Errors_and_residuals
 * Heteroskedasticity: led by non-constant variance in error terms. Usually, non-constant variance is caused by outliers or extreme values
 * Coefficient and p-value/t-statistics: coefficient measures the strength of the relationship of 2 variables, while p-value/t-statistics measures how strong the evidence that there is non-zero association
 * Anscombe's quartet comprises four datasets that have nearly identical simple statistical properties, yet appear very different when graphed: https://en.wikipedia.org/wiki/Anscombe's_quartet
 * Difference between gradient descent and stochastic gradient descent: https://www.quora.com/Whats-the-difference-between-gradient-descent-and-stochastic-gradient-descent


************************************************************************

Machine Learning Algorithms

* KNN with R example: https://www.analyticsvidhya.com/blog/2015/08/learning-concept-knn-algorithms-programming/
 * KNN unbiased and no prior assumption, fast
 * It needs good data preprocessing such as missing data imputing, categorical to numerical
 * k normally choose the square root of total data observations

* SVM with Python example: https://www.analyticsvidhya.com/blog/2015/10/understaing-support-vector-machine-example-code/?utm_content=buffer02b8d&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 
* Basic Essentials of Some Popular Machine Learning Algorithms with R & Python Examples: https://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/?utm_content=buffer00918&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 * Linear Regression: Y = aX + b, a is slope, b is intercept. When there is only 1 independent variable, it is Simple Linear Regression, when there are multiple independent variables, it is Multiple Linear Regression. For Multiple Linear Regression, we can fit Polynomial Courvilinear Regression.
 * Logistic Regression: it is classification, predicting the probability of discrete values. It chooses parameters that maximize the likelihood of observing the sample values rather than that minimize the sum of squared errors (like in ordinary regression).
 * Decision Tree: serves for both categorical and numerical data. Split with the most significant variable each time to make as distinct groups as possible, using various techniques like Gini, Information Gain = (1- entropy), Chi-square. A decision tree algorithm is known to work best to detect non – linear interactions. The reason why decision tree failed to provide robust predictions because it couldn’t map the linear relationship as good as a regression model did. 
 * SVM: seperate groups with a line and maximize the margin distance. Good for small dataset, especially those with large number of features
 * Naive Bayes: the assumption of equally importance and the independence between predictors. Very simple and good for large dataset, also majorly used in text classification and multi-class classification. <b>Likelihood</b> is the probability of classifying a given observation as 1 in presence of some other variable. For example: The probability that the word ‘FREE’ is used in previous spam message is likelihood. <b>Marginal likelihood</b> is, the probability that the word ‘FREE’ is used in any message.
 * KNN: can be used for both classification and regression. Computationally expensive since it stores all the cases. Variables should be normalized else higher range variables can bias it. Data preprocessing before using KNN, such as dealing with outliers, missing data, noise
 * K-Means
 * Random Forest: bagging, which means if the number of cases in the training set is N, then sample of N cases is taken at random but with replacement. This sample will be the training set for growing the tree. If there are M input variables, a number m<<M is specified such that at each node, m variables are selected at random out of the M and the best split on these m is used to split the node. The value of m is held constant during the forest growing. Each tree is grown to the largest extent possible. <b>There is no pruning</b>.
 * PCA: Dimensional Reduction, it selects fewer components (than features) which can explain the maximum variance in the data set, using Rotation. Personally, I like Boruta Feature Selection. Filter Methods for feature selection are my second choice. <b>Remove highly correlated variables before using PCA</b>
 * GBM (try C50, XgBoost at the same time in practice)
 
* Online Learning vs Batch Learning: https://www.analyticsvidhya.com/blog/2015/01/introduction-online-machine-learning-simplified-2/
 
 
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

************************************************************************

Interesting Data Analysis Examples

* Analysis with data visualization: https://www.analyticsvidhya.com/blog/2016/12/who-is-the-superhero-of-cricket-battlefield-an-in-depth-analysis/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


************************************************************************

AI

* Something about AI (those brief explaination about real life applications are useful and intresting): https://www.analyticsvidhya.com/blog/2016/12/artificial-intelligence-demystified/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Deep Learning Videos: https://www.analyticsvidhya.com/blog/2016/12/21-deep-learning-videos-tutorials-courses-on-youtube-from-2016/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Reinforcement Learning Open Sources: https://www.analyticsvidhya.com/blog/2016/12/getting-ready-for-ai-based-gaming-agents-overview-of-open-source-reinforcement-learning-platforms/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


************************************************************************

Experiences/Suggestions from Others

* From a data scientist (I agree with many points he said here, especially the one to get enough sleep, I also think we have to have enough good food before focusing on data science work, this is an area really needs strong focus and cost energy): https://www.analyticsvidhya.com/blog/2016/12/exclusive-ama-with-data-scientist-sebastian-raschka/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


************************************************************************

Data Science Skillset Tests

* Regression: https://www.analyticsvidhya.com/blog/2016/12/45-questions-to-test-a-data-scientist-on-regression-skill-test-regression-solution/?utm_content=buffer5229b&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Tree based skillset test: https://www.analyticsvidhya.com/blog/2016/12/detailed-solutions-for-skilltest-tree-based-algorithms/?utm_content=bufferde46d&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Interview questions: https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/


************************************************************************

Interview Tips

* Tips for Estimate Questions: https://www.analyticsvidhya.com/blog/2014/01/tips-crack-guess-estimate-case-study/?utm_content=buffer5f90d&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


************************************************************************

OTHER

* Make a wish to Data Science Santa! (I really like this idea, you just need to choose your 2017 data science learning goals and finally, they will give you a Christmas gift which is full of  relative learning resurces cater for your goals!): https://www.analyticsvidhya.com/blog/2016/12/launching-analytics-vidhya-secret-santa-kick-start-2017-with-this-gift/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Interesting Data Science videos (I mean some of them looks interesting): https://www.analyticsvidhya.com/blog/2016/12/30-top-videos-tutorials-courses-on-machine-learning-artificial-intelligence-from-2016/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
