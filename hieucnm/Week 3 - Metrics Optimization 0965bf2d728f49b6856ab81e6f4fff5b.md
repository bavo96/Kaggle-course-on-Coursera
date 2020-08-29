# Week 3 - Metrics Optimization

Created By: Hiếu Cao Nguyễn Minh
Last Edited: Aug 29, 2020 2:45 PM
Summary: Note some significant ideas from this course

# Week 3 - Metrics Optimization

## 1. Advices & general approach:

- To quantify how good a model is, ones have to employ an evaluation metric, or target metric. Depend on different purposes of business, organizers use different target metric for the same kind of problem (e.g: regression use multiple metrics: MSE, MAE, ...). In order to do modelling efficiently, competitors should explore the target metric and try to understand how it works at the very beginning.
- Note that, we have to distinguish between: *target metric*, which we want to optimize, and *loss function*, which our model optimizes. The 2 definitions sometimes can be named in different terms, which are just synonyms, not important.
- **Principle**: If your model is scored with some target metric, you get best results by optimizing **exactly** that metric.
- So, the best scenario is that target metric and loss function are the same.
- But unfortunately, not all target metrics can be optimized directly by modelling, usually due to non-defined derivation. In these cases, there are several approaches to deal:
    - Preprocess data, optimize another metric.
    - Optimize another metric, postprocess data
    - Define custom loss function ourselves.
    - Optimize another metric, use early stopping (very last choice, not neccesary to understand evaluation metric anymore)

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: General approaches for metric optimization. Optimize M1 metric, early stopping on M2 metric.

- With a particular evaluation metric, to know how good an evaluation score is, ones compare that score with some base score, namely baseline. We want to have a very good baseline so that we know how good our model is.
- The *"baseline"* term do not have formal definition, just be understood as a very simple model. In competitions, baseline usually means submitting constant predictions for all samples in test set. And that constant can be calculated based on the metric formula to obtain a best baseline.

## 2. Regression metrics

### 2.1. Mean Square Error

$$MSE = \frac{1}{N}\sum_{i=1}^{N}(y_{i} - \hat{y_{i}})^{2}$$

- Let the constant that we use to build baseline is $\alpha$. Then to find the best $\alpha$, we just derivate the above equation with all $\hat{y_{i}}$ replaced by $\alpha$, then solve the equation of the derivation equal to 0. The result is the mean of target values:

$$MSEBaseline = \frac{1}{N}\sum_{i=1}^{N}(y_{i} - \alpha )^{2}$$

$$\left( \frac{1}{N}\sum_{i=1}^{N}(y_{i} - \alpha )^{2} \right)' = 0 \Leftrightarrow \alpha = \overline{y} = \frac{1}{N}\sum_{i=1}^{N}{y_{i}}$$

### 2.2. Root Mean Square Error

$$RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_{i} - \hat{y_{i}} )^{2}} = \sqrt{MSE}$$

- Since every minimizer of MSE is a minimizer of RMSE, in other word, RMSE orders the models as MSE, we can find the best $\alpha$ for RMSE the same way as MSE.
- Taking root of MSE in RMSE make the metric value the same scale as the targets.
- Also, derivation of RMSE is alittle bit different from MSE:

$$\frac{\partial{RMSE}}{\partial{\hat{y_{i}}}} = \frac{1}{2RMSE} \frac{\partial{MSE}}{\partial{\hat{y_{i}}}}$$

- The flowing rate of RMSE is lower than MSE by a constant and depends on RMSE itself. So, using RMSE may need a higher learning rate than MSE.

### 2.3. R-squared Error

- After calculating MSE/RMSE, we can compare it to the baseline to know how far we improve our model. Instead of comparing explicitly, we can combine the baseline formula inside the formula of MSE, to create a unique metric. And that is why R-squared come into the picture.

$$R^{2} = 1 - \frac{MSE}{MSEBaseline} = 1 - \frac{\sum_{i=1}^{N}(y_{i} - \hat{y_{i}})^{2}}{\sum_{i=1}^{N}(y_{i} - \overline{y})^{2}}$$

- R-squared tell us how much our model is better than the constant baseline. It returns 0 if our model is no better than the baseline, and returns 1 if our model is perfect. Rarely, it could return a negative if our model is even worse than the baseline.

### 2.4. Mean Absolute Error

$$MAE = \frac{1}{N}\sum_{i=1}^{N}{|y_{i} - \hat{y_{i}}|}$$

- MAE is more robust than MSE because it is not sensitive with outliers.
- The best $\alpha$ for MAE is the median of target values.
- MAE is widely use in finance, where $10 error is usuall exactly twice worse than $5 error, while in MSE, the ratio is four times.
- MAE is not differentiable because its gradient is non-defined at  $\hat{y_{i}} > y_{i}$ (equal to 1 if $\hat{y_{i}} > y_{i}$ and -1 if $\hat{y_{i}} < y_{i}$ ). However, we can handle it by some simple  *if else* codes.
- If we make sure that there are outliers in our data, MAE is a better choice, otherwise, consider the others.

### 2.5. Mean Square Percentage Error & Mean Absolute Percentage Error

- MSE and MAE only consider absolute error, not relative error. Why relative error ? Consider the example from the course:

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%201.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%201.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Regression metrics review II. In a  sales prediction competition, shop 1 (or model 1) predicts 9 but the target is 10, shop 2 predicts 999 but the target is 1000. Both MSE in 2 cases is 1.

- In the example, we can commonly tell that prediction of shop 1 is more critical than the second one, and thus it should be punished severer. In an other word, we sometime concern about relative error rather than absolute error. Both MSE and MAE cannot satisfy this. That why we have MSPE and MAPE.

$$MSPE = \frac{100\%}{N}\sum_{i=1}^{N}\left(\frac{y_{i} - \hat{y_{i}}}{y_{i}}\right)^{2}$$

$$MAPE = \frac{100\%}{N}\sum_{i=1}^{N}\left|\frac{y_{i} - \hat{y_{i}}}{y_{i}}\right|$$

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%202.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%202.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Regression metrics review II. Loss changes are identical between 5 data points in MSE (left) and MAE.

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%203.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%203.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Regression metrics review II. Loss changes are different between 5 data points in MSPE (left) and MAPE.

- If the target increase, we pay less cost for error. If the target decrease, we pay more for error.
- The constant baseline for MSPE and MAPE is the corresponding weighted version of mean and median of target values. We can easily calculate it by writing down the formula.
- However, using these metrics carefully: if we have very very small targets, and the differences between predictions and targets are too large compared to the targets (usually happen at the first iteration/epoch), or we have very very small outliers in our data, then the 2 metric could become very large, even infinity, which would harm our training process.

### 2.6. Root Mean Square Logarithmic Error

- It is just RMSE calculated in log scale. But it carries about relative error, not absolute error.

$$RMSLE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(log(y_{i} + 1) - log(\hat{y_{i}} + 1)\right)^{2}}$$

- The difference here is logarithmic curve is asymmetric, so its predictions are usually higher than the targets.

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%204.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%204.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Regression metrics review II. Asymmetric RMSLE curves.

- Similar to RMSE, RMSLE can be calculated without root operation.
- To find the constant baseline for RMSLE, we firstly get log scale of all targets (remember to plus 1), then calculate the constant baseline in term of regular RMSE, and finally reverse the result by exponentiation to obtain the $\alpha$ .

### Conclusions:

- We have many metrics to evaluate regression problems: MSE, RMSE, MAE, R-squared, MSPE, MAPE, RMSLE.
- MSE and RMSE are sensitive to large outliers, while MSPE and MAPE are sensitive to small outliers.
- Absolute-type metrics are not differentiate.
- RMSLE is usually considered better than MAPE.
- Remember to find constant baseline regardless metric we are using.

## 3. Classification metrics

### 3.1. Accuracy

$$Accuracy = \frac{1}{N}\left[ \hat{y_{i}} = y_{i} \right]$$

- Baseline constant of accuracy is obviously class that has highest frequency in our data.
- Accuracy is usually hard to interpret, especially when the data is imbalanced.

### 3.2. Logarithmic Loss (Logloss)

- Binary:

$$LogLoss = -\frac{1}{N}\sum_{i=1}^{N}{\left(y_{i}log\hat{y_{i}} + (1-y_{i})log(1-\hat{y_{i}})\right)}$$

- Multiclass:

$$LogLoss = -\frac{1}{N}\sum_{i=1}^{N}{\sum_{l=1}^{L}{y_{il}log\hat{y}_{il}}}$$

- In practive, predicted scores are avoided to be exactly zeros or ones, but are clipped to be from a very small positive number to one minus a very small positive number. This variant is implemented in many libraries:

$$LogLoss = -\frac{1}{N}\sum_{i=1}^{N}{\sum_{l=1}^{L}{y_{il}log \left(min \left(max (  \hat{y}_{il}, 10^{-15}   ), 1 - 10^{-15}\right)\right)}}$$

- Here we have $\hat{y}_{il}$ is a vector of size L. So, other than accuracy, constant baseline of log loss, $\alpha$, is also a vector of size L. Once again, by solving the equation of the derivation of LogLoss equals to zero, we will find that each element of $\alpha$, $\alpha_{l}$ is the frequency of class $l$ in our data.

### 3.3. Area Under the Curve

- Only apply for binary classification
- The motivation is: when using absolute predict classes, as accuracy does, we have to choose a threshold to convert predicted scores to class indices. Different thresholds result in different accuracies. So, instead of fix a threshold, AUC try all possible ones, compute some values and aggregate them.
- There is a short but great explanation in the course, so I just rewrite it shortly here.

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%205.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%205.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Classification metrics review II. ROC curve example. Starting from 0 corner in the figure and from left to right in the axis. Each time we go through a point, if it is red (positive), we go up in the figure, if it is green, we go right in the figure. When all points passed, we end up at the upper right corner of the figure. The curve we plotted is Receiver Operating Curve - ROC, and the area under it over total area of the figure is 7/9, which is our AUC.

- Constant baseline: because AUC does not rely on hard predictions, all constants give the same AUC scores, which equal to the score of random prediction - 0.5.

### 3.4. Cohen's Kappa

- The motivation of this metric is similar to it of R-squared compared to MSE, i.e it adds the baseline score right into its formula so that we can know how far our model better than the baseline.
- The difference is Cohen's Kappa does not use the above baseline of accuracy, but a term $p_{e}$

$$Kappa = 1 - \frac{1 - accuracy}{1 - p_{e}} = 1 - \frac{error}{baselineError}$$

- We take the predictions, shuffle them many times and calculate accuracy of each shuffle. The average of those accuracies is exactly $p_{e}$ .
- In practice, we actually do not need to shuffle, we can obtain $p_{e}$ just by analysis:
    - Let say our data contain $N_{0}$ samples of class 0 and $N_{1}$samples of class 1, and our model predicts $n_{0}$ samples of class 0 and $n_1$ samples of class 1. Note that $N_0 + N_1 = n_0 + n_1 = N$. Then we have:

    $$p_e = \frac{N_0n_0 + N_1n_1}{N^2}$$

- Look closely. If our data is extremely imbalanced and biased to class 1, and our model predictions are all class 1, too. Then, accuracy will be very high, but $p_e$ will also equals to accuracy, resulting in a Cohen's Kappa score of zero.

### 3.5. Weighted Kappa

- This is just weighted version of Cohen's Kappa metric, but usually be used in competitions:

$$Weighted Kappa = 1 - \frac{weightedError}{weightedBaselineError}$$

- Weights indicate that how much severe we want our model to punish wrong predictions. We can define weights by forming a weight matrix of $L$ times $L$ where $L$ is number of classes:
- To get weighted error, we just multiply weight matrix and confusion matrix element-wise and sum up the results:

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%206.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%206.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Classification metrics review II. An example of Confusion matrix and Weight matrix.

$$WeightedError = \frac{1}{normalizer}\sum_{ij}C_{ij}W_{ij}$$

- The *normalizer* is just some constant to ensure that error is between 0 and 1, and usually is the number samples.
- The weighted accuracy can be obtained by one minus weighted error.
- The weighted baseline error can be obtained by (Oops! The course doesn't say, I am researching ...)
- In competitions, if we dont know the meaning of classes to create weight matrix, instead we have only class numbers, we can calculate weights by a linear or quadratic function:

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%207.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%207.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Classification metrics review II. Linear and Quadratic weights.

## 4. Regression metrics Optimization

### 4.1. MSE, RMSE, R-square

- are supported in most of libraries.

### 4.2. MAE

- is supported in LightGBM and sklearn's RandomForestRegressor
- isnt supported in XGBoost and sklearn's linear models (both due to MAE's second derivative, which the libraries use to update weight, is non-defined).
- The only problem with MAE is that it has no derivative at zero, so we can make it derivatiable at zero, or make it smooth, by chaging it to another loss, such as **Huber loss**, **Log cosh loss**, ...

$$ HuberLoss = \begin{cases} \frac{1}{2} (y - \hat{y})^2, \quad if \space |y - \hat{y}| \leq \delta, \\                      \delta|y - \hat{y}| - \frac{1}{2}\delta^2,   \quad otherwise                        \end{cases}$$

$$LogCosh = \sum_{i=1}^{N}{log\left(cosh\left(y_i - \hat{y}_i\right)\right)}$$

$$cosh(x) = \frac{e^x + e^-x}{2} = \frac{e^{2x}+1}{2e^x}$$

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%208.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%208.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Regression metrics optimization. Some regression loss function curves.

### 4.3. MSPE & MAPE:

- In case we dont want to make a custom loss function but still want to use these metrics, we can look a bit closer to the metrics and will realize that they are just weighted version of MSE and MAE:

$$MSPE = \frac{100\%}{N}\sum_{i=1}^{N}{\left(\frac{y_i - \hat{y}_i}{y_i}\right)}^2 \\ = \frac{100\%}{N}\sum_{i=1}^{N}\frac{1}{y_i^2}{\left(y_i - \hat{y}_i\right)}^2$$

To ensure weights are summed up to 1, we can change it a little bit:

$$w_i = \frac{1}{y_i^2} \rightarrow w_i = \frac{1/y_i^2}{\sum_{j=1}^{N}{1/y_j^2}}$$

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%209.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%209.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Regression metrics optimization. MSPE & MAPE are weighted versions of MSE & MAE respectively.

- So, we can optimize these metrics directly by passing these pre-calculated sample weights into MSE/MAE loss function. Almost common libraries accept sample weights: sklearn's models, LightGBM, XGBoost. Neural nets do not support this, but we can implement easily.
- The final way to optimize these metrics without using sample weights is loss function, is sampling our data with these sample weights before starting training. Why this works: in case of weighted loss function, samples whose weights are small will give small loss. Similarly, in case of sampling training data, samples whose weights are small also be chosen very rarely, then their influences to loss are small, too. If use this approach, for the sake of robustness, we should do the sampling multiple times, one model each, and average the predictions.

### 4.4. RMSLE:

- It's very simple, and the course explains very clearly so I just screencap it:

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2010.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2010.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Regression metrics optimization. Optimize directly RMSLE.

## 5. Calibration

### 5.1. What is this ?

- First of all, we need to understand this term before moving to classification optimization.
- In general classification problems, we already know that our model outputs a probability that a sample belongs to class 1. **This probability yet doesnt tell you about model's confidence.** It only can be interpreted as some kind of confidence level when we take all samples with probabilties around that score, for example 0.8, then there have to be around 80% of those samples are actually of class 1 (i.e their labels are ones) (note that these calculations are applied on validation set only).
- *Calibration* comes into play to do that: adjust probabilities a little bit so that they also can be interpreted as confidende (of course it only works on the models who can outputs probabilities, not only class label predictions).
- If our model optimized logloss directly, then it was well-calibrated already, nothing to do more.
- If a classifier doesnt optimize logloss directly, its predictions should be calibrated.

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2011.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2011.png)

Source: SkLearn: [https://scikit-learn.org/stable/modules/calibration.html](https://scikit-learn.org/stable/modules/calibration.html). Calibration curves of some common classifiers.

- A calibrator is basically a regressor, which receives the output probabilities of the classifiers as inputs and outputs the corresponding calibrated probabilities.
- There are 3 common calibration approaches:
    - Platt scaling: is basically a Logistic Regression.
    - Isotonic regression *(I will explain more precisely in another topic)*
    - Stacking models: XGBoost, ...

- So, we can fit any classifiers, as long as its outputs is good at some metric, for example AUC, not necessary exatly logloss, then we can calibrate our predictions by logloss later.

## 6. Classification metrics Optimization

### 6.1. Logloss

- is supported in most of popular libraries, except sklearn's RandomForestClassifier.

### 6.2. Accuracy

- We cannot optimize accuracy directly because accuracy has zero gradient at almost every points (because it force probabilities into hard label).
- The way is just optimize other metrics, and tune the threshold (for binary classification) or model's parameters according to optimize accuracy, not to the metric that the model is optimizing.

### 6.3. AUC

- The same reason as accuracy, we cannot optimize AUC directly.
- Still, there exists an algorithm to optimize AUC by gradient-based methods, called Pointwise loss. This should be explained in another topic, too.

### 6.3. Kappa

- *update soon ...*

## 7. Target/Mean Encoding

### 7.1. What is it ?

- Target encoding or Mean encoding is a way to encode categocial features, which encode each category by calculating mean of targets corresponding to that category.
- In general, the more complicated and non-linear feature-target dependency, the more effective is mean encoding.

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2012.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2012.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Concept of mean encoding. Ways to calculate mean encoding.

- Remember that the data to calculate target encoding must be different from the training data.

### 7.2. Regularization

- It is a way to make sure that we do not use training data to calcualate target encoding.

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2013.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2013.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Regularization. Ways to do regularization for mean encoding.

1. CV loop
    - Split training data into K folds
    - In each split, use K - 1 folds to estimate mean encoding and apply it to the remain fold.
    - Some rare category that only appear in one fold will have NaN mean encoding, so we should fill them by the global mean
    - Do not work with Leave-one-out scheme, so dont use this way for LOO scheme.

    ![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2014.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2014.png)

    Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Regularization. CV regularization in practice.

2. Smoothing
    - *nrows* is the number of rows that category appears.
    - If we are also using CV loop, *global_mean* is the mean of that category in all folds, otherwise, *global_mean* is the mean of all target regardless which category is used.
    - *alpha* is a regularized term. If *alpha* is zero, no regularization is applied. If *alpha* reaches to infinity, all category will be encoded nearly equally and equal to the *global_mean* value. This is a hyperparameter we have to tune.
    - This method shouldnot be used individually, because target leakage still remains. We should use it with CV loop. However, CV loop itself is good enough. That's why this method is so rarely used in practice.

3. Adding random noise
    - Noise decreases encoding quality, but very unstable
    - The amount of noise to add is a big problem.
    - Can be used in LOO regularization.

4. Sorting and calculating expanding mean:
    - It's actually CatBoost encoding
    - Fix some sorting order of data, and use only rows from zero to n - 1 to calculate encoding for row n-th

        ![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2015.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2015.png)

        Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Regularization. Expanding mean calculating.

    - Many advantages: least amount of leakage, no hyperparameters, irregular encoding quality, built-in in CatBoost library.

- CV loop or CatBoost is prefered in practice.

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2016.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2016.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Extensions. Correct validation for target encoding.

## 8. Target encoding (cont.) - Extensions

### 8.1. For regression tasks:

- Apart from mean, we can use other statistical measures to encode a category: median, std, quantiles, bins, ...
- Example of using bins, that we can seperate our continous targets into some bins, e.g. 10, and generate 10 new features, each is the mean (or median, std, ...) encoding of target values between that bin.

### 8.2. Many-to-many relations

- What is many-to-many relations ? Let's take an example: we have a dataset of users listening to some songs, in which a user can listen to many songs and a song can be listened by many users. Hence, many-to-many relations.
- In the above exaplme, we want to encode *song_id* column. To do that, we firstly need to *"flatten"* the dataset so that each row contains only one user listen to one song. After that, we can calculate target encoding for song_id normally.
- Finally, a user could listen to many songs, thus many encoding values. To get same number of features for every users, we can apply several statistic measures on those encoding: mean, std, max, min, ...

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2017.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2017.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Extensions and generalizations. Target encoding extension for many-to-many relations.

### 8.3. Time series tasks

- When time present in our dataset, we have many ways to choose rows to calculate target encoding: mean of target from previous day, previous two days, previous week, ...
- Eventually, we can consider a time unit, such as day, week, ..., as a category feature and encode it well.

![Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2018.png](Week%203%20-%20Metrics%20Optimization%200965bf2d728f49b6856ab81e6f4fff5b/Untitled%2018.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 3 Video: Extensions and generalizations. Target encoding extension for time series dataset

### 8.4. Numerical features

- In practice, we can bin numeric features and treat them as categorical features. But how can we bin it ? Very easy. We just train a tree-based model with non-encoding features, then check the tree splits, find out which splits based on our numeric feature, the split values of those splits are exactly the good values to bin our feature.

### 8.5. Interaction between features

- In practice, we usually want to combine two or more categorical features to generate a new feature. But which combination is a good choice ?
- Very easy. we just fit a tree-based model with original features, then loop over the splits and count numbers of times any couple of features appear in 2 neighbor nodes, i.e they appear in 2 splits that are adjacent. The most frequent couples would be good combinations to do target encoding.

---

---