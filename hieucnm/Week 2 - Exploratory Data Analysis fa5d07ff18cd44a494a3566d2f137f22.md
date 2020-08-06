# Week 2 - Exploratory Data Analysis

Created By: Hiếu Cao Nguyễn Minh
Last Edited: Aug 6, 2020 5:01 PM
Summary: Note some significant ideas from this course

# Week 2 - Exploratory Data Analysis

## 1. Advices

- It's very crucial that we should explore a bit domain knowledge about the given dataset before we start working on it. We can use the Internet to understand the meaning of each column in our dataset.
- Check the validity of the values for each column. If something strange turns out, ask yourself why, that if it is because of some data error (e.g: typing error) or because you misinterpret it. Sometime we can add a boolean *is_incorrect* (or *is_strange, is_abnormal, ...*) column for a specific column to indicate that each row of that column is incorrect or not.
- Understand the how the data was genereted. Check the distribution of some or all columns between train and test set. If there are some difference, try to explain. After that, if we come to the conclusion that the train and test set are generated by different sampling methods, then improving validation score (which split from train set) properly would not improve leader board score. We should have another validation strategy, or augment our train data.

## 2. Explore anonimized data

- Anonymized data is data that was hidden by the organizers, such as : encoded text data, no meaning of column names, ...
- Try to guess the type and if posible, the meaning of each anonymized feature.
- We can firstly train a simple model (such as: LGBM) on simple preprocessed training data, then print its feature importances to choose thich columns to go first.
- Try to check if a anonymized column was scaled, if yes, try to refactor it to original values (i.e. decode the feature), we might easily find the true meaning of it.
    - In case the feature was standard scaled, we can find the std by printing out the differences between neighboring unique values of that column (by *np.diff(<column>.unique()*). The invert of the most diff value may be the std of original data.
    - After dividing that feature with the found std value, checking all values might help us find out the mean and even the true meaning of the feature.

    *watch lesson Week2/Exploring anonymized data to clearly understand.*

- Rather than scaled, try to check if a column was transformed by other method, such as division by a number, moduloed by a number, ... so on. Try to generate features before being transformed and validate them to see improvements.

## 3. Visualization

- Never make a conclusion based on a single plot. If you have a hypothesis, try to plot several different figures to prove it.
- Histogram can help spot out the hidden NaNs which were imputed my the organizer.
- Multiple plots (histogram for numeric and barplot for categorical), each for one class, on a same feature also help point out if that feature can be discriminative between classes.
- Scatter plot, whose x-axis is index and y-axis is feature, is useful to check if data is shuffled properly and if there are any values repeated many times.
- Scatter plot can also help determine if a numeric feature in train and test set having same distribution.

![Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled.png](Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 2 Video: Visualization. Scatter plot between index and a numneric column, color seperated by classes of label column. Here it's obvious that our data is a little bit imbalanced, not shuffled by class, and in the 3th class there are 2 feature values repeated many times.

- Remember to use Pandas's describe function.

To visualize correlation between feature pairs, use (some plots are not taught in the course but I already knew them before, so I still note here for the sake of fully knowing about types of plots):

- Numeric vs Numeric ones: scatter plot, hexagonal plot, contour plot, correlation matrix.
- Categorical vs Categorical ones: crosstab table, barplot
- Numeric vs Categorical ones:  density plot, box plot, violin plot, groupby categorical one and calculate a statistic value (e.g: mean) of numeric one.

- Never use pieplot because it cannot contain information as high as the others.
- We can leverage color if we want to add third column, usually the target column.
- We can visualize many plots at the same time with Pandas.

We can make custom functions for 1 feature, such as: mean, std, ... , then plot a matrix of feature indices by statistic value of each one (e.g: mean). Sort feature index by the statistic values to form groups of features. Then brandstorm yourself to find out if we can generate new features based on those groups.

![Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%201.png](Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%201.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 2 Video: Visualization. Grouped features by their mean.

![Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%202.png](Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%202.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 2 Video: Visualization. Scatter plot of different kind of values by row index and feature index.

Moreover, rather than correlation between original features, we should generate new combinations between features as many as possible and visualize correlation between them. The combinations could be, such as: differences, divisions, modulos, value pair, group by this feature aggregating mean of the other, group by this feature aggregating nunique of the other, ... and so on. Then plotting any plots (e.g: correlation matrix) of these new ones with the target to see if they are useful.

![Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%203.png](Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%203.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 2 Video: Visualization. Correlation matrix.

## 4. Data Cleaning

- Remove constant columns (due to no information, of course)
    - The reason for this could be that the organizers can give only a sample of their data to competitors. Therefore, there may be some columns, are constant in the data competitors received, but are various in the original data.
- Drop duplicated columns (due to no information anymore because they are completely correlate to each other, they just make our training slower).
    - Duplicated columns can also be hidden in term of categorical ones. I.e, they have the same levels but different level names. A simple way to check this is applying label encoding with appearance order (not alphabet order), on these columns (easily done by series.factorize() function of Pandas). After that, duplicated columns will become columns having exatly the same encoded values row by row.
- For a column that test set contains some values which do not appear in train set, we should decide how much this column influence our predictions. We can use a validation set to compare results on multiple scenarios, such as: just keep it, remove it completely, or generate new feature for it (e.g: frequency encoding) and then remove it, ...
- Drop duplicated rows, included label. Try to explain why they are duplicated if possible.
- Check duplicated rows excluded label. If there are such those rows with different labels, ask yourself why this happen and decide which label to keep with each of them. This is a part of data understanding.
- Check similar rows, expluded label, between train and test set. If there are, we can manually label these rows in test set by those in train set. Sometime it can tell us something abour data generation process.
- Check if rows were shuffled by scatter plot of row index and target column. If they weren't, we can easily use row index to classify the target.
- Check if columns were shuffled by printing out number of NaNs of each one. If they weren't, then we should prioritize mining correlations between 2 adjacent columns rather than 2 random columns.

## 5. Validation

- Strategies:
    - Holdout (sklearn.model_selection.ShuffleSplit) : some data will be validated many times, and some others will never be validated. We should only use this if we have a too large train set that we cannot split it into folds for multiple validations.
    - K-folds (sklearn.model_selection.Kfold) : split data into k folds. In each fold, use that fold as validation set and the others as train set. We should use this when we have a not too large dataset. Average scores from multiple models, each from a fold, can help improve performance on test data.
    - Leave-oneout (sklearn.model_selection.LeaveOneOut) : split data into k folds that k equals to number of samples in  our data. We should only use this when our data is too small.
- Stratification: split data into subsets guaranteeing that the ratios between target values in each subsets are similar.

- Splitting strategies:
    - Row-wise: common random split
    - Time-wise: use a fix-length moving window
    - ID-wise: random split by an ID column. Sometimes the ID column is hidden, it can be a combination of some other columns. We have to determine it, by EDA of course.
    - Combined: such as multiple ID columns (e.g: predict sales of a shop by a user → split by both user_id and shop_id), ID-wise and time-wise at the same time (e.g: predict sales of a shop by time), ... and so on. It depends on the task of the competitions.

![Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%204.png](Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%204.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 2 Video: Data splitting strategies. Moving window validation.

- Set up train/validation the same way as train/test split. Different splitting strategies can differ significantly:
    - in generated features
    - in how the model learn from those features
    - in data leaks.
- **Principle**: To brainstorm new smart features, as well as to find solutions to consistently improve our model, in the very first place, **we definitely have to identify train/test split made by organizers and reproduce it**.

## 6. Problems with validation

- Although we carefully split data in a perfect way, we sometimes still find out that our validation scores are too different to each other, or to the public leader board score. There are multiple reasons for this and some useful solutions.
- There are 2 stage that usually cause the problems: valiation stage and submission stage.

- In validation stage:
    - The causes usually are:
        - Too little data: model does not have enough data to learn the patterns well. In this case, our model will learn the main patterns, which can differ in different splits.
        - Too diverse data: very similar samples with different target values, which make our model confuse.
    - Solutions:
        - Make K-fold validation with multiple random seeds and choose the one that produce lowest std of validation scores.
        - Tune model on one split, evaluate score on the other.
        - (Additional) We can calculate mean of validation scores, and decide to remove one or some folds that have highest difference with the obtained mean.

- In submission stage: LB score is consistently higher/lower than, or even not correlated with validation score.
    - The causes usually are:
        - We did a wrong train/validation split, i.e it did not as same as train/test split made by the organizers.
        - Train and test set are from different distributions.
    - Solutions:
        - Double check your validation split, is it mimic train/test split ?
        - If the distributions of target values from train and test set are completely different, our model trained on train set will produce predictions on test set that have some errors corresponding to the targets on test set. The errors may be different or constant for each sample. We can only solve this problem if it is constant. Just expect that it is and, you know, try.
            - So, we would want to shift our predictions by that constant so that the distribution of our predictions can get as closer as possible to target values of test set.
            - The constant we are seeking is assumed as the difference between mean of target values on train and test set.
            - Mean of targets of train set can be easily calculated.
            - To calculate mean of test set, we use a trick named *"leaderboard probing"* : Submit a submission file containing all unique constant values and get out the scores. Write down the formula of the evaluation metric, because we know our submission values, we can easily reverse the formula and obtain the mean of target values.
            - Finally, calculate the difference between the 2 mean values and shift our predictions. Submit again to see if our solution worked.

![Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%205.png](Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%205.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 2 Video: Problems occuring during validation. Different distributions between train set (women's height) and test set (men's height). We can try to shift our prediction by the difference between mean of target values of train and test set.

![Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%206.png](Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%206.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 2 Video: Leaderboard probing and examples of rare data leaks. Formula to calculate mean of target values of test set in a binary classification task

- **Principle**: if we did extensive validation and guaranteed every thing are correct, then **trust your validation, do not overfit Public LB**.

![Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%207.png](Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%207.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 2 Video: Problems occuring during validation. Summary of Validation.

*Reading Material:* [How to Select Your Final Models in a Kaggle Competition](http://www.chioka.in/how-to-select-your-final-models-in-a-kaggle-competitio/)

## 7. Data leakages

- Data leakages are very bad in real world of course. But, they can be exploited in competitions.
- Causes of data leaks in competitions:
    - Train/test split was not time-wise: so information about future are revealed.
    - Unexpected information:
        - Meta data: For example, meta data of images can say something about the its target (e.g: there was a competition that resolution of an image can classify its class exactly.)
        - IDs: order of ID sometime leak information about its target, too.
        - Row order: in case that data was't shuffled.
- Exploiting data leaks is extremely nontrivial, would take us a lot of time, but can improve model drastically.

*Reading Material:* [Expedia challenge: A story of data leak exploition.](https://www.coursera.org/learn/competitive-data-science/lecture/Uxcm1/expedia-challenge)

## 8. EDA Checklist

We should combine train and test set before EDA. Then:

- Get domain knowledge
- Check if data is intuitive
- Understand how the data was generated
- Explore individual features (1D EDA)
- Explore pairs and groups (2D and multiple-dimension EDA)
- Clean features up.
- Clean rows up.
- Check for leakages.

Now we have a good cleaned dataset. Let's modelling!

---

---

# Quiz

I audit this course , so I cannot submit my answers of any Quizzes (not Practice Quizzes). However, I still finish the quizzes and write my answers as well as explanation here. 

If you did enroll the course and did not finish the quizzes, please do not read my answers, you should go back to the course and do it yourself. Otherwise, please do not hesitate to comment to share your answers or point out mistakes of mine.

## 1. Exploratory data analysis

*4 questions*

1. Suppose we are given a data set with features *X*, *Y*, *Z*.

On the top figure you see a scatter plot for variables *X* and *Y*. Variable *Z* is a function of *X* and *Y* and on the bottom figure a scatter plot between *X* and *Z* is shown. Can you recover *Z* as a function of *X* and *Y* ?

![Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%208.png](Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%208.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 2 Quiz: EDA. Question 1 & 2.

*Answer:*

- *Z = X / Y*

*Explanation:*

- There is a dot on top-right position of the top figure, means that there is a data point that have X=1 and Y=100. In the lower plot, we see that scale of Z is [0, 1], thus it cannot be neither that *Z = X + Y*, nor *Z = X - Y*, nor *Z = X*Y.* It can only be *Z = X / Y*.

2. What *Y* value do the objects colored in red have?

*Answer:*

- Y = 2 (we can easily find out by looking at the grid)

3. Which hypotheses about variable *X* do NOT contradict with the plots? In other words: what hypotheses we can't reject (not in statistical sense) based on the plots and our intuition?

![Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%209.png](Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%209.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 2 Quiz: EDA. Question 3.

*Answer:*

- X is a counter or label encoded categorical feature.
- X takes only discrete values

*Explanation:*

- *"2 ≤ X< 3 happens more frequently than 3 ≤ X < 4"* : is equalvalent to "1.1 *≤ log(X+1) < 1.39 happens more frequently than 1.39 ≤ log(X+1) < 1.61" ,* which is false if we look detailly into the lower plot.
- *"X can be the temperature (in Celsius) in different cities at different times"* : is false because no citiy has temperature at 100 Celcius, and the temperatures of cities cannot happen mostly around the range of 0 - 20.
- *"X can take a value of zero"* : is equalvalent to *"log(X+1) can take a value of zero"* , which is false because no dot lines in zero row in the lower plot.

4. Suppose we are given a dataset with features *X* and *Y* and need to learn to classify objects into 22 classes. The corresponding targets for the objects from the dataset are denoted as *y*. 

Top left plot shows XX vs YY scatter plot. We use target variable yy to colorcode the points. The other three plots were produced by jittering XX and YY values. That is, we add Gaussian noise to the features before drawing scatter plot.

Select the correct statements.

![Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%2010.png](Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%2010.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 2 Quiz: EDA. Question 4.

Explanation*:*

- First of all, we have to understanding why jitter hear:
    - Regardless if there are duplicated rows in our data, on scatter plot we only see one dot for one pair of values from 2 features. There may be many data points at that dot. Even, if duplicated rows have different target, we may mistakenly conclude that that point (i.e that combination of 2 values) has that only target, but actually it has multiple targets.
    - To see how dense it is (i.e how much large of number of data points) at a dot in scatter plot, we add a small noise to both 2 features before plotting them. After that, dots wont be plot completely overlap each other anymore. If the added noise is large enough, the density is very clear.
    - This technique called *"jittering"*. Note that adding noise is just for better visualization, do not add noise to our features when modelling.
- After understanding basically, we can see that all the true statements below and the other false statements are simply based on the above explanation about jittering.

*Answer:*

- Top right plot is "better" than top left one. That is, every piece of information we can find on the top left we can also find on the top right, but not vice versa.
- Standard deviation for Jittering is the largest on the bottom right plot.
- It is always beneficial to jitter variables before building a scatter plot

## 2. Validation

*4 questions*

1. Select true statements

*Answer:*

- Underfitting refers to not capturing enough patterns in the data
- We use validation to estimate the quality of our model
- Performance increase on a fixed cross-validation split guaranties performance increase on any cross-validation split.
- The logic behind validation split should mimic the logic behind train-test split.

2. Usually on Kaggle it is allowed to select two final submissions, which will be checked against the private LB and contribute to the competitor's final position. A common practice is to select one submission with a best validation score, and another submission which scored best on Public LB. What is the logic behind this choice?

*Answer:*

- Generally, this approach is based on the assumption that the test data may have a different target distribution compared to the train data. If that would be the true, the submission which was chosen based on Public LB, will perform better. If, otherwise, the above distributions will be similar, the submission which was chosen based on validation scores, will perform better.

3. Suppose we have a competition where we are given a dataset of marketing campaigns. Each campaign runs for a few weeks and for each day in campaign we have a target - number of new customers involved. Thus the row in a dataset looks like

Campaign_id, Date, {some features}, Number_of_new_customers

Test set consists of multiple campaigns. For each of them we are given several first days in train data. For example, if a campaign runs for two weeks, we could have three first days in train set, and all next days will be present in the test set. For another campaign, running for weeks, we could have the first 6 days in the train set, and the remaining days in the test set.

Identify train/test split in a competition.

*Answer:*

- Combined split

4. Which of the following problems you usually can identify without the Leaderboard?

*Answer:*

- Train and test data are from different distributions
- Different scores/optimal parameters between folds

## 3. Data leakages

*4 questions*

1. Suppose that you have a credit scoring task, where you have to create a ML model that approximates expert evaluation of an individual's creditworthiness. Which of the following can potentially be a data leakage? Select all that apply.

*Answer:*

- Among the features you have a company_id, an identifier of a company where this person works. It turns out that this feature is very important and adding it to the model significantly improves your score.

*Explanation:*

- ID-type columns do not contain any feature that can affect model's decision in real life. Therefore, if there are improvement based on features related to any ID-type columns, between any ID-type columns and target column, there are definitely data leaks in them.
- Data leaks only relates to test set, so the other answers which only about train set cannot be applied.

2. What is the most foolproof way to set up a time series competition?

*Answer:*

- Split train, public and private parts of data by time. Remove all features except IDs (e.g. timestamp) from test set so that participants will generate all the features based on past and join them themselves.

3. Suppose that you have a binary classification task being evaluated by logloss metric. You know that there are 10000 rows in public chunk of test set and that constant 0.3 prediction gives the public score of 1.01. Mean of target variable in train is 0.44. What is the mean of target variable in public part of test data (up to 4 decimal places)?

*Answer:*

- Apply the function proved in the video *Leaderboard probing and examples of rare data leaks*, we can easily obtained the result of about 0.771068.

4. Suppose that you are solving image classification task. What is the label of this picture?

![Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%2011.png](Week%202%20-%20Exploratory%20Data%20Analysis%20fa5d07ff18cd44a494a3566d2f137f22/Untitled%2011.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 2 Quiz: Data leakages. Question 4.

*Answer:*

- This is a very excellent and practical question for us to learn this lesson. To answer this, we have to think as a competitor who are trying to find data leaks somewhere in the given data.
- Here our data is only an image, is it ?
- No, there is another information, and it is the url of this image.
- In your browser, try to inspect the image to see its information (e.g: in Chrome, we can press Ctrl + Shift + C and click on the image). What I found is the string "label_is_3" in the image's url. Then the answer is 3.
- Mining data leaks is so exciting!

---

---