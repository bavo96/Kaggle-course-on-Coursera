# Week 4 - Hyperparameters Tuning

Created By: Hiếu Cao Nguyễn Minh
Last Edited: Sep 22, 2020 9:27 AM
Summary: Note some significant ideas from this course

# Week 4 - Hyperparameters Tuning

## 1. Some imporantant hyper-parameters

### 1.1 LGBM /XGBoost 's GDBT:

- *max_depth* of 7 is recommended.
- *bagging_fraction* & *feature_fraction* is recommended to use.
- *min_data_in_leaf (min_child_weight* in XGBoost*)* is the most important hyperparam to tune the regularization.
- We can fit a reasonably *learning_rate*, e.g 0.1, then use learning curve to find out a good *num_iterations*. After that, we can do a trick that usually improve our scores: multiple our *num_rounds* by a factor, and divide *learning_rate* by a factor (e.g: double *num_rounds* and take half *learning_rate*).
- Use *random_seed* to make sure that model performance does not depend hardly on luck.

### 1.2. sklearn.RandomForest

- *n_estimators* : the higher the better, but depend on the size of data that we should use relative small *n_estimators* firstly.
- *max_depth* of 7 is recommended too.
- Gini is recommend against Entropy for *criterion*.

### 1.3. Neural Nets

- *Optimizer* is a very important hyperparam: Adam family (Adam/Adadelta/Adagrad/...) inpractice lead to more overfitting than SGD + momentum.
- Large *batch_size* leads to more overfitting, a value of 32 or 64 is recommended. Small *batch_size* can reduce overfitting, but increase training time and add more noise in gradient.
- Rule of thumb: if we increase the *batch_size* by a factor, we can also increase the *learning_rate* by the same factor.
- A technique from the top kagglers called ***static dropconnect***: we use a huge hidden layer at the beginning of our network, along with a 99% dropout layer before it. This way, overfitting can be prevented and the network still can learn useful information. Actually, DropConnect is something oppisite to Dropout: Dropout drops connections from the current layer to the next layer, while DropConnect drops connections from previous layer to current layer.
- A tips for neural nets is that: networks with different connectivity patterns (like using Dropout, DropConnect) make much nicer than those without.

### 1.4. Tips

- Never spend too much time tuning hyperparameters.
- Be patient, because it takes many rounds for GBDT/NN to fit.
- Average everything: over random_seed, over small deviations from optimal parameters.

## 2. Practical guide

### 2.1. Before starting

- Know what you want to chose suitable competitions.
- Participating in forums is recommended.
- Adter entering, everything is a hyperparameter. So mine and tune them from ones that we completely understand, to ones that we have no idea how they work.

### 2.1. Data loading

- Do basic preprocessing, then convert csv/txt to hdf5/npy for much faster loading later.
- Reduce data size by check if they can be downcast from 64-bits to 32-bits.

### 2.3. Evaluation

- Just use simple train_test_split at the beginning, CV later.
- Start with fastest models: LGBM.
- Only switch to tuning hyperparameters, sampling, CV validation, ... when you satisfied with feature engineering.

### 2.3. Fast and dirty always better

- **Primary**: understand data, EDA, feature engineering, google domain-knowlegde. Your code is secondary, dont try to make unnecessary classes.
- **Advise**: at least 3 notebooks: 1 for EDA, 1 for training, and 1 for predicting and submission. One notebook per submission is recommended.
- **Advise**: carefully build pipeline at the beginning, from read data to write sibmission.
- **Advise**: fix random seed, use git, keep everything reproducible.
- **Advise**: Write down exatly every features were generated.
- **Advise**: read forums and examine kernels first: there are always discussions happening.
- **Advise**: baseline of evaluation metrics firstly, simple EDA to make sure data is loaded correctly.
- **Advise**: List out features that we can think of, add many of them at once, not add one and evaluate.
- **Advise**: Let the model overfit, then constrain it later.
- Long notebook can leads to mistakes, usually due to global variables. Remember to sometimes restart our notebooks, or use another one.
- **Advise**: Use *"Restart and run all"* when we decide to create a submission.
- Assign paths of train/val/test data set to global variables at the top of the notebook, in order to easily exchange between dev and submit stage.
- Write custom common codes, we may use it for the next competitions.

## 3. Practical pipeline by top kaggler

*KazAnova's competition, by Marios  Michailidis*

![Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/Untitled.png](Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/Untitled.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 4 Video: KazAnova competition P1. Recommeded pipeline.

### 3.1. Understand the problem

- Type of problem
- Data size
- Hardware/Software needed
- Evaluation metric

We can reuse everything if we did this type of problem before.

### 3.2. EDA

- Check if features have similar distribution between train and test set.
- Check if features change by time, if time is available in the dataset.
- Check correlations. Try to bin continous features.

### 3.3. CV strategy

- This is the key to win the competition.
- Does time matter ? → Time-based validation.
- Does diferent entities matter ? → Stratified validation.
- Is it completely random ? → Random split.
- There are cases we may have to combine all types of validation.

### 3.4. Feature Engineering

![Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/image.jpg](Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/image.jpg)

Source: Coursera. Course: How to Win a Data Science Competition. Week 4 Video: KazAnova competition P1. The type of problem defines the feature engineering.

- If our validation strategy was consistent, then we can put all sorts of features into our model and let's the validation set(s) do the feature selection step.

### 3.5. Modeling

![Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/image_(1).jpg](Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/image_(1).jpg)

Source: Coursera. Course: How to Win a Data Science Competition. Week 4 Video: KazAnova competition P1. The type of problem defines the modeling.

- All prediction results should be saved with respect to the trained model.
- There are different ways to ensemble: from **averaging** (usually for small datasets) to **stacking**.

### 3.6. Submission

- Select the **best prediction on CV** and the **best on public LB**.
- Check if best prediction on CV correlates with prediction on LB. If yes, then we can choose the second best one on CV.

![Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/image_(2).jpg](Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/image_(2).jpg)

Source: Coursera. Course: How to Win a Data Science Competition. Week 4 Video: KazAnova competition P1. Final advices.

## 4. Advanced features

### 4.1. Statistics and distance-based features

- Statistics measures to apply group_by on : count, mean, std, min, max, mode, ...
- If there is no features to group, we can apply it on neighbors. This approach is hard to implement, but can be more flexible. We can fine tune the metrics to choose neighbors and number of neighbors.

![Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/Untitled%201.png](Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/Untitled%201.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 4 Video: Statistic and distance features. Example of using neighbors for feature engineering.

### 4.2. Matrix factorization

- Is a kind of dimenionality reduction: A matrix of user-item, which number of items is very large, is factorized into 2 matrix of shape user-latent_factor and latent_factor-item, which number of latent factors is relatively small.
- We can tune the number of latent factors. It should be only around 5-100.
- Some MF algorithms:
    - SVD
    - PCA
    - TruncatedSVD
        - Good for sparse matrices, like text data
    - Non-megative MF
        - Ensure all latent factors are non-negative
        - Good for count-like data, such as BoW of text data
        - More suitable for tree-based models.
- We can apply any transformations before applying MF, such as MF(log(X + 1)) instead of MF(X)
- We must fit MF on both train and test data, not only fit on train and then transform on test:

![Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/Untitled%202.png](Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/Untitled%202.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 4 Video: Matrix Factorization. The right way to use MF.

### 4.3. Feature interactions

- Is a kind of combine 2 or more features into 1.
- There are different ways to combine numeric features: sum, diff, multiplication, division.
- For categorical features, the traditional ways is take the combination of the 2 features
    - I.e: if feature *A* has 2 values *a1* and *a2*, feature *B* has 3 values *b1*, *b2* and *b3*, then the new combined feature, say *A_B*, has 6 values of *a1b1*, *a1b2*, *a1b3*, *a2b1*, *a2b2*, *a2b3*.

The question is: which features should be combined ?

- There are 2 common ways to find them:
    1. Just fit all features into a tree-based model. Then check out the feature importances. The 2, or sometime 3 if we want, most important features is the one that we may want to combine firstly.
    2. Another way is to fit all kind of combination of all pairs of features as new features into a tree-based model. Then check out the feature importances and take the most combination ones.

### 4.4. t-SNE

- Is another way of dimensonality reduction, but in a non-linear manifold, so-called ***Manifold Learning***.
- Manifold Learning: a method to reduce dimensionality so that the distances between data points are approximately preserved.

![Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/Untitled%203.png](Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/Untitled%203.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 4 Video: t-SNE. Comparison of manifold learning methods (source: sklearn)

- t-SNE is usually be used in EDA
- t-SNE, and other manifold methods as well, are strongly depends on hyperparameters. For example, different *perplexity* argument result in different vectors in t-SNE:

![Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/Untitled%204.png](Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/Untitled%204.png)

Source: Coursera. Course: How to Win a Data Science Competition. Week 4 Video: t-SNE. Different perplexity of t-SNE.

- Advices when using t-SNE:
    - use several different perplexities (5-100), only 1 perplexity can cause mis-interpretation.
    - train and test set should be projected together
    - dimensionality reduction should be applied before using t-SNE because it takes many time.
    - use standalone t-sne package due to its speed, should not use sklearn's one.

[How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)

[Additional Materials and Links](https://www.coursera.org/learn/competitive-data-science/supplement/3XpTg/additional-materials-and-links)

## 5. Ensemble Learning

### 5.1. Bagging

- In this method, predictions of all models will be averaged to form a more robust predictions.
- All models can be trained independently.
- We can control this ensemble method by some major hyperparameters:
    - Number of models
    - Number/Fraction of rows to train each model
    - Number/Fraction of features to train each model

### 5.2. Boosting

- In this method, errors between targets and predictions of the former model will be treated as targets for the later one. The process continues until some condition is met, such as early stopping, all models was trained, ...
- It's obvious that the models cannot be trained independently, but the laters must wait for predictions of the formers.
- We can control this ensemble method by some major hyperparameters:
    - Number of models
    - Number/Fraction of rows to train each model
    - Number/Fraction of features to train each model
    - Learning rate: is actually a constant factor multiplied with the errors of previous model to form targets for the next one.

### 5.3. Stacking

![Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/image%201.jpg](Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/image%201.jpg)

Source: Coursera. Course: How to Win a Data Science Competition. Week 4 Video: StackNet. An example of how stacking works.

- In this method, we have to split a hold-out dataset fro trainset to make prediction for predictions of trained models on the remain data. We can easily understand this method in the captured screen picture above.
- Some things to be mindful of:
    - With time sensitive data - split hold-out by time
    - Diversity (such as: using different models, different features, different encoding methods, ...) is as important as performance
    - The stacking model should be simple

### 5.4. StackNet

- Use stacking in multiple levels/layers.
- Use K-fold paradigm to split training data for stacking layers.

![Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/image%202.jpg](Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/image%202.jpg)

Source: Coursera. Course: How to Win a Data Science Competition. Week 4 Video: StackNet. K-folds strategy to do stacking.

- Can connect a layer with any previous layers, not only the directly previous one as neural nets.

![Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/image_(1)%201.jpg](Week%204%20-%20Hyperparameters%20Tuning%20901a26ebece14a109887db602c53a547/image_(1)%201.jpg)

Source: Coursera. Course: How to Win a Data Science Competition. Week 4 Video: StackNet. Connect any layers we want.

### 5.5. Tips

- Diversity:
    - Based on algorithms:
        - Using at least 3 GDBTs (LightGBM, CatBoost, ...) is recommended: 1 for higher, 1 for middle, and 1 for deeper layer.
        - 3 Neural Nets, with different architecture (different SOTA, different depth, ...)
        - 1 - 2 Random Forests/ExtracTrees (sklearn)
        - 1 - 2 linear models (sklearn)
        - 1 - 2 k-NN models (sklearn)
        - 1 Factorization machine (libfm) (quite useful)
        - 1 SVM with rbf kernel (sklearn) (if data size allows)
    - Based on input data:
        - Categorical encodings: onehot, label, target encoding
        - Numerical transformations: different binning, log or not, keep/drop outliers, ...
        - Interactions: col1 */+- col2, groupby col1 and average col2, .... (all possible interactions)
        - Unsupervised: PCA, KMeans, ...

- Simple models:
    - Tree-based: small depth (usually 3)
    - Linear models: high regularization
    - NeuralNets: shallow depth
    - kNN: BrayCurtis distance
    - Brute forcing a search for best linear weights based on CV
- Rule of thumb: for every 7.5 models in previous layer, we add one model in the current layer, and so on to the next layer.
- Do not choose too small K for CV because it still can leak the targets.

### 5.6. Libs for stacking:

- [StackNet](https://github.com/kaz-Anova/StackNet) (recommended)
- [Stacked Ensembles H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html)
- [Xcessiv](https://github.com/reiinakano/xcessiv)

## 6. CatBoost

*Coming soon ...*

---

---