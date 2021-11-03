# 7.1 ML Basics
1. Explain supervised, unsupervised, weakly supervised, semi-supervised, and active learning.
    - supervised learning - learning from a dataset with ground truth labels/values for classification/regression
    - unsupervised learning - dataset without any ground truth labels/values
    - weakly supervised - learning from a dataset with noisy or limited labels/values
    - semi-supervised - limited amount of labels/values, a subset of weakly supervised
    - active learning - the learning algorithms queries for labels in an iterative manner (see also online machine learning)
2. Empirical Risk Minimization
    - The risk in ERM is an expected value of the loss given the true distribution of the predictions
    - The risk is empirical because in ML we don't have access to the true distribution, only the distribution of a subset of the data (out training data)
    - We can minimize the 'true' risk by assuming that our empirical risk is a good approximation of it, and minimizing our empirical risk with various methods such as gradient descent etc.
3. Occam's razor states that when the simple explanation and complex explanation both work equally well, the simple explanation is usually correct. How do we apply this principle in ML?
    - One way that it can be applied is if there are two models that perform equally well, but one is simple and the other is complex, the simpler one is better.
        - a simpler model has less parameters and likely makes fewer assumptions about the data, and in theory should generalize better (assuming the two models truly are equivalent in performance)
4. What are the conditions that allowed deep learning to gain popularity in the last decade?
    - The tools to handle and use large quantities of data such as big data systems as well as advancements in GPU acceleration have allowed deep learning to achieve state-of-the-art performance on tasks such as image classification, text classification and language models
5. 

# 7.3 Objective Functions, metrics, and evaluation
## 3. Bias Variance Tradeoff
- What is the bias-variance tradeoff?
    - the bias-variance tradeoff refers to the inverse relationship between bias and variance, models with high bias tend to have low variance, and models with low variance tend to have high bias
        - bias - error caused by the model failing to capture the relationship between the data and the labels
        - variance - error caused by the model learning from a given dataset
- How is the tradeoff related to overfitting and underfitting?
    - Overfitting occurs when the model learns the noise of the data and doesn't generalize to new data, models with high variance tend to overfit
    - Underfitting occurs when the model doesn't learn the relationship between the data and the labels, models with high bias tend to underfit
- How do you know that your model is high variance, low bias? What would you do in this case?
    - you can look at the learning curve and observe the training error and the test error. A model with high variance and low bias will tend to overfit to the training data, which means the training error will be low and the test error will be high
        - specifically, the train error will be lower than acceptable and the test error will be higher than acceptable
    - Potential remedies:
        - using regularization techniques
        - switching to a simpler model
        - increasing the size of your training data to better reflect the test/validation distribution
        - bagging/ensemble methods
- How do you know that your model is low variance, high bias? What would you do in this case?
    - a model with low variance and high bias tends to underfit, it will have high train error in addition to high test/validation error, specifically the training error will be higher than acceptable
    - Potential remedies:
        - use a more complex model
        - add additional features
        - boosting
## 4. Cross Validation
- Explain different methods for cross-validation
    - **K-fold** - split your data into K chunks, train on k-1 chunks and calculate evaluation metrics on the kth chunk, and repeat until every chunk has been evaluated on
    - **Leave p out** - K-fold taken to the extreme, where `k=n-p`: the number of data points minus some number `p`, which means you train on all the data except for `p` points and predict on it, repeating until every point has been predicted on
    - **TimeSeriesSplit/Sliding Window** - for problems with a sequential or temporal component, split your data by time and gradually increase/slide your training window while keeping your validation window sequentially after the training window
- Why don’t we see more cross-validation in deep learning?
    - due to the computation time and data sizes required to train a deep learning model, it is often not feasible to train the model several times and perform something like k-fold cross-validation
    - in addition, when the dataset is large enough, a simple hold-out validation can be enough, since the size of the data in each split may have enough variance
## 7. F1 score
- What’s the benefit of F1 over the accuracy?
    - F1 score balances precision and recall, which take into account false positives and class imbalances, which accuracy doesn't
- Can we still use F1 for a problem with more than two classes. How?
    - yes, there are various methods available. In sklearn they are as follows
        - `micro` - calculate F1 globally by counting total true positives, false negatives, and false positives for all classes
        - `macro` - calculate F1 for each class and take the mean
        - `weighted` - calculate F1 for each class and take a weighted average based on the support of each label
