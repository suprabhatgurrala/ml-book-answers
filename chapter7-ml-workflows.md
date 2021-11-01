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
