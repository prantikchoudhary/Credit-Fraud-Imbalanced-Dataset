## Introduction
In this repository, we used various predictive models to see how accurate they are in detecting whether a transaction is a normal payment or a fraud. In the dataset-scaled features, the names of the features are not shown due to privacy reasons.

## Goals 
1. Understand the distribution of the data that was provided to us.
2. Create a 50/50 sub-dataframe ratio of "fraud" and "non-fraud" transactions.
3. Determine the Classifiers that help decide which one has higher accuracy.
4. Create a neural network to compare the accuracy of the best Classifier.
5. Understand the common mistakes made with imbalanced datasets since the number of either fraudulent transactions or non-fraudulent transactions are way more higher than the other which makes the predictive model biased towards the one class of data (maybe fraudulent or non-fraudulent transactions) since the class distribution is uneven.

#### Correcting previous mistakes from imbalanced datasets:
* **Never test on the oversampled or undersampled dataset -** Oversampling (duplicating fraud cases) and undersampling (removing the non-fraudulent cases) are techniques used to balance the imbalanced dataset, but are only applied to the training data. Our test set should reflect the real-world data distribution as testing on synthetic data gives unrealistic and biased performance.
* **While implementing cross validation, remember to oversample or undersample the training data during cross validation and not before -** In cross validation the data is split into "folds" for training and testing. Oversample/Undersample only on the training fold not on the entire dataset beforehand. If we balance the whole dataset before splitting, information from the test set can "leak" into the training dataset (called data leakage), this leads to overly optimistic performance results.
* **Not using accuracy score as a metric to measure with imbalanced datasets, instead using f1-score, precision/recall score or the confusion matrix -** with imbalanced datasets a model can get high accuracy just by always predicting the majority class (non-fraudulent transactions).

## Environment Setup
* Faced an issue while importing the TensorFlow library as the kernel was using Python version 3.13.5 but Tensor Flow is only supported for Python versions until 3.12.
* Deprecated the Python version of the Kernel from 3.13.5 to 3.12.11
  - Gained access as the root user in Terminal
  - Altered Python version at '/opt/anaconda3/bin/python'
  - conda create -n tf_env python=3.12.11
  - conda activate tf_env
  - Restart Anaconda Navigator
  - !pip install tensorflow
  - Restart Kernel
  - Now we can 'import  tensorflow' as required
 




