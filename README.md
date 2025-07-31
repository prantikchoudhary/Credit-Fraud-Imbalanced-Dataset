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

 ## Scaling and Distributing 
In this phase, we first scale the columns comprising of Time and Amount. Time and Amount should be scaled as other columns. On the other hand, we need to also create a sub sample of the dataframe in order to have an equal amount of Fraud and Non-Fraud cases.

**What is a sub-sample?**
For our case, the sub-sample would be a dataframe with a 50/50 ratio of fraud and non-fraud transactions (equally distributed)

**Why do we create a sub-sample?**
Using the original dataframe cause the following issues :
* Overfitting - Our classification models will assume that in most cases there are no frauds.
* Wrong Correlations - Although we do not know what the "V" feature stands for , it will be useful to understand how each of this feature influence the result (fraud or non-fraud)

**Summary**
* Scaled amount and scaled time are the columns with scaled values.
* 492 cases of fraud in our dataset hence we can get 492 cases of non-fraud to create the sub-sample.
* Concat the 492 cases of fraud and non-fraud creating a new sub-sample.

## Splitting the Dataframe 
Before proceeding with the **'Random UnderSampling'** Technique we have to separate the original dataframe. Why? For testing purposes, although we split the data while implementing **Random UnderSampling** or **OverSampling techniques**, we want to test our model on the original testing set not on the testing set created by these techniques. The main goal is to fit the model either with the datasamples that were undersample and oversample (in order for our model to detect the patterns) and then test it on the original testing set.

## Random Under-Sampling 
Implementing "Random Under Sampling" which basically consists of removing data in order to have a more balanced dataset and thus avoiding our model to overfit.

**Steps:**
* Determine how imbalanced is our class (value_counts() determine the amount for each label)
* Once we determine how many instances are considered fraud transactions (Fraud='1'). We bring the non-fraud transactions to the same amount as the fraud transactions (50/50 ratio), 492 cases of fraud and non-fraud transactions.
* After implementing this, we have a sub-sample of our dataframe with a 50/50 ratio. The next step is to implement a shuffle to see if our models can maintain a certain accuracy everytime we run the script.
**Note:** The main issue with "Random Under Sampling" is that we run the risk that our classification models will not perform as accurate as we would like to since there is information loss (bringing 492 non-fraud transaction from 284,315 non-fraud transactions)


 




