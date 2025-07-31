# Importing Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import tensorflow as tf # building, training and evaluating neaural network models to classify transactions as fraud or not

import matplotlib.pyplot as plt # Plotting charts and visualizations
import seaborn as sns # Statistical data visualization (built on top of matplotlib)
from sklearn.manifold import TSNE # visualizing high dimensional data
from sklearn.decomposition import PCA, TruncatedSVD # Reducing features to lower dimensions for better visualization and removing noise
import matplotlib.patches as mpatches # custom shapes and legend entries in plots
import time 

# Importing Classifier Libraries
from sklearn.linear_model import LogisticRegression # classifier to determine transaction is fraud or not 
from sklearn.svm import SVC # advanced classifier to capture complex, non linear patters
from sklearn.neighbors import KNeighborsClassifier # classifies transactions by looking at closest examples in training set
from sklearn.tree import DecisionTreeClassifier # builds a decision tree to distinguish between transactions
from sklearn.ensemble import RandomForestClassifier # model using multiple trees to improve prediction accuracy for fraud detection
import collections # to count class distribution or organizing grouped data

# Other Libraries
from sklearn.model_selection import train_test_split # split data into train and test sets
from sklearn.pipeline import make_pipeline # chains data-preprocessing steps and the classifier in single workflow
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline # allows oversampling/undersampling techniques
from imblearn.over_sampling import SMOTE # balances dataset by generating synthetic samples for minority class (fraud cases)
from imblearn.under_sampling import NearMiss # balances dataset by removing some majority class (non-fraud) based on distance criteria
from imblearn.metrics import classification_report_imbalanced # reports evaluation metrics tuned for imbalanced datasets
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report # measure model performance
from collections import Counter # how many fraud/non-fraud samples
from sklearn.model_selection import KFold, StratifiedKFold # data splitting for cross validation
import warnings # controls warning messages 
warnings.filterwarnings("ignore")

df = pd.read_csv('/Users/apple/Downloads/creditcard.csv')
df.head()

df.describe()

df.isnull().sum().max() # checking for null values in dataset

df.columns

print ('No Frauds' , round(df['Class'].value_counts()[0]/len(df)*100,2), '% of the dataset')
print ('Frauds' , round(df['Class'].value_counts()[1]/len(df)*100,2) , '% of the dataset')

colors = ["#0101DF", "#DF0101"]

# Fix: specify 'Class' as the x parameter instead of a positional argument
sns.countplot(x='Class', data=df, palette=colors)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)

fig, ax = plt.subplots(1, 2, figsize=(18,4)) # creates a figure with 2 side by side plots to display 2 graphs in a row

amount_val = df['Amount'].values # extracts transaction amount
time_val = df['Time'].values # extracts transaction timestamp

sns.distplot(amount_val, ax=ax[0], color='r') # plots histogram with a density curve for transaction amounts
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b') # plots the histogram with a density curve for transaction times 
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])



plt.show() # to render the output in the output window


# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
from sklearn.preprocessing import StandardScaler, RobustScaler # loads scaling tools that normalize the data features

# RobustScaler is less prone to outliers which is important because fraud datasets often contain extreme values. 
# It works by subtracting the median and dividing by IQR (Interquartile Range) making it robust to very big outliers.

std_scaler = StandardScaler() # creates object to scale data using standard scaling (zscore)
rob_scaler = RobustScaler() # creates robust scaling using medians and IQR

# Creates 2 new columns (scaled_amount and scaled_time) containing the normalized values
df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1)) # reshape(-1,1) required as scalers expect 2D arrays
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1)) 

df.drop(['Time','Amount'], axis=1, inplace=True) # dropping original 'Time' and 'Amount' columns from dataframe, 
# keeping only the scaled versions for further processing and modeling.

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!

df.head()

from sklearn.model_selection import train_test_split # imports functions to split data into train and test sets
from sklearn.model_selection import StratifiedShuffleSplit # maintain class distribution between splits

# shows the percentage split between "No Fraud" (Class=0) and "Fraud" (Class=1) in the dataset.
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

# features (X) are all the info about each transaction - like how much money, what time .etc.
X = df.drop('Class', axis=1)
# labels (y) are what we want to predict: is this a fraud (1) or not (0)
y = df['Class']

# we divide the data into 5 parts, each part ('fold') will have the same percentage of frauds as the whole dataset.
# so if our full data is 0.2% fraud, each part will also have about 0.2% fraud.
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False) 

# the loop goes through 5 possible splits, splitting the data into a training set and testing set .
# Here , we will use only the last split
for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.
# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the Distribution of the labels


# Turning the dataframe into an array for the program
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed as the original dataset provided to us
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))

# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

# mixes up all the rows in original data table randomly.
df = df.sample(frac=1)

fraud_df = df.loc[df['Class'] == 1] # finds "fraud" transactions (Class=1)
non_fraud_df = df.loc[df['Class'] == 0][:492] # finds "normal" transactions (Class-0) but only takes the first 492 of them (to balance with 492 frauds)

normal_distributed_df = pd.concat([fraud_df, non_fraud_df]) # concats the two datasets

# Shuffle the new balanced data again, random_state=42 makes sure you get the same shuffle every time we run the code
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()

print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))

# Fixed countplot function call by adding x= parameter name
sns.countplot(x='Class', data=new_df, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

