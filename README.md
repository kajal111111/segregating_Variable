# README file for the provided cross-validation Decision Tree model script:

Decision Tree Model with Stratified 5-Fold Cross-Validation
This script trains and evaluates a Decision Tree model using stratified 5-fold cross-validation. The model is trained on different subsets of the data and evaluated on the remaining parts, providing an estimate of the model's performance.

Requirements
Python 3.x
pandas
scikit-learn
You can install the necessary packages using pip:

pip install pandas scikit-learn
Script Description
The script performs the following steps:

Imports necessary libraries:

StratifiedKFold from sklearn.model_selection for stratified k-fold cross-validation.
DecisionTreeClassifier from sklearn.tree for the Decision Tree model.
accuracy_score from sklearn.metrics for evaluating model accuracy.
pandas for data manipulation.
Initializes StratifiedKFold:

Creates an instance of StratifiedKFold with 5 splits, a random state for reproducibility, and shuffling enabled.
Iterates over each fold:

Splits the dataset into training and validation sets.
Trains a DecisionTreeClassifier on the training set.
Predicts the labels for the validation set.
Calculates the accuracy of the model on the validation set.
Appends the accuracy score to a list.
Prints the mean accuracy across all folds:

Calculates and prints the average accuracy score over the 5 folds.
Usage
Ensure you have your dataset loaded into data_x (features) and data_y (target). The dataset should be compatible with pandas DataFrame and Series structures.

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Assuming data_x and data_y are your feature and target datasets
kf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
accuracy_list = []
i = 1

for train_index, test_index in kf.split(data_x, data_y):
    print('\n{} of kfold {}'.format(i, kf.get_n_splits(data_x, data_y)))
    xtr, xvl = data_x.loc[train_index], data_x.loc[test_index]
    ytr, yvl = data_y.iloc[train_index], data_y.iloc[test_index]
    
    model = DecisionTreeClassifier(random_state=1)
    model.fit(xtr, ytr)
    
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    accuracy_list.append(score)
    print('accuracy_score', score)
    
    i += 1
    
print("Mean accuracy across the folds is : ", sum(accuracy_list) / len(accuracy_list))
Replace data_x and data_y with your actual dataset variables.

Example
Here's an example of how to use the script with a sample dataset:


import pandas as pd
from sklearn.datasets import load_iris

# Load sample dataset
data = load_iris()
data_x = pd.DataFrame(data.data, columns=data.feature_names)
data_y = pd.Series(data.target)

# Run the cross-validation script
kf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
accuracy_list = []
i = 1

for train_index, test_index in kf.split(data_x, data_y):
    print('\n{} of kfold {}'.format(i, kf.get_n_splits(data_x, data_y)))
    xtr, xvl = data_x.loc[train_index], data_x.loc[test_index]
    ytr, yvl = data_y.iloc[train_index], data_y.iloc[test_index]
    
    model = DecisionTreeClassifier(random_state=1)
    model.fit(xtr, ytr)
    
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    accuracy_list.append(score)
    print('accuracy_score', score)
    
    i += 1
    
print("Mean accuracy across the folds is : ", sum(accuracy_list) / len(accuracy_list))
License
This project is licensed under the MIT License. See the LICENSE file for details.
