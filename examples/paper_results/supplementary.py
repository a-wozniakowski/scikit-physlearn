"""
============================
Supplementary test error
============================

This example generates the machine learned test error in the
supplementary, wherein the machine learner is not inductively
biased by the incumbent.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import pandas as pd

from physlearn import Regressor
from physlearn.datasets import load_benchmark, supplementary_params


# Here we load the the training data, as well as the test data.
# Each example corresponds to 9 raw control voltage features,
# and the multi-targets are the sorted eigenenergies, as in the
# benchmark task.
n_features = 9
n_targets = 5
data = load_benchmark()
X_train, X_test = data['X_train'].iloc[:, :n_features], data['X_test'].iloc[:, :n_features]
y_train, y_test = data['y_train'].iloc[:, :n_targets], data['y_test'].iloc[:, :n_targets]

# We choose the Sklearn MLPRegressor.
model = 'mlpregressor'

print('Building scoring DataFrame for each single-target subtask.')
test_error = []
for index in range(5):
    # We make an instance of Regressor with our choice of
    # fully connected neural network for the single-target
    # regression subtask: 1.
    reg = Regressor(regressor_choice=model, params=supplementary_params(index),
                    target_index=index)

    # We invoke the fit and predict methods, then we
    # compute the single-target test error.
    y_pred = reg.fit(X_train, y_train).predict(X_test)
    score = reg.score(y_test, y_pred)
    test_error.append(score)

test_error = pd.concat(test_error).round(decimals=2)
print('Finished building the scoring DataFrame.')
print(test_error)
print('Finished computing the multi-target scores.')
print(test_error.mean().round(decimals=2))
