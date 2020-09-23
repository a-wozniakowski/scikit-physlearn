"""
============================
Multi-target regression
============================

This example introduces the Regressor object in a
multi-target regression task.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import pandas as pd

from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split

from physlearn import Regressor


# Load the data from Sklearn
bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
X, y = bunch['data'], bunch['target']

# Split the data, using the default test_size=0.25.
# X_train has shape (15, 3), y_train has shape (15, 3)
# X_test has shape (5, 3), and y_test has shape (5, 3).
# Namely, there are 3 features and 3 single-target regression subtasks.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Choose the underlying regressor to be the Sklearn
# histogram-based gradient boosting regressor.
regressor_choice = 'HistGradientBoostingRegressor'

# Choose the Sklearn QuantileTransformer as the data preprocessor.
# The output distribution is the Gaussian, e.g., 'normal'.
# The number of quantiles is the number of examples in y_train,
# e.g., 15.
pipeline_transform = 'quantilenormal'

# Make an instance of the Regressor object.
reg = Regressor(regressor_choice=regressor_choice, pipeline_transform=pipeline_transform)

# Generate test data predictions
y_pred = reg.fit(X_train, y_train).predict(X_test)

# Evaluate the test error, and store
# the results as a DataFrame
score = reg.score(y_test, y_pred)

# Print the mean absolute error, mean squared error,
# root mean squared error, R2, the expected variance,
# and the mean squared log error for each individual
# single-target regression subtask.
print(score)
