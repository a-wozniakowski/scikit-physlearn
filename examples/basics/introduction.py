"""
============================
Single-target regression
============================

This example introduces the Regressor object in a
single-target regression task.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from physlearn import Regressor


# Load the data from Sklearn
X, y = fetch_california_housing(return_X_y=True)
X, y = pd.DataFrame(X), pd.Series(y)

# Split the data, using the default test_size=0.25.
# X_train has shape (15480, 8), y_train has shape (15480,)
# X_test has shape (5160, 8), and y_test has shape (5160,).
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42)

# Choose the underlying regressor to be the LightGBM LGBMRegressor.
# The regressor choice is a special parameter and it is case-insensitive.
regressor_choice = 'lgbmregressor'

# Choose the Sklearn StandardScaler as the data preprocessor.
# This parameter is case-sensitive.
pipeline_transform = 'standardscaler'

# Make an instance of the Regressor object.
reg = Regressor(regressor_choice=regressor_choice,
                pipeline_transform=pipeline_transform)

# Generate test data predictions
y_pred = reg.fit(X_train, y_train).predict(X_test)

# Evaluate the test error, and store
# the results as a DataFrame
score = reg.score(y_test, y_pred)

# Print the mean absolute error, mean squared error,
# root mean squared error, R2, the expected variance,
# and the mean squared log error for the single-target
# regression task.
print(score)
