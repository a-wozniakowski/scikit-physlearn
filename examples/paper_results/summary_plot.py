"""
============================
SHAP summary plot
============================

This example generates a SHAP summary plot for a
quantum device calibration application. 
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

from physlearn import Regressor
from physlearn.datasets import load_benchmark, supplementary_params
from physlearn.supervised import ShapInterpret


# Here we load the the training data, as well as the test data.
# Each example corresponds to 9 raw control voltage features,
# and the multi-targets are the sorted eigenenergies, as in the
# benchmark task.
n_features = 9
n_targets = 5
data = load_benchmark()
X_train, X_test = data['X_train'].iloc[:, :n_features], data['X_test'].iloc[:, :n_features]
y_train, y_test = data['y_train'].iloc[:, :n_targets], data['y_test'].iloc[:, :n_targets]

# We focus on the first single-target regression subtask,
# as this is the most difficult subtask for the base regressor.
# Note that the index corresponds to the Python convention.
index = 0

# We choose the Sklearn MLPRegressor.
model = 'mlpregressor'

# We make an instance of ShapInterpret with our choice
# of neural network for the single-target regression subtask: 1.
# We set the show parameter as True, which enables SHAP to display
# the plot.
interpret = ShapInterpret(regressor_choice=model, params=supplementary_params(index),
                          target_index=index, show=True)

# We generate the SHAP summary plot using the training data.
interpret.summary_plot(X_train, y_train)
