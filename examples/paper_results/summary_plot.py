"""
============================
SHAP summary plot
============================

This example generates a SHAP summary plot for a
quantum device calibration application. 

Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>
"""

from physlearn import Regressor
from physlearn.datasets import load_benchmark, supplementary_params
from physlearn.supervised import ShapInterpret


n_features = 9
n_targets = 5
data = load_benchmark()
X_train, X_test = data['X_train'].iloc[:, :n_features], data['X_test'].iloc[:, :n_features]
y_train, y_test = data['y_train'].iloc[:, :n_targets], data['y_test'].iloc[:, :n_targets]

index = 0
model = 'mlpregressor'

interpret = ShapInterpret(regressor_choice=model, params=supplementary_params(index), target_index=index)
interpret.summary_plot(X_train, y_train)
