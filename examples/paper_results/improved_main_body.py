"""
============================
Improved test error
============================

This example improves upon the machine learned test error
in the main body.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import pandas as pd

from physlearn import Regressor
from physlearn.datasets import load_benchmark, paper_params


# Here we load the the training data, as well as the test data.
# The shapes of X_train and y_train are (95, 5), and the shapes
# of X_test and y_test are (41, 5).
X_train, X_test, y_train, y_test = load_benchmark(return_split=True)

# We choose the Sklearn StackingRegressor as the basis
# function b in Eq. 2 for the single-target regression subtasks:
# 1, 2, 4, and 5. The first stacking layer consists of
# the Sklearn MLPRegressor and the LightGBM LGBMRegressor. The
# second stacking layer consists of the Sklearn MLPRegressor.
stacking_basis_fn = 'stackingregressor'
stack = dict(regressors=['mlpregressor', 'lgbmregressor'],
             final_regressor='mlpregressor')

# We choose the Sklearn Ridge as the basis function b in Eq. 2
# for the single-target regression subtask: 3. This choice of
# basis function results in a modest improvement during
# model selection.
linear_basis_fn = 'ridge'

# The number of regressors corresponds to K in Eq. 2.
n_regressors = 1

# The boosting loss is the Huber loss function, which is
# utilized in the computation of the pseudo-residuals, e.g., the
# negative gradient. This choice of loss function is less sensitive
# to outliers than our previous choice of the squared error loss
# function. See the loss module for its implementation.
boosting_loss = 'huber'


# Here we set the line search regularization strength and optimization
# algorithm.
line_search_regularization = 0.1
line_search_options = dict(init_guess=1, opt_method='minimize',
                           alg='Nelder-Mead', tol=1e-7,
                           options={"maxiter": 10000},
                           niter=None, T=None)

print('Building scoring DataFrame for each single-target regression subtask.')
test_error = []
for index in range(5):
    if index != 2:
        # We make an instance of Regressor with our choice of stacking
        # for the single-target regression subtasks: 1, 2, 4, and 5.
        reg = Regressor(regressor_choice=stacking_basis_fn, n_regressors=n_regressors,
                        boosting_loss=boosting_loss, params=paper_params(index),
                        line_search_regularization=line_search_regularization,
                        line_search_options=line_search_options,
                        stacking_layer=stack, target_index=index)
    else:
        # We make an instance of Regressor with our choice of ridge
        # regression for the single-target regression subtask: 3.
        reg = Regressor(regressor_choice=linear_basis_fn, n_regressors=n_regressors,
                        boosting_loss=boosting_loss, params=dict(alpha=0.1),
                        line_search_regularization=line_search_regularization,
                        line_search_options=line_search_options, target_index=index)

    # We use the baseboostcv method, which utilizes a private
    # inbuilt model selection method to choose either the
    # incumbent or the candidate.
    y_pred = reg.baseboostcv(X_train, y_train).predict(X_test)

    # We can check if the incumbent won model selection with
    # the return_incumbent_ attribute.
    if hasattr(reg, 'return_incumbent_'):
        print(f'Incumbent was chosen in subtask {index + 1}')

    # We compute the single-target test error.
    score = reg.score(y_test, y_pred)
    test_error.append(score)

test_error = pd.concat(test_error).round(decimals=2)
print('Finished building the scoring DataFrame.')
print(test_error)
print('Finished computing the multi-target scores.')
print(test_error.mean().round(decimals=2))
print('To gain the improvement, we computed the negative gradient of the Huber loss function',
      'instead of the negative gradient of the least squares loss function.',
      'Additionally, we used ridge regression as the basis function in the',
      'third single-target subtask.', sep='\n')
