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

# In the single-target regression subtask: 4, we will use
# the squared error loss function in the computation of the
# pseudo-residuals, e.g., negative gradient. This choice
# results in stable model selection results. In the other
# single-target regression subtasks, namely: 1, 2, 3, and 5,
# we will use the Huber loss function in the computation of the
# pseudo-residuals. See the loss module for its implementation,
# it is written as in the M-regression section 4.4 of refrerence:
# Jerome Friedman. Greedy function approximation: A gradient
# boosting machine. Annals of Statistics, 29(5):1189–1232, 2001.
huber_loss = 'huber'
ls_loss = 'ls'


# Here we set the line search regularization strength, as well as
# the optimization algorithm and its parameters. Moreover, we
# specify the loss function utilized in the line search.
# Namely, lad is the key for absolute error in the single-target
# regression subtask: 4. Otherwise, 'huber' is the key for the
# Huber loss function in the single-target regression subtasks:
# 1, 2, 3, and 5.
line_search_options_with_lad = dict(init_guess=1, opt_method='minimize',
                                    method='Nelder-Mead', tol=1e-7,
                                    options={"maxiter": 10000},
                                    niter=None, T=None, loss='lad',
                                    regularization=0.1)
line_search_options_with_huber = dict(init_guess=1, opt_method='minimize',
                                      method='Nelder-Mead', tol=1e-7,
                                      options={"maxiter": 10000},
                                      niter=None, T=None, loss='huber',
                                      regularization=0.1)

# We collect the various base boosting (hyper)parameters.
options_with_lad = dict(n_regressors=n_regressors,
                        boosting_loss=ls_loss,
                        line_search_options=line_search_options_with_lad)
options_with_huber = dict(n_regressors=n_regressors,
                          boosting_loss=huber_loss,
                          line_search_options=line_search_options_with_huber)

print('Building scoring DataFrame for each single-target regression subtask.')
test_error = []
for index in range(5):
    if index in [0, 1, 4]:
        # We make an instance of Regressor with our choice of stacking
        # for the single-target regression subtasks: 1, 2, and 5.
        reg = Regressor(regressor_choice=stacking_basis_fn, params=paper_params(index),
                        target_index=index, stacking_options=dict(layers=stack),
                        base_boosting_options=options_with_huber)
    elif index == 3:
        # We make an instance of Regressor with our choice of stacking
        # for the single-target regression subtask: 4.
        reg = Regressor(regressor_choice=stacking_basis_fn, params=paper_params(index),
                        target_index=index, stacking_options=dict(layers=stack),
                        base_boosting_options=options_with_lad)
    else:
        # We make an instance of Regressor with our choice of ridge
        # regression for the single-target regression subtask: 3.
        # The parameter alpha denotes the regularization strength
        # in ridge regression, where the Tikhonov matrix is the
        # scalar alpha times the identity matrix.
        reg = Regressor(regressor_choice=linear_basis_fn, params=dict(alpha=0.1),
                        target_index=index, base_boosting_options=options_with_huber)

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

test_error = pd.concat(test_error)
print('Finished building the scoring DataFrame.')
print(test_error.round(decimals=2))
print('Finished computing the multi-target scores.')
print(test_error.mean().round(decimals=2))
print('To gain the improvement, we used the Huber loss function in the',
      'negative gradient and line search computations in the single-target',
      'regression subtasks: 1, 2, 3, and 5. We used the squared error loss',
      'function in the negative gradient computation and the absolute error',
      'loss function in the line search computation in the single-target',
      'regression subtask: 4. Additionally, we used ridge regression as the',
      'basis function in the single-target regression subtask: 3.', sep='\n')
