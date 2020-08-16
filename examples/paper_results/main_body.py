"""
============================
Main body test error
============================

This example generates the machine learned regressor's
test error, which we report in the main body.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import pandas as pd

from physlearn import Regressor
from physlearn.datasets import load_benchmark, paper_params


# Here we load the the training data, as well as the test data.
# The shapes of X_train and y_train are (95, 5), and the shapes
# of X_test and y_test are (41, 5).
X_train, X_test, y_train, y_test = load_benchmark(return_split=True)

# We choose the Sklearn StackingRegressor as the basis function b
# in Eq. 2 of the main body. The first stacking layer consists of
# the Sklearn MLPRegressor and the LightGBM LGBMRegressor. The
# second stacking layer consists of the Sklearn MLPRegressor.
basis_fn = 'stackingregressor'
stack = dict(regressors=['mlpregressor', 'lgbmregressor'],
             final_regressor='mlpregressor')

# The number of regressors corresponds to K in Eq. 2.
n_regressors = 1

# The boosting loss is the squared error loss function, which is
# utilized in the computation of the pseudo-residuals, e.g., the
# negative gradient. 
boosting_loss = 'ls'

# Here we set the line search regularization strength, as well as
# the optimization algorithm and its parameters. Moreover, we
# specify the loss function utilized in the line search.
# Namely, lad is the key for absolute error.
line_search_options = dict(init_guess=1, opt_method='minimize',
                           method='Nelder-Mead', tol=1e-7,
                           options={"maxiter": 10000},
                           niter=None, T=None, loss='lad',
                           regularization=0.1)

# We collect the various base boosting (hyper)parameters.
base_boosting_options = dict(n_regressors=n_regressors,
                             boosting_loss=boosting_loss,
                             line_search_options=line_search_options)

print('Building scoring DataFrame for each single-target regression subtask.')
test_error = []
for index in range(5):
    # We make an instance of Regressor with our choice of stacking
    # for each single-target regression subtask.
    reg = Regressor(regressor_choice=basis_fn, params=paper_params(index),
                    target_index=index, stacking_options=dict(layers=stack),
                    base_boosting_options=base_boosting_options)

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
