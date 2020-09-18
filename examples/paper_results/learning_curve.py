"""
============================
Augmented learning curve
============================

This example generates an augmented learning curve for a
quantum device calibration application. 
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import numpy as np

from physlearn import Regressor
from physlearn.datasets import load_benchmark, paper_params
from physlearn.supervised import plot_learning_curve


# Here we load the the training data. The shapes of X_train
# and y_train are (95, 5).
X_train, _, y_train, _ = load_benchmark(return_split=True)

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

# We focus on the third single-target regression subtask,
# as this is the least difficult subtask for the base regressor.
# Note that the index corresponds to the Python convention.
index = 2

# We generate the augmented learning curve with our choice
# of stacking in the single-target regression subtask: 3,
# wherein we use 40 different training data sizes. The
# first training data size corresponds to a quarter of 
# the training data, and the last training data size
# corresponds to the full amount of training data.
plot_learning_curve(regressor_choice=basis_fn,
                    title='Augmented learning curve',
                    X=X_train, y=y_train, verbose=1, cv=5,
                    train_sizes=np.linspace(0.25, 1.0, 40),
                    alpha=0.1, train_color='b',
                    cv_color='orange', y_ticks_step=0.15,
                    fill_std=False, legend_loc='best',
                    save_plot=False, path=None,
                    pipeline_transform='quantilenormal',
                    pipeline_memory=None, chain_order=None,
                    params=paper_params(index), target_index=index,
                    ylabel='Mean absolute error', stacking_options=dict(layers=stack),
                    base_boosting_options=base_boosting_options,
                    return_incumbent_score=True)
