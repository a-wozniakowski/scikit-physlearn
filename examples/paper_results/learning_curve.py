"""
============================
Augmented learning curve
============================

This example generates an augmented learning curve for a
quantum device calibration application. 

Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>
"""

import numpy as np
import pandas as pd

from physlearn import Regressor
from physlearn.datasets import load_benchmark, paper_params
from physlearn.supervised import plot_learning_curve


X_train, X_test, y_train, y_test = load_benchmark(return_split=True)

index = 0
model = 'stackingregressor'
n_regressors = 1
boosting_loss = 'ls'
line_search_regularization = 0.1
line_search_options = dict(init_guess=1, opt_method='minimize',
                           alg='Nelder-Mead', tol=1e-7,
                           options={"maxiter": 10000},
                           niter=None, T=None)

stack = dict(regressors=['mlpregressor', 'lgbmregressor'],
             final_regressor='mlpregressor')

plot_learning_curve(regressor_choice=model, title='', X=X_train, y=y_train,
                    verbose=0, cv=3, train_sizes=np.linspace(0.25, 1.0, 40),
                    alpha=0.1, train_color='b', cv_color='orange', y_ticks_step=0.15,
                    fill_std=False, legend_loc='best', save_plot=False,
                    path=None, pipeline_transform='quantilenormal',
                    pipeline_memory=None, params=paper_params(index), chain_order=None,
                    stacking_layer=stack, target_index=index,
                    n_regressors=n_regressors, boosting_loss=boosting_loss,
                    line_search_regularization=line_search_regularization,
                    line_search_options=line_search_options, ylabel='Mean absolute error',
                    return_incumbent_score=True)
