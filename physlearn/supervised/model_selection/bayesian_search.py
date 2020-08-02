"""
Bayesian optimization (hyper)parameter search.
"""

# Author: Alex Wozniakowski
# License: MIT

import sklearn.model_selection

from bayes_opt import BayesianOptimization
from physlearn.supervised.utils._model_checks import _check_bayesoptcv_parameter_type


def _bayesoptcv(X, y, estimator, search_params, cv,
                scoring, n_jobs, verbose, random_state,
                init_points, n_iter):

    def regressor_cross_val_mean(**pbounds):
        pbounds = _check_bayesoptcv_parameter_type(pbounds)
        estimator.set_params(**pbounds)
        cross_val = sklearn.model_selection.cross_val_score(estimator=estimator,
                                                            X=X, y=y, scoring=scoring,
                                                            cv=cv, n_jobs=n_jobs)
        return cross_val.mean()

    search = BayesianOptimization(f=regressor_cross_val_mean, pbounds=search_params,
                                  verbose=verbose, random_state=random_state)
    search.maximize(init_points=init_points, n_iter=n_iter)

    return search
