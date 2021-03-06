"""
The :mod:`physlearn.supervised.model_selection.search` module provides
basic utilities for automated (hyper)parameter search.
"""

# Author: Alex Wozniakowski
# License: MIT

from __future__ import annotations

import typing

import numpy as np

import bayes_opt
import sklearn.model_selection

from physlearn.supervised.utils._estimator_checks import _check_bayesoptcv_param_type
from physlearn.supervised.utils._definition import _SEARCH_METHOD

search_method = typing.Union[sklearn.model_selection.GridSearchCV,
                             sklearn.model_selection.RandomizedSearchCV,
                             bayes_opt.BayesianOptimization]


def _bayesoptcv(X, y, estimator, search_params, cv,
                scoring, n_jobs, verbose, random_state,
                init_points, n_iter):

    def regressor_cross_val_mean(**pbounds):
        estimator.set_params(**_check_bayesoptcv_param_type(pbounds=pbounds))
        cross_val = sklearn.model_selection.cross_val_score(estimator=estimator,
                                                            X=X, y=y,
                                                            scoring=scoring,
                                                            cv=cv,
                                                            n_jobs=n_jobs)
        return cross_val.mean()

    search = bayes_opt.BayesianOptimization(f=regressor_cross_val_mean,
                                            pbounds=search_params,
                                            verbose=verbose,
                                            random_state=random_state)

    search.maximize(init_points=init_points, n_iter=n_iter)

    return search


def _search_method(search_method: str, pipeline: ModifiedPipeline,
                   search_params: dict, scoring: str, refit=True,
                   n_jobs=-1, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                   error_score=np.nan, return_train_score=None,
                   randomizedcv_n_iter=None, X=None, y=None,
                   random_state=None, init_points=None,
                   bayesoptcv_n_iter=None) -> search_method:
    """Helper (hyper)parameter search function.

    Parameters
    ----------
    search_method : str
        Specifies the search method. If ``'gridsearchcv'``, ``'randomizedsearchcv'``,
        or ``'bayesoptcv'`` then the search method is GridSearchCV, RandomizedSearchCV,
        or Bayesian Optimization.

    pipeline : ModifiedPipeline
        A ModifiedPipeline object.

    search_params : dict
        Dictionary with (hyper)parameter names as keys, and either lists of
        (hyper)parameter settings to try as values or tuples of (hyper)parameter
        lower and upper bounds to try as values.

    scoring : str, callable, list/tuple, or dict, optional (default='neg_mean_absolute_error')
        Determines scoring in the k-fold cross-validation methods.

    refit : bool, optional (default=True)
        Determines whether to return the refit ModifiedPipeline object.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel in GridSearchCV and RandomizedSearchCV.

    cv : int, cross-validation generator, an iterable, or None, optional (default=None)
        Determines the cross-validation strategy. If None, then the default
        is 5-fold cross-validation.

    verbose : int, optional (default=0)
        Determines verbosity.

    pre_dispatch : int or str, optional (default='2*n_jobs')
        Controls the number of jobs that get dispatched during parallel execution in
        GridSearchCV and RandomizedSearchCV.

    error_score : 'raise' or numeric, optional (default=np.nan)
        The assigned value if an error occurs while inducing a regressor.
        If set to 'raise', then the specific error is raised. Else if set
        to a numeric value, then FitFailedWarning is raised in GridSearchCV
        and RandomizedSearchCV.

    return_train_score : bool or None, optional (default=None)
        Determines whether to return the training scores from the k-fold
        cross-validation methods in GridSearchCV and RandomizedSearchCV.

    randomizedcv_n_iter : int or None, optional (default=None)
        Determines the number of (hyper)parameter settings that are
        sampled in RandomizedSearchCV.

    X : array-like of shape = [n_samples, n_features] or None, optional (default=None)
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s). Used in Bayesian Optimization.

    y : array-like of shape = [n_samples] or shape = [n_samples, n_targets] or None, optional (default=None)
        The target matrix, where each row corresponds to an example and the
        column(s) correspond to the single-target(s). Used in Bayesian Optimization.

    random_state : int, RandomState instance, or None, optional (default=0)
        Determines the random number generation in Bayesian Optimization.

    init_points : int or None, optional (default=None)
        Determines the number of random exploration steps in Bayesian
        Optimization. Increasing the number corresponds to diversifying
        the exploration space.

    bayesoptcv_n_iter : int or None, optional (default=None)
        Determines the number of optimization steps in in Bayesian
        Optimization.
    """

    assert search_method in _SEARCH_METHOD

    if search_method == 'gridsearchcv':
        search = sklearn.model_selection.GridSearchCV(estimator=pipeline,
                                                      param_grid=search_params,
                                                      scoring=scoring,
                                                      refit=refit,
                                                      n_jobs=n_jobs,
                                                      cv=cv,
                                                      verbose=verbose,
                                                      pre_dispatch=pre_dispatch,
                                                      error_score=error_score,
                                                      return_train_score=return_train_score)
    elif search_method == 'randomizedsearchcv':
        search = sklearn.model_selection.RandomizedSearchCV(estimator=pipeline,
                                                            param_distributions=search_params,
                                                            n_iter=randomizedcv_n_iter,
                                                            scoring=scoring,
                                                            refit=refit,
                                                            n_jobs=n_jobs,
                                                            cv=cv,
                                                            verbose=verbose,
                                                            pre_dispatch=pre_dispatch,
                                                            error_score=error_score,
                                                            return_train_score=return_train_score)
    elif search_method == 'bayesoptcv':
        search = _bayesoptcv(X=X, y=y,
                             estimator=pipeline,
                             search_params=search_params,
                             cv=cv,
                             scoring=scoring,
                             n_jobs=n_jobs,
                             verbose=verbose,
                             random_state=random_state,
                             init_points=init_points,
                             n_iter=bayesoptcv_n_iter)
    else:
        raise KeyError('The search method: %s is not a recognized option. '
                       % (search_method))

    return search
