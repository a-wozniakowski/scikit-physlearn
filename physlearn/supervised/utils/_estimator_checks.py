"""
The :mod:`physlearn.supervised.utils._estimator_checks` module provides
basic utilities for automated estimator checking. 
"""

# Author: Alex Wozniakowski
# License: MIT

import Levenshtein
import os
import re
import warnings

import numpy as np

from physlearn.loss import LOSS_FUNCTIONS
from physlearn.supervised.utils._definition import (_BAYESOPTCV_INIT_PARAMS,
                                                    _ESTIMATOR_DICT,
                                                    _OPTIMIZE_METHOD,
                                                    _PIPELINE_PARAMS,
                                                    _SEARCH_METHOD)


def _basic_autocorrect(init_choice: str, candidate_choices: list) -> str:
    """Chooses the candidate string that minimizes the edit distance.

    Parameters
    ----------
    init_choice : str
        Specify the initial choice as a string, e.g., the Scikit-Learn class
        Ridge as 'ridge', 'Ridge', 'RIDGE', etc.

    candidate_choices : list
        A list of candidate choices, where each candidate is a string.

    Returns
    -------
    out_choice : str

    Notes
    -----
    The edit distance between the initial choice and each possible choice corresponds
    to the Levenshtein distance, which uses the operations of insertion, removal, or
    substitution to count the distance.
    """

    assert isinstance(init_choice, str)
    assert isinstance(candidate_choices, list)

    min_dist = np.inf
    out_choice = init_choice
    for candidate_choice in candidate_choices:
        dist = Levenshtein.distance(init_choice, candidate_choice)
        if dist < min_dist:
            min_dist = dist
            out_choice = candidate_choice

    if min_dist > 0:
        warnings.warn(f'{init_choice} was misspelled, so we replaced it with {out_choice}.',
                      UserWarning)
    return out_choice


def _check_estimator_choice(estimator_choice: str, estimator_type: str,
                            estimator_choices=None) -> str:
    """Chooses the candidate estimator that minimizes the edit distance.

    Parameters
    ----------
    estimator_choice : str
        Specify the estimator choice as a string, e.g., the Scikit-Learn
        class Ridge as 'ridge', 'Ridge', 'RIDGE', etc.

    estimator_type : str
        Specify the supervised learning task, e.g., regression.

    estimator_choices : list or None, optional (default=None)
        A list of estimator choices, where each estimator is a string.

    Returns
    -------
    estimator_choice : str
    """

    assert all(isinstance(arg, str) for arg in [estimator_choice, estimator_type])

    if estimator_choices is not None:
        assert isinstance(estimator_choices, list)
        assert all(isinstance(choices, str) for choices in estimator_choices)
    else:
        estimator_choices = [choice for choice in _ESTIMATOR_DICT[estimator_type].keys()]

    estimator_choice  = _basic_autocorrect(init_choice=estimator_choice.strip().lower(),
                                           candidate_choices=estimator_choices)
    return estimator_choice


def _check_stacking_layer(stacking_layer: dict, estimator_type: str) -> dict:
    """Chooses the the first and second stacking layer estimators.

    Parameters
    ----------
    stacking_layer : dict
        Specify the estimator(s) in the first stacking layer, and the
        final estimator in the second stacking layer.

    estimator_type : str
        Specify the supervised learning task, e.g., regression.

    Returns
    -------
    stacking_layer : dict
    """

    assert isinstance(stacking_layer, dict)
    assert isinstance(estimator_type, str)

    estimator_choices = [choice for choice in _ESTIMATOR_DICT[estimator_type].keys()]

    # Loop through the first and second stacking layers,
    # and compute the edit distance for each estimator.
    # If the edit distance is positive, then replace
    # the estimator with the minimal distance estimator.
    for key, layer in stacking_layer.items():
        if key in ['estimators', 'regressors']:
            stacking_layer[key] = [_check_estimator_choice(estimator_choice=est.strip().lower(),
                                                           estimator_type=estimator_type, 
                                                           estimator_choices=estimator_choices)
                                  for est in layer]
        elif key in ['final_estimator', 'final_regressor']:
            stacking_layer[key] = _check_estimator_choice(estimator_choice=layer.strip().lower(),
                                                          estimator_type=estimator_type, 
                                                          estimator_choices=estimator_choices)
        else:
            raise KeyError('The key: %s is not a valid choice for stacking_layer.'
                           % (key))
    return stacking_layer


def _check_line_search_options(line_search_options: dict) -> None:
    """Checks the line search computation options for base boosting.

    Parameters
    ----------
    init_guess : int, float, or ndarray
        The initial guess for the expansion coefficient.

    opt_method : str
        Choice of optimization method. If ``'minimize'``, then
        :class:`scipy.optimize.minimize`, else if ``'basinhopping'``,
        then :class:`scipy.optimize.basinhopping`.

    method : str or None
        The type of solver utilized in the optimization method.

    tol : float or None
        The epsilon tolerance for terminating the optimization method.

    options : dict or None
        A dictionary of solver options.

    niter : int or None
        The number of iterations in basin-hopping.

    T : float or None
        The temperature paramter utilized in basin-hopping,
        which determines the accept or reject criterion.

    loss : str
        The loss function utilized in the line search computation, where 'ls'
        denotes the squared error loss function, 'lad' denotes the absolute error
        loss function, 'huber' denotes the Huber loss function, and 'quantile'
        denotes the quantile loss function.

    regularization : int or float
        The regularization strength in the line search computation.
    """

    for search_key, search_option in line_search_options.items():
        if search_key == 'init_guess':
            assert isinstance(search_option, (int, float, np.array))
        elif search_key == 'opt_method':
            assert search_option in ['minimize', 'basinhopping']
        elif search_key == 'method':
            assert search_option in _OPTIMIZE_METHOD
        elif search_key == 'tol':
            assert isinstance(search_option, float)
        elif search_key == 'options':
            assert isinstance(search_option, dict)
        elif search_key == 'niter':
            if search_option is not None:
                assert isinstance(search_option, int)
        elif search_key == 'T':
            if search_option is not None:
                assert isinstance(search_option, float)
        elif search_key == 'loss':
            assert search_option in LOSS_FUNCTIONS
        elif search_key == 'regularization':
            assert isinstance(search_option, (int, float))
        else:
            raise KeyError('The key: %s is not a valid line search option.'
                           % (search_key))


def _check_bayesoptcv_param_type(pbounds: dict) -> dict:
    """Checks if the Bayesian optimization utility changed the (hyper)parameter type.

    Parameters
    ----------
    pbounds: dict
        A dictionary, wherein the keys are the (hyper)parameter names
        and the values are the (hyper)parameter values.

    Returns
    -------
    pbounds : dict

    Notes
    -----
    During the sequential Bayesian optimization, the utility occasionally sets
    the value of a (hyper)parameter with type int to a value with type float.
    """

    assert isinstance(pbounds, dict)

    for key, param in pbounds.items():
        if key in _BAYESOPTCV_INIT_PARAMS:
            pbounds[key] = int(value)
    return pbounds


def _preprocess_hyperparams(raw_params: dict, multi_target: bool,
                            chain: bool) -> dict:
    """Preprocesses the (hyper)parameters.

    The preprocessing is determined by the regression task, and the assumption
    on the single-targets, if the task is multi-target regression.

    Parameters
    ----------
    raw_params : dict
        The user provided (hyper)parameters.

    multi_target : bool
        Distinguishes between single-target and multi-target regression.
        If True, then the expected task is multi-target regression.

    chain : bool
        Distinguishes between independent single-target regression
        subtasks and chaining. If true, then the expected multi-target
        combination is chaining.

    Returns
    -------
    out_params : dict
    """

    assert isinstance(raw_params, (dict))

    out_params = {}
    if multi_target and chain:
        for key, value in raw_params.items():
            if re.match('tr__', key):
                out_params[key] = value
            elif re.match('reg__', key):
                out_params['reg__base_estimator__' + key[5:]] = value
            elif key in _PIPELINE_PARAMS:
                out_params[key] = value
            else:
                raise KeyError('The key: %s is not a valid (hyper)parameter name.'
                               % (key))
    elif multi_target and not chain:
        for key, value in raw_params.items():
            if re.match('tr__', key):
                out_params[key] = value
            elif re.match('reg__', key):
                out_params['reg__estimator__' + key[5:]] = value
            elif key in _PIPELINE_PARAMS:
                out_params[key] = value
            else:
                raise KeyError('The key: %s is not a valid (hyper)parameter name.'
                               % (key))
    else:
        for key, value in raw_params.items():
            if re.match('tr__', key):
                out_params[key] = value
            elif re.match('reg__', key):
                out_params[key] = value
            elif key in _PIPELINE_PARAMS:
                out_params[key] = value
            else:
                raise KeyError('The key: %s is not a valid (hyper)parameter name.'
                               % (key))

    return out_params


def _check_search_method(search_method: str) -> str:
    """Chooses the (hyper)parameter search method that minimizes the edit distance.

    Parameters
    ----------

    search_method : str
        Specifies the Scikit-learn or Bayesian optimization
        (hyper)parameter search method.

    Returns
    -------
    search_method : str
    """

    return _basic_autocorrect(init_choice=search_method.strip().lower(),
                              candidate_choices=_SEARCH_METHOD)
