"""
Utilities for automated model checking.
"""

# Author: Alex Wozniakowski
# License: MIT

import Levenshtein
import os
import re

import numpy as np

from physlearn.loss import LOSS_FUNCTIONS

from ._definition import _ESTIMATOR_DICT, _OPTIMIZE_METHOD, _SEARCH_METHOD


def _basic_autocorrect(model_choice, model_choices, model_type):
    """Choose a model in the model dictionary, which minimizes the edit distance."""

    assert isinstance(model_choice, str)
    assert isinstance(model_choices, list)
    assert isinstance(model_type, str)

    min_ = np.inf
    output_model_choice = model_choice
    for candidate_model_choice in model_choices:
        dist = Levenshtein.distance(model_choice, candidate_model_choice)
        if dist < min_:
            min_ = dist
            output_model_choice = candidate_model_choice

    if min_ > 0:
        print(model_choice, 'is not a key in the model dictionary. Instead we used',
              output_model_choice + '.')
    return output_model_choice


def _check_model_choice(model_choice, model_type):
    """Choose a model in the model dictionary."""

    assert all(isinstance(arg, str) for arg in [model_choice, model_type])

    model_choices = [choice for choice in _ESTIMATOR_DICT[model_type].keys()]
    model_choice  = _basic_autocorrect(model_choice=model_choice.strip().lower(),
                                       model_choices=model_choices,
                                       model_type=model_type)
    return model_choice


def _check_stacking_layer(stacking_layer, model_type):
    """Choose the stacking layer models in the model dictionary."""

    try:
        assert isinstance(stacking_layer, dict)
        assert isinstance(model_type, str)
        
        model_choices = [choice for choice in _ESTIMATOR_DICT[model_type].keys()]
        if 'regressors' in stacking_layer:
            stacking_layer['regressors'] = [_basic_autocorrect(model_choice=model.strip().lower(),
                                                               model_choices=model_choices,
                                                               model_type=model_type)
                                           for model in stacking_layer['regressors']]

        if 'final_regressor' in stacking_layer:
            model = stacking_layer['final_regressor']
            stacking_layer['final_regressor'] = _basic_autocorrect(model_choice=model.strip().lower(),
                                                                   model_choices=model_choices,
                                                                   model_type=model_type)
    finally:
        return stacking_layer


def _check_line_search_options(line_search_options):
    """
    Ensure that the specified options are valid for
    the line search in base boosting.
    """

    for search_key, search_option in line_search_options.items():
        if search_key == 'init_guess':
            assert isinstance(search_option, (int, float))
        elif search_key == 'opt_method':
            assert search_option in ['minimize', 'basinhopping']
        elif search_key == 'alg':
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


def _check_bayesoptcv_parameter_type(params):
    """Ensure that the Bayesian optimization utility returns an int for select int parameters."""

    assert isinstance(params, dict)

    for key, param in params.items():
        if 'max_iter' in key:
            params[key] = int(param)
        elif 'n_estimators' in key:
            params[key] = int(param)
        elif 'max_depth' in key:
            params[key] = int(param)
    return params


def _preprocessing(raw_params, multi_target, chain):
    """Preprocesses the search parameters based on the multi_target and chain filters."""

    assert isinstance(raw_params, (dict))

    tr_params = {}
    reg_params = {}
    pipe_params = {}
    if multi_target and chain:
        for key, value in raw_params.items():
            if re.match('tr__', key):
                tr_params[key] = value
            elif re.match('reg__', key):
                reg_params['reg__base_estimator__' + key[5:]] = value
            else:
                pipe_params[key] = value
    elif multi_target and not chain:
        for key, value in raw_params.items():
            if re.match('tr__', key):
                tr_params[key] = value
            elif re.match('reg__', key):
                reg_params['reg__estimator__' + key[5:]] = value
            else:
                pipe_params[key] = value
    else:
        for key, value in raw_params.items():
            if re.match('tr__', key):
                tr_params[key] = value
            elif re.match('reg__', key):
                reg_params[key] = value
            else:
                pipe_params[key] = value

    return dict(**tr_params, **reg_params, **pipe_params)


def _check_search_method(search_method):
    """
    Ensure that the search method is from either Sklearn
    or the Bayesian Optimization package.
    """

    assert isinstance(search_method, str)
    search_method = search_method.strip().lower()
    assert search_method in _SEARCH_METHOD
    return search_method


def _convert_filename_to_csv_path(filename):
    """Saves filename as a csv in the current working directory."""

    assert isinstance(filename, str)
    
    root = os.getcwd()
    if re.search('C:', root):
        path = root + '\\' + filename + '.csv'
    else:
        path = root + '/' + filename + '.csv'
    return path
