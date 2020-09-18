"""
The :mod:`physlearn.supervised.utils._data_checks` module provides basic
utilities for automated data checking.
"""

# Author: Alex Wozniakowski
# License: MIT

import os
import re
import typing

import numpy as np
import pandas as pd

import sklearn.utils.multiclass

DataFrame_or_Series = typing.Union[pd.DataFrame, pd.Series]


def _check_X(X: DataFrame_or_Series) -> DataFrame_or_Series:
    """Checks if the design matrix uses a pandas data representation.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        The design matrix, where each row corresponds to an example and the
        column(s) correspond to the feature(s).

    Returns
    -------
    X : pd.Series or pd.DataFrame
    """

    assert isinstance(X, (pd.Series, pd.DataFrame))
    return X

def _check_y(y: DataFrame_or_Series) -> DataFrame_or_Series:
    """Checks if the target matrix uses a pandas data representation.

    Parameters
    ----------
    y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
        The target matrix, where each row corresponds to an example and the
        column(s) correspond to the single-target(s).

    Returns
    -------
    y : pd.Series or pd.DataFrame
    """

    assert isinstance(y, (pd.Series, pd.DataFrame))
    return y


def _check_X_y(X: DataFrame_or_Series, y: DataFrame_or_Series) -> DataFrame_or_Series:
    """Checks if the design and target matrices use a pandas data representations.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        The design matrix, where each row corresponds to an example and the
        column(s) correspond to the feature(s).

    y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
        The target matrix, where each row corresponds to an example and the
        column(s) correspond to the single-target(s).

    Returns
    -------
    data : tuple
    """

    X = _check_X(X=X)
    y = _check_y(y=y)

    assert X.index.equals(y.index)

    target_type = sklearn.utils.multiclass.type_of_target(y)
    assert any(target_type != continuous_target_type
           for continuous_target_type in ['continous', 'continuous-multioutput'])

    if X.ndim > 1:
        n_features = X.shape[-1]
    else:
        n_features = 1

    data = pd.concat([X, y], axis=1).dropna(how='any', axis=0)
    if target_type == 'continuous-multioutput':
        return data.iloc[:, :n_features], data.iloc[:, n_features:]
    else:
        return data.iloc[:, :n_features], data.iloc[:, n_features:].squeeze()


def _validate_data(X=None, y=None):
    """Bundles the pandas data checks together.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features] or None, optional (default=None)
        The design matrix, where each row corresponds to an example and the
        column(s) correspond to the feature(s).

    y : array-like of shape = [n_samples] or shape = [n_samples, n_targets] or None, optional (default=None)
        The target matrix, where each row corresponds to an example and the
        column(s) correspond to the single-target(s).

    Returns
    -------
    data : tuple, pd.Series, or pd.DataFrame
    """

    if X is not None and y is not None:
        data = _check_X_y(X=X, y=y)
    elif X is not None:
        data = _check_X(X=X)
    elif y is not None:
        data = _check_y(y=y)
    else:
        raise ValueError('Both the design matrix X and the target matrix y are None. '
                         'Thus, there is no data to validate.')
    return data


def _n_features(X: DataFrame_or_Series) -> int:
    """Counts the number of features in the design matrix.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features] or None, optional (default=None)
        The design matrix, where each row corresponds to an example and the
        column(s) correspond to the feature(s).

    Returns
    -------
    dim : int
    """

    if X.ndim > 1:
        dim = X.shape[-1]
    else:
        dim = 1
    return dim


def _n_targets(y: DataFrame_or_Series) -> int:
    """Counts the number of targets in the target matrix.

    Parameters
    ----------
    y : array-like of shape = [n_samples] or shape = [n_samples, n_targets] or None, optional (default=None)
        The target matrix, where each row corresponds to an example and the
        column(s) correspond to the single-target(s).

    Returns
    -------
    dim : int
    """

    if y.ndim > 1:
        dim = y.shape[-1]
    else:
        dim = 1
    return dim


def _n_samples(y: DataFrame_or_Series) -> int:
    """Counts the number of observations in the target matrix.
    
    Parameters
    ----------
    y : array-like of shape = [n_samples] or shape = [n_samples, n_targets] or None, optional (default=None)
        The target matrix, where each row corresponds to an example and the
        column(s) correspond to the single-target(s).

    Returns
    -------
    samples : int
    """

    return y.shape[0]
