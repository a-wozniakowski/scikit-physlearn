"""
The :mod:`physlearn.loss` module enables computation of the average loss
or the negative gradient in either the single-target or the multi-target
regression setting, whereby data can be represented heterogeneously with
Numpy or Pandas. It includes the :class:`physlearn.LeastSquaresError`,
:class:`physlearn.LeastAbsoluteError`, :class:`physlearn.HuberLossFunction`,
:class:`physlearn.QuantileLossFunction` classes, and the helper
:func:`physlearn.loss._difference` function.
"""

# Author: Alex Wozniakowski
# License: MIT

import typing

import numpy as np
import pandas as pd

import sklearn.ensemble
import sklearn.metrics

pandas_or_numpy = typing.Union[pd.DataFrame, pd.Series, np.ndarray]


def _difference(y: pandas_or_numpy, raw_predictions: pandas_or_numpy) -> pandas_or_numpy:
    """Subtract the raw predictions from the single-target(s).

    The function supports heterogeneous usage of Numpy and pandas
    data representations.

    Parameters
    ----------

    y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
        The target matrix, where each row corresponds to an example and the
        column(s) correspond to the single-target(s).

    raw_predictions : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
        The estimate matrix, where each row corresponds to an example and the
        column(s) correspond to the prediction(s) for the single-target(s).

    Returns
    -------
    diff : DataFrame, Series, or ndarray
        The difference between the single-target(s) and the raw predictions.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_linnerud
    >>> from physlearn.loss import _difference
    >>> X, y = load_linnerud(return_X_y=True)
    >>> _difference(y=pd.DataFrame(y), raw_predictions=X).iloc[:2]
           0      1     2
    0  186.0 -126.0 -10.0
    1  187.0  -73.0  -8.0
    """

    if isinstance(y, (pd.DataFrame, pd.Series)):
        if hasattr(raw_predictions, 'values'):
            diff = y.subtract(raw_predictions.values)
        else:
            diff = y.subtract(raw_predictions)
    else:
        diff = y - raw_predictions
    return diff


class LeastSquaresError(sklearn.ensemble._gb_losses.LeastSquaresError):
    """Least squares loss function.

    The object modifies the original Scikit-learn LeastSquaresError such that
    the average loss and pseudo-residual computations support heterogeneous
    usage of Numpy and pandas data representations. Moreover, the modification
    supports both single-target and multi-target data.

    References
    ----------
    - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
      "Boosting on the shoulders of giants in quantum device calibration",
      arXiv preprint arXiv:2005.06194 (2020).
    
    - Jerome Friedman. "Greedy function approximation: A gradient boosting machine,"
      Annals of Statistics, 29(5):1189–1232 (2001).
      """

    def __call__(self, y: pandas_or_numpy, raw_predictions: pandas_or_numpy,
                 sample_weight=None) -> pandas_or_numpy:
        """Computes the average loss.

        Parameters
        ----------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each target. If the weight is a float, then
            every target will have the same weight.

        Returns
        -------
        mse : DataFrame, Series, or ndarray

        Examples
        --------
        >>> from sklearn.datasets import load_linnerud
        >>> from physlearn import LeastSquaresError
        >>> X, y = load_linnerud(return_X_y=True)
        >>> ls = LeastSquaresError()
        >>> ls(y=y, raw_predictions=X)
        16048.6
        """

        if sample_weight is None:
            return sklearn.metrics.mean_squared_error(y_true=y, y_pred=raw_predictions)
        else:
            return sklearn.metrics.mean_squared_error(y_true=y, y_pred=raw_predictions,
                                                      sample_weight=sample_weight)

    def negative_gradient(self, y: pandas_or_numpy,
                          raw_predictions: pandas_or_numpy) -> pandas_or_numpy:
        """Computes the pseudo-residuals.

        Parameters
        ----------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).

        Returns
        -------
        residual : DataFrame, Series, or ndarray

        Examples
        --------
        >>> import pandas as pd
        >>> from sklearn.datasets import load_linnerud
        >>> from physlearn import LeastSquaresError
        >>> X, y = load_linnerud(return_X_y=True)
        >>> ls = LeastSquaresError()
        >>> ls.negative_gradient(y=pd.DataFrame(y), raw_predictions=X).iloc[:2]
               0      1     2
        0  186.0 -126.0 -10.0
        1  187.0  -73.0  -8.0
        """

        return _difference(y=y, raw_predictions=raw_predictions)


class LeastAbsoluteError(sklearn.ensemble._gb_losses.LeastAbsoluteError):
    """Absolute error loss function.

    The object modifies the original Scikit-learn LeastAbsoluteError such that
    the average loss and pseudo-residual computations support heterogeneous
    usage of Numpy and pandas data representations. Moreover, the modification
    supports both single-target and multi-target data.

    References
    ----------
    - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
      "Boosting on the shoulders of giants in quantum device calibration",
      arXiv preprint arXiv:2005.06194 (2020).
    
    - Jerome Friedman. "Greedy function approximation: A gradient boosting machine,"
      Annals of Statistics, 29(5):1189–1232 (2001).
      """

    def __call__(self, y: pandas_or_numpy, raw_predictions: pandas_or_numpy,
                 sample_weight=None) -> pandas_or_numpy:
        """Computes the average loss.

        Parameters
        ----------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each target. If the weight is a float, then
            every target will have the same weight.

        Returns
        -------
        mae : DataFrame, Series, or ndarray

        Examples
        --------
        >>> from sklearn.datasets import load_linnerud
        >>> from physlearn import LeastAbsoluteError
        >>> X, y = load_linnerud(return_X_y=True)
        >>> lad = LeastAbsoluteError()
        >>> lad(y=y, raw_predictions=X)
        104.23333333333333
        """

        if sample_weight is None:
            return sklearn.metrics.mean_absolute_error(y_true=y, y_pred=raw_predictions)
        else:
            return sklearn.metrics.mean_absolute_error(y_true=y, y_pred=raw_predictions,
                                                       sample_weight=sample_weight)

    def negative_gradient(self, y: pandas_or_numpy,
                          raw_predictions: pandas_or_numpy) -> pandas_or_numpy:
        """Computes the pseudo-residuals.

        Parameters
        ----------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).

        Returns
        -------
        residual : DataFrame, Series, or ndarray

        Examples
        --------
        >>> import pandas as pd
        >>> from sklearn.datasets import load_linnerud
        >>> from physlearn import LeastAbsoluteError
        >>> X, y = load_linnerud(return_X_y=True)
        >>> lad = LeastAbsoluteError()
        >>> lad.negative_gradient(y=pd.DataFrame(y), raw_predictions=X).iloc[:2]
             0    1    2
        0  1.0 -1.0 -1.0
        1  1.0 -1.0 -1.0
        """

        if isinstance(y, (pd.DataFrame, pd.Series)):
            return _difference(y=y, raw_predictions=raw_predictions).apply(np.sign)
        else:
            return np.sign(_difference(y=y, raw_predictions=raw_predictions))


class HuberLossFunction(sklearn.ensemble._gb_losses.HuberLossFunction):
    """Huber loss function.

    The object modifies the original Scikit-learn HuberLossFunction such that
    the average loss and pseudo-residual computations support heterogeneous
    usage of Numpy and pandas data representations. Moreover, the modification
    supports both single-target and multi-target data.

    References
    ----------
    - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
      "Boosting on the shoulders of giants in quantum device calibration",
      arXiv preprint arXiv:2005.06194 (2020).
    
    - Jerome Friedman. "Greedy function approximation: A gradient boosting machine,"
      Annals of Statistics, 29(5):1189–1232 (2001).
      """

    def _delta(self, difference: pandas_or_numpy,
               sample_weight=None) -> np.float64:
        """Computes the delta threshold.

        This threshold determines whether to use the squared error or
        the absolute error loss function.

        Parameters
        ----------
        difference : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The difference between the single-target(s) and the raw prediction(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each target. If the weight is a float, then
            every target will have the same weight.

        Returns
        -------
        delta : np.float64
        """

        if hasattr(difference, 'abs'):
            abs_diff = difference.abs()
        else:
            abs_diff = np.abs(difference)
        if sample_weight is None:
            delta = np.percentile(abs_diff, self.alpha * 100)
        else:
            delta = sklearn.utils.stats._weighted_percentile(abs_diff, sample_weight,
                                                             self.alpha * 100)
        return delta


    def __call__(self, y: pandas_or_numpy, raw_predictions: pandas_or_numpy,
                 sample_weight=None) -> pandas_or_numpy: 
        """Computes the average loss.

        Parameters
        ----------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each target. If the weight is a float, then
            every target will have the same weight.

        Returns
        -------
        huber : DataFrame, Series, or ndarray

        Examples
        --------
        >>> from sklearn.datasets import load_linnerud
        >>> from physlearn import HuberLossFunction
        >>> X, y = load_linnerud(return_X_y=True)
        >>> huber = HuberLossFunction()
        >>> huber(y=y, raw_predictions=X)
        7989.893
        """

        diff = _difference(y=y, raw_predictions=raw_predictions)
        delta = self._delta(difference=diff, sample_weight=sample_weight)

        if hasattr(diff, 'abs'):
            mask = diff.abs() > delta
            if sample_weight is None:
                diff[mask] = delta * (diff[mask].abs() - delta/2)
                diff[~mask] = 0.5 * diff[~mask].pow(other=2)
                return diff.mean()
            else:
                diff[mask] = delta * (diff[mask].abs().multiply(sample_weight[mask]) - delta/2)
                diff[~mask] = 0.5 * diff[~mask].pow(other=2).multiply(sample_weight[~mask])
        else:
            mask = np.abs(diff) > delta
            if sample_weight is None:
                diff[mask] = delta * (np.abs(diff[mask]) - delta/2)
                diff[~mask] = 0.5 * diff[~mask]**2
                return diff.mean()
            else:
                diff[mask] = delta * (sample_weight[mask]@np.abs(diff[mask]) - delta/2)
                diff[~mask] = 0.5 * sample_weight[~mask]@diff[~mask]**2
        
        return diff.sum() / sample_weight.sum()

    def negative_gradient(self, y: pandas_or_numpy, raw_predictions: pandas_or_numpy,
                          sample_weight=None) -> pandas_or_numpy:
        """Computes the pseudo-residuals.

        Parameters
        ----------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).

        Returns
        -------
        residual : DataFrame, Series, or ndarray

        Examples
        --------
        >>> import pandas as pd
        >>> from sklearn.datasets import load_linnerud
        >>> from physlearn import HuberLossFunction
        >>> X, y = load_linnerud(return_X_y=True)
        >>> huber = HuberLossFunction()
        >>> huber.negative_gradient(y=pd.DataFrame(y), raw_prediction=X).iloc[:2]
               0      1     2
        0  186.0 -126.0 -10.0
        1  187.0  -73.0  -8.0
        """

        diff = _difference(y=y, raw_predictions=raw_predictions)
        delta = self._delta(difference=diff, sample_weight=sample_weight)

        if hasattr(diff, 'abs'):
            mask = diff.abs() > delta
            diff[mask] = delta * diff[mask].apply(np.sign)
        else:
            mask = np.abs(diff) > delta
            diff[mask] = delta * np.sign(diff[mask])
        
        return diff


class QuantileLossFunction(sklearn.ensemble._gb_losses.QuantileLossFunction):
    """Quantile loss function.

    The object modifies the original Scikit-learn QuantileLossFunction such that
    the average loss and pseudo-residual computations support heterogeneous
    usage of Numpy and pandas data representations. Moreover, the modification
    supports both single-target and multi-target data.

    References
    ----------
    - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
      "Boosting on the shoulders of giants in quantum device calibration",
      arXiv preprint arXiv:2005.06194 (2020).
    
    - Jerome Friedman. "Greedy function approximation: A gradient boosting machine,"
      Annals of Statistics, 29(5):1189–1232 (2001).
      """

    def __call__(self, y: pandas_or_numpy, raw_predictions: pandas_or_numpy,
                 sample_weight=None) -> pandas_or_numpy:
        """Computes the average loss.

        Parameters
        ----------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each target. If the weight is a float, then
            every target will have the same weight.

        Returns
        -------
        quantile : DataFrame, Series, or ndarray

        Examples
        --------
        >>> from sklearn.datasets import load_linnerud
        >>> from physlearn import QuantileLossFunction
        >>> X, y = load_linnerud(return_X_y=True)
        >>> quantile = QuantileLossFunction()
        >>> quantile(y=y, raw_predictions=X)
        174.27
        """

        diff = _difference(y=y, raw_predictions=raw_predictions)
        mask = y > raw_predictions
        if sample_weight is None:
            return (self.alpha * diff[mask].sum() - (1-self.alpha) * diff[~mask].sum()) / y.shape[0]
        else:
            if hasattr(diff, 'multiply'):
                with_mask = diff[mask].multiply(sample_weight[mask]).sum()
                without_mask = diff[~mask].multiply(sample_weight[~mask]).sum()
            else:
                with_mask = (sample_weight[mask]@diff[mask]).sum()
                without_mask = (sample_weight[~mask]@diff[~mask]).sum()
            return (self.alpha*with_mask - (1-self.alpha)*without_mask) / sample_weight.sum()


    def negative_gradient(self, y: pandas_or_numpy,
                          raw_predictions: pandas_or_numpy) -> pandas_or_numpy:
        """Computes the pseudo-residuals.

        Parameters
        ----------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).

        Returns
        -------
        residual : DataFrame, Series, or ndarray

        Examples
        --------
        >>> import pandas as pd
        >>> from sklearn.datasets import load_linnerud
        >>> from physlearn import QuantileLossFunction
        >>> X, y = load_linnerud(return_X_y=True)
        >>> quantile = QuantileLossFunction()
        >>> quantile.negative_gradient(y=pd.DataFrame(y), raw_predictions=X).iloc[:2]
             0    1    2
        0  0.9 -0.1 -0.1
        1  0.9 -0.1 -0.1
        """

        if hasattr(raw_predictions, 'values'):
            mask = y.gt(raw_predictions.values)
        else:
            mask = y.gt(raw_predictions)
        return self.alpha*mask - (1-self.alpha)*~mask


LOSS_FUNCTIONS = dict(ls=LeastSquaresError,
                      lad=LeastAbsoluteError,
                      huber=HuberLossFunction,
                      quantile=QuantileLossFunction)
