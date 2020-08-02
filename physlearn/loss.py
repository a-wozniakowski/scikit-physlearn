"""
The :mod:`physlearn.loss` module enables computation of the average loss
or the negative gradient in either the single-target or the multi-target
regression setting, whereby data can be represented heterogeneously with
Numpy or Pandas. The choice of loss functions are: squared error,
absolute error, Huber, or quantile.
"""

# Author: Alex Wozniakowski
# License: MIT

import numpy as np
import pandas as pd

import sklearn.ensemble
import sklearn.metrics


def _difference(y, raw_predictions):
    """
    Subtract the raw predictions from the single-target(s)
    regardless of the Numpy or Pandas data representation(s).

    Parameters
    ----------

    y : DataFrame, Series, or ndarray
        The target matrix, where each row corresponds to an example and the
        column(s) correspond to the single-target(s).

    raw_predictions : DataFrame, Series, or ndarray
        The estimate matrix, where each row corresponds to an example and the
        column(s) correspond to the prediction(s) for the single-target(s).
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
    """Least squares loss function, which is used in base boosting."""

    def __call__(self, y, raw_predictions, sample_weight=None):
        """
        Compute the average loss.

        Parameters
        ----------

        y : DataFrame, Series, or ndarray
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : DataFrame, Series, or ndarray
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each example. If the weight is a float, then
            every example will have the same weight.
        """

        if sample_weight is None:
            return sklearn.metrics.mean_squared_error(y_true=y, y_pred=raw_predictions)
        else:
            return sklearn.metrics.mean_squared_error(y_true=y, y_pred=raw_predictions,
                                                      sample_weight=sample_weight)

    def negative_gradient(self, y, raw_predictions):
        """
        Compute the pseduoresiduals.

        Parameters
        ----------

        y : DataFrame, Series, or ndarray
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : DataFrame, Series, or ndarray
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).
        """

        return _difference(y=y, raw_predictions=raw_predictions)


class LeastAbsoluteError(sklearn.ensemble._gb_losses.LeastAbsoluteError):
    """Absolute error loss function, which is used in base boosting."""

    def __call__(self, y, raw_predictions, sample_weight=None):
        """
        Compute the average loss.

        Parameters
        ----------

        y : DataFrame, Series, or ndarray
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : DataFrame, Series, or ndarray
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each example. If the weight is a float, then
            every example will have the same weight.
        """

        if sample_weight is None:
            return sklearn.metrics.mean_absolute_error(y_true=y, y_pred=raw_predictions)
        else:
            return sklearn.metrics.mean_absolute_error(y_true=y, y_pred=raw_predictions,
                                                       sample_weight=sample_weight)

    def negative_gradient(self, y, raw_predictions):
        """
        Compute the pseduoresiduals.

        Parameters
        ----------

        y : DataFrame, Series, or ndarray
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : DataFrame, Series, or ndarray
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).
        """

        if isinstance(y, (pd.DataFrame, pd.Series)):
            return _difference(y=y, raw_predictions=raw_predictions).apply(np.sign)
        else:
            return np.sign(_difference(y=y, raw_predictions=raw_predictions))


class HuberLossFunction(sklearn.ensemble._gb_losses.HuberLossFunction):
    """Huber loss function, which is used in base boosting."""

    def _delta(self, difference, sample_weight=None):
        """
        Compute the delta threshold, which determines whether to
        use the squared error or the absolute error loss function.

        difference : DataFrame, Series, or ndarray
            The difference between the single-target(s) and the raw prediction(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each example. If the weight is a float, then
            every example will have the same weight.

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


    def __call__(self, y, raw_predictions, sample_weight=None):
        """
        Compute the average loss.

        Parameters
        ----------

        y : DataFrame, Series, or ndarray
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : DataFrame, Series, or ndarray
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each example. If the weight is a float, then
            every example will have the same weight.
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

    def negative_gradient(self, y, raw_predictions, sample_weight=None):
        """
        Compute the pseduoresiduals.

        Parameters
        ----------

        y : DataFrame, Series, or ndarray
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : DataFrame, Series, or ndarray
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).
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
    """Quantile loss function, which is used in base boosting"""

    def __call__(self, y, raw_predictions, sample_weight=None):
        """
        Compute the average loss.

        Parameters
        ----------

        y : DataFrame, Series, or ndarray
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : DataFrame, Series, or ndarray
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each example. If the weight is a float, then
            every example will have the same weight.
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


    def negative_gradient(self, y, raw_predictions):
        """
        Compute the pseduoresiduals.

        Parameters
        ----------

        y : DataFrame, Series, or ndarray
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        raw_predictions : DataFrame, Series, or ndarray
            The estimate matrix, where each row corresponds to an example and the
            column(s) correspond to the prediction(s) for the single-target(s).
        """

        mask = y.gt(raw_predictions.values)
        return self.alpha*mask - (1-self.alpha)*~mask


LOSS_FUNCTIONS = dict(ls=LeastSquaresError,
                      lad=LeastAbsoluteError,
                      huber=HuberLossFunction,
                      quantile=QuantileLossFunction)
