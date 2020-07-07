"""
Loss functions built for multi-target regression.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import numpy as np

import sklearn.ensemble
import sklearn.metrics


class LeastSquaresError(sklearn.ensemble._gb_losses.LeastSquaresError):
    """Least squares loss function for boosting."""

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the average loss."""

        if sample_weight is None:
            return sklearn.metrics.mean_squared_error(y_true=y, y_pred=raw_predictions)
        else:
            return sklearn.metrics.mean_squared_error(y_true=y, y_pred=raw_predictions,
                                                      sample_weight=sample_weight)

    def negative_gradient(self, y, raw_predictions):
        """Compute the pseduoresiduals."""

        return y.subtract(raw_predictions.values)


class LeastAbsoluteError(sklearn.ensemble._gb_losses.LeastAbsoluteError):
    """Absolute error loss function for boosting."""

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the average loss."""

        if sample_weight is None:
            return sklearn.metrics.mean_absolute_error(y_true=y, y_pred=raw_predictions)
        else:
            return sklearn.metrics.mean_absolute_error(y_true=y, y_pred=raw_predictions,
                                                       sample_weight=sample_weight)

    def negative_gradient(self, y, raw_predictions):
        """Compute the pseduoresiduals."""

        return y.subtract(raw_predictions.values).apply(np.sign)


class HuberLossFunction(sklearn.ensemble._gb_losses.HuberLossFunction):
    """Huber loss function for boosting."""

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the average loss."""

        diff = y.subtract(raw_predictions.values)
        if sample_weight is None:
            delta = np.percentile(diff.abs(), self.alpha * 100)
        else:
            delta = sklearn.utils.stats._weighted_percentile(diff.abs(), sample_weight,
                                                             self.alpha * 100)

        mask = diff.abs() > delta
        if sample_weight is None:
            diff[mask] = delta * (diff[mask].abs() - delta/2)
            diff[~mask] = 0.5 * diff[~mask].pow(other=2)
            return diff.mean()
        else:
            diff[mask] = delta * (diff[mask].abs().multiply(sample_weight[mask]) - delta/2)
            diff[~mask] = 0.5 * diff[~mask].pow(other=2).multiply(sample_weight[~mask])
            return diff.sum() / sample_weight.sum()

    def negative_gradient(self, y, raw_predictions, sample_weight=None):
        """Compute the pseduoresiduals."""

        diff = y.subtract(raw_predictions.values)
        if sample_weight is None:
            delta = np.percentile(diff.abs(), self.alpha * 100)
        else:
            delta = sklearn.utils.stats._weighted_percentile(diff.abs(), sample_weight,
                                                             self.alpha * 100)
        mask = diff.abs() > delta
        diff[mask] = delta * diff[mask].apply(np.sign)
        return diff


class QuantileLossFunction(sklearn.ensemble._gb_losses.QuantileLossFunction):

    def __call__(self, y, raw_predictions, sample_weight=None):
        """Compute the average loss."""

        diff = y.subtract(raw_predictions.values)
        mask = y.gt(raw_predictions.values)
        if sample_weight is None:
            return (self.alpha * diff[mask].sum() - (1-self.alpha) * diff[~mask].sum()) / y.shape[0]
        else:
            return ((self.alpha * diff[mask].multiply(sample_weight[mask]).sum() -
                    (1-self.alpha) * diff[~mask].multiply(sample_weight[~mask]).sum())) / sample_weight.sum()


    def negative_gradient(self, y, raw_predictions):
        """Compute the pseduoresiduals."""

        mask = y.gt(raw_predictions.values)
        return (self.alpha*mask) - ((1-self.alpha)*~mask)


LOSS_FUNCTIONS = {'ls': LeastSquaresError,
                  'lad': LeastAbsoluteError,
                  'huber': HuberLossFunction,
                  'quantile': QuantileLossFunction}
