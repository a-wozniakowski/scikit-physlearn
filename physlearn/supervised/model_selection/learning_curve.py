"""
The :mod:`physlearn.supervised.model_selection.learning_curve` module provides
utilities for plotting learning curves. It includes the
:class:`physlearn.LearningCurve` class and the
:func:`physlearn.plot_learning_curve` function.
"""

# Author: Alex Wozniakowski
# License: MIT

from __future__ import annotations

import joblib
import matplotlib
import re
import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn.base
import sklearn.metrics
import sklearn.model_selection._split
import sklearn.model_selection._validation
import sklearn.utils.multiclass
import sklearn.utils.validation

from dataclasses import dataclass, field

from physlearn.supervised.regression import BaseRegressor
from physlearn.supervised.utils._data_checks import _n_samples, _validate_data

DataFrame_or_Series = typing.Union[pd.DataFrame, pd.Series]


@dataclass
class LearningCurve(BaseRegressor):
    """Learning curve object that supports base boosting.

    The object retains the original functionality provided by the
    Scikit-learn learning curve utility, which performs a
    cross-validation procedure with varying training sizes. It extends
    the utility to support augmented cross-validation procedures, which
    score an incumbent model and a candidate model on the same withheld
    folds.

    Parameters
    ----------
    regressor_choice : str, optional (default='ridge')
        Specifies the case-insensitive regressor choice.

    cv : int, cross-validation generator, an iterable, or None, optional (default=5)
        Determines the cross-validation strategy if the regressor choice is stacking,
        if the task is multi-target regression and the single-targets are chained,
        and as the default in the k-fold cross-validation methods.

    random_state : int, RandomState instance, or None, optional (default=0)
        Determines the random number generation in the regressor choice
        :class:`mlxtend.regressor.StackingCVRegressor` and in the modified
        pipeline construction.

    verbose : int, optional (default=0)
        Determines verbosity in either regressor choice:
        :class:`mlxtend.regressor.StackingRegressor` and
        :class:`mlxtend.regressor.StackingCVRegressor`, in the modified
        pipeline construction, and in the k-fold cross-validation methods.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel if the regressor choice is stacking
        or voting, in the modified pipeline construction, and in the k-fold
        cross-validation methods.

    score_multioutput : str, optional (default='raw_values')
        Defines aggregating of multiple output values in the score method,
        wherein the string must be either ``'raw_values'``, ``'uniform_average'``, or
        ``'variance_weighted'``.

    scoring : str, callable, list/tuple, or dict, optional (default='neg_mean_absolute_error')
        Determines scoring in the k-fold cross-validation methods.

    return_train_score : bool, optional (default=True)
        Determines whether to return the training scores from the k-fold
        cross-validation methods.

    pipeline_transform : str, list, tuple, or None, optional (default='quantilenormal')
        Choice of transform(s) used in the modified pipeline construction.
        If the specified choice is a string, then it must be a default option,
        where ``'standardscaler'``, ``'boxcox'``, ``'yeojohnson'``, ``'quantileuniform'``,
        and ``'quantilenormal'`` denote :class:`sklearn.preprocessing.StandardScaler`,
        :class:`sklearn.preprocessing.PowerTransformer` with ``method='box-cox'``
        or ``method='yeo-johnson'``, and :class:`sklearn.preprocessing.QuantileTransformer`
        with ``output_distribution='uniform'`` or ``output_distribution='normal'``,
        respectively.

    pipeline_memory : str or object with the joblib.Memory interface, optional (default=None)
        Enables fitted transform caching in the modified pipeline construction.

    params : dict, list, or None, optional (default=None)
        The choice of (hyper)parameters for the regressor choice.
        If None, then the default (hyper)parameters are utilized.

    target_index : int, or None, optional (default=None)
        Specifies the single-target regression subtask in the multi-target
        regression task.

    chain_order : list or None
        Determines the target order in :class:`sklearn.multioutput.RegressorChain`
        during the modified pipeline construction.

    stacking_options : dict or None, optional (default=None)
        A dictionary of stacking options, whereby ``layers``
        must be specified:

        layers :obj:`dict`
            A dictionary of stacking layer(s).

        shuffle :obj:`bool` or None, (default=True)
            Determines whether to shuffle the training data in
            :class:`mlxtend.regressor.StackingCVRegressor`.

        refit :obj:`bool` or None, (default=True)
            Determines whether to clone and refit the regressors in
            :class:`mlxtend.regressor.StackingCVRegressor`.

        passthrough :obj:`bool` or None, (default=True)
            Determines whether to concatenate the original features with
            the first stacking layer predictions in
            :class:`sklearn.ensemble.StackingRegressor`,
            :class:`mlxtend.regressor.StackingRegressor`, or
            :class:`mlxtend.regressor.StackingCVRegressor`.

        meta_features : :obj:`bool` or None, (default=True)
            Determines whether to make the concatenated features
            accessible through the attribute ``train_meta_features_``
            in :class:`mlxtend.regressor.StackingRegressor` and
            :class:`mlxtend.regressor.StackingCVRegressor`.

        voting_weights : :obj:`ndarray` of shape (n_regressors,) or None, (default=None)
            Sequence of weights for :class:`sklearn.ensemble.VotingRegressor`.

    base_boosting_options : dict or None, optional (default=None)
        A dictionary of base boosting options used in the modified pipeline construction,
        wherein the following options must be specified:

        n_estimators :obj:`int`
            The number of basis functions in the noise term of the additive expansion.
            Note that this option may also be specified as ``n_regressors``.

        boosting_loss :obj:`str` 
            The loss function utilized in the pseudo-residual computation, where 'ls'
            denotes the squared error loss function, 'lad' denotes the absolute error
            loss function, 'huber' denotes the Huber loss function, and 'quantile'
            denotes the quantile loss function.

        line_search_options :obj:`dict` 
            init_guess :obj:`int`, :obj:`float`, or :obj:`ndarray`
                The initial guess for the expansion coefficient.

            opt_method :obj:`str`
                Choice of optimization method. If ``'minimize'``, then
                :class:`scipy.optimize.minimize`, else if ``'basinhopping'``,
                then :class:`scipy.optimize.basinhopping`.

            method :obj:`str` or None
                The type of solver utilized in the optimization method.

            tol :obj:`float` or None
                The epsilon tolerance for terminating the optimization method.

            options :obj:`dict` or None
                A dictionary of solver options.

            niter :obj:`int` or None
                The number of iterations in basin-hopping.

            T :obj:`float` or None
                The temperature paramter utilized in basin-hopping,
                which determines the accept or reject criterion.

            loss :obj:`str`
                The loss function utilized in the line search computation, where 'ls'
                denotes the squared error loss function, 'lad' denotes the absolute error
                loss function, 'huber' denotes the Huber loss function, and 'quantile'
                denotes the quantile loss function.

            regularization :obj:`int` or :obj:`float`
                The regularization strength in the line search computation.

    See Also
    --------
    :class:`physlearn.supervised.regression.BaseRegressor` : Base class for regressor amalgamation."""

    pipeline_transform: str = field(default='quantilenormal')

    def __post_init__(self):
        self._validate_regressor_options()
        self._get_regressor()

    def _modified_learning_curve(self, X: DataFrame_or_Series, y: DataFrame_or_Series,
                                 train_sizes=np.linspace(0.1, 1.0, 5),
                                 return_train_score=True, return_times=False,
                                 return_estimator=False, error_score=np.nan,
                                 return_incumbent_score=False, cv=None,
                                 fit_params=None):
        """Performs an (augmented) cross-validation procedure with varying training sizes.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        train_sizes : array-like of shape (n_ticks,), optional (default=np.linspace(0.1, 1.0, 5))
            The array elements determine the amount of traning examples used in each
            cross-validation procedure.

        return_train_score : bool, optional (default=True)
            Determines whether to return the candidate's training fold scores.

        return_times : bool, optional (default=False)
            Determines whether to return the candidate's fit and score times.

        return_estimator : bool, optional (default=False)
            Determines whether to return the induced estimator.

        error_score : 'raise' or numeric, optional (default=np.nan)
            The assigned value if an error occurs while inducing an estimator.
            If set to 'raise', then the specific error is raised. Else if set
            to a numeric value, then FitFailedWarning is raised.

        return_incumbent_score : bool, optional (default=False)
            Determines whether to score the incumbent on the withheld folds,
            whereby the incumbent is assumed to be an example in the design
            matrix.

        cv : int, cross-validation generator, an iterable, or None, optional (default=None)
            Determines the cross-validation strategy. If None, then the default
            is 5-fold cross-validation.

        fit_params : dict, optional (default=None)
            (Hyper)parameters to pass to the regressor's fit method.
        """

        X, y = self._validate_data(X=X, y=y)

        # Automates single-target slicing.
        y = self._check_target_index(y=y)

        X, y, groups = sklearn.utils.validation.indexable(X, y, None)

        if cv is None:
            cv = self.cv

        if not hasattr(self, 'pipe'):
            self.get_pipeline(y=y,
                              n_quantiles=self._estimate_fold_size(y=y,
                                                                   cv=cv))

        check_classifier = sklearn.base.is_classifier(self.pipe)
        cv = sklearn.model_selection._split.check_cv(cv=self.cv, y=y,
                                                     classifier=check_classifier)
        cv_iter = list(cv.split(X, y, groups))

        scorer = sklearn.metrics.check_scoring(estimator=self.pipe,
                                               scoring=self.scoring)

        # Modify n_max_training_samples
        n_max_training_samples = X.shape[0]
        train_sizes_abs = sklearn.model_selection._validation._translate_train_sizes(
            train_sizes=train_sizes, n_max_training_samples=n_max_training_samples
        )

        parallel = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                                   pre_dispatch='2*n_jobs')

        n_unique_ticks = train_sizes_abs.shape[0]

        train_test_proportions = []
        for train, test in cv_iter:
            for n_train_samples in train_sizes_abs:
                train_test_proportions.append((train[:n_train_samples], test))

        results = parallel(
            joblib.delayed(sklearn.model_selection._validation._fit_and_score)(
                estimator=sklearn.base.clone(self.pipe), X=X, y=y, scorer=scorer,
                train=train, test=test, verbose=self.verbose, parameters=None,
                fit_params=fit_params, return_train_score=return_train_score,
                return_parameters=False, return_n_test_samples=False,
                return_times=return_times, return_estimator=return_estimator,
                error_score=error_score)
            for train, test in train_test_proportions)

        results = sklearn.model_selection._validation._aggregate_score_dicts(results)
        train_scores = results["train_scores"].reshape(-1, n_unique_ticks).T
        test_scores = results["test_scores"].reshape(-1, n_unique_ticks).T

        if re.match('neg', self.scoring):
            train_scores *= -1
            test_scores *= -1

        out = [train_scores, test_scores]

        if return_incumbent_score:
            if self.target_index is not None:
                y_pred = X.iloc[:, self.target_index]
            else:
                y_pred = X

            if self.scoring == 'neg_mean_absolute_error':
                scoring = 'mae'
            elif self.scoring == 'neg_mean_squared_error':
                scoring = 'mse'

            # Avoids recomputing the incumbent score for each
            # training size, as the score does not depend upon
            # the training size.
            incumbent_score = parallel(
                joblib.delayed(self.score)(
                    y_true=y.loc[pair[1]], y_pred=y_pred.loc[pair[1]],
                    scoring=scoring, multioutput='raw_values')
                for index, pair in enumerate(train_test_proportions)
                if index % len(train_sizes) == 0)
            
            incumbent_score = np.array(incumbent_score).transpose()

            # Check if the incumbent won the inbuilt model selection
            # step in the augmented version of base boosting. If the
            # incumbent won, then we replace the cross-validation score
            # with the incumbent score, as base boosting would select
            # the incumbent.
            for index, row_score in enumerate(out[1]):
                if np.mean(a=row_score) > np.mean(a=incumbent_score):
                    out[1][index] = incumbent_score

            ret = train_sizes_abs, out[0], out[1], incumbent_score
        else:
            ret = train_sizes_abs, out[0], out[1]

        if return_times:
            ret = ret + (out[2], out[3])

        return ret


def plot_learning_curve(regressor_choice, title, X, y, verbose=0, cv=5,
                        train_sizes=np.linspace(0.2, 1.0, 5), alpha=0.1,
                        train_color='b', cv_color='orange', y_ticks_step=0.15,
                        fill_std=False, legend_loc='best', save_plot=False,
                        scoring='neg_mean_absolute_error', 
                        pipeline_transform='quantilenormal',
                        path=None, pipeline_memory=None, params=None,
                        target_index=None, chain_order=None, ylabel=None,
                        stacking_options=None, base_boosting_options=None,
                        return_incumbent_score=False):

    lcurve = LearningCurve(regressor_choice=regressor_choice,
                           verbose=verbose, cv=cv,
                           pipeline_transform=pipeline_transform,
                           pipeline_memory=pipeline_memory,
                           params=params, scoring=scoring,
                           target_index=target_index,
                           chain_order=chain_order,
                           stacking_options=stacking_options,
                           base_boosting_options=base_boosting_options)

    ret = lcurve._modified_learning_curve(X=X, y=y,
                                          train_sizes=train_sizes,
                                          return_incumbent_score=return_incumbent_score)
    if return_incumbent_score:
        train_sizes, train_score, cv_score, incumbent_score = ret
    else:
        train_sizes, train_score, cv_score = ret

    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Lucida Console']
    plt.figure()
    plt.title(title)
    plt.xlabel('Number of ordered pairs')

    if ylabel is not None:
        assert isinstance(ylabel, str)
        plt.ylabel(ylabel)
    else:
        if lcurve.scoring == 'neg_mean_absolute_error':
            plt.ylabel('MAE')
        elif lcurve.scoring == 'neg_mean_squared_error':
            plt.ylabel('MSE')
        else:
            plt.ylabel('Empirical error')

    train_score_mean = np.mean(a=train_score, axis=1)
    train_score_std = np.std(a=train_score, axis=1)
    cv_score_mean = np.mean(a=cv_score, axis=1)
    cv_score_std = np.std(a=cv_score, axis=1)
    
    if return_incumbent_score:
        incumbent_score_mean = np.repeat(a=np.mean(incumbent_score, axis=1),
                                         repeats=len(train_sizes))
        incumbent_score_std = np.repeat(a=np.std(incumbent_score, axis=1),
                                        repeats=len(train_sizes))
        _score_min = np.min([train_score.min(), cv_score.min(),
                             incumbent_score_mean.min()])
        global_score_min = np.around(a=_score_min, decimals=2)
        _score_max = np.max([train_score.max(), cv_score.max(),
                             incumbent_score_mean.max()])
        global_score_max = np.around(a=_score_max, decimals=2)
    else:
        global_score_min = np.around(a=np.min([train_score.min(), cv_score.min()]),
                                     decimals=2)
        global_score_max = np.around(a=np.max([train_score.max(), cv_score.max()]),
                                     decimals=2)

    plt.grid()

    if fill_std:
        plt.fill_between(train_sizes, train_score_mean - train_score_std,
                         train_score_mean + train_score_std,
                         alpha=alpha, color=train_color)
        plt.fill_between(train_sizes, cv_score_mean - cv_score_std,
                         cv_score_mean + cv_score_std,
                         alpha=alpha, color=cv_color)

    plt.plot(train_sizes, train_score_mean, 'o-', color=train_color,
             label='Training error')
    plt.plot(train_sizes, cv_score_mean, 'o-', color=cv_color,
             label='Cross-validation error')
    
    if return_incumbent_score:
        plt.plot(train_sizes, incumbent_score_mean, 'o-', color='r',
                 label='Incumbent error')
        if fill_std:
            plt.fill_between(train_sizes, incumbent_score_mean - incumbent_score_std,
                             incumbent_score_mean + incumbent_score_std,
                             alpha=alpha, color='r')

    plt.yticks(np.arange(global_score_min, global_score_max, step=y_ticks_step))
    plt.legend(loc=legend_loc)

    if save_plot:
        assert path is not None and isinstance(path, str)
        plt.savefig(path)
    else:
        plt.show(block=True)
