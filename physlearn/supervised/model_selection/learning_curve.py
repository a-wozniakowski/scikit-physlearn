"""
Utility for plotting (augmented) learning curves.
"""

# Author: Alex Wozniakowski
# License: MIT

import joblib
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn.base
import sklearn.metrics
import sklearn.model_selection._split
import sklearn.model_selection._validation
import sklearn.utils.multiclass
import sklearn.utils.validation

from physlearn.supervised.regression import BaseRegressor
from physlearn.supervised.utils._data_checks import _n_samples, _validate_data


class LearningCurve(BaseRegressor):
    """(Augmented) learning curve for base boosting."""

    def __init__(self, regressor_choice='ridge', cv=5, verbose=1,
                 n_jobs=-1, scoring='neg_mean_absolute_error',
                 return_train_score=True, pipeline_transform='quantilenormal',
                 pipeline_memory=None, params=None, target_index=None,
                 chain_order=None, stacking_options=None,
                 base_boosting_options=None):

        super().__init__(regressor_choice=regressor_choice,
                         cv=cv,
                         verbose=verbose,
                         n_jobs=n_jobs,
                         scoring=scoring,
                         return_train_score=return_train_score,
                         pipeline_transform=pipeline_transform,
                         pipeline_memory=pipeline_memory,
                         params=params,
                         target_index=target_index,
                         chain_order=chain_order,
                         stacking_options=stacking_options,
                         base_boosting_options=base_boosting_options)

    def _modified_learning_curve(self, X, y, train_sizes=np.linspace(0.1, 1.0, 5),
                                 return_train_score=True, return_times=False,
                                 return_estimator=False, error_score=np.nan,
                                 return_incumbent_score=False):

        # Automates single-target slicing
        y = super()._check_target_index(y=y)

        X, y = _validate_data(X=X, y=y)
        X, y, groups = sklearn.utils.validation.indexable(X, y, None)

        if not hasattr(self, 'pipe'):
            n_samples = _n_samples(y)
            fold_size =  np.full(shape=n_samples, fill_value=n_samples // self.cv,
                                 dtype=np.int)
            estimate_fold_size = n_samples - (np.max(fold_size) + 1)
            self.get_pipeline(y=y, n_quantiles=estimate_fold_size)

        cv = sklearn.model_selection._split.check_cv(cv=self.cv, y=y,
                                                     classifier=sklearn.base.is_classifier(self.pipe))
        cv_iter = list(cv.split(X, y, groups))

        scorer = sklearn.metrics.check_scoring(estimator=self.pipe, scoring=self.scoring)

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

        out = parallel(
            joblib.delayed(sklearn.model_selection._validation._fit_and_score)(
                estimator=sklearn.base.clone(self.pipe), X=X, y=y, scorer=scorer,
                train=train, test=test, verbose=self.verbose, parameters=None,
                fit_params=None, return_train_score=return_train_score,
                return_parameters=False, return_n_test_samples=False,
                return_times=return_times, return_estimator=return_estimator,
                error_score=error_score)
            for train, test in train_test_proportions)
        out = np.array(out)
        n_cv_folds = out.shape[0] // n_unique_ticks
        dim = 4 if return_times else 2
        out = out.reshape(n_cv_folds, n_unique_ticks, dim)
        
        out = np.asarray(a=out).transpose((2, 1, 0))

        # Sklearn returns negative MAE and MSE scores,
        # so we restore nonnegativity
        if self.scoring in ['neg_mean_absolute_error', 'neg_mean_squared_error']:
            out *= -1

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
                        path=None, pipeline_transform='quantilenormal',
                        pipeline_memory=None, params=None, target_index=None,
                        chain_order=None, ylabel=None, stacking_options=None,
                        base_boosting_options=None, return_incumbent_score=False):

    lcurve = LearningCurve(regressor_choice=regressor_choice,
                           verbose=verbose, cv=cv,
                           pipeline_transform=pipeline_transform,
                           pipeline_memory=pipeline_memory,
                           params=params, target_index=target_index,
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
