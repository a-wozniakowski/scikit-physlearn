import copy
import joblib
import matplotlib
import numbers
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn.base
import sklearn.metrics
import sklearn.metrics._scorer
import sklearn.model_selection._split
import sklearn.model_selection._validation
import sklearn.utils.metaestimators
import sklearn.utils.validation

from traceback import format_exc

from ..regression import Regressor
from ..utils._data_checks import _validate_data


class LearningCurve(Regressor):

    def __init__(self, model_choice, cv=5, verbose=0, n_jobs=-1,
                 scoring='mae', scoring_multioutput='raw_values',
                 search_scoring='neg_mean_absolute_error',
                 transform_feature='quantile_normal', transform_target=None,
                 params=None, stacking_layer=None, chain_order=None,
                 n_regressors=None, target_index=None):

        super().__init__(model_choice=model_choice, cv=cv, verbose=verbose, 
                         n_jobs=n_jobs, scoring=scoring,
                         scoring_multioutput=scoring_multioutput,
                         search_scoring=search_scoring,
                         transform_feature=transform_feature,
                         transform_target=transform_target, params=params,
                         stacking_layer=stacking_layer, chain_order=chain_order,
                         n_regressors=n_regressors, target_index=target_index)

    def _modified_fit_and_score(self, regressor, X, y, scorer, train, test,
                                verbose, parameters, fit_params,
                                return_train_score=False, return_parameters=False,
                                return_n_test_samples=False, return_times=False,
                                return_regressor=False, error_score=np.nan):
        if verbose > 1:
            if parameters is None:
                msg = ''
            else:
                msg = '%s' % (', '.join('%s=%s' % (k, v)
                              for k, v in parameters.items()))
            print('[CV] %s %s' % (msg, (64 - len(msg)) * '.'))

        # Adjust length of sample weights
        fit_params = fit_params if fit_params is not None else {}
        fit_params = sklearn.utils.validation._check_fit_params(X, fit_params, train)

        train_scores = {}
        if parameters is not None:
            # clone after setting parameters in case any parameters
            # are regressors (like pipeline steps)
            # because pipeline doesn't clone steps in fit
            cloned_parameters = {}
            for k, v in parameters.items():
                cloned_parameters[k] = clone(v, safe=False)

            regressor = regressor.set_params(**cloned_parameters)

        start_time = time.time()

        X_train, y_train = sklearn.utils.metaestimators._safe_split(regressor, X, y, train)
        X_test, y_test = sklearn.utils.metaestimators._safe_split(regressor, X, y, test, train)

        try:
            if y_train is None:
                regressor.fit(X=X_train, **fit_params)
            else:
                regressor.fit(X=X_train, y=y_train, **fit_params)

        except Exception as e:
            # Note fit time as time until error
            fit_time = time.time() - start_time
            score_time = 0.0
            if error_score == 'raise':
                raise
            elif isinstance(error_score, numbers.Number):
                if isinstance(scorer, dict):
                    test_scores = {name: error_score for name in scorer}
                    if return_train_score:
                        train_scores = test_scores.copy()
                else:
                    test_scores = error_score
                    if return_train_score:
                        train_scores = error_score
                warnings.warn("Regressor fit failed. The score on this train-test"
                              " partition for these parameters will be set to %f. "
                              "Details: \n%s" %
                              (error_score, format_exc()),
                              FitFailedWarning)
            else:
                raise ValueError("error_score must be the string 'raise' or a"
                                 " numeric value. (Hint: if using 'raise', please"
                                 " make sure that it has been spelled correctly.)")

        else:
            fit_time = time.time() - start_time
            if self.target_index is not None and y.ndim > 1:
                y_test = y_test.iloc[:, self.target_index]
            test_scores = sklearn.model_selection._validation._score(regressor, X_test, y_test, scorer)
            score_time = time.time() - start_time - fit_time
            if return_train_score:
                if self.target_index is not None and y.ndim > 1:
                    y_train = y_train.iloc[:, self.target_index]
                train_scores = sklearn.model_selection._validation._score(
                    regressor, X_train, y_train, scorer
                )
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    msg += ', %s=' % scorer_name
                    if return_train_score:
                        msg += '(train=%.3f,' % train_scores[scorer_name]
                        msg += ' test=%.3f)' % test_scores[scorer_name]
                    else:
                        msg += '%.3f' % test_scores[scorer_name]
            else:
                msg += ', score='
                msg += ('%.3f' % test_scores if not return_train_score else
                        '(train=%.3f, test=%.3f)' % (train_scores, test_scores))

        if verbose > 1:
            total_time = score_time + fit_time
            print(_message_with_time('CV', msg, total_time))

        ret = [train_scores, test_scores] if return_train_score else [test_scores]

        if return_n_test_samples:
            ret.append(_num_samples(X_test))
        if return_times:
            ret.extend([fit_time, score_time])
        if return_parameters:
            ret.append(parameters)
        if return_regressor:
            ret.append(regressor)
        return ret

    def _modified_learning_curve(self, X, y, train_sizes=np.linspace(0.1, 1.0, 5),
                                 return_train_score=True, return_times=False,
                                 return_regressor=False, error_score=np.nan,
                                 return_incumbent_score=False):

        X, y = _validate_data(X=X, y=y)
        self._get_pipeline(X=X, y=y)
        X, y, groups = sklearn.utils.validation.indexable(X, y, None)
        
        cv = sklearn.model_selection._split.check_cv(cv=self.cv, y=y,
                                                     classifier=sklearn.base.is_classifier(self.pipe))
        cv_iter = list(cv.split(X, y, groups))

        scorer = sklearn.metrics.check_scoring(estimator=self.pipe, scoring=self.search_scoring)

        # modified n_max_training_samples from sklearn
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

        out = parallel(joblib.delayed(self._modified_fit_and_score)(
            regressor=sklearn.base.clone(self.pipe), X=X, y=y, scorer=scorer,
            train=train, test=test, verbose=self.verbose, parameters=None,
            fit_params=None, return_train_score=return_train_score,
            return_parameters=False, return_n_test_samples=False,
            return_times=return_times, return_regressor=return_regressor,
            error_score=error_score)
            for train, test in train_test_proportions)
        out = np.array(out)
        n_cv_folds = out.shape[0] // n_unique_ticks
        dim = 4 if return_times else 2
        out = out.reshape(n_cv_folds, n_unique_ticks, dim)
        
        out = np.asarray(out).transpose((2, 1, 0))

        if self.search_scoring in ['neg_mean_absolute_error', 'neg_mean_squared_error']:
            out *= -1

        if return_incumbent_score:
            if self.target_index is not None:
                y_pred = X.iloc[:, self.target_index]
                y_true = y.iloc[:, self.target_index]
            else:
                y_pred = X

            incumbent_score = parallel(joblib.delayed(self.score_summary)(
                y_true=y_true.loc[test], y_pred=y_pred.loc[test])
                for _, test in train_test_proportions)
            
            if self.search_scoring == 'neg_mean_absolute_error':
                incumbent_score = [score['mae'] for score in incumbent_score]
            elif self.search_scoring == 'neg_mean_squared_error':
                incumbent_score = [score['mse'] for score in incumbent_score]            
            incumbent_score = np.array(incumbent_score).reshape(n_cv_folds, n_unique_ticks).transpose()

            # Base boosting cross-validation check
            # if initialization step produces a
            # better cross-validation score return
            # the identity function
            get_index = []
            for index, row_score in enumerate(out[1]):
                if np.mean(row_score) > np.mean(incumbent_score[index]):
                    out[1][index] = incumbent_score[index]
                    get_index.append(index)

            ret = train_sizes_abs, out[0], out[1], incumbent_score
            self.init_score_index_ = get_index
        else:
            ret = train_sizes_abs, out[0], out[1]

        if return_times:
            ret = ret + (out[2], out[3])

        return ret


def plot_learning_curve(model_choice, title, X, y, verbose=0,
                        train_sizes=np.linspace(0.2, 1.0, 5), alpha=0.1,
                        train_color='b', cv_color='orange',
                        y_ticks_step=0.15, fill_std=False,
                        legend_loc='best', save_plot=False,
                        path=None, params=None, stacking_layer=None, 
                        ylabel=None, n_regressors=None, target_index=None,
                        return_incumbent_score=False):

    lcurve = LearningCurve(model_choice=model_choice, params=params,
                           stacking_layer=stacking_layer, verbose=verbose,
                           n_regressors=n_regressors, target_index=target_index)

    if target_index is not None:
            y = y.iloc[:, target_index]

    if return_incumbent_score:
        train_sizes, train_score, cv_score, incumbent_score = lcurve._modified_learning_curve(
        X=X, y=y, train_sizes=train_sizes, return_incumbent_score=return_incumbent_score
    )
    else:
        train_sizes, train_score, cv_score = lcurve._modified_learning_curve(X=X, y=y, train_sizes=train_sizes)

    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Lucida Console']
    plt.figure()
    plt.title(title)
    plt.xlabel('Number of ordered pairs')

    if ylabel is not None:
        assert isinstance(ylabel, str)
        plt.ylabel(ylabel)
    else:
        if lcurve.search_scoring == 'neg_mean_absolute_error':
            plt.ylabel('MAE')
        elif lcurve.search_scoring == 'neg_mean_squared_error':
            plt.ylabel('MSE')
        else:
            plt.ylabel('Empirical error')

    train_score_mean = np.mean(train_score, axis=1)
    train_score_std = np.std(train_score, axis=1)
    cv_score_mean = np.mean(cv_score, axis=1)
    cv_score_std = np.std(cv_score, axis=1)
    
    if return_incumbent_score:
        incumbent_score_mean = np.mean(incumbent_score, axis=1)
        incumbent_score_std = np.std(incumbent_score, axis=1)
        for index in lcurve.init_score_index_:
            cv_score_std[index] = incumbent_score_std[index]
        global_score_min = np.around(np.min([train_score.min(), cv_score.min(),
                                     incumbent_score_mean.min()]), decimals=2)
        global_score_max = np.around(np.max([train_score.max(), cv_score.max(),
                                     incumbent_score_mean.max()]), decimals=2)
    else:
        global_score_min = np.around(np.min([train_score.min(), cv_score.min()]), decimals=2)
        global_score_max = np.around(np.max([train_score.max(), cv_score.max()]), decimals=2)

    plt.grid()

    if fill_std:
        plt.fill_between(train_sizes, train_score_mean - train_score_std,
                         train_score_mean + train_score_std,
                         alpha=alpha, color=train_color)
        plt.fill_between(train_sizes, cv_score_mean - cv_score_std,
                         cv_score_mean + cv_score_std,
                         alpha=alpha, color=cv_color)

    plt.plot(train_sizes, train_score_mean, 'o-', color=train_color, label='Training error')
    plt.plot(train_sizes, cv_score_mean, 'o-', color=cv_color, label='Cross-validation error')
    
    if return_incumbent_score:
        plt.plot(train_sizes, incumbent_score_mean, 'o-', color='r', label='Incumbent error')
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
        return plt