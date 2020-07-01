"""
Single-target and multi-target regression.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import copy
import inspect
import joblib
import warnings

import numpy as np
import pandas as pd

import scipy.linalg

import sklearn.base
import sklearn.metrics
import sklearn.metrics._scorer
import sklearn.model_selection._split
import sklearn.model_selection._validation
import sklearn.utils
import sklearn.utils.multiclass
import sklearn.utils.validation

from ..base import AdditionalRegressorMixin
from ..loss import LOSS_FUNCTIONS
from ..pipeline import _make_modified_pipeline

from .interface import RegressorDictionaryInterface
from .model_selection.search import (_helper_gridsearchcv, _helper_randomizedsearchcv,
                                     _helper_bayesianoptimizationcv)
from .utils._data_checks import _n_features, _n_targets, _n_samples, _validate_data
from .utils._definition import (_MODEL_DICT, _MODEL_SEARCH_METHOD, _PIPELINE_TRANSFORM_CHOICE,
                                _SCORE_CHOICE)
from .utils._model_checks import (_check_bayesianoptimization_parameter_type, _check_model_choice,
                                  _check_model_search_style, _check_stacking_layer,
                                  _convert_filename_to_csv_path, _parallel_model_search_preprocessing,
                                  _sequential_model_search_preprocessing)


class BaseRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin, AdditionalRegressorMixin):
    """
    Base class for main regressor object.
    """

    def __init__(self, regressor_choice='ridge', cv=5, random_state=0,
                 verbose=0, n_jobs=-1, score_multioutput='raw_values',
                 search_scoring='neg_mean_absolute_error',
                 return_train_score=True, pipeline_transform=None,
                 pipeline_memory=None, params=None, chain_order=None,
                 stacking_layer=None, stacking_cv_shuffle=True,
                 stacking_cv_refit=True, stacking_passthrough=True,
                 stacking_meta_features=True, target_index=None,
                 n_regressors=None, boosting_loss=None,
                 line_search_regularization=None,
                 line_search_options=None):
    
        assert isinstance(regressor_choice, str)
        assert isinstance(cv, int) and cv > 1
        assert isinstance(random_state, int) and random_state >= 0
        assert isinstance(verbose, int) and verbose >= 0
        assert isinstance(n_jobs, int)
        assert isinstance(score_multioutput, str)
        assert isinstance(search_scoring, str)
        assert isinstance(return_train_score, bool)
        assert isinstance(stacking_cv_shuffle, bool)
        assert isinstance(stacking_cv_refit, bool)
        assert isinstance(stacking_passthrough, bool)
        assert isinstance(stacking_meta_features, bool)

        if pipeline_transform is not None:
            assert any(pipeline_transform == transform for transform in _PIPELINE_TRANSFORM_CHOICE), (
                'Choose from ' f'{_PIPELINE_TRANSFORM_CHOICE}')

        if pipeline_memory is not None:
            assert isinstance(pipeline_memory, bool)

        if chain_order is not None:
            assert isinstance(chain_order, list)

        if target_index is not None:
            assert isinstance(target_index, int)

        if n_regressors is not None:
            assert isinstance(n_regressors, int)
            if n_regressors <= 0:
                raise ValueError(f'{n_regressors} must be greater than zero.')

        if boosting_loss is not None:
            assert isinstance(boosting_loss, str)
            if boosting_loss not in LOSS_FUNCTIONS:
                raise KeyError(f'{boosting_loss} is not an available loss function.')

        if line_search_regularization is not None:
            assert isinstance(line_search_regularization, float) and line_search_regularization >= 0.0

        if line_search_options is not None:
            assert isinstance(line_search_options, dict)

        self.regressor_choice = _check_model_choice(model_choice=regressor_choice,
                                                    model_type='regression')
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.score_multioutput = score_multioutput
        self.search_scoring = search_scoring
        self.return_train_score = return_train_score
        self.pipeline_transform = pipeline_transform
        self.pipeline_memory = pipeline_memory
        self.chain_order = chain_order
        self.stacking_layer = _check_stacking_layer(stacking_layer=stacking_layer,
                                                    model_type='regression')
        self.stacking_cv_shuffle = stacking_cv_shuffle
        self.stacking_cv_refit = stacking_cv_refit
        self.stacking_passthrough = stacking_passthrough
        self.stacking_meta_features = stacking_meta_features
        self.target_index = target_index
        self.n_regressors = n_regressors
        self.boosting_loss = boosting_loss
        self.line_search_regularization = line_search_regularization
        self.line_search_options = line_search_options

        # Prepare regressor
        _regressor = RegressorDictionaryInterface(regressor_choice=self.regressor_choice,
                                                  params=params,
                                                  stacking_layer=self.stacking_layer)

        self._regressor, self.params = _regressor.set_params(cv=self.cv,
                                                             verbose=self.verbose,
                                                             random_state=self.random_state,
                                                             n_jobs=self.n_jobs,
                                                             shuffle=self.stacking_cv_shuffle,
                                                             refit=self.stacking_cv_refit,
                                                             passthrough=self.stacking_passthrough,
                                                             meta_features=self.stacking_meta_features)

    def get_params(self, deep=True):
        """Retrieve parameters."""

        # Override method in BaseEstimator
        return self.params

    def dump(self, value, filename):
        """Save a file in joblib format."""

        assert isinstance(filename, str)
        joblib.dump(value=value, filename=filename)

    def load(self, filename):
        """Load a file in joblib format."""

        assert isinstance(filename, str)        
        return joblib.load(filename=filename)
    
    def get_pipeline(self, y):
        """Create pipe attribute for downstream tasks."""

        self.pipe =  _make_modified_pipeline(estimator=self._regressor,
                                             transform=self.pipeline_transform,
                                             n_targets=_n_targets(y),
                                             random_state=self.random_state,
                                             verbose=self.verbose,
                                             n_jobs=self.n_jobs,
                                             cv=self.cv,
                                             memory=self.pipeline_memory,
                                             n_quantiles=_n_samples(y),
                                             chain_order=self.chain_order,
                                             n_estimators=self.n_regressors,
                                             target_index=self.target_index,
                                             boosting_loss=self.boosting_loss,
                                             regularization=self.line_search_regularization,
                                             line_search_options=self.line_search_options)

    def regattr(self, attr):
        """Get regressor attribute from pipeline."""

        assert hasattr(self, 'pipe') and isinstance(attr, str)

        attr = {f'target {index}': getattr(self.pipe, attr)
               for index, self.pipe
               in enumerate(self.pipe.named_steps['reg'].estimators_)}

        return attr

    @staticmethod
    def _fit(regressor, X, y, sample_weight=None):
        """Helper fit method."""

        if sample_weight is not None:
            try:
                regressor.fit(X=X, y=y, sample_weight=sample_weight)
            except TypeError as exc:
                if 'unexpected keyword argument sample_weight' in str(exc):
                    raise TypeError(
                        f'{regressor.__class__.__name__} does not support sample weights.'
                    ) from exc
            raise
        else:
            regressor.fit(X=X, y=y)

    def fit(self, X, y, sample_weight=None):
        """Fit regressor."""

        X, y = _validate_data(X=X, y=y)

        if self.target_index is not None and \
        sklearn.utils.multiclass.type_of_target(y) == 'continuous-multioutput':
            # Selects a particular single-target
            y = y.iloc[:, self.target_index]

        self.get_pipeline(y=y)
        self._fit(regressor=self.pipe, X=X, y=y, sample_weight=sample_weight)

        return self.pipe

    def _inbuilt_model_selection_step(self, X, y):
        """Cross-validates the incumbent and the candidate regressor."""

        cross_val_score = self.cross_val_score(X=X, y=y,
                                               return_incumbent_score=True)
        mean_cross_val_score = cross_val_score.mean(axis=0)

        if mean_cross_val_score[0] >= mean_cross_val_score[1]:
            # Base boosting did not improve performance
            setattr(self, 'return_incumbent_', True)

    def baseboostcv(self, X, y, sample_weight=None):
        """Base boosting with inbuilt cross-validation"""

        self._inbuilt_model_selection_step(X=X, y=y)
        if not hasattr(self, 'return_incumbent_'):
            # Base boosting improves performance,
            # so we fit the candidate
            self.fit(X=X, y=y, sample_weight=sample_weight)
        else:
            self.get_pipeline(y=y)

        return self.pipe

    def predict(self, X):
        """Generate predictions."""

        assert hasattr(self, 'pipe')
        X = _validate_data(X=X)

        if hasattr(self, 'return_incumbent_'):
            # Generate predictions with the incumbent
            if self.target_index is not None:
                y_pred = X.iloc[:, self.target_index]
            else:
                y_pred = X
        else:
            y_pred = self.pipe.predict(X=X)

        if _n_targets(y_pred) > 1:
            return pd.DataFrame(y_pred, index=X.index)
        else:
            return pd.Series(y_pred, index=X.index)

    def score(self, y_true, y_pred, scoring, multioutput):
        """Compute score in supervised fashion."""

        assert any(scoring for method in _SCORE_CHOICE) and isinstance(scoring, str)

        if scoring in ['r2', 'ev']:
            possible_multioutputs = ['raw_values', 'uniform_average', 'variance_weighted']
            assert any(multioutput for output in possible_multioutputs)
        else:
            possible_multioutputs = ['raw_values', 'uniform_average']
            assert any(multioutput for output in possible_multioutputs)

        if scoring == 'mae':
            score = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred,
                                                        multioutput=multioutput)
        elif scoring == 'mse':
            score = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred,
                                                       multioutput=multioutput)
        elif scoring == 'rmse':
            score = np.sqrt(sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred,
                                                               multioutput=multioutput))
        elif scoring == 'r2':
            score = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred,
                                             multioutput=multioutput)
        elif scoring == 'ev':
            score = sklearn.metrics.explained_variance_score(y_true=y_true, y_pred=y_pred,
                                                             multioutput=multioutput)
        elif scoring == 'msle':
            if any(y_true < 0) or any(y_pred < 0):
                score = np.nan
            else:
                score = sklearn.metrics.mean_squared_log_error(y_true=y_true, y_pred=y_pred,
                                                               multioutput=multioutput)

        return score

    def _process_search_params(self, y, search_params, search_method='parallel'):
        """Search parameter helper function."""

        assert search_method in _MODEL_SEARCH_METHOD

        if search_method == 'parallel':
            search_params = _parallel_model_search_preprocessing(raw_params=search_params)
        elif search_method == 'sequential':
            search_params = _sequential_model_search_preprocessing(raw_pbounds=search_params)

        return search_params

    def _modified_cross_validate(self, X, y, return_regressor=False,
                                 error_score=np.nan, return_incumbent_score=False):
        """Perform cross-validation for regressor
           and incumbent, if return_incumbent_score is True."""

        if self.target_index is not None and y.ndim > 1:
            y = y.iloc[:, self.target_index]

        X, y = _validate_data(X=X, y=y)
        X, y, groups = sklearn.utils.validation.indexable(X, y, None)
        
        cv = sklearn.model_selection._split.check_cv(cv=self.cv, y=y, classifier=False)
        
        self.get_pipeline(y=y)
        scorers, _ = sklearn.metrics._scorer._check_multimetric_scoring(estimator=self.pipe,
                                                                        scoring=self.search_scoring)

        parallel = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                                   pre_dispatch='2*n_jobs')

        scores = parallel(joblib.delayed(sklearn.model_selection._validation._fit_and_score)(
            estimator=sklearn.base.clone(self.pipe), X=X, y=y, scorer=scorers,
            train=train, test=test, verbose=self.verbose, parameters=None,
            fit_params=None, return_train_score=self.return_train_score,
            return_parameters=False, return_n_test_samples=False,
            return_times=True, return_estimator=return_regressor,
            error_score=np.nan)
            for train, test in cv.split(X, y, groups))

        if return_incumbent_score:
            if self.target_index is not None:
                y_pred = X.iloc[:, self.target_index]
            else:
                y_pred = X
            incumbent_test_score = parallel(joblib.delayed(self.score)(
                y_true=y.loc[test], y_pred=y_pred.loc[test])
                for _, test in cv.split(X, y, groups))
            if self.search_scoring == 'neg_mean_absolute_error':
                incumbent_test_score = [score['mae'].values[0] for score in incumbent_test_score]
            elif self.search_scoring == 'neg_mean_squared_error':
                incumbent_test_score = [score['mse'].values[0] for score in incumbent_test_score]

        zipped_scores = list(zip(*scores))
        if self.return_train_score:
            train_scores = zipped_scores.pop(0)
            train_scores = sklearn.model_selection._validation._aggregate_score_dicts(train_scores)
        if return_regressor:
            fitted_regressors = zipped_scores.pop()
        test_scores, fit_times, score_times = zipped_scores
        test_scores = sklearn.model_selection._validation._aggregate_score_dicts(test_scores)

        ret = {}
        ret['fit_time'] = np.array(fit_times)
        ret['score_time'] = np.array(score_times)

        if return_regressor:
            ret['regressor'] = fitted_regressors

        for name in scorers:
            ret['test_%s' % name] = np.array(test_scores[name])
            if self.return_train_score:
                key = 'train_%s' % name
                ret[key] = np.array(train_scores[name])

        if return_incumbent_score:
            ret['incumbent_test_score'] = incumbent_test_score

        return ret

    def cross_validate(self, X, y, return_incumbent_score=False):
        """Retrieve cross-validation results for regressor
           and incumbent, if return_incumbent_score is True."""

        scores_dict = self._modified_cross_validate(X=X, y=y,
                                                    return_incumbent_score=return_incumbent_score)

        # sklearn returns negative MAE and MSE scores,
        # so we restore nonnegativity
        if self.search_scoring in ['neg_mean_absolute_error', 'neg_mean_squared_error']:
            scores_dict['train_score'] = np.array([np.abs(score) for score in scores_dict['train_score']])
            scores_dict['test_score'] = np.array([np.abs(score) for score in scores_dict['test_score']])

        return pd.DataFrame(scores_dict)

    def cross_val_score(self, X, y, return_incumbent_score=False):
        """Retrieve withheld fold errors for regressor
           and incumbent, if return_incumbent_score is True."""

        scores_dict = self.cross_validate(X=X, y=y,
                                          return_incumbent_score=return_incumbent_score)

        if return_incumbent_score:
            return scores_dict[['test_score', 'incumbent_test_score']]
        else:
            return scores_dict['test_score']


class Regressor(BaseRegressor):
    """
    Main regressor class for building a prediction model.

    Important methods are fit, baseboostcv, predict, and search.
    """

    def __init__(self, regressor_choice='ridge', cv=5, random_state=0,
                 verbose=1, n_jobs=-1, score_multioutput='raw_values',
                 search_scoring='neg_mean_absolute_error', search_refit=True,
                 search_randomizedcv_n_iter=20, search_bayesianoptimization_init_points=2,
                 search_bayesianoptimization_n_iter=20, return_train_score=True,
                 pipeline_transform='quantile_normal', pipeline_memory=None,
                 params=None, chain_order=None, stacking_layer=None,
                 target_index=None, n_regressors=None, boosting_loss=None,
                 line_search_regularization=None, line_search_options=None):

        super().__init__(regressor_choice=regressor_choice,
                         cv=cv,
                         random_state=random_state,
                         verbose=verbose,
                         n_jobs=n_jobs,
                         score_multioutput=score_multioutput,
                         search_scoring=search_scoring,
                         return_train_score=return_train_score,
                         pipeline_transform=pipeline_transform,
                         pipeline_memory=pipeline_memory,
                         params=params,
                         chain_order=chain_order,
                         stacking_layer=stacking_layer,
                         target_index=target_index,
                         n_regressors=n_regressors,
                         boosting_loss=boosting_loss,
                         line_search_regularization=line_search_regularization,
                         line_search_options=line_search_options)

        assert isinstance(search_refit, bool)
        assert isinstance(search_randomizedcv_n_iter, int)
        assert isinstance(search_bayesianoptimization_init_points, int)
        assert isinstance(search_bayesianoptimization_n_iter, int)

        self.search_refit = search_refit
        self.search_randomizedcv_n_iter = search_randomizedcv_n_iter
        self.search_bayesianoptimization_init_points = search_bayesianoptimization_init_points
        self.search_bayesianoptimization_n_iter = search_bayesianoptimization_n_iter

    def get_params(self, deep=True):
        """Retrieve parameters."""

        return super().get_params(deep=deep)

    def dump(self, value, filename):
        """Save a file in joblib format."""

        super().dump(value=value, filename=filename)

    def load(self, filename):
        """Load a file in joblib format."""

        return super().load(filename=filename)

    def regattr(self, attr):
        """Get regressor attribute from pipeline."""

        return super().regattr(attr=attr)

    def fit(self, X, y, sample_weight=None):
        """Fit regressor."""

        return super().fit(X=X, y=y, sample_weight=sample_weight)

    def baseboostcv(self, X, y, sample_weight=None):
        """Base boosting with inbuilt cross-validation"""

        return super().baseboostcv(X=X, y=y, sample_weight=sample_weight)

    def predict(self, X):
        """Generate predictions."""

        return super().predict(X=X)

    def score(self, y_true, y_pred, filename=None):
        """Compute score DataFrame in supervised fashion."""

        if filename is not None:
            assert isinstance(filename, str)

        score_summary = {}
        for scoring in _SCORE_CHOICE:
            score_summary[scoring] = super().score(y_true=y_true, y_pred=y_pred,
                                                   scoring=scoring, multioutput=self.score_multioutput)

        score_summary_df = pd.DataFrame(score_summary).dropna(how='any', axis=1)
        score_summary_df.index.name = 'target'

        if filename is not None:
            path = _convert_filename_to_csv_path(filename=filename)
            score_summary_df.to_csv(path_or_buf=path)

        return score_summary_df

    def cross_validate(self, X, y, return_incumbent_score=False):
        """Retrieve cross-validation results for regressor
           and incumbent, if return_incumbent_score is True."""

        return super().cross_validate(X=X, y=y,
                                      return_incumbent_score=return_incumbent_score)

    def cross_val_score(self, X, y, return_incumbent_score=False):
        """Retrieve withheld fold errors for regressor
           and incumbent, if return_incumbent_score is True."""

        return super().cross_val_score(X=X, y=y,
                                       return_incumbent_score=return_incumbent_score)

    def search(self, X, y, search_params, search_style='gridsearchcv', filename=None):
        """(Hyper)parameter search method."""

        if filename is not None:
            assert isinstance(filename, str)

        if self.target_index is not None:
            y = y.iloc[:, self.target_index]

        self._helper_model_search(X=X, y=y, search_params=search_params,
                                  search_style=search_style)
        try:
            self.model_search.fit(X=X, y=y)
        except AttributeError:
            print('fit method requires model_search attribute')

        if search_style in ['gridsearchcv', 'randomizedsearchcv']:
            self.best_params_ = pd.Series(self.model_search.best_params_)
            self.best_score_ = pd.Series({'best_score': self.model_search.best_score_})
        elif search_style == 'bayesianoptimization':
            try:
                self.best_params_ = pd.Series(self.optimization.max['params'])
                self.best_score_ = pd.Series({'best_score': self.optimization.max['target']})
            except AttributeError:
                print('best_params_ and best_score_ require optimization attribute')

        # Avert negative mae or mse
        # returned by sklearn and bayes-opt
        if len(self.search_scoring) > 3 and self.search_scoring[:3] == 'neg':
            self.best_score_.loc['best_score'] *= -1.0

        self.search_summary_ = pd.concat([self.best_score_, self.best_params_], axis=0)

        # Filter based on sklearn model search attributes
        _sklearn_list = ['best_estimator_', 'cv_results_', 'refit_time_']
        if all(hasattr(self.model_search, attr) for attr in _sklearn_list):
            self.best_regressor_ = self.model_search.best_estimator_
            self.cv_results_ = pd.DataFrame(self.model_search.cv_results_)
            self.refit_time_ = pd.Series({'refit_time':self.model_search.refit_time_})
            self.search_summary_ = pd.concat([self.search_summary_, self.refit_time_], axis=0)

        if filename is not None:
            path = _convert_filename_to_csv_path(filename=filename)
            self.search_summary_.to_csv(path_or_buf=path, header=True)

    def _helper_model_search(self, X, y, search_params, search_style='gridsearchcv'):
        """Prepare model search attribute according to search_style."""

        # The returned search method is either
        # sequential or parallell. The former
        # identifies Bayesian optimization, while
        # the latter identifies grid or randomized
        # search by sklearn. 
        search_style, search_method = _check_model_search_style(search_style)
        search_params = super()._process_search_params(X=X, y=y, search_params=search_params,
                                                       search_method=search_method)

        self.get_pipeline(y=y)

        if search_style == 'gridsearchcv':
            self.model_search = _helper_gridsearchcv(estimator=self.pipe,
                                                     param_grid=search_params,
                                                     search_scoring=self.search_scoring,
                                                     refit=self.search_refit,
                                                     n_jobs=self.n_jobs,
                                                     cv=self.cv,
                                                     verbose=self.verbose,
                                                     pre_dispatch='2*n_jobs',
                                                     error_score=np.nan,
                                                     return_train_score=self.return_train_score)
        elif search_style == 'randomizedsearchcv':
            self.model_search = _helper_randomizedsearchcv(estimator=self.pipe,
                                                           param_distributions=search_params,
                                                           n_iter=self.search_randomizedcv_n_iter,
                                                           search_scoring=self.search_scoring,
                                                           n_jobs=self.n_jobs,
                                                           refit=self.search_refit,
                                                           cv=self.cv,
                                                           verbose=self.verbose,
                                                           pre_dispatch='2*n_jobs',
                                                           error_score=np.nan,
                                                           return_train_score=self.return_train_score)
        elif search_style == 'bayesianoptimization':
            self.optimization = _helper_bayesianoptimizationcv(X=X, y=y,
                                                               estimator=self.pipe,
                                                               search_params=search_params,
                                                               cv=self.cv,
                                                               scoring=self.search_scoring,
                                                               n_jobs=self.n_jobs,
                                                               verbose=self.verbose,
                                                               random_state=self.random_state,
                                                               init_points=self.search_bayesianoptimization_init_points,
                                                               n_iter=self.search_bayesianoptimization_n_iter)

            if self.search_refit:
                max_params = self.optimization.max['params']
                get_best_params_ = _check_bayesianoptimization_parameter_type(max_params)
                self.model_search = self.pipe.set_params(**get_best_params_)
        
    def subsample(self, X, y, subsample_proportion=None):
        """Generate subsamples from data X, y."""

        if subsample_proportion is not None:
            assert subsample_proportion > 0 and subsample_proportion < 1
            n_samples = int(len(X) * subsample_proportion)
            X, y = sklearn.utils.resample(X, y, replace=False,
                                          n_samples=n_samples, random_state=self.random_state)
        else:
            X, y = sklearn.utils.resample(X, y, replace=False,
                                          n_samples=len(X), random_state=self.random_state)

        return X, y

    def diagonalize_tridiagonal(self, y_true, y_pred, sort=True):
        """Retrieve eigenvalues from tridiagonal matrix."""

        assert isinstance(sort, bool)
        test_size = y_pred.shape[0]
        n_nonzero_entries = y_pred.shape[-1]
        n_diag_entries = int(np.ceil(y_pred.shape[-1] / 2))

        get_eigs = []
        for i in range(test_size):
            # Constructs tridiagonal matrix entries
            # then diagonalizes the matrix
            diag = [y_pred.iloc[i, j] for j in range(n_diag_entries)]
            off_diag = [y_pred.iloc[i, j] for j in range(n_diag_entries, n_nonzero_entries)]
            pos_eigs, _ = scipy.linalg.eigh_tridiagonal(diag, off_diag)
            get_eigs.append(pos_eigs)
        eigs = pd.DataFrame(get_eigs)

        # Sorts eigenvalues according to
        # Google's quantum lab convention
        if sort:
            return eigs.apply(lambda eig: -1*eig).sort_values(by=eigs.index.values[0], axis=1)
        else:
            return eigs.apply(lambda eig: -1*eig)
