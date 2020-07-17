"""
Single-target and multi-target regression.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import joblib

import numpy as np
import pandas as pd

import scipy.linalg

import sklearn.base
import sklearn.metrics
import sklearn.metrics._scorer
import sklearn.model_selection._search
import sklearn.model_selection._split
import sklearn.model_selection._validation
import sklearn.utils
import sklearn.utils.estimator_checks
import sklearn.utils.multiclass
import sklearn.utils.validation

from collections import defaultdict

from ..base import AdditionalRegressorMixin
from ..loss import LOSS_FUNCTIONS
from ..pipeline import _make_pipeline

from .interface import RegressorDictionaryInterface
from .model_selection.bayesian_search import _bayesoptcv
from .utils._data_checks import _n_features, _n_targets, _n_samples, _validate_data
from .utils._definition import (_MODEL_DICT, _MULTI_TARGET, _SEARCH_METHOD,
                                _SCORE_CHOICE, _SEARCH_TAXONOMY)
from .utils._model_checks import (_check_bayesoptcv_parameter_type, _check_model_choice,
                                  _check_search_method, _check_stacking_layer,
                                  _convert_filename_to_csv_path, _parallel_search_preprocessing,
                                  _sequential_search_preprocessing)


_MODEL_DICT = _MODEL_DICT['regression']


class BaseRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin, AdditionalRegressorMixin):
    """
    Base class for main regressor object.
    """

    def __init__(self, regressor_choice='ridge', cv=5, random_state=0,
                 verbose=0, n_jobs=-1, score_multioutput='raw_values',
                 scoring='neg_mean_absolute_error', return_train_score=True,
                 pipeline_transform=None, pipeline_memory=None,
                 params=None, chain_order=None, stacking_layer=None,
                 stacking_cv_shuffle=True, stacking_cv_refit=True,
                 stacking_passthrough=True, stacking_meta_features=True,
                 target_index=None, n_regressors=None,
                 boosting_loss=None, line_search_regularization=None,
                 line_search_options=None):
    
        assert isinstance(regressor_choice, str)
        assert isinstance(cv, int) and cv > 1
        assert isinstance(random_state, int) and random_state >= 0
        assert isinstance(verbose, int) and verbose >= 0
        assert isinstance(n_jobs, int)
        assert isinstance(score_multioutput, str)
        assert isinstance(scoring, str)
        assert isinstance(return_train_score, bool)
        assert isinstance(stacking_cv_shuffle, bool)
        assert isinstance(stacking_cv_refit, bool)
        assert isinstance(stacking_passthrough, bool)
        assert isinstance(stacking_meta_features, bool)

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
        self.scoring = scoring
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
                                                  params=params, stacking_layer=self.stacking_layer)

        self._regressor, self.params = _regressor.set_params(cv=self.cv, verbose=self.verbose,
                                                             random_state=self.random_state,
                                                             n_jobs=self.n_jobs, 
                                                             shuffle=self.stacking_cv_shuffle,
                                                             refit=self.stacking_cv_refit,
                                                             passthrough=self.stacking_passthrough,
                                                             meta_features=self.stacking_meta_features)

    @property
    def check_regressor(self):
        """Check if regressor adheres to scikit-learn conventions."""

        return sklearn.utils.estimator_checks.check_estimator(self._regressor)

    def get_params(self, deep=True):
        """Retrieve parameters."""

        # Override method in BaseEstimator
        return self.params

    def set_params(self, **params):
        """Set parameters of regressor choice."""

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for regressor %s. '
                                 'Check the list of available parameters '
                                 'with `regressor.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self._regressor, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def dump(self, value, filename):
        """Save a file in joblib format."""

        assert isinstance(filename, str)
        joblib.dump(value=value, filename=filename)

    def load(self, filename):
        """Load a file in joblib format."""

        assert isinstance(filename, str)        
        return joblib.load(filename=filename)
    
    def get_pipeline(self, y, n_quantiles=None):
        """Create pipe attribute for downstream tasks."""

        if n_quantiles is None:
            n_quantiles = _n_samples(y)

        self.pipe =  _make_pipeline(estimator=self._regressor,
                                    transform=self.pipeline_transform,
                                    n_targets=_n_targets(y),
                                    random_state=self.random_state,
                                    verbose=self.verbose,
                                    n_jobs=self.n_jobs,
                                    cv=self.cv,
                                    memory=self.pipeline_memory,
                                    n_quantiles=n_quantiles,
                                    chain_order=self.chain_order,
                                    n_estimators=self.n_regressors,
                                    target_index=self.target_index,
                                    boosting_loss=self.boosting_loss,
                                    regularization=self.line_search_regularization,
                                    line_search_options=self.line_search_options)

    def regattr(self, attr):
        """Get regressor attribute from pipeline."""

        assert hasattr(self, 'pipe') and isinstance(attr, str)

        try:
            attr = {f'target {index}': getattr(self.pipe, attr)
                   for index, self.pipe
                   in enumerate(self.pipe.named_steps['reg'].estimators_)}
            return attr
        except AttributeError:
            print(f'{self.pipe.named_steps["reg"]} needs to have an estimators_ attribute.')

    def _check_target_index(self, y):
        """Automates single-target regression subtask slicing."""

        if self.target_index is not None and \
        sklearn.utils.multiclass.type_of_target(y) in _MULTI_TARGET:
            # Selects a particular single-target
            return y.iloc[:, self.target_index]
        else:
            return y

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

        if not hasattr(self, '_validated_data'):
            X, y = _validate_data(X=X, y=y)
            setattr(self, '_validated_data', True)

        # Automates single-target slicing
        y = self._check_target_index(y=y)

        if not hasattr(self, 'pipe'):
            self.get_pipeline(y=y)

        self._fit(regressor=self.pipe, X=X, y=y,
                  sample_weight=sample_weight)

        return self.pipe

    def predict(self, X):
        """Generate predictions."""

        if not hasattr(self, '_validated_data'):
            X = _validate_data(X=X)

        if hasattr(self, 'return_incumbent_'):
            # Generate predictions with the incumbent
            if self.target_index is not None:
                y_pred = X.iloc[:, self.target_index]
            else:
                y_pred = X
        else:
            assert hasattr(self, 'pipe')
            y_pred = self.pipe.predict(X=X)

        return y_pred

    def score(self, y_true, y_pred, scoring, multioutput):
        """Compute score in supervised fashion."""

        assert any(scoring for method in _SCORE_CHOICE) and isinstance(scoring, str)

        if scoring in ['r2', 'ev']:
            possible_multioutputs = ['raw_values', 'uniform_average',
                                     'variance_weighted']
            assert any(multioutput for output in possible_multioutputs)
        else:
            possible_multioutputs = ['raw_values', 'uniform_average']
            assert any(multioutput for output in possible_multioutputs)

        # Automates single-target slicing
        y_true = self._check_target_index(y=y_true)

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
            try:
                score = sklearn.metrics.mean_squared_log_error(y_true=y_true, y_pred=y_pred,
                                                               multioutput=multioutput)
            except ValueError:
                # Sklearn will raise a ValueError if
                # either statement is true, so we circumvent
                # this error and score with a NaN
                score = np.nan

        return score

    def _preprocess_search_params(self, y, search_params, search_taxonomy='parallel'):
        """Search parameter helper function."""

        assert search_taxonomy in _SEARCH_TAXONOMY

        if search_taxonomy == 'parallel':
            if sklearn.utils.multiclass.type_of_target(y) in _MULTI_TARGET:
                if self.chain_order is not None:
                    search_params = _parallel_search_preprocessing(raw_params=search_params,
                                                                   multi_target=True, chain=True)
                else:
                    search_params = _parallel_search_preprocessing(raw_params=search_params,
                                                                   multi_target=True, chain=False)
            else:
                search_params = _parallel_search_preprocessing(raw_params=search_params,
                                                               multi_target=False, chain=False)
        elif search_taxonomy == 'sequential':
            if sklearn.utils.multiclass.type_of_target(y) in _MULTI_TARGET:
                if self.chain_order is not None:
                    search_params = _sequential_search_preprocessing(raw_pbounds=search_params,
                                                                     multi_target=True, chain=True)
                else:
                    search_params = _sequential_search_preprocessing(raw_pbounds=search_params,
                                                                     multi_target=True, chain=False)
            else:
                search_params = _sequential_search_preprocessing(raw_pbounds=search_params,
                                                                 multi_target=False, chain=False)

        return search_params

    def _modified_cross_validate(self, X, y, return_regressor=False,
                                 error_score=np.nan, return_incumbent_score=False):
        """Perform cross-validation for regressor
           and incumbent, if return_incumbent_score is True."""

        if not hasattr(self, '_validated_data'):
            X, y = _validate_data(X=X, y=y)
            setattr(self, '_validated_data', True)

        X, y, groups = sklearn.utils.validation.indexable(X, y, None)

        if not hasattr(self, 'pipe'):
            n_samples = _n_samples(y)
            fold_size =  np.full(shape=n_samples, fill_value=n_samples // self.cv,
                                 dtype=np.int)
            estimate_fold_size = n_samples - (np.max(fold_size) + 1)
            self.get_pipeline(y=y, n_quantiles=estimate_fold_size)

        cv = sklearn.model_selection._split.check_cv(cv=self.cv, y=y, classifier=self.pipe)

        scorers, _ = sklearn.metrics._scorer._check_multimetric_scoring(estimator=self.pipe,
                                                                        scoring=self.scoring)

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

            if self.scoring == 'neg_mean_absolute_error':
                incumbent_test_score = [score['mae'].values[0] for score in incumbent_test_score]
            elif self.scoring == 'neg_mean_squared_error':
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

        # Sklearn returns negative MAE and MSE scores,
        # so we restore nonnegativity
        if self.scoring in ['neg_mean_absolute_error', 'neg_mean_squared_error']:
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
                 scoring='neg_mean_absolute_error', refit=True,
                 randomizedcv_n_iter=20, bayesoptcv_init_points=2,
                 bayesoptcv_n_iter=20, return_train_score=True,
                 pipeline_transform='quantilenormal', pipeline_memory=None,
                 params=None, chain_order=None, stacking_layer=None,
                 target_index=None, n_regressors=None, boosting_loss=None,
                 line_search_regularization=None, line_search_options=None):

        super().__init__(regressor_choice=regressor_choice,
                         cv=cv,
                         random_state=random_state,
                         verbose=verbose,
                         n_jobs=n_jobs,
                         score_multioutput=score_multioutput,
                         scoring=scoring,
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

        assert isinstance(refit, bool)
        assert isinstance(randomizedcv_n_iter, int)
        assert isinstance(bayesoptcv_init_points, int)
        assert isinstance(bayesoptcv_n_iter, int)

        self.refit = refit
        self.randomizedcv_n_iter = randomizedcv_n_iter
        self.bayesoptcv_init_points = bayesoptcv_init_points
        self.bayesoptcv_n_iter = bayesoptcv_n_iter

    @property
    def check_regressor(self):
        """Check if regressor adheres to scikit-learn conventions."""

        # Sklearn and Mlxtend stacking regressors, as well as 
        # LightGBM, XGBoost, and CatBoost regressor 
        # do not adhere to the convention.
        try:
            super().check_regressor
        except:
            print(f'{_MODEL_DICT[self.regressor_choice]} does not adhere to sklearn conventions.')

    def get_params(self, deep=True):
        """Retrieve parameters."""

        return super().get_params(deep=deep)

    def set_params(self, **params):
        """Set parameters of regressor choice."""

        return super().set_params(**params)

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

    def _inbuilt_model_selection_step(self, X, y):
        """Cross-validates the incumbent and the candidate regressor."""

        cross_val_score = super().cross_val_score(X=X, y=y,
                                                  return_incumbent_score=True)
        mean_cross_val_score = cross_val_score.mean(axis=0)

        if mean_cross_val_score[0] >= mean_cross_val_score[1]:
            # Base boosting did not improve performance
            setattr(self, '_return_incumbent', True)

    def baseboostcv(self, X, y, sample_weight=None):
        """Base boosting with inbuilt cross-validation"""

        if not hasattr(self, '_validated_data'):
            X, y = _validate_data(X=X, y=y)
            setattr(self, '_validated_data', True)

        # Automates single-target slicing
        y = super()._check_target_index(y=y)

        # Performs augmented k-fold cross-validation,
        # then selects the incumbent or the candidate
        self._inbuilt_model_selection_step(X=X, y=y)

        if not hasattr(self, 'pipe'):
            super().get_pipeline(y=y)

        if not hasattr(self, '_return_incumbent'):
            # Base boosting improves performance,
            # so we fit the candidate
            super().fit(X=X, y=y, sample_weight=sample_weight)
            return self.pipe
        else:
            setattr(self, 'return_incumbent_', True) 
            return self

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
        
        # Shifts the index origin by one
        if self.target_index is not None:
            score_summary_df.index = pd.RangeIndex(start=self.target_index + 1,
                                                   stop=self.target_index + 2, step=1)

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

    def _search(self, X, y, search_params, search_method='gridsearchcv'):
        """Helper (hyper)parameter search method."""

        # The returned search method is either
        # sequential or parallell. The former
        # identifies Bayesian optimization, while
        # the latter identifies grid or randomized
        # search by Sklearn. 
        search_method, search_taxonomy = _check_search_method(search_method=search_method)
        search_params = super()._preprocess_search_params(y=y, search_params=search_params,
                                                          search_taxonomy=search_taxonomy)
        if not hasattr(self, 'pipe'):
            n_samples = _n_samples(y)
            fold_size =  np.full(shape=n_samples, fill_value=n_samples // self.cv,
                                 dtype=np.int)
            estimate_fold_size = n_samples - (np.max(fold_size) + 1)
            self.get_pipeline(y=y, n_quantiles=estimate_fold_size)

        if search_method == 'gridsearchcv':
            self._regressor_search = sklearn.model_selection._search.GridSearchCV(
                estimator=self.pipe, param_grid=search_params,
                scoring=self.scoring, refit=self.refit, n_jobs=self.n_jobs,
                cv=self.cv, verbose=self.verbose, pre_dispatch='2*n_jobs',
                error_score=np.nan, return_train_score=self.return_train_score)
        elif search_method == 'randomizedsearchcv':
            self._regressor_search = sklearn.model_selection._search.RandomizedSearchCV(
                estimator=self.pipe, param_distributions=search_params,
                n_iter=self.randomizedcv_n_iter, scoring=self.scoring,
                n_jobs=self.n_jobs, refit=self.refit, cv=self.cv,
                verbose=self.verbose, pre_dispatch='2*n_jobs',
                error_score=np.nan, return_train_score=self.return_train_score)
        elif search_method == 'bayesoptcv':
            self.optimization = _bayesoptcv(X=X, y=y, estimator=self.pipe,
                                            search_params=search_params,
                                            cv=self.cv,
                                            scoring=self.scoring,
                                            n_jobs=self.n_jobs,
                                            verbose=self.verbose,
                                            random_state=self.random_state,
                                            init_points=self.bayesoptcv_init_points,
                                            n_iter=self.bayesoptcv_n_iter)

            if self.refit:
                max_params = self.optimization.max['params']
                get_best_params_ = _check_bayesoptcv_parameter_type(max_params)
                self._regressor_search = self.pipe.set_params(**get_best_params_)

    def search(self, X, y, search_params, search_method='gridsearchcv', filename=None):
        """(Hyper)parameter search method."""

        if filename is not None:
            assert isinstance(filename, str)

        # Automates single-target slicing
        y = self._check_target_index(y=y)

        self._search(X=X, y=y, search_params=search_params,
                     search_method=search_method)

        try:
            self._regressor_search.fit(X=X, y=y)
        except AttributeError:
            print('Fit method requires _regressor_search attribute')

        if search_method in ['gridsearchcv', 'randomizedsearchcv']:
            self.best_params_ = pd.Series(self._regressor_search.best_params_)
            self.best_score_ = pd.Series({'best_score': self._regressor_search.best_score_})
        elif search_method == 'bayesoptcv':
            try:
                self.best_params_ = pd.Series(self.optimization.max['params'])
                self.best_score_ = pd.Series({'best_score': self.optimization.max['target']})
            except AttributeError:
                print('best_params_ and best_score_ require optimization attribute')

        # Sklearn and bayes-opt return negative MAE and MSE scores,
        # so we restore nonnegativity
        if len(self.scoring) > 3 and self.scoring[:3] == 'neg':
            self.best_score_.loc['best_score'] *= -1.0

        self.search_summary_ = pd.concat([self.best_score_, self.best_params_], axis=0)

        # Filter based on sklearn model search attributes
        _sklearn_list = ['best_estimator_', 'cv_results_', 'refit_time_']
        if all(hasattr(self._regressor_search, attr) for attr in _sklearn_list):
            self.best_regressor_ = self._regressor_search.best_estimator_
            self.cv_results_ = pd.DataFrame(self._regressor_search.cv_results_)
            self.refit_time_ = pd.Series({'refit_time':self._regressor_search.refit_time_})
            self.search_summary_ = pd.concat([self.search_summary_, self.refit_time_], axis=0)

        if self.refit:
            try:
                self.pipe = self._regressor_search.best_estimator_
            except AttributeError:
                print('The regressor search object does not have the attribute: base_estimator_.')

        if filename is not None:
            path = _convert_filename_to_csv_path(filename=filename)
            self.search_summary_.to_csv(path_or_buf=path, header=True)
        
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
