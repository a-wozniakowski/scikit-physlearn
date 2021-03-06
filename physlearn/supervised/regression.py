"""
The :mod:`physlearn.supervised.regression` module provides machine learning
utilities, which solve single-target and multi-target regression tasks. It
includes the :class:`physlearn.BaseRegressor` and :class:`physlearn.Regressor`
classes.
"""

# Author: Alex Wozniakowski
# License: MIT

from __future__ import annotations

import joblib
import re
import typing

import numpy as np
import pandas as pd

import sklearn.base
import sklearn.metrics
import sklearn.metrics._scorer
import sklearn.model_selection
import sklearn.model_selection._split
import sklearn.model_selection._validation
import sklearn.utils
import sklearn.utils.estimator_checks
import sklearn.utils.metaestimators
import sklearn.utils.multiclass
import sklearn.utils.validation

from collections import defaultdict
from dataclasses import dataclass, field

from physlearn.base import AdditionalRegressorMixin
from physlearn.loss import LOSS_FUNCTIONS
from physlearn.pipeline import make_pipeline
from physlearn.supervised.interface import RegressorDictionaryInterface
from physlearn.supervised.utils._data_checks import (_n_features, _n_targets,
                                                     _n_samples, _validate_data)
from physlearn.supervised.utils._definition import (_MULTI_TARGET, _REGRESSOR_DICT,
                                                    _SEARCH_METHOD, _SCORE_CHOICE,
                                                    _SCORE_MULTIOUTPUT)
from physlearn.supervised.utils._estimator_checks import (_check_bayesoptcv_param_type,
                                                          _check_estimator_choice,
                                                          _check_search_method,
                                                          _check_stacking_layer,
                                                          _preprocess_hyperparams)
from physlearn.supervised.utils._search import _search_method

DataFrame_or_Series = typing.Union[pd.DataFrame, pd.Series]
pandas_or_numpy = typing.Union[pd.DataFrame, pd.Series, np.ndarray]
str_list_or_tuple = typing.Union[str, list, tuple]


@dataclass
class BaseRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin,
                    AdditionalRegressorMixin):
    """Base class for regressor amalgamation.

    The object is designed to amalgamate regressors from 
    `Scikit-learn <https://scikit-learn.org/>`_,
    `LightGBM <https://lightgbm.readthedocs.io/en/latest/index.html>`_,
    `XGBoost <https://xgboost.readthedocs.io/en/latest/>`_,
    `CatBoost <https://catboost.ai/>`_,
    and `Mlxtend <http://rasbt.github.io/mlxtend/>`_ into a unified framework,
    which follows the Scikit-learn API. Important methods include ``fit``,
    ``predict``, ``score``, ``dump``, ``load``, ``cross_validate``, and
    ``cross_val_score``.

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

    pipeline_transform : str, list, tuple, or None, optional (default=None)
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

    Notes
    -----
    The ``score`` method differs from the Scikit-learn usage, as the method is designed
    to abstract the regressor metrics, e.g., :class:`sklearn.metrics.mean_absolute_error`.

    See Also
    --------
    :class:`physlearn.pipeline.ModifiedPipeline` : Class for creating a pipeline.
    :class:`physlearn.supervised.regression.Regressor` : Main class for regressor amalgamation.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import train_test_split
    >>> from physlearn import BaseRegressor
    >>> X, y = load_boston(return_X_y=True)
    >>> X, y = pd.DataFrame(X), pd.Series(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)
    >>> reg = BaseRegressor(regressor_choice='lgbmregressor',
                            pipeline_transform='standardscaler')
    >>> y_pred = reg.fit(X_train, y_train).predict(X_test)
    >>> reg.score(y_test, y_pred)
    array([11.63706835])
    """

    regressor_choice: str = field(default='ridge')
    cv: int = field(default=5)
    random_state: int = field(default=0)
    verbose: int = field(default=0)
    n_jobs: int = field(default=-1)
    score_multioutput: str = field(default='raw_values')
    scoring: str = field(default='neg_mean_absolute_error')
    return_train_score: bool = field(default=True)
    pipeline_transform: typing.Union[str, list, tuple] = field(default=None)
    pipeline_memory: str = field(default=None)
    params: typing.Union[dict, list] = field(default=None)
    target_index: int = field(default=None)
    chain_order: list = field(default=None)
    stacking_options: dict = field(default=None)
    base_boosting_options: dict = field(default=None)

    def __post_init__(self):
        self._validate_regressor_options()
        self._get_regressor()

    def _validate_regressor_options(self):
        self.regressor_choice = _check_estimator_choice(estimator_choice=self.regressor_choice,
                                                        estimator_type='regression')

        assert isinstance(self.cv, int) and self.cv > 1
        assert isinstance(self.random_state, int) and self.random_state >= 0
        assert isinstance(self.verbose, int) and self.verbose >= 0
        assert isinstance(self.n_jobs, int)
        assert isinstance(self.score_multioutput, str)
        assert isinstance(self.scoring, str)
        assert isinstance(self.return_train_score, bool)

        if self.pipeline_transform is not None:
            assert any(isinstance(self.pipeline_transform, built_in)
                   for built_in in (str, list, tuple))

        if self.pipeline_memory is not None:
            assert isinstance(self.pipeline_memory, bool)

        if self.params is not None:
            assert isinstance(self.params, (dict, list))

        if self.target_index is not None:
            assert isinstance(self.target_index, int)

        if self.chain_order is not None:
            assert isinstance(self.chain_order, list)

        if self.stacking_options is not None:
            for key, option in self.stacking_options.items():
                if key == 'layers':
                    self.stacking_options[key] = _check_stacking_layer(stacking_layer=option,
                                                                       estimator_type='regression')
                elif key not in ['shuffle', 'refit', 'passthrough', 'meta_features']:
                    raise KeyError('The key: %s is not a stacking option.'
                                   % (key))

        if self.base_boosting_options is not None:
            # The options are checked in the
            # ModifiedPipeline constructor.
            assert isinstance(self.base_boosting_options, dict)

    def _get_regressor(self):
        """Helper method which instantiates the regressor choice."""

        reg = RegressorDictionaryInterface(regressor_choice=self.regressor_choice,
                                           params=self.params,
                                           stacking_options=self.stacking_options)

        kwargs = dict(cv=self.cv,
                      verbose=self.verbose,
                      random_state=self.random_state,
                      n_jobs=self.n_jobs,
                      stacking_options=self.stacking_options)

        # The (hyper)parameters must be set
        # before retrieval.
        self._regressor = reg.set_params(**kwargs)
        self.params = reg.get_params(regressor=self._regressor)

    @property
    def check_regressor(self):
        """Checks if regressor adheres to scikit-learn conventions.

        Namely, it runs :class:`sklearn.utils.estimator_checks.check_estimator`.
        """

        return sklearn.utils.estimator_checks.check_estimator(self._regressor)

    def get_params(self, deep=True) -> dict:
        """Retrieves the (hyper)parameters.

        Parameters
        ----------
        deep : bool, optional (default=True)
            Although we do not use this parameter, it is required as
            various Scikit-learn utilities require it.

        Returns
        -------
        self.params : dict
            (Hyper)parameter names mapped to their values.
        """

        return self.params

    def set_params(self, **params) -> BaseRegressor:
        """Sets the regressor's (hyper)parameters.

        Parameters
        ----------
        **params : dict
            The regressor's (hyper)parameters.

        Returns
        -------
        self : BaseRegressor
            The base regressor object.
        """

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
                                 'with `regressor.get_params().keys()`.'
                                 % (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self._regressor, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def _validate_data(self, X=None, y=None):
        """Checks the validity of the data representation(s).

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
        out : validated data
        """

        if X is not None and y is not None:
            if not hasattr(self, '_validated_data'):
                out = _validate_data(X=X, y=y)
                setattr(self, '_validated_data', True)
            else:
                out = X, y
        elif X is not None:
            if not hasattr(self, '_validated_data'):
                out = _validate_data(X=X)
            else:
                out = X
        elif y is not None:
            if not hasattr(self, '_validated_data'):
                out = _validate_data(y=y)
            else:
                out = y
        else:
            raise ValueError('Both the data matrix X and the target matrix y are None. '
                             'Thus, there is no data to validate.')

        return out


    def dump(self, value, filename) -> list:
        """Serializes the value with joblib.

        Parameters
        ----------
        value: any Python object
            The object to store to disk.

        filename : str, joblib.pathlib.Path, or file object
            The file object or path of the file.

        Returns
        -------
        filenames: list of str
            The list of file names in which the data is stored.
        """

        assert isinstance(filename, str)
        joblib.dump(value=value, filename=filename)

    def load(self, filename):
        """Deserializes the file object.

        Parameters
        ----------
        filename : str, joblib.pathlib.Path, or file object
            The file object or path of the file.

        Returns
        -------
        joblib.load : any Python object
            The object stored in the file.
        """

        assert isinstance(filename, str)        
        return joblib.load(filename=filename)
    
    def get_pipeline(self, y: DataFrame_or_Series, n_quantiles=None):
        """Creates pipe attribute for downstream tasks.

        This method constructs a ModifiedPipeline from the given base regressor.

        Parameters
        ----------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s). The targets are used to
            determine the type of the target, and the number of samples if the
            ``pipeline_transform`` involves quantile transformers.

        n_quantiles : int or None, optional (default=None)
            Number of quantiles in :class:`sklearn.preprocessing.QuantileTransformer`, if
            ``pipeline_transform`` is either ```quantileuniform``` or ```quantilenormal```.

        Attributes
        ----------
        pipe : :class:`physlearn.pipeline.ModifiedPipeline`
            A ModifiedPipeline object.
        """

        y = self._validate_data(y=y)

        if n_quantiles is None and isinstance(self.pipeline_transform, str):
            if re.search('quantile', self.pipeline_transform):
                n_quantiles = _n_samples(y)

        kwargs = dict(random_state=self.random_state,
                      verbose=self.verbose,
                      n_jobs=self.n_jobs,
                      cv=self.cv,
                      memory=self.pipeline_memory,
                      target_index=self.target_index,
                      target_type = sklearn.utils.multiclass.type_of_target(y),
                      n_quantiles=n_quantiles,
                      chain_order=self.chain_order,
                      base_boosting_options=self.base_boosting_options)

        self.pipe =  make_pipeline(estimator=self._regressor,
                                   transform=self.pipeline_transform,
                                   **kwargs)

    def regattr(self, attr: str) -> str:
        """Gets a regressor's attribute from the ModifiedPipeline object.

        The pipe attribute must exist in order to use this method. 

        Parameters
        ----------
        attr : str
            The name of the regressor's attribute.

        Returns
        -------
        attr : type of attribute
        """

        assert hasattr(self, 'pipe') and isinstance(attr, str)

        try:
            attr = {f'target {index}': getattr(self.pipe, attr)
                   for index, self.pipe
                   in enumerate(self.pipe.named_steps['reg'].estimators_)}
            return attr
        except:
            raise AttributeError('%s needs to have an estimators_ attribute '
                                 'in order to access the attribute: %s.'
                                 % (self.pipe.named_steps['reg'], attr))

    def _check_target_index(self, y: DataFrame_or_Series) -> DataFrame_or_Series:
        """Automates subtask slicing in multi-target regression.

        Parameters
        ----------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s). The targets are used to
            determine the type of the target, and the number of samples if the
            ``pipeline_transform`` involves quantile transformers.

        Returns
        -------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
        """

        y = self._validate_data(y=y)

        if self.target_index is not None and \
        sklearn.utils.multiclass.type_of_target(y) in _MULTI_TARGET:
            # Selects a particular single-target
            return y.iloc[:, self.target_index]
        else:
            return y

    @staticmethod
    def _fit(regressor, X: DataFrame_or_Series, y: DataFrame_or_Series,
             sample_weight=None, **fit_params):
        """Helper fit method.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each example. If the weight is a float, then
            every example will have the same weight.

        **fit_params : dict of string -> object
            If base boosting, then these parameters are passed to the stagewise
            ``_fit_stages`` method.
        """

        if sample_weight is not None:
            try:
                regressor.fit(X=X, y=y, sample_weight=sample_weight)
            except TypeError as exc:
                if 'unexpected keyword argument sample_weight' in str(exc):
                    raise TypeError('%s does not support sample weights.'
                                    % (regressor.__class__.__name__)) from exc
        elif sample_weight is None and fit_params:
            try:
                regressor.fit(X=X, y=y, **fit_params)
            except ValueError:
                raise ('%s is not a valid fit parameter for this regressor.'
                       % (fit_params.values()))
        else:
            regressor.fit(X=X, y=y)

    def fit(self, X: DataFrame_or_Series, y: DataFrame_or_Series,
            sample_weight=None) -> ModifiedPipeline:
        """Fits the ModifiedPipeline object.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each example. If the weight is a float, then
            every example will have the same weight.

        Returns
        -------
        self.pipe : ModifiedPipeline
            The induced pipeline object.
        """

        X, y = self._validate_data(X=X, y=y)

        # Automates single-target slicing.
        y = self._check_target_index(y=y)

        if not hasattr(self, 'pipe'):
            self.get_pipeline(y=y)

        self._fit(regressor=self.pipe, X=X, y=y,
                  sample_weight=sample_weight)

        return self.pipe

    def predict(self, X: DataFrame_or_Series) -> DataFrame_or_Series:
        """Generates predictions with the ModifiedPipeline object.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        Returns
        -------
        y_pred : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The predictions generated by the induced ModifiedPipeline object.
        """

        assert hasattr(self, 'pipe')
        X = self._validate_data(X=X)

        return self.pipe.predict(X=X)

    def score(self, y_true: pandas_or_numpy, y_pred: pandas_or_numpy, scoring='mse',
              multioutput='raw_values') -> pandas_or_numpy:
        """Computes the supervised score.

        Parameters
        ----------
        y_true : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The observed target matrix, where each row corresponds to an example and the
            column(s) correspond to the observed single-target(s).

        y_pred : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The predicted target matrix, where each row corresponds to an example and the
            column(s) correspond to the predicted single-target(s).

        scoring : str, optional (default='mse')
            The scoring name, which may be `mae`, `mse`, `rmse`, `r2`, `ev`, or
            `msle`.

        multioutput : str, optional (default='raw_values')
            Defines aggregating of multiple output values, wherein the string
            must be either ``'raw_values'``, ``'uniform_average'``, or
            ``'variance_weighted'``.

        Returns
        -------
        score : float or ndarray of floats
            The computed score.
        """

        assert any(scoring for method in _SCORE_CHOICE) and isinstance(scoring, str)

        if scoring in ['r2', 'ev']:
            possible_multioutputs = _SCORE_MULTIOUTPUT + ['variance_weighted']
            assert any(multioutput for output in possible_multioutputs)
        else:
            possible_multioutputs = _SCORE_MULTIOUTPUT
            assert any(multioutput for output in possible_multioutputs)

        # Automates single-target slicing
        y_true = self._check_target_index(y=y_true)

        if scoring == 'mae':
            score = sklearn.metrics.mean_absolute_error(y_true=y_true,
                                                        y_pred=y_pred,
                                                        multioutput=multioutput)
        elif scoring == 'mse':
            score = sklearn.metrics.mean_squared_error(y_true=y_true,
                                                       y_pred=y_pred,
                                                       multioutput=multioutput)
        elif scoring == 'rmse':
            score = np.sqrt(sklearn.metrics.mean_squared_error(y_true=y_true,
                                                               y_pred=y_pred,
                                                               multioutput=multioutput))
        elif scoring == 'r2':
            score = sklearn.metrics.r2_score(y_true=y_true,
                                             y_pred=y_pred,
                                             multioutput=multioutput)
        elif scoring == 'ev':
            score = sklearn.metrics.explained_variance_score(y_true=y_true,
                                                             y_pred=y_pred,
                                                             multioutput=multioutput)
        elif scoring == 'msle':
            try:
                score = sklearn.metrics.mean_squared_log_error(y_true=y_true,
                                                               y_pred=y_pred,
                                                               multioutput=multioutput)
            except:
                # Sklearn will raise a ValueError if either
                # statement is true, so we circumvent
                # this error and score with a NaN.
                score = np.nan

        return score

    def _estimate_fold_size(self, y: DataFrame_or_Series, cv) -> int:
        """Helper method to estimate cross-validation fold size.

        Parameters
        ----------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        cv : int, cross-validation generator, or an iterable
            Used in order to determine the fold size.

        Returns
        -------
        estimate : int
        """

        n_samples = _n_samples(y)
        if isinstance(cv, int):
            fold_size =  np.full(shape=n_samples,
                                 fill_value=n_samples // cv,
                                 dtype=np.int)
        else:
            fold_size =  np.full(shape=n_samples,
                                 fill_value=n_samples // cv.n_splits,
                                 dtype=np.int)
        return n_samples - (np.max(fold_size) + 1)

    def _modified_cross_validate(self, X: DataFrame_or_Series, y: DataFrame_or_Series,
                                 return_regressor=False, error_score=np.nan,
                                 return_incumbent_score=False, cv=None,
                                 fit_params=None) -> dict:
        """Performs (augmented) cross-validation.

        If ``return_incumbent_score`` is True, then the incumbent is scored
        on the withheld folds. Otherwise, the behavior is the same as in
        Scikit-learn.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        return_regressor : bool, optional (default=False)
            Determines whether to return the induced regressor.

        error_score : 'raise' or numeric, optional (default=np.nan)
            The assigned value if an error occurs while inducing a regressor.
            If set to 'raise', then the specific error is raised. Else if set
            to a numeric value, then FitFailedWarning is raised.

        return_incumbent_score : bool, optional (default=True)
            Determines whether to score the incumbent on the withheld folds,
            whereby the incumbent is assumed to be an example in the design
            matrix.

        cv : int, cross-validation generator, an iterable, or None, optional (default=None)
            Determines the cross-validation strategy. If None, then the default
            is 5-fold cross-validation.

        fit_params : dict, optional (default=None)
            (Hyper)parameters to pass to the regressor's fit method.

        Returns
        -------
        scores : dict of float arrays of shape (n_splits,)
            Array of scores for each run of the cross-validation procedure.

        References
        ----------
        - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
          "Boosting on the shoulders of giants in quantum device calibration",
          arXiv preprint arXiv:2005.06194 (2020).
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

        cv = sklearn.model_selection._split.check_cv(cv=cv, y=y,
                                                     classifier=False)

        if isinstance(self.scoring, str):
            scorers = sklearn.metrics._scorer.check_scoring(estimator=self.pipe,
                                                            scoring=self.scoring)
        else:
            scorers, _ = sklearn.metrics._scorer._check_multimetric_scoring(estimator=self.pipe,
                                                                            scoring=self.scoring)

        parallel = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                                   pre_dispatch='2*n_jobs')

        results = parallel(
            joblib.delayed(sklearn.model_selection._validation._fit_and_score)(
                estimator=sklearn.base.clone(self.pipe), X=X, y=y,
                scorer=scorers, train=train, test=test, verbose=self.verbose,
                parameters=None, fit_params=fit_params,
                return_train_score=self.return_train_score,
                return_parameters=False, return_n_test_samples=False,
                return_times=True, return_estimator=return_regressor,
                error_score=np.nan)
            for train, test in cv.split(X, y, groups))

        if return_incumbent_score:
            if self.target_index is not None:
                y_pred = X.iloc[:, self.target_index]
            else:
                y_pred = X

            incumbent_test_score = parallel(
                joblib.delayed(self.score)(
                    y_true=y.loc[test], y_pred=y_pred.loc[test])
                for _, test in cv.split(X, y, groups))

            if self.scoring == 'neg_mean_absolute_error':
                incumbent_test_score = [score['mae'].values[0] for score in incumbent_test_score]
            elif self.scoring == 'neg_mean_squared_error':
                incumbent_test_score = [score['mse'].values[0] for score in incumbent_test_score]


        results = sklearn.model_selection._validation._aggregate_score_dicts(results)

        ret = {}
        ret['fit_time'] = results["fit_time"]
        ret['score_time'] = results["score_time"]

        if return_regressor:
            ret['regressor'] = results["estimator"]

        test_scores_dict = sklearn.model_selection._validation._normalize_score_results(
            results["test_scores"])

        if self.return_train_score:
            train_scores_dict = sklearn.model_selection._validation._normalize_score_results(
                results["train_scores"])

        for name in test_scores_dict:
            ret['test_%s' % name] = test_scores_dict[name]
            if self.return_train_score:
                key = 'train_%s' % name
                ret[key] = train_scores_dict[name]

        if return_incumbent_score:
            ret['incumbent_test_score'] = incumbent_test_score

        return ret

    def cross_validate(self, X: DataFrame_or_Series, y: DataFrame_or_Series,
                       return_regressor=False, error_score=np.nan,
                       return_incumbent_score=False, cv=None,
                       fit_params=None) -> pd.DataFrame:
        """Performs (augmented) cross-validation, and wraps the result in a DataFrame.

        If ``return_incumbent_score`` is True, then the incumbent is scored
        on the withheld folds. Otherwise, the behavior is the same as in
        Scikit-learn.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        return_regressor : bool, optional (default=False)
            Determines whether to return the induced regressor.

        error_score : 'raise' or numeric, optional (default=np.nan)
            The assigned value if an error occurs while inducing a regressor.
            If set to 'raise', then the specific error is raised. Else if set
            to a numeric value, then FitFailedWarning is raised.

        return_incumbent_score : bool, optional (default=True)
            Determines whether to score the incumbent on the withheld folds,
            whereby the incumbent is assumed to be an example in the design
            matrix.

        cv : int, cross-validation generator, an iterable, or None, optional (default=None)
            Determines the cross-validation strategy. If None, then the default
            is 5-fold cross-validation.

        fit_params : dict, optional (default=None)
            (Hyper)parameters to pass to the regressor's fit method.

        Returns
        -------
        scores : pd.DataFrame
            DataFrame of scores for each run of the cross-validation procedure.

        Notes
        -----
        Scikit-learn returns negative scores for some metrics, such as
        mean absolute error (MAE) or mean squared error (MSE). However,
        we only return nonnegativie scores.

        References
        ----------
        - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
          "Boosting on the shoulders of giants in quantum device calibration",
          arXiv preprint arXiv:2005.06194 (2020).
        """

        scores = self._modified_cross_validate(X=X, y=y,
                                               return_regressor=return_regressor,
                                               error_score=error_score,
                                               return_incumbent_score=return_incumbent_score,
                                               cv=cv, fit_params=fit_params)

        if re.match('neg', self.scoring):
            scores['train_score'] = np.abs(scores['train_score'])
            scores['test_score'] = np.abs(scores['test_score'])

        return pd.DataFrame(scores)

    def cross_val_score(self, X: DataFrame_or_Series, y: DataFrame_or_Series,
                        error_score=np.nan, return_incumbent_score=False,
                        cv=None, fit_params=None) -> DataFrame_or_Series:
        """Performs (augmented) cross-validation, then returns the withheld fold score.

        If ``return_incumbent_score`` is True, then the incumbent is scored
        on the withheld folds. Otherwise, the behavior is the same as in
        Scikit-learn.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        error_score : 'raise' or numeric, optional (default=np.nan)
            The assigned value if an error occurs while inducing a regressor.
            If set to 'raise', then the specific error is raised. Else if set
            to a numeric value, then FitFailedWarning is raised.

        return_incumbent_score : bool, optional (default=True)
            Determines whether to score the incumbent on the withheld folds,
            whereby the incumbent is assumed to be an example in the design
            matrix.

        cv : int, cross-validation generator, an iterable, or None, optional (default=None)
            Determines the cross-validation strategy. If None, then the default
            is 5-fold cross-validation.

        fit_params : dict, optional (default=None)
            (Hyper)parameters to pass to the regressor's fit method.

        Returns
        -------
        scores : pd.Series or pd.DataFrame
            The withheld fold scores for each run of the cross-validation procedure.

        Notes
        -----
        Scikit-learn returns negative scores for some metrics, such as
        mean absolute error (MAE) or mean squared error (MSE). However,
        we only return nonnegativie scores.

        References
        ----------
        - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
          "Boosting on the shoulders of giants in quantum device calibration",
          arXiv preprint arXiv:2005.06194 (2020).
        """

        scores = self.cross_validate(X=X, y=y,
                                     error_score=error_score,
                                     return_incumbent_score=return_incumbent_score,
                                     cv=cv, fit_params=fit_params)

        if return_incumbent_score:
            return scores[['test_score', 'incumbent_test_score']]
        else:
            return scores['test_score']


@dataclass
class Regressor(BaseRegressor):
    """Main class for regressor amalgamation.

    The object is designed to amalgamate regressors from 
    `Scikit-learn <https://scikit-learn.org/>`_,
    `LightGBM <https://lightgbm.readthedocs.io/en/latest/index.html>`_,
    `XGBoost <https://xgboost.readthedocs.io/en/latest/>`_,
    `CatBoost <https://catboost.ai/>`_,
    and `Mlxtend <http://rasbt.github.io/mlxtend/>`_ into a unified framework,
    which follows the Scikit-learn API. Important methods include ``fit``,
    ``predict``, ``score``, ``baseboostcv``, ``search``, ``dump``, ``load``,
    ``cross_val_score``, and ``nested_cross_validate``.

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

    verbose : int, optional (default=1)
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

    refit : bool, optional (default=True)
        Determines whether to return the refit regressor in the search method.

    randomizedcv_n_iter : int, optional (default=20)
        Determines the number of (hyper)parameter settings that are
        sampled in the search method, when the chosen search is
        ``'randomizedsearchcv'``, e.g., RandomizedSearchCV from
        Scikit-learn.

    bayesoptcv_init_points : int, optional (default=2)
        Determines the number of random exploration steps in the search method,
        when the chose search method is ``'bayesoptcv'``, e.g., `Bayesian
        Optimization <https://github.com/fmfn/BayesianOptimization>`_.
        Increasing the number corresponds to diversifying the exploration
        space.

    bayesoptcv_n_iter : int, optional (default=20)
        Determines the number of Bayesian optimization steps in the search method,
        when the chose search method is ``'bayesoptcv'``, e.g., `Bayesian
        Optimization <https://github.com/fmfn/BayesianOptimization>`_.

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

    Notes
    -----
    The ``score`` method differs from the Scikit-learn usage, as the method is designed
    to abstract the regressor metrics, e.g., :class:`sklearn.metrics.mean_absolute_error`.
    Moreover, it computes multiple metrics, and returns the scores in a pandas object.

    See Also
    --------
    :class:`physlearn.pipeline.ModifiedPipeline` : Class for creating a pipeline.
    :class:`physlearn.supervised.regression.BaseRegressor` : Base class for regressor amalgamation.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.pipeline import FeatureUnion
    >>> from physlearn import Regressor
    >>> X, y = load_boston(return_X_y=True)
    >>> X, y = pd.DataFrame(X), pd.Series(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)
    >>> transformer_list = [('pca', PCA(n_components=1)),
                            ('svd', TruncatedSVD(n_components=2))]
    >>> union = FeatureUnion(transformer_list=transformer_list, n_jobs=-1)
    >>> stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
    >>> reg = Regressor(regressor_choice='stackingregressor',
                        pipeline_transform=('tr', union),
                        stacking_options=dict(layers=stack))
    >>> y_pred = reg.fit(X_train, y_train).predict(X_test)
    >>> reg.score(y_test, y_pred)
                 mae        mse      rmse        r2       ev      msle
    target
    0       4.775145  42.874253  6.547843  0.387748  0.40836  0.079818
    """

    verbose: int = field(default=1)
    pipeline_transform: str_list_or_tuple = field(default='quantilenormal')
    refit: bool = field(default=True)
    randomizedcv_n_iter: int = field(default=20)
    bayesoptcv_init_points: int = field(default=2)
    bayesoptcv_n_iter: int = field(default=20)

    def __post_init__(self):
        self._validate_regressor_options()
        self._validate_search_options()
        self._get_regressor()

    def _validate_search_options(self):
        assert isinstance(self.refit, bool)
        assert isinstance(self.randomizedcv_n_iter, int)
        assert isinstance(self.bayesoptcv_init_points, int)
        assert isinstance(self.bayesoptcv_n_iter, int)

    @property
    def check_regressor(self):
        """Checks if regressor adheres to scikit-learn conventions.

        Namely, it runs :class:`sklearn.utils.estimator_checks.check_estimator`.
        Scikit-learn and Mlxtend stacking regressors, as well as LightGBM,
        XGBoost, and CatBoost regressor do not adhere to the convention.
        """
        try:
            super().check_regressor
        except:
            raise TypeError('%s does not adhere to the Scikit-learn estimator convention.'
                            % (_REGRESSOR_DICT[self.regressor_choice]))

    def get_params(self, deep=True) -> dict:
        """Retrieves the (hyper)parameters.

        Parameters
        ----------
        deep : bool, optional (default=True)
            Although we do not use this parameter, it is required as
            various Scikit-learn utilities require it.

        Returns
        -------
        self.params : dict
            (Hyper)parameter names mapped to their values.
        """

        return super().get_params(deep=deep)

    def set_params(self, **params) -> BaseRegressor:
        """Sets the regressor's (hyper)parameters.

        Parameters
        ----------
        **params : dict
            The regressor's (hyper)parameters.

        Returns
        -------
        self : BaseRegressor
            The base regressor object.
        """

        return super().set_params(**params)

    def dump(self, value, filename) -> list:
        """Serializes the value with joblib.

        Parameters
        ----------
        value: any Python object
            The object to store to disk.

        filename : str, joblib.pathlib.Path, or file object
            The file object or path of the file.

        Returns
        -------
        filenames: list of str
            The list of file names in which the data is stored.
        """

        super().dump(value=value, filename=filename)

    def load(self, filename):
        """Deserializes the file object.

        Parameters
        ----------
        filename : str, joblib.pathlib.Path, or file object
            The file object or path of the file.

        Returns
        -------
        joblib.load : any Python object
            The object stored in the file.
        """

        return super().load(filename=filename)

    def regattr(self, attr: str) -> str:
        """Gets a regressor's attribute from the ModifiedPipeline object.

        The pipe attribute must exist in order to use this method. 

        Parameters
        ----------
        attr : str
            The name of the regressor's attribute.

        Returns
        -------
        attr : type of attribute
        """

        return super().regattr(attr=attr)

    def fit(self, X: DataFrame_or_Series, y: DataFrame_or_Series,
            sample_weight=None) -> ModifiedPipeline:
        """Fits the ModifiedPipeline object.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        sample_weight : float, ndarray, or None, optional (default=None)
            Individual weights for each example. If the weight is a float, then
            every example will have the same weight.

        Returns
        -------
        self.pipe : ModifiedPipeline
            The induced pipeline object.
        """

        return super().fit(X=X, y=y, sample_weight=sample_weight)

    def _inbuilt_model_selection_step(self, X: DataFrame_or_Series,
                                      y: DataFrame_or_Series) -> None:
        """Performs augmented cross-validation.

        This method is designed to be utilized within
        :meth:`physlearn.supervised.regression.Regressor.baseboostcv`,
        as the inbuilt model selection step.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        Attributes
        ----------
        _return_incumbent : bool
            This flag implies that the incumbent won the inbuilt model
            selection step.

        Returns
        -------
        None

        References
        ----------
        - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
          "Boosting on the shoulders of giants in quantum device calibration",
          arXiv preprint arXiv:2005.06194 (2020).
        """

        cross_val_score = super().cross_val_score(X=X, y=y,
                                                  return_incumbent_score=True)
        mean_cross_val_score = cross_val_score.mean(axis=0)

        if mean_cross_val_score[0] >= mean_cross_val_score[1]:
            # Base boosting did not improve performance.
            setattr(self, '_return_incumbent', True)

    def baseboostcv(self, X: DataFrame_or_Series, y: DataFrame_or_Series,
                    **fit_params) -> typing.Union[Regressor, ModifiedPipeline]:
        """Base boosting with inbuilt cross-validation.

        This method starts with inbuilt cross-validation, which scores both
        the incumbent and the candidate base boosting algorithm. If the
        incumbent wins, then the explict model of the domain is the single-target
        regressor. Otherwise, base boosting greedily boosts the explict model of
        the domain in a stagewise fashion.

        In essence, this method acts as a fit method.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        **fit_params : dict of string -> object
            If base boosting, then these parameters are passed to the stagewise
            ``_fit_stages`` method.

        Attributes
        ----------
        return_incumbent_ : bool
            This flag implies that the incumbent won the inbuilt model
            selection step, and it notifies the predict method.

        Returns
        -------
        single-target regressor : Regressor or ModifiedPipeline

        References
        ----------
        - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
          "Boosting on the shoulders of giants in quantum device calibration",
          arXiv preprint arXiv:2005.06194 (2020).
        """

        X, y = super()._validate_data(X=X, y=y)

        # Automates single-target slicing
        y = super()._check_target_index(y=y)

        # Performs augmented k-fold cross-validation, then it
        # selects either the incumbent or the candidate.
        self._inbuilt_model_selection_step(X=X, y=y)

        if not hasattr(self, 'pipe'):
            super().get_pipeline(y=y)

        if not hasattr(self, '_return_incumbent'):
            # This checks if the candidate was chosen
            # in model selection.
            super()._fit(regressor=self.pipe, X=X, y=y, **fit_params)
            return self.pipe
        else:
            setattr(self, 'return_incumbent_', True) 
            return self

    def predict(self, X: DataFrame_or_Series) -> pd.DataFrame:
        """Generates predictions with the ModifiedPipeline object.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        Returns
        -------
        y_pred : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The predictions generated by the induced ModifiedPipeline object.

        References
        ----------
        - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
          "Boosting on the shoulders of giants in quantum device calibration",
          arXiv preprint arXiv:2005.06194 (2020).
        """

        X = self._validate_data(X=X)

        if hasattr(self, 'return_incumbent_'):
            # This checks if the incumbent was chosen in the
            # inbuilt model selection of base boosting with
            # augmented cross-validation.
            if self.target_index is not None:
                y_pred = X.iloc[:, self.target_index]
            else:
                y_pred = X
        else:
            assert hasattr(self, 'pipe')
            y_pred = self.pipe.predict(X=X)

        return y_pred

    def score(self, y_true: DataFrame_or_Series, y_pred: DataFrame_or_Series,
              path=None) -> pd.DataFrame:
        """Computes the DataFrame of supervised scores.

        The scoring metrics include mean squared error, mean absolute error,
        root mean squared error, R^2, explained variance, and mean squared
        logarithmic error. If the observed or predicted single-targets contain
        negative values, then the mean squared logarithmic error is not included,
        as the score is considered a NaN.

        Parameters
        ----------
        y_true : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The observed target matrix, where each row corresponds to an example
            and the column(s) correspond to the observed single-target(s).

        y_pred : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The predicted target matrix, where each row corresponds to an example
            and the column(s) correspond to the predicted single-target(s).

        path : str or file handle, optional (default=None)
            The file path or object, if the scoring DataFrame is to be saved
            to a comma-seperated values (csv) file.

        Returns
        -------
        scores : pd.DataFrame or pd.Series
            The pandas object of computed scores.
        """

        assert any(self.score_multioutput for output in _SCORE_MULTIOUTPUT)

        scores = {}
        for scoring in _SCORE_CHOICE:
            scores[scoring] = super().score(y_true=y_true,
                                            y_pred=y_pred,
                                            scoring=scoring,
                                            multioutput=self.score_multioutput)

        if self.score_multioutput == 'raw_values':
            scores = pd.DataFrame(scores).dropna(how='any', axis=1)
            scores.index.name = 'target'
            
            # Shifts the index origin by one.
            if self.target_index is not None:
                scores.index = pd.RangeIndex(start=self.target_index + 1,
                                             stop=self.target_index + 2,
                                             step=1)
        else:
            scores = pd.Series(scores).dropna(how='any', axis=0)

        if path is not None:
            assert isinstance(path, str)
            scores.to_csv(path_or_buf=path)

        return scores

    def cross_validate(self, X: DataFrame_or_Series, y: DataFrame_or_Series,
                       return_regressor=False, error_score=np.nan,
                       return_incumbent_score=False, cv=None,
                       fit_params=None) -> pd.DataFrame:
        """Performs (augmented) cross-validation, and wraps the result in a DataFrame.

        If ``return_incumbent_score`` is True, then the incumbent is scored
        on the withheld folds. Otherwise, the behavior is the same as in
        Scikit-learn.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        return_regressor : bool, optional (default=False)
            Determines whether to return the induced regressor.

        error_score : 'raise' or numeric, optional (default=np.nan)
            The assigned value if an error occurs while inducing a regressor.
            If set to 'raise', then the specific error is raised. Else if set
            to a numeric value, then FitFailedWarning is raised.

        return_incumbent_score : bool, optional (default=True)
            Determines whether to score the incumbent on the withheld folds,
            whereby the incumbent is assumed to be an example in the design
            matrix.

        cv : int, cross-validation generator, an iterable, or None, optional (default=None)
            Determines the cross-validation strategy. If None, then the default
            is 5-fold cross-validation.

        fit_params : dict, optional (default=None)
            (Hyper)parameters to pass to the regressor's fit method.

        Returns
        -------
        scores : pd.DataFrame
            DataFrame of scores for each run of the cross-validation procedure.

        Notes
        -----
        Scikit-learn returns negative scores for some metrics, such as
        mean absolute error (MAE) or mean squared error (MSE). However,
        we only return nonnegativie scores.

        References
        ----------
        - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
          "Boosting on the shoulders of giants in quantum device calibration",
          arXiv preprint arXiv:2005.06194 (2020).
        """

        return super().cross_validate(X=X, y=y,
                                       return_regressor=return_regressor,
                                       error_score=error_score,
                                       return_incumbent_score=return_incumbent_score,
                                       cv=cv,
                                       fit_params=fit_params)

    def cross_val_score(self, X: DataFrame_or_Series, y: DataFrame_or_Series,
                        error_score=np.nan, return_incumbent_score=False,
                        cv=None, fit_params=None) -> DataFrame_or_Series:
        """Performs (augmented) cross-validation, then returns the withheld fold score.

        If ``return_incumbent_score`` is True, then the incumbent is scored
        on the withheld folds. Otherwise, the behavior is the same as in
        Scikit-learn.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        error_score : 'raise' or numeric, optional (default=np.nan)
            The assigned value if an error occurs while inducing a regressor.
            If set to 'raise', then the specific error is raised. Else if set
            to a numeric value, then FitFailedWarning is raised.

        return_incumbent_score : bool, optional (default=True)
            Determines whether to score the incumbent on the withheld folds,
            whereby the incumbent is assumed to be an example in the design
            matrix.

        cv : int, cross-validation generator, an iterable, or None, optional (default=None)
            Determines the cross-validation strategy. If None, then the default
            is 5-fold cross-validation.

        fit_params : dict, optional (default=None)
            (Hyper)parameters to pass to the regressor's fit method.

        Returns
        -------
        scores : pd.Series or pd.DataFrame
            The withheld fold scores for each run of the cross-validation procedure.

        Notes
        -----
        Scikit-learn returns negative scores for some metrics, such as
        mean absolute error (MAE) or mean squared error (MSE). However,
        we only return nonnegativie scores.

        References
        ----------
        - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
          "Boosting on the shoulders of giants in quantum device calibration",
          arXiv preprint arXiv:2005.06194 (2020).
        """

        return super().cross_val_score(X=X, y=y,
                                       return_regressor=return_regressor,
                                       error_score=error_score,
                                       return_incumbent_score=return_incumbent_score,
                                       cv=cv,
                                       fit_params=fit_params)

    def _preprocess_search_params(self, y: DataFrame_or_Series, search_params: dict) -> dict:
        """Helper method for preprocessing (hyper)parameters.

        This method automatically preprocesses (hyper)parameter names for the
        exhaustive search method by determining whether the task is single-target
        or multi-target regression. In the latter case, it further determines the
        user's assumption on the single-targets's independence. Namely, it asks if
        the user wishes to chain the single-targets.

        Parameters
        ----------
        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        search_params : dict
            Dictionary with (hyper)parameter names as keys, and either lists of
            (hyper)parameter settings to try as values or tuples of (hyper)parameter
            lower and upper bounds to try as values.

        Returns
        -------
        search_params : dict
            The preprocessed (hyper)parameters.
        """

        if sklearn.utils.multiclass.type_of_target(y) in _MULTI_TARGET:
            if self.chain_order is not None:
                search_params = _preprocess_hyperparams(raw_params=search_params,
                                                        multi_target=True,
                                                        chain=True)
            else:
                search_params = _preprocess_hyperparams(raw_params=search_params,
                                                        multi_target=True,
                                                        chain=False)
        else:
            search_params = _preprocess_hyperparams(raw_params=search_params,
                                                    multi_target=False,
                                                    chain=False)

        return search_params

    def _search(self, X: DataFrame_or_Series, y: DataFrame_or_Series, search_params: dict,
                search_method='gridsearchcv', cv=None) -> None:
        """Helper (hyper)parameter search method.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        search_params : dict
            Dictionary with (hyper)parameter names as keys, and either lists of
            (hyper)parameter settings to try as values or tuples of (hyper)parameter
            lower and upper bounds to try as values.

        search_method : str, optional (default='gridsearchcv')
            Specifies the search method. If ``'gridsearchcv'``, ``'randomizedsearchcv'``,
            or ``'bayesoptcv'`` then the search method is GridSearchCV, RandomizedSearchCV,
            or Bayesian Optimization.

        cv : int, cross-validation generator, an iterable, or None, optional (default=None)
            Determines the cross-validation strategy. If None, then the default
            is 5-fold cross-validation.

        Attributes
        ----------
        _method : GridSearchCV, RandomizedSearchCV, BayesianOptimization
            An instance of the (hyper)parameter search object.
        """

        search_method = _check_search_method(search_method=search_method)
        search_params = self._preprocess_search_params(y=y, search_params=search_params)

        if cv is None:
            cv = self.cv

        if not hasattr(self, 'pipe'):
            self.get_pipeline(y=y,
                              n_quantiles=super()._estimate_fold_size(y=y,
                                                                      cv=cv))

        self._method = _search_method(search_method=search_method,
                                      pipeline=self.pipe,
                                      search_params=search_params,
                                      scoring=self.scoring,
                                      refit=self.refit,
                                      n_jobs=self.n_jobs,
                                      cv=cv,
                                      verbose=self.verbose,
                                      pre_dispatch='2*n_jobs',
                                      error_score=np.nan,
                                      return_train_score=self.return_train_score,
                                      randomizedcv_n_iter=self.randomizedcv_n_iter,
                                      X=X, y=y,
                                      init_points=self.bayesoptcv_init_points,
                                      bayesoptcv_n_iter=self.bayesoptcv_n_iter)

    def search(self, X: DataFrame_or_Series, y: DataFrame_or_Series, search_params: dict,
                search_method='gridsearchcv', cv=None, path=None) -> None:
        """(Hyper)parameter search method.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        search_params : dict
            Dictionary with (hyper)parameter names as keys, and either lists of
            (hyper)parameter settings to try as values or tuples of (hyper)parameter
            lower and upper bounds to try as values.

        search_method : str, optional (default='gridsearchcv')
            Specifies the search method. If ``'gridsearchcv'``, ``'randomizedsearchcv'``,
            or ``'bayesoptcv'`` then the search method is GridSearchCV, RandomizedSearchCV,
            or Bayesian Optimization.

        cv : int, cross-validation generator, an iterable, or None, optional (default=None)
            Determines the cross-validation strategy. If None, then the default
            is 5-fold cross-validation.

        path : str or file handle, optional (default=None)
            The file path or object, if the scoring DataFrame is to be saved
            to a comma-seperated values (csv) file.

        Attributes
        ----------
        best_params_ : pd.Series
            The optimal (hyper)parameters.

        best_score_ : pd.Series
            The scores for the optimal (hyper)parameters.

        search_summary_ : pd.DataFrame
            Bundles the ``best_params_``, ``best_score_``, and ``refit_time``
            into one attribute.

        Notes
        -----
        Scikit-learn returns negative scores for some metrics, such as
        mean absolute error (MAE) or mean squared error (MSE). However,
        we only return nonnegativie scores.
        """

        X, y = super()._validate_data(X=X, y=y)

        # Automates single-target slicing.
        y = self._check_target_index(y=y)

        self._search(X=X, y=y, search_params=search_params,
                     search_method=search_method, cv=cv)

        if search_method == 'bayesoptcv' and self.refit:
            self.pipe = sklearn.base.clone(sklearn.base.clone(self.pipe).set_params(
                **_check_bayesoptcv_param_type(pbounds=self._method.max['params'])))
            self.pipe.fit(X=X, y=y)
        else:
            try:
                self._method.fit(X=X, y=y)
            except:
                raise AttributeError('Performing the search requires the '
                                     'attribute: %s. However, the attribute '
                                     'is not set.'
                                     % (_method))

        if search_method in ['gridsearchcv', 'randomizedsearchcv']:
            self.best_params_ = pd.Series(self._method.best_params_)
            self.best_score_ = pd.Series({'best_score': self._method.best_score_})
        elif search_method == 'bayesoptcv':
            try:
                self.best_params_ = pd.Series(self._method.max['params'])
                self.best_score_ = pd.Series({'best_score': self._method.max['target']})
            except:
                raise AttributeError('In order to set the attributes: %s and %s, '
                                     'there must be the attribute: %s.'
                                     % (best_params_, best_score_, optimization))

        if re.match('neg', self.scoring):
            self.best_score_.loc['best_score'] *= -1.0

        self.search_summary_ = pd.concat([self.best_score_, self.best_params_], axis=0)

        _sklearn_list = ['best_estimator_', 'cv_results_', 'refit_time_']
        if all(hasattr(self._method, attr) for attr in _sklearn_list):
            self.pipe = self._method.best_estimator_
            self.best_regressor_ = self._method.best_estimator_
            self.pipe = self._method.best_estimator_
            self.cv_results_ = pd.DataFrame(self._method.cv_results_)
            self.refit_time_ = pd.Series({'refit_time':self._method.refit_time_})
            self.search_summary_ = pd.concat([self.search_summary_, self.refit_time_], axis=0)

        if path is not None:
            assert isinstance(path, str)
            self.search_summary_.to_csv(path_or_buf=path, header=True)

    def _search_and_score(self, pipeline: ModifiedPipeline, X: DataFrame_or_Series,
                          y: DataFrame_or_Series, scorer: dict,
                          train: list, test: list, verbose: int,
                          search_params: dict, search_method='gridsearchcv',
                          cv=None) -> tuple:
        """Helper method for nested cross-validation.

        Exhaustively searches over the specified (hyper)parameters in the inner
        loop then scores the best performing regressor in the outer loop.

        Parameters
        ----------
        pipeline : ModifiedPipeline
            A ModifiedPipeline object.

        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        scorer : dict
            A dict mapping each scorer name to its validated scorer.

        train : list
            A list of indices for the training folds.

        test : list
            A list of indices for the withheld folds.

        verbose : int
            Determines verbosity.

        search_params : dict
            Dictionary with (hyper)parameter names as keys, and either lists of
            (hyper)parameter settings to try as values or tuples of (hyper)parameter
            lower and upper bounds to try as values.

        search_method : str, optional (default='gridsearchcv')
            Specifies the search method. If ``'gridsearchcv'``, ``'randomizedsearchcv'``,
            or ``'bayesoptcv'`` then the search method is GridSearchCV, RandomizedSearchCV,
            or Bayesian Optimization.

        cv : int, cross-validation generator, an iterable, or None, optional (default=None)
            Determines the cross-validation strategy. If None, then the default
            is 5-fold cross-validation.

        Returns
        -------
        score : tuple

        Notes
        -----
        Scikit-learn returns negative scores for some metrics, such as
        mean absolute error (MAE) or mean squared error (MSE). However,
        we only return nonnegativie scores.
        """

        X_train, y_train = sklearn.utils.metaestimators._safe_split(estimator=pipeline,
                                                                    X=X, y=y,
                                                                    indices=train)
        X_test, y_test = sklearn.utils.metaestimators._safe_split(estimator=pipeline,
                                                                  X=X, y=y,
                                                                  indices=test,
                                                                  train_indices=train)

        self.search(X=X_train, y=y_train, search_params=search_params,
                    search_method=search_method, cv=cv)

        if not self.refit:
            self.pipe = sklearn.base.clone(sklearn.base.clone(self.pipe).set_params(
                **self.best_params_))
            self.pipe._fit(X=X_train, y=y_train)

        test_score = sklearn.model_selection._validation._score(estimator=self.pipe,
                                                                X_test=X_test,
                                                                y_test=y_test,
                                                                scorer=scorer)

        return (self.best_score_.values, test_score)

    def nested_cross_validate(self, X: DataFrame_or_Series, y: DataFrame_or_Series,
                              search_params: dict, search_method='gridsearchcv',
                              outer_cv=None, inner_cv=None,
                              return_inner_loop_score=False) ->typing.Union[pd.Series, tuple]:
        """Performs a nested cross-validation procedure.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        search_params : dict
            Dictionary with (hyper)parameter names as keys, and either lists of
            (hyper)parameter settings to try as values or tuples of (hyper)parameter
            lower and upper bounds to try as values.

        search_method : str, optional (default='gridsearchcv')
            Specifies the search method. If ``'gridsearchcv'``, ``'randomizedsearchcv'``,
            or ``'bayesoptcv'`` then the search method is GridSearchCV, RandomizedSearchCV,
            or Bayesian Optimization.

        outer_cv : int, cross-validation generator, an iterable, or None, optional (default=None)
            Determines the outer loop cross-validation strategy. If None, then the default
            is 5-fold cross-validation.

        inner_cv : int, cross-validation generator, an iterable, or None, optional (default=None)
            Determines the inner loop cross-validation strategy. If None, then the default
            is 5-fold cross-validation.

        return_inner_loop_score : bool, optional (default=False)
            If True, then we return the inner loop score in addition to the
            outer loop score.

        Returns
        -------
        score : pd.Series or tuple

        Notes
        -----
        The procedure does not compute the single best set of (hyper)parameters,
        as each inner loop may return a different set of optimal (hyper)parameters.

        Scikit-learn returns negative scores for some metrics, such as
        mean absolute error (MAE) or mean squared error (MSE). However,
        we only return nonnegativie scores.

        References
        ----------
        Jacques Wainer and Gavin Cawley. "Nested cross-validation when selecting
        classifiers is overzealous for most practical applications," arXiv preprint
        arXiv:1809.09446 (2018).
        """

        X, y = super()._validate_data(X=X, y=y)

        # Automates single-target slicing
        y = self._check_target_index(y=y)

        X, y, groups = sklearn.utils.validation.indexable(X, y, None)

        if outer_cv is None:
            outer_cv = self.cv

        if inner_cv is None:
            inner_cv = self.cv

        if not hasattr(self, 'pipe'):
            self.get_pipeline(y=y,
                              n_quantiles=super()._estimate_fold_size(y=y,
                                                                      cv=outer_cv))

        outer_cv = sklearn.model_selection._split.check_cv(cv=outer_cv, y=y,
                                                           classifier=False)

        scorers, _ = sklearn.metrics._scorer._check_multimetric_scoring(estimator=self.pipe,
                                                                        scoring=self.scoring)

        parallel = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                                   pre_dispatch='2*n_jobs')

        # Parallelized nested cross-validation: the helper method utilizes
        # the search method to select a regressor from the inner loop, then
        # the performance of this regressor is evaluated in the outer loop.
        scores = parallel(
            joblib.delayed(self._search_and_score)(
                pipeline=sklearn.base.clone(self.pipe), X=X, y=y, scorer=scorers,
                train=train, test=test, verbose=self.verbose, search_params=search_params,
                search_method='gridsearchcv', cv=inner_cv)
            for train, test in outer_cv.split(X, y, groups))

        outer_loop_scores = pd.Series([np.abs(pair[1]['score']) for pair in scores])

        if return_inner_loop_score:
            inner_loop_scores = pd.Series(np.concatenate([np.abs(pair[0]) for pair in scores]))
            return outer_loop_scores, inner_loop_scores
        else:
            return outer_loop_scores
        
    def subsample(self, X: DataFrame_or_Series, y: DataFrame_or_Series,
                  subsample_proportion=None) -> tuple: 
        """Subsamples from the design and target matrices.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        subsample_proportion : float or None, optional (default=None)
            Determines the proportion of observations to use in the
            subsampling procedure.

        Returns
        -------
        out : tuple
            A tuple with the X and y data.
        """

        if subsample_proportion is not None:
            assert subsample_proportion > 0 and subsample_proportion < 1
            out = sklearn.utils.resample(X, y, replace=False,
                                         n_samples=int(len(X) * subsample_proportion),
                                         random_state=self.random_state)
        else:
            out = sklearn.utils.resample(X, y, replace=False,
                                         n_samples=len(X),
                                         random_state=self.random_state)

        return out
