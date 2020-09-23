"""
The :mod:`physlearn.pipeline` module enhances the original Scikit-learn
pipeline with an implementation of base boosting. It includes a
:class:`physlearn.pipeline.ModifiedPipeline` class, as well as a
:func:`physlearn.pipeline.make_pipeline` convenience
function.
"""

# Author: Alex Wozniakowski
# License: MIT

from __future__ import annotations

import copy
import joblib
import typing

import numpy as np
import pandas as pd

import scipy.optimize

import sklearn.base
import sklearn.multioutput
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.utils
import sklearn.utils.metaestimators
import sklearn.utils.multiclass

from physlearn.loss import LOSS_FUNCTIONS
from physlearn.supervised.utils._data_checks import _n_targets
from physlearn.supervised.utils._definition import (_CATBOOST_FLAG, _CHAIN_FLAG,
                                                    _MULTI_TARGET,
                                                    _PIPELINE_TRANSFORM_CHOICE,
                                                    _XGBOOST_FLAG,
                                                    _SCORE_CHOICE)
from physlearn.supervised.utils._estimator_checks import _check_line_search_options

DataFrame_or_Series = typing.Union[pd.DataFrame, pd.Series]
pandas_or_numpy = typing.Union[pd.DataFrame, pd.Series, np.ndarray]


class ModifiedPipeline(sklearn.pipeline.Pipeline):
    """Custom pipeline object that supports base boosting.

    The object inherits from the original Scikit-learn Pipeline, thus it is
    designed to sequentially compose a list of named transforms and a final
    estimator into a new estimator. The modification extends this
    functionality such that the composed estimator supports base boosting.
    In other words, the ``base_boosting_options`` parameter enables a user
    to boost an explicit model of the domain by fitting an additive
    expansion, wherein the intercept term is generated by the explicit
    model. As such, the final estimator may be any estimator contained
    in the dictionary of estimators, i.e., the final estimator is not
    restricted to the decision tree hypothesis class.

    Parameters
    ----------
    steps : list
        List of tuples, wherein the preceding tuple(s) (name, transform) are transform(s)
        and the last tuple (name, estimator) is an estimator

    memory : str or object with the joblib.Memory interface, default=None
        Enables fitted transform caching.

    verbose : int
        Determines verbosity.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel.

    target_index : int or None, optional (default=None)
        Specifies the single-target subtask in the multi-target task.

    base_boosting_options : dict or None, optional (default=None)
        A dictionary of base boosting options, wherein the following options
        must be specified:

        n_estimators :obj:`int`
            The number of basis functions in the noise term of the additive expansion.
            Note that this option may also be specified as ``n_regressors``; see the
            example below.

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

    Attributes
    ----------
    named_steps : :class:`sklearn.utils.Bunch`
        Dictionary-like object, wherein the keys are the user provided
        step names and the values are the steps parameters.

    See Also
    --------
    :func:`physlearn.pipeline.make_pipeline` : Convenience function for constructing a modified pipeline.
    :mod:`physlearn.supervised.utils._definition` : Dictionary of final estimator options.

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.preprocessing import StandardScaler
    >>> from physlearn import ModifiedPipeline
    >>> from physlearn.datasets import load_benchmark
    >>> X_train, X_test, y_train, y_test = load_benchmark(return_split=True)
    >>> line_search_options = dict(init_guess=1, opt_method='minimize',
                                   method='Nelder-Mead', tol=1e-7,
                                   options={"maxiter": 10000},
                                   niter=None, T=None, loss='lad',
                                   regularization=0.1)
    >>> base_boosting_options = dict(n_regressors=3, boosting_loss='lad',
                                     line_search_options=line_search_options)
    >>> pipe = ModifiedPipeline(steps=[('scaler', StandardScaler()), ('reg', Ridge())],
                                base_boosting_options=base_boosting_options)
    >>> pipe.fit(X_train, y_train)
    >>> pipe.score(X_test, y_test).round(decimals=2)
        mae    mse  rmse    r2    ev
    0  2.17  10.01  3.16  0.97  0.98
    1  1.17   3.09  1.76  0.99  0.99
    2  0.78   1.20  1.09  1.00  1.00
    3  0.83   1.12  1.06  1.00  1.00
    4  0.99   2.00  1.42  1.00  1.00

    References
    ----------
    - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
      "Boosting on the shoulders of giants in quantum device calibration",
      arXiv preprint arXiv:2005.06194 (2020).

    - John Tukey. "Exploratory Data Analysis", Addison-Wesley (1977).
    
    - Jerome Friedman. "Greedy function approximation: A gradient boosting machine,"
      Annals of Statistics, 29(5):1189–1232 (2001).
    
    - Trevor Hastie, Robert Tibshirani, and Jerome Friedman.
      "The Elements of Statistical Learning", Springer (2009).

    - Lars Buitinck et al.
      "API design for machine learning software: experiences from the scikit-learn project"
      arXiv preprint arXiv:1309.0238 (2013).
    """

    _required_parameters = ['steps']

    def __init__(self, steps: list, memory=None,
                 verbose=False, n_jobs=None,
                 target_index=None,
                 base_boosting_options=None):

        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.target_index = target_index
        self.base_boosting_options = base_boosting_options
        self._validate_steps()

    def _validate_base_boosting_options(self):
        if self.base_boosting_options is not None:
            for key, option in self.base_boosting_options.items():
                if key in ['n_estimators', 'n_regressors']:
                    if option is not None:
                        assert isinstance(option, int) and option > 0
                        self.n_estimators = option
                    else:
                        raise ValueError('The value of key: %s cannot be None. '
                                         'Specify a positive integer for the number '
                                         'of basis functions to fit in the additive '
                                         'expansion.'
                                         % (key))
                elif key == 'boosting_loss':
                    if option is not None:
                        assert option in LOSS_FUNCTIONS
                        self.boosting_loss = option
                    else:
                        raise ValueError('The value of key: %s cannot be None. '
                                         'The choice of loss function is required '
                                         'in the pseudo-residual computation.'
                                         % (key))
                elif key == 'line_search_options':
                    if option is not None:
                        assert isinstance(option, dict)
                        _check_line_search_options(line_search_options=option)
                        self.line_search_options = option
                    else:
                        raise ValueError('The value of key: %s cannot be None. '
                                         'These options determine the line search '
                                         'optimization procedure.'
                                         % (key))
                else:
                    raise KeyError('The key: %s is not a base boosting option.'
                                   % (key))
        else:
            # This attribute instructs the fit method to avoid
            # base boosting. Consequently, the predict method
            # will not generate predictions from the decomposition
            # into a smooth term and a noise term.
            setattr(self, '_default_fit', True)

    @staticmethod
    def line_search(function: typing.Callable[[np.ndarray], float],
                    init_guess: typing.Union[int, float],
                    opt_method: str, method=None, tol=None,
                    options=None, niter=None, T=None) -> np.ndarray:
        """Computes the expansion coefficient in :meth:`physlearn.pipeline.ModifiedPipeline._fit_stages`.

        Parameters
        ----------
        function : callable
            The objective function for the line search.

        init_guess : int, float, or ndarray
            The initial guess for the expansion coefficient.

        opt_method : str
            Choice of optimization method. If ``'minimize'``, then
            :class:`scipy.optimize.minimize`, else if ``'basinhopping'``,
            then :class:`scipy.optimize.basinhopping`.

        method : str or None, optional (default=None)
            The type of solver utilized in the optimization method.

        tol : float or None, optional (default=None)
            The epsilon tolerance for terminating the optimization method.

        options : dict or None, optional (default=None)
            A dictionary of solver options.

        niter : int or None, optional (default=None)
            The number of iterations in basin-hopping.

        T : float or None, optional (default=None)
            The temperature paramter utilized in basin-hopping,
            which determines the accept or reject criterion.

        Returns
        -------
        res.x[0] : float
            The expansion coefficient, i.e., the first element in the solution array.

        Notes
        -----
        The supported optimization methods include: :class:`scipy.optimize.minimize`
        and :class:`scipy.optimize.basinhopping`; see the Scipy optimization
        `documentation <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_ for
        further details.
        """

        if opt_method == 'minimize':
            assert method is not None and tol is not None
            minimize_object = scipy.optimize.minimize(fun=function,
                                                      x0=init_guess,
                                                      method=method,
                                                      tol=tol,
                                                      options=options)            
        elif opt_method == 'basinhopping':
            assert niter is not None and T is not None
            minimize_object = scipy.optimize.basinhopping(func=function,
                                                          x0=init_guess,
                                                          niter=niter,
                                                          T=T)
        else:
            raise ValueError('The optimization method: %s has not been implemented.'
                             % (opt_method))

        return minimize_object.x[0]

    def _fit_stage(self, X: DataFrame_or_Series, pseudo_residual: pandas_or_numpy,
                   **fit_params_last_step) -> None:
        """Induces a basis function, which is a map from the domain to the pseudo-residual space.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        pseudo_residual : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The negative gradient of the loss function.

        **fit_params_last_step : dict of string -> object
            Parameters passed to the estimator's ``fit`` method during the stage.
        """

        for k in range(self.loss.K):
            if self._final_estimator.__class__ in _CHAIN_FLAG:
                self._final_estimator.fit(X=X, Y=pseudo_residual, **fit_params_last_step)
            else:
                self._final_estimator.fit(X=X, y=pseudo_residual, **fit_params_last_step)

    def _fit_stages(self, X: DataFrame_or_Series, y: pandas_or_numpy,
                    init_expansion: pandas_or_numpy,
                    **fit_params_last_step) -> None:
        """Fits the additive expansion in a greedy stagewise fashion.

        This method transfers prior domain knowledge to gradient boosting through the
        ``init_expansion`` parameter, and it is designed to be utilized within
        :meth:`physlearn.pipeline.ModifiedPipeline.fit`. The induced basis functions
        and the learned expansion coefficients can be retrieved with the ``estimators_``
        and the ``coefs_`` attributes, respectively.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        init_expansion : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The smooth term in the additive expansion, i.e., the initial guess
            in gradient boosting.

        **fit_params_last_step : dict of string -> object
            Parameters passed to the estimator's ``fit`` method during the stage.

        Attributes
        ----------
        estimators_ : list
            A list of induced basis functions.
        coefs_ : list
            A list of learned expansion coefficients.

        Notes
        -----
        This greedy stagewise algorithm fits an additive expansion, which differs
        from the standard additve expansion. Namely, the constant term is a random
        variable, which depends upon the input example, e.g., an element in
        ``init_expansion``.
        """

        self._estimators = []
        self._coefs = []

        if getattr(self, "_estimator_type", None) == 'regressor':
            # The boosting loss attribute determines, which loss function
            # is employed in the negative gradient computation.
            if self.boosting_loss == 'huber':
                self.loss = LOSS_FUNCTIONS[self.boosting_loss](n_classes=1,
                                                               alpha=0.9)
            else:
                self.loss = LOSS_FUNCTIONS[self.boosting_loss](n_classes=1)
        
        pseudo_residual = self.loss.negative_gradient(y=y,
                                                      raw_predictions=init_expansion)

        # Greedily builds the additive expansion in a stagewise fashion,
        # wherein gradient boosting initializes with init_expansion. Thereby,
        # enabling the transfer of prior domain knowledge to gradient boosting.
        current_expansion = init_expansion
        for k in range(self.n_estimators):
            self._fit_stage(X=X, pseudo_residual=pseudo_residual,
                            **fit_params_last_step)

            # Copy the basis function for the predict method.
            # We use deepcopy instead of clone, since a clone
            # instance will not have invoked the fit method.
            self._estimators.append(copy.deepcopy(self))

            # Generate predictions for the line search computation.
            y_pred = self._final_estimator.predict(X=X)

            # This loss key determines, which loss function
            # is employed in the line search computation.
            if self.line_search_options['loss'] == 'huber':
                line_search_loss = LOSS_FUNCTIONS[self.line_search_options['loss']](n_classes=1,
                                                                                    alpha=0.9)
            else:
                line_search_loss = LOSS_FUNCTIONS[self.line_search_options['loss']](n_classes=1)
            
            def regularized_loss(alpha):
                current_expansion_ref = current_expansion
                loss = line_search_loss(y=y, 
                                        raw_predictions=current_expansion_ref.add(alpha*y_pred))
                return loss + self.line_search_options['regularization']*np.abs(alpha)

            coef = self.line_search(function=regularized_loss,
                                    init_guess=self.line_search_options['init_guess'],
                                    opt_method=self.line_search_options['opt_method'],
                                    method=self.line_search_options['method'],
                                    tol=self.line_search_options['tol'],
                                    options=self.line_search_options['options'],
                                    niter=self.line_search_options['niter'],
                                    T=self.line_search_options['T'])
            
            # Store the learned expansion coefficient for the predict method.
            self._coefs.append(coef)

            # These computations are not required in the last stage.
            if self.n_estimators - 1 > k:
                current_expansion = current_expansion.add(coef*y_pred)
                pseudo_residual = self.loss.negative_gradient(y=pseudo_residual,
                                                              raw_predictions=current_expansion)

        self.estimators_ = self._estimators
        self.coefs_ = self._coefs

    def fit(self, X: DataFrame_or_Series, y: pandas_or_numpy,
            **fit_params) -> ModifiedPipeline:
        """Sequentially fits the transform(s) then the final estimator.

        This method supports base boosting.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step or the stagewise
            ``_fit_stage`` method.

        Returns
        -------
        self : ModifiedPipeline
            The induced pipeline object.
        """

        fit_params_steps = self._check_fit_params(**fit_params)

        if X.ndim == 1:
            Xt = self._fit(X=X.values.reshape(-1, 1), y=y, **fit_params_steps)
        else:
            Xt = self._fit(X=X, y=y, **fit_params_steps)
        
        with sklearn.utils._print_elapsed_time('Pipeline',
                                               self._log_message(len(self.steps) - 1)):
            # This check distinguishes between a
            # default fit and base boosting.
            self._validate_base_boosting_options()
            if self._final_estimator != 'passthrough':
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                if not hasattr(self, '_default_fit'):
                    if self.target_index is not None and \
                    sklearn.utils.multiclass.type_of_target(y) == 'continuous':
                        smooth_term = X.iloc[:, self.target_index]
                    else:
                        smooth_term = X
                    self._fit_stages(X=Xt, y=y, init_expansion=smooth_term,
                                     **fit_params_last_step)
                else:
                    if self._final_estimator.__class__ in _CHAIN_FLAG:
                        self._final_estimator.fit(X=Xt, Y=y, **fit_params_last_step)
                    else:
                        self._final_estimator.fit(X=Xt, y=y, **fit_params_last_step)

        return self

    def _predict(self, estimator, Xt: pandas_or_numpy, coef: float,
                 **predict_params) -> np.ndarray:
        """Helper method for parallelizing the noise term predictions.

        Parameters
        ----------
        estimator : estimator
            An estimator that follows the Scikit-learn API.

        Xt : array-like of shape = [n_samples, n_features]
            The transformed design matrix.

        coef : float
            The learned expansion coefficient.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline.

        Returns
        -------
        y_pred : ndarray
            A Numpy array of predictions.
        """

        return coef * estimator.steps[-1][-1].predict(X=Xt, **predict_params)

    @sklearn.utils.metaestimators.if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X: DataFrame_or_Series, **predict_params) -> DataFrame_or_Series:
        """Applies transform(s) to the data, then predicts with the final estimator.

        The method supports base boosting.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        **predict_params : dict of string -> object
            Parameters to the ``predict`` method, which are called after completing
            all of the pipeline transformations.

        Returns
        -------
        y_pred : DataFrame or Series
            A pandas DataFrame or Series of predictions.

        Notes
        -----
        In base boosting, we decompose the predictions in accord with Tukey's
        notion of reroughing. Namely, data = smooth + rough.
        """

        Xt = X
        for _, name, transform in self._iter(with_final=False):
            if Xt.ndim == 1:
                Xt = transform.transform(X=Xt.values.reshape(-1, 1))
            else:
                Xt = transform.transform(X=Xt)
                
        if hasattr(self, 'coefs_') and hasattr(self, 'estimators_'):
            # Tukey's reroughing: smooth term plus noise term(s).
            if self.target_index is not None:
                smooth_term = X.iloc[:, self.target_index]
            else:
                smooth_term = X
            
            parallel = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                                       pre_dispatch='2*n_jobs')

            noise_term = parallel(
                joblib.delayed(self._predict)(
                    estimator=estimator, Xt=Xt, coef=coef)
                for coef, estimator in zip(self.coefs_, self.estimators_))
            y_pred = smooth_term.add(noise_term[0])
        else:
            # Generate predictions without reroughing.
            if self.target_index is not None:
                if self._final_estimator.__class__ in [_CATBOOST_FLAG, _XGBOOST_FLAG]:
                    y_pred = pd.DataFrame(self.steps[-1][-1].predict(data=Xt, **predict_params),
                                          index=X.iloc[:, self.target_index].index)
                else:
                    y_pred = pd.DataFrame(self.steps[-1][-1].predict(X=Xt, **predict_params),
                                          index=X.iloc[:, self.target_index].index)
            else:
                if self._final_estimator.__class__ in [_CATBOOST_FLAG, _XGBOOST_FLAG]:
                    y_pred = pd.DataFrame(self.steps[-1][-1].predict(data=Xt, **predict_params),
                                          index=X.index)
                else:
                    y_pred = pd.DataFrame(self.steps[-1][-1].predict(X=Xt, **predict_params),
                                          index=X.index)

        return y_pred

    @sklearn.utils.metaestimators.if_delegate_has_method(delegate='_final_estimator')
    def score(self, X: DataFrame_or_Series, y: DataFrame_or_Series,
              multioutput='raw_values', **predict_params) -> pd.DataFrame:
        """Computes the supervised score.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : array-like of shape = [n_samples] or shape = [n_samples, n_targets]
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        multioutput : str, optional (default='raw_values')
            Defines aggregating of multiple output values, wherein the string
            must be either ``'raw_values'`` or ``'uniform_average'``.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` method, which are called after completing
            all of the pipeline transformations.

        Returns
        -------
        scores : pd.DataFrame or pd.Series
            The pandas object of computed scores.
        """

        assert any(multioutput for output in ['raw_values', 'uniform_average'])

        y_pred = self.predict(X=X, **predict_params)
        y_true = y

        scores = {}
        for scoring in _SCORE_CHOICE:
            if scoring == 'mae':
                scores[scoring] = sklearn.metrics.mean_absolute_error(y_true=y_true,
                                                                      y_pred=y_pred,
                                                                      multioutput=multioutput)
            elif scoring == 'mse':
                scores[scoring] = sklearn.metrics.mean_squared_error(y_true=y_true,
                                                                     y_pred=y_pred,
                                                                     multioutput=multioutput)
            elif scoring == 'rmse':
                scores[scoring] = np.sqrt(sklearn.metrics.mean_squared_error(y_true=y_true,
                                                                             y_pred=y_pred,
                                                                             multioutput=multioutput))
            elif scoring == 'r2':
                scores[scoring] = sklearn.metrics.r2_score(y_true=y_true,
                                                           y_pred=y_pred,
                                                           multioutput=multioutput)
            elif scoring == 'ev':
                scores[scoring] = sklearn.metrics.explained_variance_score(y_true=y_true,
                                                                           y_pred=y_pred,
                                                                           multioutput=multioutput)
            elif scoring == 'msle':
                try:
                    scores[scoring] = sklearn.metrics.mean_squared_log_error(y_true=y_true,
                                                                             y_pred=y_pred,
                                                                             multioutput=multioutput)
                except:
                    # Sklearn will raise a ValueError if either
                    # statement is true, so we circumvent
                    # this error and score with a NaN.
                    scores[scoring] = np.nan

        if multioutput == 'raw_values':
            return pd.DataFrame(scores).dropna(how='any', axis=1)
        else:
            return pd.Series(scores).dropna(how='any', axis=0)


def make_pipeline(estimator, transform=None, **kwargs) -> ModifiedPipeline:
    """Constructs a ModifiedPipeline from the given base estimator.

    Parameters
    ----------
    estimator : estimator
        A base estimator that follows the Scikit-learn API.

    transform : str, list, tuple, or None, optional (default=None)
        Choice of transform(s). If the specified choice is a string,
        then it must be a default option, where ``'standardscaler'``,
        ``'boxcox'``, ``'yeojohnson'``, ``'quantileuniform'``, and
        ``'quantilenormal'`` denote :class:`sklearn.preprocessing.StandardScaler`,
        :class:`sklearn.preprocessing.PowerTransformer` with ``method='box-cox'``
        or ``method='yeo-johnson'``, and :class:`sklearn.preprocessing.QuantileTransformer`
        with ``output_distribution='uniform'`` or ``output_distribution='normal'``,
        respectively.

    memory : str or object with the joblib.Memory interface
        Enables fitted transform caching.

    verbose : int
        Determines verbosity.

    n_jobs : int or None
        The number of jobs to run in parallel.

    target_index : int or None
        Specifies the single-target subtask in the multi-target task.

    target_type : str
        Specifies the type of target according to :class:`sklearn.utils.multiclass.type_of_target`.

    base_boosting_options : dict or None
        A dictionary of base boosting options, wherein the following options
        must be specified:

        n_estimators :obj:`int`
            The number of basis functions in the noise term of the additive expansion.

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

    random_state : int, RandomState instance, or None
        Determines the random number generation in
        :class:`sklearn.preprocessing.QuantileTransformer`, if ``pipeline_transform``
        is either ```quantileuniform``` or ```quantilenormal```, and also in
        :class:`sklearn.multioutput.RegressorChain`.

    n_quantiles : int or None
        Number of quantiles in :class:`sklearn.preprocessing.QuantileTransformer`, if
        ``pipeline_transform`` is either ```quantileuniform``` or ```quantilenormal```.

    cv : int, cross-validation generator, an iterable, or None
        Determines which targets are utilized in :class:`sklearn.multioutput.RegressorChain`.

    chain_order : list or None
        Determines the target order in :class:`sklearn.multioutput.RegressorChain`.

    Returns
    -------
    pipe : ModifiedPipeline

    See Also
    --------
    :class:`physlearn.pipeline.ModifiedPipeline` : Class for creating a modified pipeline of
        transforms with a final estimator, which supports base boosting.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.utils.multiclass import type_of_target
    >>> from physlearn import make_pipeline, Regressor
    >>> X, y = make_regression(n_targets=3, random_state=42)
    >>> X, y = pd.DataFrame(X), pd.DataFrame(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)
    >>> pipe = make_pipeline(Ridge(), 'yeojohnson',
                             target_type=type_of_target(y))
    >>> pipe.fit(X_train, y_train)
    >>> pipe.score(X_test, y_test).round(decimals=2)
          mae       mse    rmse    r2    ev
    0   58.68   5884.12   76.71  0.67  0.67
    1  101.19  14627.70  120.95  0.36  0.36
    2   96.31  14450.54  120.21  0.40  0.40

    References
    ----------
    - Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
      "Boosting on the shoulders of giants in quantum device calibration",
      arXiv preprint arXiv:2005.06194 (2020).

    - Jerome Friedman. "Greedy function approximation: A gradient boosting machine,"
      Annals of Statistics, 29(5):1189–1232 (2001).
    
    - Trevor Hastie, Robert Tibshirani, and Jerome Friedman.
      "The Elements of Statistical Learning", Springer (2009).
    """

    memory = kwargs.pop('memory', None)
    verbose = kwargs.pop('verbose', None)
    n_jobs = kwargs.pop('n_jobs', None)
    target_index = kwargs.pop('target_index', None)
    target_type = kwargs.pop('target_type', None)
    base_boosting_options = kwargs.pop('base_boosting_options', None)
    random_state = kwargs.pop('random_state', None)
    n_quantiles = kwargs.pop('n_quantiles', None)
    cv = kwargs.pop('cv', None)
    chain_order = kwargs.pop('chain_order', None)
    if kwargs:
        raise TypeError('Unknown keyword arguments: %s'
                        % (list(kwargs.keys())[0]))

    if transform is not None:
        if isinstance(transform, tuple):
            steps = [transform]
        elif isinstance(transform, list):
            steps = transform
        elif transform in _PIPELINE_TRANSFORM_CHOICE:
            # Utulize the in-built transforma options.
            if transform == 'standardscaler':
                transform = sklearn.preprocessing.StandardScaler()
            elif transform == 'boxcox':
                transform = sklearn.preprocessing.PowerTransformer(method='box-cox')
            elif transform == 'yeojohnson':
                transform = sklearn.preprocessing.PowerTransformer(method='yeo-johnson')
            elif transform == 'quantileuniform':
                transform = sklearn.preprocessing.QuantileTransformer(n_quantiles=n_quantiles,
                                                                      output_distribution='uniform',
                                                                      random_state=random_state)
            elif transform == 'quantilenormal':  
                transform = sklearn.preprocessing.QuantileTransformer(n_quantiles=n_quantiles,
                                                                      output_distribution='normal',
                                                                      random_state=random_state)
            steps = [('tr', transform)]
        else:
            raise TypeError('The transform: %s was not a default str option, '
                            'tuple with (name, transform), or a list of such '
                            'tuple(s).'
                            % (transform))
    else:
        steps = []

    # Distinguishes between single-target and multi-target regression.
    if target_type in _MULTI_TARGET:
        if chain_order is not None:
            estimator = sklearn.multioutput.RegressorChain(base_estimator=estimator,
                                                           order=chain_order,
                                                           cv=cv,
                                                           random_state=random_state)
            steps.append(('reg', estimator))
        else:
            estimator = sklearn.multioutput.MultiOutputRegressor(estimator=estimator,
                                                                 n_jobs=n_jobs)
            steps.append(('reg', estimator))
    else:
        steps.append(('reg', estimator))

    return ModifiedPipeline(steps=steps,
                            memory=memory,
                            verbose=verbose,
                            n_jobs=n_jobs,
                            target_index=target_index,
                            base_boosting_options=base_boosting_options)
