"""
The :mod:`physlearn.pipeline` module enhances the original Scikit-learn
pipeline with an implementation of base boosting. Thereby, enabling the
transfer of prior domain knowledge to gradient boosting.
"""

# Author: Alex Wozniakowski
# License: MIT

import copy
import joblib

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

from .loss import LOSS_FUNCTIONS

from physlearn.supervised.utils._data_checks import _n_targets
from physlearn.supervised.utils._definition import (_CATBOOST_FLAG, _CHAIN_FLAG,
                                                    _MULTI_TARGET,
                                                    _PIPELINE_TRANSFORM_CHOICE,
                                                    _XGBOOST_FLAG)
from physlearn.supervised.utils._model_checks import _check_line_search_options


def _make_pipeline(estimator, transform, n_targets,
                   random_state, verbose, n_jobs=-1,
                   cv=5, memory=None, n_quantiles=None,
                   chain_order=None, target_index=None,
                   base_boosting_options=None):
    """Construct a ModifiedPipeline from the given estimator.

    Parameters
    ----------
    estimator : An estimator that supports the Scikit-learn API.

    transform : str, list, or tuple
        Choice of transform(s).

    n_targets : int
        Number of targets in the supervised learning task. If greater than
        one, then sklearn.multioutput.RegressorChain is utilized if chain_order
        is not None otherwise sklearn.multioutput.MultiOutputRegressor is utilized.

    random_state : int, RandomState instance or None.
        Determines random number generation in sklearn.preprocessing.QuantileTransformer
        or sklearn.multioutput.RegressorChain.

    verbose : int
        Determines verbosity.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel for fit.

    cv : int, cross-validation generator or an iterable, optional (default=None)
        Determines which targets are utilized in sklearn.multioutput.RegressorChain.

    memory : str or object with the joblib.Memory interface, default=None
        Enables fitted transform caching.

    n_quantiles : int or None, optional (default=None)
        Number of quantiles in sklearn.preprocessing.QuantileTransformer.

    chain_order : list or None, optional (default=None)
        Target order used in chaining.

    target_index : int or None, optional (default=None)
        Index of single-target slice in the multi-target task.

    base_boosting_options : dict or None, optional (default=None)
        The required keys are n_estimators :int: which specifies
        the number of basis functions in the noise term,
        boosting_loss :str: which specifies the loss function
        utilized in the pseudo-residual computation-- 'ls' denotes
        the squared error loss function, 'lad' denotes the absolute
        error loss function, 'huber' denotes the Huber loss function,
        and 'quantile' denotes the quantile loss function, and
        line_search_options :dict: which specifies the line search
        algorithm and its parameters.
    """

    if isinstance(transform, (list, tuple)):
        # Allows custom transform options.
        if not hasattr(transform, 'append'):
            # Wrap a list around transform as downstream
            # tasks require append.
            steps = [transform]
        else:
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
        # No transform steps were provided.
        steps = []

    # Automatically prepares pipeline for single-target
    # or multi-target regression task
    if n_targets == 1:
        steps.append(('reg', estimator))
    elif n_targets > 1:
        if chain_order is not None:
            estimator = sklearn.multioutput.RegressorChain(base_estimator=estimator,
                                                           order=chain_order, cv=cv,
                                                           random_state=random_state)
            steps.append(('reg', estimator))
        else:
            estimator = sklearn.multioutput.MultiOutputRegressor(estimator=estimator,
                                                                 n_jobs=n_jobs)
            steps.append(('reg', estimator))

    return ModifiedPipeline(steps=steps,
                            memory=memory,
                            verbose=verbose,
                            n_jobs=n_jobs,
                            target_index=target_index,
                            base_boosting_options=base_boosting_options)


class ModifiedPipeline(sklearn.pipeline.Pipeline):
    """Custom pipeline object that supports base boosting.

    The object inherits from the Scikit-learn Pipeline object, so it retains
    the ability to sequentially assemble several steps of transforms and a
    final estimator that can be cross-validated together, while setting
    different parameters. Further, it supports base boosting, which enables
    the transfer of prior domain knowledge to gradient boosting.

    Parameters
    ----------
    steps: list
        List of tuples, wherein the last tuple (name, estimator) is an estimator
        and the preceding tuple(s) (name, transform) are transform(s).

    memory: str or object with the joblib.Memory interface, default=None
        Enables fitted transform caching.

    verbose : int
        Determines verbosity.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel for fit.

    target_index : int or None, optional (default=None)
        Index of single-target slice in the multi-target task.

    base_boosting_options : dict or None, optional (default=None)
        The required keys are n_estimators :int: which specifies
        the number of basis functions in the noise term,
        boosting_loss :str: which specifies the loss function
        utilized in the pseudo-residual computation-- 'ls' denotes
        the squared error loss function, 'lad' denotes the absolute
        error loss function, 'huber' denotes the Huber loss function,
        and 'quantile' denotes the quantile loss function, and
        line_search_options :dict: which specifies the line search
        algorithm and its parameters.

    Attributes
    -----------
    named_steps : :class:`sklearn.utils.Bunch 
    <https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html>`_
        Dictionary-like object, wherein the keys are the user provided
        step names and the values are the steps parameters.

    See Also
    --------
    physlearn.pipeline.make_pipeline : Convenience function for modified pipeline construction.

    References
    ----------
    Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
    "Boosting on the shoulders of giants in quantum device calibration",
    arXiv preprint arXiv:2005.06194 (2020).
    """

    _required_parameters = ['steps']

    def __init__(self, steps, memory=None, verbose=False,
                 n_jobs=None, target_index=None,
                 base_boosting_options=None):

        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.target_index = target_index
        self.base_boosting_options = base_boosting_options
        self._validate_steps()
        self._validate_base_boosting_options()

    def _validate_base_boosting_options(self):
        if self.base_boosting_options is not None:
            for key, boosting_option in self.base_boosting_options.items():
                if key in ['n_estimators', 'n_regressors']:
                    if boosting_option is not None:
                        assert isinstance(boosting_option, int) and boosting_option > 0
                        self.n_estimators = boosting_option
                    else:
                        raise ValueError('The value of key: %s cannot be None. '
                                         'Specify a positive integer for the number '
                                         'of basis functions to fit in the additive '
                                         'expansion.'
                                         % (key))
                elif key == 'boosting_loss':
                    if boosting_option is not None:
                        assert boosting_option in LOSS_FUNCTIONS
                        self.boosting_loss = boosting_option
                    else:
                        raise ValueError('The value of key: %s cannot be None. '
                                         'The choice of loss function is required '
                                         'in the pseudo-residual computation.'
                                         % (key))
                elif key == 'line_search_options':
                    if boosting_option is not None:
                        assert isinstance(boosting_option, dict)
                        _check_line_search_options(line_search_options=boosting_option)
                        self.line_search_options = boosting_option
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
    def line_search(function, init_guess, opt_method,
                    alg=None, tol=None, options=None,
                    niter=None, T=None):
        """Optimization method used within a stage of the
        greedy stagewise algorithm, which computes an 
        expansion coefficient.

        Parameters
        ----------

        function : callable
            The objective function for the line search.

        init_guess : int, float, ndarray
            The initial guess for the expansion coefficient.

        opt_method : str
            Choice of optimization method.

        alg : str, callable, or None, optional (default=None)
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
        """

        if opt_method == 'minimize':
            assert alg is not None and tol is not None
            minimize_object = scipy.optimize.minimize(fun=function, x0=init_guess,
                                                      method=alg, tol=tol,
                                                      options=options)            
        elif opt_method == 'basinhopping':
            assert niter is not None and T is not None
            minimize_object = scipy.optimize.basinhopping(func=function,
                                                          x0=init_guess,
                                                          niter=niter, T=T)
        else:
            raise ValueError('The argument: %s is not an implemented optimization method.'
                             % (opt_method))

        return minimize_object.x

    def _fit_stage(self, X, pseudo_residual, current_expansion, **fit_params_last_step):
        """Fit a stage of the stagewise additive expansion.

        Parameters
        ----------

        X : DataFrame or Series
        The design matrix, where each row corresponds to an example and the
        column(s) correspond to the feature(s).

        pseudo_residual : DataFrame or Series
        The negative gradient of the loss function.

        current_expansion : DataFrame or Series
        The current estimate of the training target(s). This data is invariant
        during the call to _fit_stage.

        **fit_params_last_step : dict of string -> object
            Parameters passed to the estimator's ``fit`` method during the stage.
        """

        for k in range(self.loss.K):
            if self._final_estimator.__class__ in _CHAIN_FLAG:
                self._final_estimator.fit(X=X, Y=pseudo_residual, **fit_params_last_step)
            else:
                self._final_estimator.fit(X=X, y=pseudo_residual, **fit_params_last_step)

        return current_expansion

    def _fit_stages(self, X, y, init_expansion, **fit_params_last_step):
        """
        Fit the additive expansion in a stagewise fashion.

        Parameters
        ----------

        X : DataFrame or Series
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : DataFrame or Series
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        init_expansion : DataFrame or Series
            The smooth term in the additive expansion, i.e., the initial guess
            in gradient boosting.

        **fit_params_last_step : dict of string -> object
            Parameters passed to the estimator's ``fit`` method during the stage.

        Notes
        -----
        This greedy stagewise algorithm fits an additive expansion, which differs
        from the standard additve expansion. Namely, the constant term is a random
        variable, which depends upon the input example, e.g., init_expansion.

        References
        ---------
        Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
        "Boosting on the shoulders of giants in quantum device calibration",
        arXiv preprint arXiv:2005.06194 (2020).
        Jerome Friedman. "Greedy function approximation: A gradient boosting machine,"
        Annals of Statistics, 29(5):1189–1232 (2001).
        Trevor Hastie, Robert Tibshirani, and Jerome Friedman. "The Elements of
        Statistical Learning", Springer (2009).
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
        # wherein we initialize with init_expansion. This modification of
        # the initialization step enables the transfer of prior domain
        # knowledge to gradient boosting.
        current_expansion = init_expansion
        for k in range(self.n_estimators):
            # The current expansion is invariant during the fit stage.
            current_expansion = self._fit_stage(X=X, pseudo_residual=pseudo_residual,
                                                current_expansion=current_expansion,
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
                                    alg=self.line_search_options['alg'],
                                    tol=self.line_search_options['tol'],
                                    options=self.line_search_options['options'],
                                    niter=self.line_search_options['niter'],
                                    T=self.line_search_options['T'])
            
            # Store the learned expansion coefficient for the predict method.
            self._coefs.append(coef[0])

            # These computations are not required in the last stage.
            if self.n_estimators - 1 > k:
                current_expansion = current_expansion.add(coef[0]*y_pred)
                pseudo_residual = self.loss.negative_gradient(y=pseudo_residual,
                                                              raw_predictions=current_expansion)

        self.estimators_ = self._estimators
        self.coefs_ = self._coefs

    def fit(self, X, y, **fit_params):
        """
        Fit the model, wherein transforms are seqeuntially fit,
        then the final estimator is fit. This method supports
        base boosting.

        Parameters
        ----------

        X : DataFrame or Series
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        y : DataFrame or Series
            The target matrix, where each row corresponds to an example and the
            column(s) correspond to the single-target(s).

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step or the stagewise
            ``_fit_stage`` method.

        References
        ---------
        Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
        "Boosting on the shoulders of giants in quantum device calibration",
        arXiv preprint arXiv:2005.06194 (2020).
        Jerome Friedman. "Greedy function approximation: A gradient boosting machine,"
        Annals of Statistics, 29(5):1189–1232 (2001).
        Trevor Hastie, Robert Tibshirani, and Jerome Friedman. "The Elements of
        Statistical Learning", Springer (2009).
        """

        fit_params_steps = self._check_fit_params(**fit_params)

        if X.ndim == 1:
            Xt = self._fit(X=X.values.reshape(-1, 1), y=y, **fit_params_steps)
        else:
            Xt = self._fit(X=X, y=y, **fit_params_steps)
        
        with sklearn.utils._print_elapsed_time('Pipeline',
                                               self._log_message(len(self.steps) - 1)):
            if self._final_estimator != 'passthrough':
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                # Start checks for base boosting.
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

    def _predict(self, estimator, Xt, coef, **predict_params):
        """
        Helper method for generating the noise term predictions in parallel.

        Parameters
        ----------

        estimator : estimator object
            An estimator object implementing :term:`fit` and :term:`predict`.

        Xt : DataFrame or Series
            The transformed design matrix.

        coef : float
            The learned expansion coefficient.

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline.
        """

        return coef * estimator.steps[-1][-1].predict(X=Xt, **predict_params)

    @sklearn.utils.metaestimators.if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, **predict_params):
        """
        Apply transforms to the data, then predict with the final estimator.
        The method supports base boosting.

        Parameters
        ----------

        X : DataFrame or Series
            The design matrix, where each row corresponds to an example and the
            column(s) correspond to the feature(s).

        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline.

        Notes
        -----
        The prediction decomposition in base boosting emanates from
        Tukey's reroughing, whereby data = smooth + rough.

        References
        ---------
        Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
        "Boosting on the shoulders of giants in quantum device calibration",
        arXiv preprint arXiv:2005.06194 (2020).
        John Tukey. "Exploratory Data Analysis", Addison-Wesley (1977).
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
