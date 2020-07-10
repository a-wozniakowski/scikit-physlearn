"""
Custom pipeline that handles base boosting.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import copy

import numpy as np
import pandas as pd

import scipy.optimize

import sklearn.dummy
import sklearn.metrics
import sklearn.multioutput
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.utils
import sklearn.utils.metaestimators
import sklearn.utils.multiclass

from .loss import LOSS_FUNCTIONS
from .supervised.utils._data_checks import _n_targets


_CHAIN_FLAG = sklearn.multioutput.RegressorChain(base_estimator=sklearn.dummy.DummyRegressor()).__class__


def _make_pipeline(estimator, transform, n_targets,
                   random_state, verbose, n_jobs=-1,
                   cv=5, memory=None, n_quantiles=None,
                   chain_order=None, n_estimators=None,
                   target_index=None, boosting_loss=None,
                   regularization=None, line_search_options=None):

    # Feature transformation options
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

    # Automatically manages single-target 
    # and multi-target regression
    if transform is not None:
        steps = [('tr', transform)]
    else:
        steps = []
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
                            n_estimators=n_estimators,
                            target_index=target_index,
                            boosting_loss=boosting_loss,
                            regularization=regularization,
                            line_search_options=line_search_options)


class ModifiedPipeline(sklearn.pipeline.Pipeline):
    """
    Custom pipeline object capable of base boosting.
    """

    _required_parameters = ['steps']

    def __init__(self, steps, memory=None, verbose=False,
                 n_estimators=None, target_index=None,
                 boosting_loss=None, regularization=None,
                 line_search_options=None):

        if n_estimators is not None:
            assert isinstance(n_estimators, int) and n_estimators > 0

        if target_index is not None:
            assert isinstance(target_index, int)

        if boosting_loss is not None:
            assert isinstance(boosting_loss, str)

        if regularization is not None:
            assert isinstance(regularization, (float, int))

        if line_search_options is not None:
            assert isinstance(line_search_options, dict)

        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.target_index = target_index
        self.boosting_loss = boosting_loss
        self.regularization = regularization
        self.line_search_options = line_search_options
        self._validate_steps()

    @staticmethod
    def line_search(function, init_guess, opt_method,
                    alg=None, tol=None, options=None,
                    niter=None, T=None):

        assert opt_method in ['minimize', 'basinhopping']

        if opt_method == 'minimize':
            assert alg is not None and tol is not None
            minimize_object = scipy.optimize.minimize(fun=function, x0=init_guess,
                                                      method=alg, tol=tol,
                                                      options=options)            
        else:
            assert niter is not None and T is not None
            minimize_object = scipy.optimize.basinhopping(func=function,
                                                          x0=init_guess,
                                                          niter=niter, T=T)            
        
        return minimize_object.x

    def _fit_stages(self, X, y, raw_predictions, **fit_params_last_step):
        self._estimators = []
        self._coefs = []

        loss = LOSS_FUNCTIONS[self.boosting_loss](self.n_estimators)
        residual = y

        for k in range(loss.K):
            residual = loss.negative_gradient(y=residual,
                                              raw_predictions=raw_predictions)
            
            self._final_estimator.fit(X=X, y=residual, **fit_params_last_step)            
            self._estimators.append(copy.deepcopy(self))
            y_pred = self._final_estimator.predict(X=X)
            
            def regularized_loss(alpha):
                line_search_df = raw_predictions
                line_search_df = line_search_df.add(alpha * y_pred)
                loss = sklearn.metrics.mean_absolute_error(y, line_search_df)
                loss += self.regularization * np.abs(alpha)                    
                return loss

            coef = self.line_search(function=regularized_loss,
                                    init_guess=self.line_search_options['init_guess'],
                                    opt_method=self.line_search_options['opt_method'],
                                    alg=self.line_search_options['alg'],
                                    tol=self.line_search_options['tol'],
                                    options=self.line_search_options['options'],
                                    niter=self.line_search_options['niter'],
                                    T=self.line_search_options['T'])
            
            self._coefs.append(coef[0])
            raw_predictions = raw_predictions.add(coef[0] * y_pred)

        self.estimators_ = self._estimators
        self.coefs_ = self._coefs

    def fit(self, X, y, **fit_params):
        fit_params_steps = self._check_fit_params(**fit_params)

        if X.ndim == 1:
            Xt = pd.DataFrame(self._fit(X=X.values.reshape(-1, 1), y=y, **fit_params_steps), index=X.index)
        else:
            Xt = pd.DataFrame(self._fit(X=X, y=y, **fit_params_steps), index=X.index)
        
        with sklearn.utils._print_elapsed_time('Pipeline',
                                               self._log_message(len(self.steps) - 1)):
            if self._final_estimator != 'passthrough':
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                # Start checks for base boosting 
                if self.n_estimators is not None:
                    if self.target_index is not None and \
                    sklearn.utils.multiclass.type_of_target(y) == 'continuous':
                        raw_predictions = X.iloc[:, self.target_index]
                    else:
                        raw_predictions = X
                    self._fit_stages(X=Xt, y=y, raw_predictions=raw_predictions, **fit_params_last_step)
                else:
                    if self._final_estimator.__class__ == _CHAIN_FLAG:
                        # RegressorChain capitilizes the target
                        self._final_estimator.fit(X=Xt, Y=y, **fit_params_last_step)
                    else:
                        self._final_estimator.fit(X=Xt, y=y, **fit_params_last_step)

        return self

    @sklearn.utils.metaestimators.if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, **predict_params):
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            if Xt.ndim == 1:
                Xt = pd.DataFrame(transform.transform(X=Xt.values.reshape(-1, 1)), index=Xt.index)
            else:
                Xt = pd.DataFrame(transform.transform(X=Xt), index=Xt.index)
                
        if hasattr(self, 'coefs_') and hasattr(self, 'estimators_'):
            # Generate predictions with base boosting model
            if self.target_index is not None:
                y_pred = X.iloc[:, self.target_index]
            else:
                y_pred = X
            for coef, estimator in zip(self.coefs_, self.estimators_):
                y_pred = y_pred.add(coef * estimator.steps[-1][-1].predict(X=Xt, **predict_params))
        else:
            # Generate predictions without base boosting model
            y_pred = pd.DataFrame(self.steps[-1][-1].predict(Xt, **predict_params), index=Xt.index)

        return y_pred
