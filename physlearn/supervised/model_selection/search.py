import copy
import time

import numpy as np

import sklearn.base
import sklearn.model_selection
import sklearn.model_selection._search
import sklearn.metrics._scorer
import sklearn.model_selection._split
import sklearn.model_selection._validation
import sklearn.utils.validation

from bayes_opt import BayesianOptimization
from itertools import product
from joblib import Parallel, delayed

from ..utils._model_checks import _check_bayesianoptimization_parameter_type


def _bayesianoptimizationcv(X, y, estimator, search_params, cv,
                            scoring, n_jobs, verbose, random_state,
                            init_points, n_iter):

    def regressor_cross_val_mean(**pbounds):
        pbounds = _check_bayesianoptimization_parameter_type(pbounds)
        estimator.set_params(**pbounds)
        cross_val = sklearn.model_selection.cross_val_score(estimator=estimator,
                                                            X=X, y=y, scoring=scoring,
                                                            cv=cv, n_jobs=n_jobs)
        return cross_val.mean()

    search = BayesianOptimization(f=regressor_cross_val_mean, pbounds=search_params,
                                  verbose=verbose, random_state=random_state)
    search.maximize(init_points=init_points, n_iter=n_iter)

    return search


class ModifiedBaseSearchCV(sklearn.model_selection._search.BaseSearchCV):

    def __init__(self, estimator, n_jobs=-1, search_scoring=None,
                 refit=True, cv=5, verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=True):

        super().__init__(estimator=estimator, scoring=search_scoring,
                         n_jobs=n_jobs, iid='deprecated', refit=refit,
                         cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
                         error_score=error_score,
                         return_train_score=return_train_score)

    def fit(self, X, y=None, groups=None, **fit_params):

        cv = sklearn.model_selection._split.check_cv(cv=self.cv, y=y,
                                                     classifier=sklearn.base.is_classifier(estimator))

        scorers, self.multimetric_ = sklearn.metrics._scorer._check_multimetric_scoring(
            estimator=self, scoring=self.scoring
        )

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, str) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers) and not callable(self.refit):
                raise ValueError('For multi-metric scoring, the parameter '
                                 'refit must be set to a scorer key or a '
                                 'callable to refit an estimator with the '
                                 'best parameter setting on the whole '
                                 'data and make the best_* attributes '
                                 'available for that metric. If this is '
                                 'not needed, refit should be set to '
                                 'False explicitly. %r was passed.'
                                 % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X, y, groups = sklearn.utils.validation.indexable(X, y, None)
        fit_params = sklearn.utils.validation._check_fit_params(X=X, fit_params=fit_params)
        n_splits = cv.get_n_splits(X=X, y=y, groups=groups)

        estimator = sklearn.base.clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(scorer=scorers, fit_params=fit_params,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True, return_times=True,
                                    return_parameters=False, error_score=self.error_score,
                                    verbose=self.verbose)
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []

            def evaluate_candidates(candidate_params):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print('Fitting {0} folds for each of {1} candidates,'
                          ' totalling {2} fits'.format(n_splits, n_candidates, n_candidates * n_splits))

                out = parallel(
                    delayed(sklearn.model_selection._validation._fit_and_score)(
                        estimator=sklearn.base.clone(estimator), X=X, y=y, train=train,
                        test=test, parameters=parameters, **fit_and_score_kwargs)
                    for parameters, (train, test) 
                    in product(candidate_params, cv.split(X, y, groups))
                )

                if len(out) < 1:
                    raise ValueError('No fits were performed. '
                                     'Was the CV iterator empty? '
                                     'Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned '
                                     'inconsistent results. Expected {} '
                                     'splits, got {}'
                                     .format(n_splits,
                                             len(out) // n_candidates))

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, scorers, n_splits, all_out)
                return results

            self._run_search(evaluate_candidates)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is 'score'
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or
                   self.best_index_ >= len(results['params'])):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results['rank_test_%s'
                                           % refit_metric].argmin()
                self.best_score_ = results['mean_test_%s' % refit_metric][
                                           self.best_index_]
            self.best_params_ = results['params'][self.best_index_]

        if self.refit:
            self.best_estimator_ = sklearn.base.clone(
                sklearn.base.clone(estimator).set_params(**self.best_params_))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self


class GridSearchCV(ModifiedBaseSearchCV):

    _required_parameters = ['estimator', 'param_grid']

    def __init__(self, estimator, param_grid, search_scoring=None,
                 refit=True, n_jobs=-1, cv=5, verbose=0,
                 pre_dispatch='2*n_jobs', error_score=np.nan,
                 return_train_score=True):

        super().__init__(estimator=estimator,
                         search_scoring=search_scoring,
                         n_jobs=n_jobs,
                         refit=refit,
                         cv=cv,
                         verbose=verbose,
                         pre_dispatch=pre_dispatch,
                         error_score=error_score,
                         return_train_score=return_train_score)

        self.param_grid = param_grid
        sklearn.model_selection._search._check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        evaluate_candidates(sklearn.model_selection._search.ParameterGrid(param_grid=self.param_grid))


class RandomizedSearchCV(ModifiedBaseSearchCV):

    _required_parameters = ['estimator', 'param_distributions']

    def __init__(self, estimator, param_distributions, n_iter=10,
                 search_scoring=None, n_jobs=-1, refit=True,
                 cv=5, verbose=0, pre_dispatch='2*n_jobs', random_state=None,
                 error_score=np.nan, return_train_score=True):

        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

        super().__init__(estimator=estimator, search_scoring=search_scoring,
                         n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score,
                         return_train_score=return_train_score)

    def _run_search(self, evaluate_candidates):
        evaluate_candidates(
            sklearn.model_selection._search.ParameterSampler(param_distributions=self.param_distributions,
                                                             n_iter=self.n_iter,
                                                             random_state=self.random_state))


def _gridsearchcv(estimator, param_grid, search_scoring, refit,
                  n_jobs, cv, verbose, pre_dispatch, error_score,
                  return_train_score):

    search = GridSearchCV(estimator=estimator, param_grid=param_grid,
                          search_scoring=search_scoring, refit=refit,
                          n_jobs=n_jobs, cv=cv, verbose=verbose,
                          pre_dispatch=pre_dispatch, error_score=error_score,
                          return_train_score=return_train_score)

    return search


def _randomizedsearchcv(estimator, param_distributions, n_iter,
                        search_scoring, n_jobs, refit, cv,
                        verbose, pre_dispatch, error_score,
                        return_train_score):

    search = RandomizedSearchCV(estimator=estimator, param_distributions=param_distributions,
                                n_iter=n_iter, search_scoring=search_scoring, n_jobs=n_jobs,
                                refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
                                error_score=error_score, return_train_score=return_train_score)

    return search
