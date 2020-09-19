"""
Unit tests for Mlxtend compatibility.
"""

# Author: Alex Wozniakowski
# License: MIT

import unittest

import pandas as pd

from scipy.stats import randint

from sklearn import __version__ as sk_version
from sklearn.base import clone
from sklearn.datasets import load_boston, load_linnerud
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

from physlearn import Regressor
from physlearn.datasets import load_benchmark
from physlearn.supervised import ShapInterpret


class TestMlxtend(unittest.TestCase):

    def test_stacking_regressor_without_cv_gridsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack))
        search_params = dict(reg__kneighborsregressor__n_neighbors=[2, 4, 5],
                             reg__bayesianridge__alpha_1=[1e-7, 1e-6],
                             reg__meta_regressor__alpha=[1.0],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 3.0)
        self.assertIn(reg.best_params_['reg__kneighborsregressor__n_neighbors'], [2, 4, 5])
        self.assertIn(reg.best_params_['reg__bayesianridge__alpha_1'], [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__meta_regressor__alpha'], [1.0])

    def test_stacking_regressor_with_cv_gridsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingcvregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack))
        search_params = dict(reg__kneighborsregressor__n_neighbors=[2, 4, 5],
                             reg__bayesianridge__alpha_1=[1e-7, 1e-6],
                             reg__meta_regressor__alpha=[1.0],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 2.8)
        self.assertIn(reg.best_params_['reg__kneighborsregressor__n_neighbors'], [2, 4, 5])
        self.assertIn(reg.best_params_['reg__bayesianridge__alpha_1'], [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__meta_regressor__alpha'], [1.0])

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressor_without_cv_gridsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack))
        search_params = dict(reg__kneighborsregressor__n_neighbors=[2, 4, 5],
                             reg__bayesianridge__alpha_1=[1e-7, 1e-6],
                             reg__meta_regressor__alpha=[1.0],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 10.0)
        self.assertIn(reg.best_params_['reg__estimator__kneighborsregressor__n_neighbors'],
                      [2, 4, 5])
        self.assertIn(reg.best_params_['reg__estimator__bayesianridge__alpha_1'], [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__estimator__meta_regressor__alpha'], [1.0])

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressor_with_cv_gridsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingcvregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack))
        search_params = dict(reg__kneighborsregressor__n_neighbors=[2, 4, 5],
                             reg__bayesianridge__alpha_1=[1e-7, 1e-6],
                             reg__meta_regressor__alpha=[1.0],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 10.0)
        self.assertIn(reg.best_params_['reg__estimator__kneighborsregressor__n_neighbors'],
                      [2, 4, 5])
        self.assertIn(reg.best_params_['reg__estimator__bayesianridge__alpha_1'], [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__estimator__meta_regressor__alpha'], [1.0])

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressorchain_without_cv_gridsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack),
                        chain_order=[2, 0, 1])
        search_params = dict(reg__kneighborsregressor__n_neighbors=[2, 4, 5],
                             reg__bayesianridge__alpha_1=[1e-7, 1e-6],
                             reg__meta_regressor__alpha=[1.0],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 12.0)
        self.assertIn(reg.best_params_['reg__base_estimator__kneighborsregressor__n_neighbors'],
                      [2, 4, 5])
        self.assertIn(reg.best_params_['reg__base_estimator__bayesianridge__alpha_1'],
                      [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__base_estimator__meta_regressor__alpha'],
                      [1.0])

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressorchain_with_cv_gridsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingcvregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack),
                        chain_order=[2, 0, 1])
        search_params = dict(reg__kneighborsregressor__n_neighbors=[2, 4, 5],
                             reg__bayesianridge__alpha_1=[1e-7, 1e-6],
                             reg__meta_regressor__alpha=[1.0],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 12.0)
        self.assertIn(reg.best_params_['reg__base_estimator__kneighborsregressor__n_neighbors'],
                      [2, 4, 5])
        self.assertIn(reg.best_params_['reg__base_estimator__bayesianridge__alpha_1'],
                      [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__base_estimator__meta_regressor__alpha'],
                      [1.0])

    def test_stacking_regressor_without_cv_randomizedsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack),
                        randomizedcv_n_iter=6)
        search_params = dict(reg__kneighborsregressor__n_neighbors=randint(low=2, high=5),
                             reg__bayesianridge__alpha_1=[1e-7, 1e-6],
                             reg__meta_regressor__alpha=[1.0],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 3.0)
        self.assertLessEqual(reg.best_params_['reg__kneighborsregressor__n_neighbors'], 5)
        self.assertGreaterEqual(reg.best_params_['reg__kneighborsregressor__n_neighbors'], 2)
        self.assertIn(reg.best_params_['reg__bayesianridge__alpha_1'], [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__meta_regressor__alpha'], [1.0])

    def test_stacking_regressor_with_cv_randomizedsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingcvregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack),
                        randomizedcv_n_iter=6)
        search_params = dict(reg__kneighborsregressor__n_neighbors=randint(low=2, high=5),
                             reg__bayesianridge__alpha_1=[1e-7, 1e-6],
                             reg__meta_regressor__alpha=[1.0],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 3.0)
        self.assertLessEqual(reg.best_params_['reg__kneighborsregressor__n_neighbors'], 5)
        self.assertGreaterEqual(reg.best_params_['reg__kneighborsregressor__n_neighbors'], 2)
        self.assertIn(reg.best_params_['reg__bayesianridge__alpha_1'], [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__meta_regressor__alpha'], [1.0])

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressor_without_cv_randomizedsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack),
                        randomizedcv_n_iter=6)
        search_params = dict(reg__kneighborsregressor__n_neighbors=randint(low=2, high=5),
                             reg__bayesianridge__alpha_1=[1e-7, 1e-6],
                             reg__meta_regressor__alpha=[1.0],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 12.8)
        self.assertLessEqual(reg.best_params_['reg__estimator__kneighborsregressor__n_neighbors'],
                             5)
        self.assertGreaterEqual(reg.best_params_['reg__estimator__kneighborsregressor__n_neighbors'],
                                2)
        self.assertIn(reg.best_params_['reg__estimator__bayesianridge__alpha_1'], [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__estimator__meta_regressor__alpha'], [1.0])

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressor_with_cv_randomizedsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingcvregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack),
                        randomizedcv_n_iter=6)
        search_params = dict(reg__kneighborsregressor__n_neighbors=randint(low=2, high=5),
                             reg__bayesianridge__alpha_1=[1e-7, 1e-6],
                             reg__meta_regressor__alpha=[1.0],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 12.8)
        self.assertLessEqual(reg.best_params_['reg__estimator__kneighborsregressor__n_neighbors'],
                             5)
        self.assertGreaterEqual(reg.best_params_['reg__estimator__kneighborsregressor__n_neighbors'],
                                2)
        self.assertIn(reg.best_params_['reg__estimator__bayesianridge__alpha_1'], [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__estimator__meta_regressor__alpha'], [1.0])

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressorchain_without_cv_randomizedsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack),
                        randomizedcv_n_iter=6,
                        chain_order=[2, 0, 1])
        search_params = dict(reg__kneighborsregressor__n_neighbors=randint(low=2, high=5),
                             reg__bayesianridge__alpha_1=[1e-7, 1e-6],
                             reg__meta_regressor__alpha=[1.0],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 12.8)
        self.assertLessEqual(reg.best_params_['reg__base_estimator__kneighborsregressor__n_neighbors'],
                             5)
        self.assertGreaterEqual(reg.best_params_['reg__base_estimator__kneighborsregressor__n_neighbors'],
                                2)
        self.assertIn(reg.best_params_['reg__base_estimator__bayesianridge__alpha_1'],
                      [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__base_estimator__meta_regressor__alpha'],
                      [1.0])

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressorchain_with_cv_randomizedsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingcvregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack),
                        randomizedcv_n_iter=6,
                        chain_order=[2, 0, 1])
        search_params = dict(reg__kneighborsregressor__n_neighbors=randint(low=2, high=5),
                             reg__bayesianridge__alpha_1=[1e-7, 1e-6],
                             reg__meta_regressor__alpha=[1.0],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 12.8)
        self.assertLessEqual(reg.best_params_['reg__base_estimator__kneighborsregressor__n_neighbors'],
                             5)
        self.assertGreaterEqual(reg.best_params_['reg__base_estimator__kneighborsregressor__n_neighbors'],
                                2)
        self.assertIn(reg.best_params_['reg__base_estimator__bayesianridge__alpha_1'],
                      [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__base_estimator__meta_regressor__alpha'],
                      [1.0])

    def test_stacking_regressor_without_cv_fit_score(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack))
        reg.fit(X_train, y_train)
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred)
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'].values, 0.0)
        self.assertGreaterEqual(score['mse'].values, 0.0)
        self.assertLess(score['mae'].values, 2.7)
        self.assertLess(score['mse'].values, 19.0)

    def test_stacking_regressor_with_cv_fit_score(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingcvregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack))
        reg.fit(X_train, y_train)
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred)
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'].values, 0.0)
        self.assertGreaterEqual(score['mse'].values, 0.0)
        self.assertLess(score['mae'].values, 2.7)
        self.assertLess(score['mse'].values, 19.0)

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressor_without_cv_fit_score(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack))
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred).mean()
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'], 0.0)
        self.assertGreaterEqual(score['mse'], 0.0)
        self.assertLess(score['mae'], 11.0)
        self.assertLess(score['mse'], 260.0)

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressor_with_cv_fit_score(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingcvregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack))
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred).mean()
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'], 0.0)
        self.assertGreaterEqual(score['mse'], 0.0)
        self.assertLess(score['mae'], 9.0)
        self.assertLess(score['mse'], 190.0)

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressorchain_without_cv_fit_score(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack),
                        chain_order=[2, 0, 1])
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred).mean()
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'], 0.0)
        self.assertGreaterEqual(score['mse'], 0.0)
        self.assertLess(score['mae'], 10.0)
        self.assertLess(score['mse'], 180.0)

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressorchain_with_cv_fit_score(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingcvregressor',
                        pipeline_transform='standardscaler',
                        stacking_options=dict(layers=stack),
                        chain_order=[2, 0, 1])
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred).mean()
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'], 0.0)
        self.assertGreaterEqual(score['mse'], 0.0)
        self.assertLess(score['mae'], 7.0)
        self.assertLess(score['mse'], 110.0)

    def test_without_cv_pipeline_clone_fit_score(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)
        transformer_list = [('pca', PCA(n_components=1)),
                            ('svd', TruncatedSVD(n_components=2))]
        union = FeatureUnion(transformer_list=transformer_list, n_jobs=-1)
        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingregressor',
                        pipeline_transform=('tr', union),
                        stacking_options=dict(layers=stack))
        reg.get_pipeline(y=y_train)
        _class_before_clone = reg.pipe.__class__
        reg.pipe = clone(reg.pipe)
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred)
        self.assertEqual(_class_before_clone, reg.pipe.__class__)
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'].values, 0.0)
        self.assertGreaterEqual(score['mse'].values, 0.0)
        self.assertLess(score['mae'].values, 11.0)
        self.assertLess(score['mse'].values, 232.0)

    def test_with_cv_pipeline_clone_fit_score(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)
        transformer_list = [('pca', PCA(n_components=1)),
                            ('svd', TruncatedSVD(n_components=2))]
        union = FeatureUnion(transformer_list=transformer_list, n_jobs=-1)
        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingcvregressor',
                        pipeline_transform=('tr', union),
                        stacking_options=dict(layers=stack))
        reg.get_pipeline(y=y_train)
        _class_before_clone = reg.pipe.__class__
        reg.pipe = clone(reg.pipe)
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred)
        self.assertEqual(_class_before_clone, reg.pipe.__class__)
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'].values, 0.0)
        self.assertGreaterEqual(score['mse'].values, 0.0)
        self.assertLess(score['mae'].values, 11.0)
        self.assertLess(score['mse'].values, 232.0)

    def test_without_cv_shap_explainer(self):
        X_train, _, y_train, _ = load_benchmark(return_split=True)
        index = 3
        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        interpret = ShapInterpret(regressor_choice='mlxtendstackingregressor',
                                  target_index=index,
                                  stacking_options=dict(layers=stack))
        interpret.fit(X=X_train, y=y_train, index=index)
        explainer, shap_values = interpret.explainer(X=X_train)
        self.assertEqual(X_train.shape, shap_values.shape)

    def test_with_cv_shap_explainer(self):
        X_train, _, y_train, _ = load_benchmark(return_split=True)
        index = 3
        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        interpret = ShapInterpret(regressor_choice='mlxtendstackingcvregressor',
                                  target_index=index,
                                  stacking_options=dict(layers=stack))
        interpret.fit(X=X_train, y=y_train, index=index)
        explainer, shap_values = interpret.explainer(X=X_train)
        self.assertEqual(X_train.shape, shap_values.shape)


if __name__ == '__main__':
    unittest.main()
