"""
Unit tests for LightGBM compatibility.
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


class TestLightGBM(unittest.TestCase):

    def test_regressor_gridsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        params = dict(n_estimators=3, objective='mean_squared_error')
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler',
                        params=params)
        search_params = dict(reg__n_estimators=[3, 5, 10],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 3.8)
        self.assertIn(reg.best_params_['reg__n_estimators'], [3, 5, 10])

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressor_gridsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        params = dict(n_estimators=3, objective='mean_squared_error')
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler',
                        params=params)
        search_params = dict(reg__n_estimators=[3, 5, 10],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 10.0)
        self.assertIn(reg.best_params_['reg__estimator__n_estimators'], [3, 5, 10])

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressorchain_gridsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        params = dict(n_estimators=3, objective='mean_squared_error')
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler',
                        params=params, chain_order=[2, 0, 1])
        search_params = dict(reg__n_estimators=[3, 5, 10],
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 10.0)
        self.assertIn(reg.best_params_['reg__base_estimator__n_estimators'], [3, 5, 10])

    def test_regressor_randomizedsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        params = dict(n_estimators=3, objective='mean_squared_error')
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler',
                        params=params, randomizedcv_n_iter=6)
        search_params = dict(reg__n_estimators=randint(low=3, high=10),
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 4.5)
        self.assertLessEqual(reg.best_params_['reg__n_estimators'], 10)
        self.assertGreaterEqual(reg.best_params_['reg__n_estimators'], 3)

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressor_randomizedsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        params = dict(n_estimators=3, objective='mean_squared_error')
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler',
                        params=params, randomizedcv_n_iter=6)
        search_params = dict(reg__n_estimators=randint(low=3, high=10),
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 10.0)
        self.assertLessEqual(reg.best_params_['reg__estimator__n_estimators'], 10)
        self.assertGreaterEqual(reg.best_params_['reg__estimator__n_estimators'], 3)

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressorchain_randomizedsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        params = dict(n_estimators=3, objective='mean_squared_error')
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler',
                        params=params, randomizedcv_n_iter=6, chain_order=[2, 0, 1])
        search_params = dict(reg__n_estimators=randint(low=3, high=10),
                             tr__with_std=[True, False])
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 10.0)
        self.assertLessEqual(reg.best_params_['reg__base_estimator__n_estimators'], 10)
        self.assertGreaterEqual(reg.best_params_['reg__base_estimator__n_estimators'], 3)

    def test_regressor_fit_score(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        params = dict(n_estimators=10, objective='mean_squared_error',
                      boosting_type='goss')
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler',
                        params=params)
        reg.fit(X_train, y_train)
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred)
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'].values, 0.0)
        self.assertGreaterEqual(score['mse'].values, 0.0)
        self.assertLess(score['mae'].values, 3.4)
        self.assertLess(score['mse'].values, 24.0)

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressor_fit_score(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        params = dict(n_estimators=3, objective='mean_squared_error')
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler',
                        params=params)
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred).mean()
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'], 0.0)
        self.assertGreaterEqual(score['mse'], 0.0)
        self.assertLess(score['mae'], 8.1)
        self.assertLess(score['mse'], 122.5)

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressorchain_fit_score(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        params = dict(n_estimators=3, objective='mean_squared_error')
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler',
                        params=params, chain_order=[0, 2, 1])
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred).mean()
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'], 0.0)
        self.assertGreaterEqual(score['mse'], 0.0)
        self.assertLess(score['mae'], 8.1)
        self.assertLess(score['mse'], 122.5)

    def test_pipeline_clone_fit_score(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)
        transformer_list = [('pca', PCA(n_components=1)),
                            ('svd', TruncatedSVD(n_components=2))]
        union = FeatureUnion(transformer_list=transformer_list, n_jobs=-1)
        params = dict(n_estimators=3, objective='mean_squared_error')
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform=('tr', union),
                        params=params)
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

    def test_shap_explainer(self):
        X_train, _, y_train, _ = load_benchmark(return_split=True)
        index = 3
        params = dict(n_estimators=3, objective='mean_squared_error')
        interpret = ShapInterpret(regressor_choice='lgbmregressor', target_index=index,
                                  params=params)
        interpret.fit(X=X_train, y=y_train, index=index)
        explainer, shap_values = interpret.explainer(X=X_train)
        self.assertEqual(X_train.shape, shap_values.shape)


if __name__ == '__main__':
    unittest.main()
