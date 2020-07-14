"""
Unit tests for LightGBM compatibility.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import unittest

import pandas as pd

from scipy.stats import randint

from sklearn import __version__ as sk_version
from sklearn.datasets import load_boston, load_linnerud
from sklearn.model_selection import train_test_split

from physlearn import Regressor


class TestBasic(unittest.TestCase):

    def test_regressor_gridsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        search_params = dict(n_estimators=[3, 5, 10])
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler')
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

        search_params = dict(n_estimators=[3, 5, 10])
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler')
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

        search_params = dict(n_estimators=[3, 5, 10])
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler',
                        chain_order=[2, 0, 1])
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 10.0)
        self.assertIn(reg.best_params_['reg__base_estimator__n_estimators'], [3, 5, 10])

    def test_regressor_randomizedsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        search_params = dict(n_estimators=randint(low=3, high=10))
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler')
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 4.0)
        self.assertLessEqual(reg.best_params_['reg__n_estimators'], 10)
        self.assertGreaterEqual(reg.best_params_['reg__n_estimators'], 3)

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressor_randomizedsearchcv(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        search_params = dict(n_estimators=randint(low=3, high=10))
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler')
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

        search_params = dict(n_estimators=randint(low=3, high=10))
        reg = Regressor(regressor_choice='lgbmregressor', pipeline_transform='standardscaler',
                        chain_order=[2, 0, 1])
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 10.0)
        self.assertLessEqual(reg.best_params_['reg__base_estimator__n_estimators'], 10)
        self.assertGreaterEqual(reg.best_params_['reg__base_estimator__n_estimators'], 3)

    def test_regressor_fit_score(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

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

if __name__ == '__main__':
    unittest.main()
