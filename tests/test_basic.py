import unittest

import pandas as pd

from scipy.stats import uniform
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

        search_params = dict(alpha=[0.1, 0.2, 0.5],
                             fit_intercept=[True, False])
        reg = Regressor(regressor_choice='ridge', pipeline_transform='standardscaler')
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 3.6)
        self.assertIn(reg.best_params_['reg__alpha'], [0.1, 0.2, 0.5])
        self.assertIn(reg.best_params_['reg__fit_intercept'], [True, False])

    def test_regressor_randomizedsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        search_params = dict(alpha=uniform(loc=0.01, scale=1.5),
                             fit_intercept=[True, False])
        reg = Regressor(regressor_choice='ridge', pipeline_transform='standardscaler')
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 3.6)
        self.assertLessEqual(reg.best_params_['reg__alpha'], 1.51)
        self.assertGreaterEqual(reg.best_params_['reg__alpha'], 0.01)
        self.assertIn(reg.best_params_['reg__fit_intercept'], [True, False])

    def test_regressor_bayesoptcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        search_pbounds = dict(gamma=(0.1, 2.0), epsilon=(0.1, 0.4))
        reg = Regressor(regressor_choice='svr', pipeline_transform='standardscaler')
        reg.search(X_train, y_train, search_params=search_pbounds,
                   search_method='bayesoptcv')
        self.assertLess(reg.best_score_.values, 3.7)
        self.assertLessEqual(reg.best_params_['reg__gamma'], 2.0)
        self.assertGreaterEqual(reg.best_params_['reg__gamma'], 0.1)
        self.assertLessEqual(reg.best_params_['reg__epsilon'], 0.4)
        self.assertGreaterEqual(reg.best_params_['reg__epsilon'], 0.1)

    def test_regressor_fit_score(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        reg = Regressor(regressor_choice='ridge', pipeline_transform='standardscaler')
        reg.fit(X_train, y_train)
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred)
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'].values, 0.0)
        self.assertGreaterEqual(score['mse'].values, 0.0)
        self.assertLess(score['mae'].values, 3.1)
        self.assertLess(score['mse'].values, 23.0)

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressor(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        reg = Regressor(regressor_choice='ridge', pipeline_transform='standardscaler')
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred).mean()
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'], 0.0)
        self.assertGreaterEqual(score['mse'], 0.0)
        self.assertLess(score['mae'], 11.0)
        self.assertLess(score['mse'], 232.0)

    # sklearn < 0.23 does not have as_frame parameter
    @unittest.skipIf(sk_version < '0.23.0', 'scikit-learn version is less than 0.23')
    def test_multioutput_regressorchain(self):
        bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
        X, y = bunch['data'], bunch['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        reg = Regressor(regressor_choice='ridge', pipeline_transform='standardscaler',
                        chain_order=[0, 2, 1])
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred).mean()
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'], 0.0)
        self.assertGreaterEqual(score['mse'], 0.0)
        self.assertLess(score['mae'], 11.0)
        self.assertLess(score['mse'], 237.0)

if __name__ == '__main__':
    unittest.main()