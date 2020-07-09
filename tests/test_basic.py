import unittest

import pandas as pd

from scipy.stats import uniform
from sklearn.datasets import load_boston
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
        reg = Regressor(regressor_choice='ridge', pipeline_transform='standard_scaler')
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
        reg = Regressor(regressor_choice='ridge', pipeline_transform='standard_scaler')
        reg.search(X_train, y_train, search_params=search_params,
                   search_style='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 3.6)
        self.assertLessEqual(reg.best_params_['reg__alpha'], 1.51)
        self.assertGreaterEqual(reg.best_params_['reg__alpha'], 0.01)
        self.assertIn(reg.best_params_['reg__fit_intercept'], [True, False])

    def test_regressor_fit_score(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        reg = Regressor(regressor_choice='ridge', pipeline_transform='standard_scaler')
        reg.fit(X_train, y_train)
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred)
        self.assertCountEqual(X_test.index, y_test.index)
        self.assertLess(score['mae'].values, 3.1)
        self.assertLess(score['mse'].values, 23.0)

if __name__ == '__main__':
    unittest.main()