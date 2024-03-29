"""
Unit tests for base boost capability.
"""

# Author: Alex Wozniakowski
# License: MIT

import unittest
import sklearn.ensemble

import pandas as pd

from sklearn.datasets import fetch_california_housing, load_linnerud
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from physlearn import ModifiedPipeline, Regressor
from physlearn.datasets import load_benchmark
from physlearn.loss import LOSS_FUNCTIONS
from physlearn.supervised import ShapInterpret


class TestBaseBoost(unittest.TestCase):

    def test_benchmark(self):
        _, X_test, _, y_test = load_benchmark(return_split=True)
        reg = Regressor()
        score = reg.score(y_test, X_test)
        self.assertEqual(score['mae'].mean().round(decimals=2), 1.34)
        self.assertEqual(score['mse'].mean().round(decimals=2), 4.19)
        self.assertEqual(score['rmse'].mean().round(decimals=2), 1.88)
        self.assertEqual(score['r2'].mean().round(decimals=2), 0.99)
        self.assertEqual(score['ev'].mean().round(decimals=2), 0.99)

    def test_pipeline_score_uniform_average(self):
        X_train, X_test, y_train, y_test = load_benchmark(return_split=True)
        line_search_options = dict(init_guess=1, opt_method='minimize',
                                   method='Nelder-Mead', tol=1e-7,
                                   options={"maxiter": 10000},
                                   niter=None, T=None, loss='lad',
                                   regularization=0.1)
        base_boosting_options = dict(n_regressors=3, boosting_loss='lad',
                                     line_search_options=line_search_options)
        pipe = ModifiedPipeline(steps=[('scaler', StandardScaler()), ('reg', Ridge())],
                                base_boosting_options=base_boosting_options)
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test, multioutput='uniform_average')
        self.assertEqual(score['mae'].mean().round(decimals=2), 1.19)
        self.assertEqual(score['mse'].mean().round(decimals=2), 3.49)
        self.assertEqual(score['rmse'].mean().round(decimals=2), 1.87)
        self.assertEqual(score['r2'].mean().round(decimals=2), 0.99)
        self.assertEqual(score['ev'].mean().round(decimals=2), 0.99)

    def test_return_incumbent(self):
        X_train, X_test, y_train, y_test = load_benchmark(return_split=True)
        linear_basis_fn = 'ridge'
        n_regressors = 1
        boosting_loss = 'ls'
        line_search_options = dict(init_guess=1, opt_method='minimize',
                                   method='Nelder-Mead', tol=1e-7,
                                   options={"maxiter": 10000},
                                   niter=None, T=None, loss='lad',
                                   regularization=0.1)

        base_boosting_options = dict(n_regressors=n_regressors,
                                     boosting_loss=boosting_loss,
                                     line_search_options=line_search_options)
        index = 3
        reg = Regressor(regressor_choice=linear_basis_fn, params=dict(alpha=0.1),
                        target_index=index, base_boosting_options=base_boosting_options)
        reg.baseboostcv(X_train.iloc[:10, :], y_train.iloc[:10, :])
        self.assertHasAttr(reg, 'return_incumbent_')

    def test_baseboostcv_score(self):
        X_train, X_test, y_train, y_test = load_benchmark(return_split=True)
        stack = dict(regressors=['ridge', 'lgbmregressor'],
                     final_regressor='ridge')
        line_search_options = dict(init_guess=1, opt_method='minimize',
                                   method='Nelder-Mead', tol=1e-7,
                                   options={"maxiter": 10000},
                                   niter=None, T=None, loss='lad',
                                   regularization=0.1)
        base_boosting_options = dict(n_regressors=3,
                                     boosting_loss='ls',
                                     line_search_options=line_search_options)
        reg = Regressor(regressor_choice='stackingregressor', target_index=0,
                        stacking_options=dict(layers=stack),
                        base_boosting_options=base_boosting_options)
        y_pred = reg.baseboostcv(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred)
        self.assertNotHasAttr(reg, 'return_incumbent_')
        self.assertGreaterEqual(score['mae'].values, 0.0)
        self.assertGreaterEqual(score['mse'].values, 0.0)
        self.assertLess(score['mae'].values, 2.0)
        self.assertLess(score['mse'].values, 6.2)

    def test_squared_error(self):
        X, y = fetch_california_housing(return_X_y=True)
        loss = LOSS_FUNCTIONS['ls']()
        score = loss(y=y, raw_predictions=X[:, 0]).round(decimals=2)
        sklearn_loss = sklearn.ensemble._gb_losses.LOSS_FUNCTIONS['ls']()
        sklearn_score = sklearn_loss(y=y, raw_predictions=X[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)
        score = loss(y=pd.Series(y), raw_predictions=X[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)
        score = loss(y=pd.Series(y),
                     raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)
        score = loss(y=y,
                     raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)

        X, y = load_linnerud(return_X_y=True)
        for i in range(y.shape[1]):
            score = loss(y=y[:, i], raw_predictions=X[:, 0]).round(decimals=2)
            sklearn_score = sklearn_loss(y=y[:, i],
                                         raw_predictions=X[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)
            score = loss(y=pd.DataFrame(y).iloc[:, i],
                         raw_predictions=X[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)
            score = loss(y=pd.DataFrame(y).iloc[:, i],
                         raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)
            score = loss(y=y[:, i],
                         raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)

    def test_absolute_error(self):
        X, y = fetch_california_housing(return_X_y=True)
        loss = LOSS_FUNCTIONS['lad']()
        score = loss(y=y, raw_predictions=X[:, 0]).round(decimals=2)
        sklearn_loss = sklearn.ensemble._gb_losses.LOSS_FUNCTIONS['lad']()
        sklearn_score = sklearn_loss(y=y, raw_predictions=X[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)
        score = loss(y=pd.Series(y), raw_predictions=X[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)
        score = loss(y=pd.Series(y),
                     raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)
        score = loss(y=y,
                     raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)

        X, y = load_linnerud(return_X_y=True)
        for i in range(y.shape[1]):
            score = loss(y=y[:, i], raw_predictions=X[:, 0]).round(decimals=2)
            sklearn_score = sklearn_loss(y=y[:, i],
                                         raw_predictions=X[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)
            score = loss(y=pd.DataFrame(y).iloc[:, i],
                         raw_predictions=X[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)
            score = loss(y=pd.DataFrame(y).iloc[:, i],
                         raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)
            score = loss(y=y[:, i],
                         raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)

    def test_huber_loss(self):
        X, y = fetch_california_housing(return_X_y=True)
        loss = LOSS_FUNCTIONS['huber'](alpha=0.9)
        score = loss(y=y, raw_predictions=X[:, 0]).round(decimals=2)
        sklearn_loss = sklearn.ensemble._gb_losses.LOSS_FUNCTIONS['huber'](alpha=0.9)
        sklearn_score = sklearn_loss(y=y, raw_predictions=X[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)
        score = loss(y=pd.Series(y), raw_predictions=X[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)
        score = loss(y=pd.Series(y),
                     raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)
        score = loss(y=y,
                     raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)

        X, y = load_linnerud(return_X_y=True)
        for i in range(y.shape[1]):
            score = loss(y=y[:, i], raw_predictions=X[:, 0]).round(decimals=2)
            sklearn_score = sklearn_loss(y=y[:, i],
                                         raw_predictions=X[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)
            score = loss(y=pd.DataFrame(y).iloc[:, i],
                         raw_predictions=X[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)
            score = loss(y=pd.DataFrame(y).iloc[:, i],
                         raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)
            score = loss(y=y[:, i],
                         raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)

    def test_quantile_loss(self):
        X, y = fetch_california_housing(return_X_y=True)
        loss = LOSS_FUNCTIONS['quantile']()
        score = loss(y=y, raw_predictions=X[:, 0]).round(decimals=2)
        sklearn_loss = sklearn.ensemble._gb_losses.LOSS_FUNCTIONS['quantile']()
        sklearn_score = sklearn_loss(y=y, raw_predictions=X[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)
        score = loss(y=pd.Series(y), raw_predictions=X[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)
        score = loss(y=pd.Series(y),
                     raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)
        score = loss(y=y,
                     raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)

        X, y = load_linnerud(return_X_y=True)
        for i in range(y.shape[1]):
            score = loss(y=y[:, i], raw_predictions=X[:, 0]).round(decimals=2)
            sklearn_score = sklearn_loss(y=y[:, i],
                                         raw_predictions=X[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)
            score = loss(y=pd.DataFrame(y).iloc[:, i],
                         raw_predictions=X[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)
            score = loss(y=pd.DataFrame(y).iloc[:, i],
                         raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)
            score = loss(y=y[:, i],
                         raw_predictions=pd.DataFrame(X).iloc[:, 0]).round(decimals=2)
            self.assertEqual(score, sklearn_score)

    def assertHasAttr(self, obj, attr):
        testBool = hasattr(obj, attr)
        self.assertTrue(testBool,
                        msg='object: %s lacks attribute: %s' % (obj, attr))

    def assertNotHasAttr(self, obj, attr):
        testBool = hasattr(obj, attr)
        self.assertFalse(testBool,
                         msg='object: %s has attribute: %s' % (obj, attr))


if __name__ == '__main__':
    unittest.main()
