"""
Unit tests for base boost capability.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import unittest
import sklearn.ensemble

import pandas as pd

from sklearn.datasets import load_boston

from physlearn import Regressor
from physlearn.datasets import load_benchmark, paper_params
from physlearn.loss import LOSS_FUNCTIONS


class TestBasic(unittest.TestCase):

    def test_benchmark(self):
        _, X_test, _, y_test = load_benchmark(return_split=True)
        reg = Regressor()
        score = reg.score(y_test, X_test)
        self.assertEqual(score['mae'].mean().round(decimals=2), 1.34)
        self.assertEqual(score['mse'].mean().round(decimals=2), 4.19)

    def test_return_incumbent(self):
        X_train, X_test, y_train, y_test = load_benchmark(return_split=True)
        linear_basis_fn = 'ridge'
        n_regressors = 1
        boosting_loss = 'ls'
        line_search_regularization = 0.1
        line_search_options = dict(init_guess=1, opt_method='minimize',
                                   alg='Nelder-Mead', tol=1e-7,
                                   options={"maxiter": 10000},
                                   niter=None, T=None,
                                   loss='lad')
        index = 3
        reg = Regressor(regressor_choice=linear_basis_fn, n_regressors=n_regressors,
                        boosting_loss=boosting_loss, params=dict(alpha=0.1),
                        line_search_regularization=line_search_regularization,
                        line_search_options=line_search_options, target_index=index)
        y_pred = reg.baseboostcv(X_train.iloc[:20, :], y_train.iloc[:20, :]).predict(X_test)
        self.assertHasAttr(reg, 'return_incumbent_')

    def test_squared_error(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        loss = LOSS_FUNCTIONS['ls'](n_classes=1)
        score = loss(y=y, raw_predictions=X.iloc[:, 0]).round(decimals=2)
        sklearn_loss = sklearn.ensemble._gb_losses.LOSS_FUNCTIONS['ls'](n_classes=1)
        sklearn_score = sklearn_loss(y=y, raw_predictions=X.iloc[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)

    def test_absolute_error(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        loss = LOSS_FUNCTIONS['lad'](n_classes=1)
        score = loss(y=y, raw_predictions=X.iloc[:, 0]).round(decimals=2)
        sklearn_loss = sklearn.ensemble._gb_losses.LOSS_FUNCTIONS['lad'](n_classes=1)
        sklearn_score = sklearn_loss(y=y, raw_predictions=X.iloc[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)

    def test_huber_loss(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        loss = LOSS_FUNCTIONS['huber'](n_classes=1, alpha=0.9)
        score = loss(y=y, raw_predictions=X.iloc[:, 0]).round(decimals=2)
        sklearn_loss = sklearn.ensemble._gb_losses.LOSS_FUNCTIONS['huber'](n_classes=1,
                                                                           alpha=0.9)
        sklearn_score = sklearn_loss(y=y, raw_predictions=X.iloc[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)

    def test_quantile_loss(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        loss = LOSS_FUNCTIONS['quantile'](n_classes=1)
        score = loss(y=y, raw_predictions=X.iloc[:, 0]).round(decimals=2)
        sklearn_loss = sklearn.ensemble._gb_losses.LOSS_FUNCTIONS['quantile'](n_classes=1)
        sklearn_score = sklearn_loss(y=y, raw_predictions=X.iloc[:, 0]).round(decimals=2)
        self.assertEqual(score, sklearn_score)

    def assertHasAttr(self, obj, attr):
        testBool = hasattr(obj, attr)
        self.assertTrue(testBool,
                        msg='obj lacking an attribute. obj: %s, attr: %s' % (obj, attr))

if __name__ == '__main__':
    unittest.main()
