"""
Unit tests for Mlxtend compatibility.
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

    # sklearn < 0.22 does not have a stacking regressor
    @unittest.skipIf(sk_version < '0.22.0', 'scikit-learn version is less than 0.22')
    def test_stacking_regressor_without_cv_gridsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler')
        search_params = {'kneighborsregressor__n_neighbors': [2, 4, 5],
                         'bayesianridge__alpha_1': [1e-7, 1e-6],
                         'meta_regressor__alpha': [1.0]}
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 3.0)
        self.assertIn(reg.best_params_['reg__kneighborsregressor__n_neighbors'], [2, 4, 5])
        self.assertIn(reg.best_params_['reg__bayesianridge__alpha_1'], [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__meta_regressor__alpha'], [1.0])

    # sklearn < 0.22 does not have a stacking regressor
    @unittest.skipIf(sk_version < '0.22.0', 'scikit-learn version is less than 0.22')
    def test_stacking_regressor_with_cv_gridsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingcvregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler')
        search_params = {'kneighborsregressor__n_neighbors': [2, 4, 5],
                         'bayesianridge__alpha_1': [1e-7, 1e-6],
                         'meta_regressor__alpha': [1.0]}
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

        reg = Regressor(regressor_choice='mlxtendstackingregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler')
        search_params = {'kneighborsregressor__n_neighbors': [2, 4, 5],
                         'bayesianridge__alpha_1': [1e-7, 1e-6],
                         'meta_regressor__alpha': [1.0]}
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

        reg = Regressor(regressor_choice='mlxtendstackingcvregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler')
        search_params = {'kneighborsregressor__n_neighbors': [2, 4, 5],
                         'bayesianridge__alpha_1': [1e-7, 1e-6],
                         'meta_regressor__alpha': [1.0]}
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

        reg = Regressor(regressor_choice='mlxtendstackingregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler', chain_order=[2, 0, 1])
        search_params = {'kneighborsregressor__n_neighbors': [2, 4, 5],
                         'bayesianridge__alpha_1': [1e-7, 1e-6],
                         'meta_regressor__alpha': [1.0]}
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 10.0)
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

        reg = Regressor(regressor_choice='mlxtendstackingcvregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler', chain_order=[2, 0, 1])
        search_params = {'kneighborsregressor__n_neighbors': [2, 4, 5],
                         'bayesianridge__alpha_1': [1e-7, 1e-6],
                         'meta_regressor__alpha': [1.0]}
        reg.search(X_train, y_train, search_params=search_params)
        self.assertLess(reg.best_score_.values, 10.0)
        self.assertIn(reg.best_params_['reg__base_estimator__kneighborsregressor__n_neighbors'],
                      [2, 4, 5])
        self.assertIn(reg.best_params_['reg__base_estimator__bayesianridge__alpha_1'],
                      [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__base_estimator__meta_regressor__alpha'],
                      [1.0])

    # sklearn < 0.22 does not have a stacking regressor
    @unittest.skipIf(sk_version < '0.22.0', 'scikit-learn version is less than 0.22')
    def test_stacking_regressor_without_cv_randomizedsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler')
        search_params = {'kneighborsregressor__n_neighbors': randint(low=2, high=5),
                         'bayesianridge__alpha_1': [1e-7, 1e-6],
                         'meta_regressor__alpha': [1.0]}
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 3.0)
        self.assertLessEqual(reg.best_params_['reg__kneighborsregressor__n_neighbors'], 5)
        self.assertGreaterEqual(reg.best_params_['reg__kneighborsregressor__n_neighbors'], 2)
        self.assertIn(reg.best_params_['reg__bayesianridge__alpha_1'], [1e-7, 1e-6])
        self.assertIn(reg.best_params_['reg__meta_regressor__alpha'], [1.0])

    # sklearn < 0.22 does not have a stacking regressor
    @unittest.skipIf(sk_version < '0.22.0', 'scikit-learn version is less than 0.22')
    def test_stacking_regressor_with_cv_randomizedsearchcv(self):
        X, y = load_boston(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')

        reg = Regressor(regressor_choice='mlxtendstackingcvregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler')
        search_params = {'kneighborsregressor__n_neighbors': randint(low=2, high=5),
                         'bayesianridge__alpha_1': [1e-7, 1e-6],
                         'meta_regressor__alpha': [1.0]}
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 2.8)
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

        reg = Regressor(regressor_choice='mlxtendstackingregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler')
        search_params = {'kneighborsregressor__n_neighbors': randint(low=2, high=5),
                         'bayesianridge__alpha_1': [1e-7, 1e-6],
                         'meta_regressor__alpha': [1.0]}
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 10.4)
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

        reg = Regressor(regressor_choice='mlxtendstackingcvregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler')
        search_params = {'kneighborsregressor__n_neighbors': randint(low=2, high=5),
                         'bayesianridge__alpha_1': [1e-7, 1e-6],
                         'meta_regressor__alpha': [1.0]}
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 10.0)
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

        reg = Regressor(regressor_choice='mlxtendstackingregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler', chain_order=[2, 0, 1])
        search_params = {'kneighborsregressor__n_neighbors': randint(low=2, high=5),
                         'bayesianridge__alpha_1': [1e-7, 1e-6],
                         'meta_regressor__alpha': [1.0]}
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 10.8)
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

        reg = Regressor(regressor_choice='mlxtendstackingcvregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler', chain_order=[2, 0, 1])
        search_params = {'kneighborsregressor__n_neighbors': randint(low=2, high=5),
                         'bayesianridge__alpha_1': [1e-7, 1e-6],
                         'meta_regressor__alpha': [1.0]}
        reg.search(X_train, y_train, search_params=search_params,
                   search_method='randomizedsearchcv')
        self.assertLess(reg.best_score_.values, 10.0)
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler')
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingcvregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler')
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler')
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingcvregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler')
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler', chain_order=[2, 0, 1])
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        stack = dict(regressors=['kneighborsregressor', 'bayesianridge'],
                     final_regressor='lasso')
        reg = Regressor(regressor_choice='mlxtendstackingcvregressor', stacking_layer=stack,
                        pipeline_transform='standardscaler', chain_order=[2, 0, 1])
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        score = reg.score(y_test, y_pred).mean()
        self.assertCountEqual(y_pred.index, y_test.index)
        self.assertGreaterEqual(score['mae'], 0.0)
        self.assertGreaterEqual(score['mse'], 0.0)
        self.assertLess(score['mae'], 7.0)
        self.assertLess(score['mse'], 110.0)

if __name__ == '__main__':
    unittest.main()
