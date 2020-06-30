import os
import re
import joblib
import numpy as np

from scipy.stats import randint, uniform

from ..utils._model_checks import _prepare_best_params_filename


one_hidden_layer_sizes = [(i, ) for i in range(2, 15)]
two_hidden_layers_sizes = [(i, j, ) for i in range(2, 7) for j in range(2, 9)]
mlp_hidden_layers_sizes = one_hidden_layer_sizes + two_hidden_layers_sizes


mlp_grid_search_params = {'hidden_layer_sizes': mlp_hidden_layers_sizes,
                          'alpha': list(range(8, 20)),
                          'max_iter': list(range(4200, 6000, 20)),
                          'activation': ['relu', 'tanh']}

mlp_random_search_params = {'hidden_layer_sizes': [(2, 4, )],
                            'alpha': uniform(loc=5, scale=3),
                            'max_iter': uniform(loc=2000, scale=300)}

mlp_bayesian_search_pbounds = {'max_iter': (1500, 4000),
                               'alpha': (4.0, 18.0)}

lgb_grid_search_params = {'n_estimators': list(range(10, 20, 2)),
                          'max_depth': list(range(14, 30, 1)),
                          'num_leaves': list(range(5, 40, 1)),
                          'learning_rate': np.arange(0.01, 0.15, 0.01),
                          'reg_alpha': np.arange(0.1, 0.7, 0.1),
                          'reg_lambda': np.arange(0.1, 0.7, 0.1)}

mlp_grid_search_params = {'hidden_layer_sizes': mlp_hidden_layers_sizes,
                          'alpha': list(range(8, 20)),
                          'max_iter': list(range(4200, 6000, 20)),
                          'activation': ['relu', 'tanh']}

mlp_random_search_params = {'hidden_layer_sizes': [(2, 4, )],
                            'alpha': uniform(loc=5, scale=3),
                            'max_iter': uniform(loc=2000, scale=300)}

mlp_bayesian_search_pbounds = {'max_iter': (1500, 4000),
                               'alpha': (4.0, 18.0)}

stacking_gridsearchcv_params = {'0__n_estimators': list(range(50, 250, 10)),
                                '0__max_depth': list(range(14, 30, 4)),
                                '0__num_leaves': list(range(20, 40, 10)),
                                '0__learning_rate': np.arange(0.1, 0.7, 0.1),
                                '0__reg_alpha': np.arange(0.1, 0.7, 0.2),
                                '0__reg_lambda': np.arange(0.1, 0.7, 0.2),
                                '1__hidden_layer_sizes': mlp_hidden_layers_sizes,
                                '1__alpha': list(range(4, 20)),
                                '1__max_iter': list(range(4000, 6000, 200)),
                                'final_estimator__alpha': list(range(4, 20)),
                                'final_estimator__max_iter': list(range(2000, 5000, 50)),
                                'final_estimator__hidden_layer_sizes': mlp_hidden_layers_sizes}

mlxtendstacking_gridsearchcv_params = {'lgbmregressor__n_estimators': list(range(30, 200, 20)),
                                       'lgbmregressor__max_depth': list(range(14, 30, 4)),
                                       'lgbmregressor__num_leaves': list(range(20, 40, 10)),
                                       'lgbmregressor__learning_rate': np.arange(0.1, 0.7, 0.2),
                                       'lgbmregressor__reg_alpha': np.arange(0.1, 0.7, 0.2),
                                       'lgbmregressor__reg_lambda': np.arange(0.1, 0.7, 0.2),
                                       'mlpregressor__hidden_layer_sizes': mlp_hidden_layers_sizes,
                                       'mlpregressor__alpha': list(range(2, 20)),
                                       'mlpregressor__max_iter': list(range(1000, 5000, 250)),
                                       'meta_regressor__alpha': list(range(4, 20)),
                                       'meta_regressor__max_iter': list(range(1000, 5000, 250)),
                                       'meta_regressor__hidden_layer_sizes': mlp_hidden_layers_sizes}

stacking_randomizedsearchcv_params = {'0__n_estimators': list(range(30, 1500, 50)),
                                      '0__max_depth': list(range(14, 30, 3)),
                                      '0__num_leaves': randint(low=10, high=50),
                                      '0__learning_rate': uniform(loc=0.5, scale=0.25),
                                      '0__reg_alpha': uniform(loc=0.5, scale=0.25),
                                      '0__reg_lambda': uniform(loc=0.5, scale=0.25),
                                      '1__hidden_layer_sizes': mlp_hidden_layers_sizes,
                                      '1__alpha': list(range(4, 18)),
                                      '1__max_iter': uniform(loc=2000, scale=1000),
                                      'final_estimator__alpha': list(range(4, 18)),
                                      'final_estimator__max_iter': uniform(loc=2500, scale=1500),
                                      'final_estimator__hidden_layer_sizes': mlp_hidden_layers_sizes}

mlxtendstacking_randomizedsearchcv_params = {'lgbmregressor__n_estimators': list(range(30, 1500, 50)),
                                             'lgbmregressor__max_depth': list(range(14, 30, 3)),
                                             'lgbmregressor__num_leaves': randint(low=10, high=50),
                                             'lgbmregressor__learning_rate': uniform(loc=0.5, scale=0.25),
                                             'lgbmregressor__reg_alpha': uniform(loc=0.5, scale=0.25),
                                             'lgbmregressor__reg_lambda': uniform(loc=0.5, scale=0.25),
                                             'mlpregressor__hidden_layer_sizes': mlp_hidden_layers_sizes,
                                             'mlpregressor__alpha': list(range(4, 18)),
                                             'mlpregressor__max_iter': uniform(loc=2000, scale=1000),
                                             'meta_regressor__alpha': list(range(4, 18)),
                                             'meta_regressor__max_iter': uniform(loc=2500, scale=1500),
                                             'meta_regressor__hidden_layer_sizes': mlp_hidden_layers_sizes}

mlxtendstacking_bayesiansearch_pbounds = {'lgbmregressor__n_estimators': (30, 3000),
                                          'lgbmregressor__max_depth': (12, 40),
                                          'lgbmregressor__learning_rate': (0.05, 1.5),
                                          'lgbmregressor__reg_alpha': (0.05, 1.5),
                                          'lgbmregressor__reg_lambda': (0.05, 1.5),
                                          'mlpregressor__alpha': (2.0, 24.0),
                                          'mlpregressor__max_iter': (500, 5000),
                                          'meta_regressor__alpha': (2.0, 24.0),
                                          'meta_regressor__max_iter': (500, 5000)}

search_params = {'lgbm_gridsearchcv': lgb_grid_search_params,
                 'mlp_gridsearchcv': mlp_grid_search_params,
                 'mlp_randomizedsearchcv': mlp_random_search_params,
                 'mlp_bayesianoptimization': mlp_bayesian_search_pbounds,
                 'sklearn_stacking_gridsearchcv': stacking_gridsearchcv_params,
                 'sklearn_stacking_randomizedsearchcv': stacking_randomizedsearchcv_params,
                 'mlxtend_stacking_gridsearchcv': mlxtendstacking_gridsearchcv_params,
                 'mlxtend_stacking_randomizedsearchcv': mlxtendstacking_randomizedsearchcv_params,
                 'mlxtend_stacking_bayesianoptimization': mlxtendstacking_bayesiansearch_pbounds}
