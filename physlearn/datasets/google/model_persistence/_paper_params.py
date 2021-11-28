"""
The :mod:`physlearn.datasets.google.model_persistence._paper_params` module 
stores the (hyper)parameters used in Boosting on the shoulders of giants in
quantum device calibration. Moreover, it provides utilities for retrieving
these (hyper)parameters.

References
----------
- Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
  "A new formulation of gradient boosting",
  Machine Learning: Science and Technology, 2 045022 (2021).
"""

# Author: Alex Wozniakowski
# License: MIT

import copy


###########################################################
# BaseBoost params
###########################################################


mlp_param_template = dict(activation='tanh',
                          solver='lbfgs')

gbm_param_template = dict(objective='mean_absolute_error',
                          boosting_type='goss',
                          num_leaves=32,
                          max_depth=20,
                          learning_rate=0.2,
                          reg_alpha=0.3,
                          reg_lambda=0.3,
                          max_bin=512,
                          subsample_for_bin=200)


# Target 1
mlp_params_1 = copy.deepcopy(mlp_param_template)
mlp_params_1.update(dict(hidden_layer_sizes=(3, ),
                         alpha=17.0, max_iter=4390))
gbm_params_1 = copy.deepcopy(gbm_param_template)
gbm_params_1.update(dict(n_estimators=1060))
meta_params_1 = copy.deepcopy(mlp_param_template)
meta_params_1.update(dict(hidden_layer_sizes=(10, ),
                          alpha=15.0, max_iter=4070))

# Target 2
mlp_params_2 = copy.deepcopy(mlp_param_template)
mlp_params_2.update(dict(hidden_layer_sizes=(11, ),
                         alpha=15.0, max_iter=4440))
gbm_params_2 = copy.deepcopy(gbm_param_template)
gbm_params_2.update(dict(n_estimators=1100))
meta_params_2 = copy.deepcopy(mlp_param_template)
meta_params_2.update(dict(hidden_layer_sizes=(13, ),
                          alpha=15.0, max_iter=3700))

# Target 3
mlp_params_3 = copy.deepcopy(mlp_param_template)
mlp_params_3.update(dict(hidden_layer_sizes=(8, ),
                         alpha=5.0, max_iter=3500))
gbm_params_3 = copy.deepcopy(gbm_param_template)
gbm_params_3.update(dict(n_estimators=750))
meta_params_3 = copy.deepcopy(mlp_param_template)
meta_params_3.update(dict(hidden_layer_sizes=(8, ),
                          alpha=8.0, max_iter=2000))

# Target 4
mlp_params_4 = copy.deepcopy(mlp_param_template)
mlp_params_4.update(dict(hidden_layer_sizes=(10, ),
                         alpha=9.0, max_iter=4500))
gbm_params_4 = copy.deepcopy(gbm_param_template)
gbm_params_4.update(dict(n_estimators=380))
meta_params_4 = copy.deepcopy(mlp_param_template)
meta_params_4.update(dict(hidden_layer_sizes=(8, ),
                          alpha=10.0, max_iter=4250))

# Target 5
mlp_params_5 = copy.deepcopy(mlp_param_template)
mlp_params_5.update(dict(hidden_layer_sizes=(12, ),
                         alpha=8.0, max_iter=3450))
gbm_params_5 = copy.deepcopy(gbm_param_template)
gbm_params_5.update(dict(n_estimators=990))
meta_params_5 = copy.deepcopy(mlp_param_template)
meta_params_5.update(dict(hidden_layer_sizes=(7, ),
                          alpha=4.0, max_iter=3650))


paper_params_dict = dict(target_1=[[mlp_params_1, gbm_params_1], meta_params_1],
                         target_2=[[mlp_params_2, gbm_params_2], meta_params_2],
                         target_3=[[mlp_params_3, gbm_params_3], meta_params_3],
                         target_4=[[mlp_params_4, gbm_params_4], meta_params_4],
                         target_5=[[mlp_params_5, gbm_params_5], meta_params_5])

def paper_params(index=0) -> list:
    """Retrieves a list for StackingRegressor.

    Parameters
    ----------
    index : int
        Specifies the single-target regression subtask,
        using the Python indexing convention.

    Returns
    -------
    params : list

    Examples
    --------
    >>> from physlearn.datasets import paper_params
    >>> paper_params(index=0)
    [[{'activation': 'tanh',
       'solver': 'lbfgs',
       'hidden_layer_sizes': (3,),
       'alpha': 17.0,
       'max_iter': 4390},
      {'objective': 'mean_absolute_error',
       'boosting_type': 'goss',
       'num_leaves': 32,
       'max_depth': 20,
       'learning_rate': 0.2,
       'reg_alpha': 0.3,
       'reg_lambda': 0.3,
       'max_bin': 512,
       'subsample_for_bin': 200,
       'n_estimators': 1060}],
     {'activation': 'tanh',
      'solver': 'lbfgs',
      'hidden_layer_sizes': (10,),
      'alpha': 15.0,
      'max_iter': 4070}]
    """

    # Changes to the paper's index convention.
    index += 1
    assert index >=1 and index <=5
    return paper_params_dict[f'target_{index}']


###########################################################
#LGBMRegessor params
###########################################################

lgbm_param_template = dict(objective='mean_absolute_error',
                           boosting_type='goss',
                           num_leaves=20,
                           reg_alpha=0.01,
                           reg_lambda=0.01)


# Target 1
lgbm_params_1 = copy.deepcopy(lgbm_param_template)
lgbm_params_1.update(dict(n_estimators=60))

# Target 2
lgbm_params_2 = copy.deepcopy(lgbm_param_template)
lgbm_params_2.update(dict(n_estimators=70, reg_lambda=0.1))

# Target 3
lgbm_params_3 = copy.deepcopy(lgbm_param_template)
lgbm_params_3.update(dict(n_estimators=60))

# Target 4
lgbm_params_4 = copy.deepcopy(lgbm_param_template)
lgbm_params_4.update(dict(n_estimators=50, reg_lambda=0.1))

# Target 5
lgbm_params_5 = copy.deepcopy(lgbm_param_template)
lgbm_params_5.update(dict(n_estimators=50))


additional_paper_dict = dict(target_1=lgbm_params_1,
                             target_2=lgbm_params_2,
                             target_3=lgbm_params_3,
                             target_4=lgbm_params_4,
                             target_5=lgbm_params_5)

def additional_paper_params(index=0) -> dict:
    """Retrieves a dict for LGBMRegressor.

    Parameters
    ----------
    index : int
        Specifies the single-target regression subtask,
        using the Python indexing convention.

    Returns
    -------
    params : dict

    Examples
    --------
    >>> from physlearn.datasets import additional_paper_params
    >>> additional_paper_params(index=0)
    {'objective': 'mean_absolute_error',
     'boosting_type': 'goss',
     'num_leaves': 20,
     'reg_alpha': 0.01,
     'reg_lambda': 0.01,
     'n_estimators': 60}
    """
    
    # Changes to the paper's index convention.
    index += 1
    assert index >=1 and index <=5
    return additional_paper_dict[f'target_{index}']


###########################################################
#XGBRegessor params
###########################################################

xgb_param_template = dict(objective='reg:squarederror')


# Target 1
xgb_params_1 = copy.deepcopy(xgb_param_template)
xgb_params_1.update(dict(n_estimators=80, max_depth=8, booster='dart'))

# Target 2
xgb_params_2 = copy.deepcopy(xgb_param_template)
xgb_params_2.update(dict(n_estimators=50, max_depth=4, booster='gbtree'))

# Target 3
xgb_params_3 = copy.deepcopy(xgb_param_template)
xgb_params_3.update(dict(n_estimators=90, max_depth=6, booster='gbtree'))

# Target 4
xgb_params_4 = copy.deepcopy(xgb_param_template)
xgb_params_4.update(dict(n_estimators=40, max_depth=4, booster='dart'))

# Target 5
xgb_params_5 = copy.deepcopy(xgb_param_template)
xgb_params_5.update(dict(n_estimators=75, max_depth=16, booster='dart'))


xgb_paper_dict = dict(target_1=xgb_params_1,
                      target_2=xgb_params_2,
                      target_3=xgb_params_3,
                      target_4=xgb_params_4,
                      target_5=xgb_params_5)

def xgb_paper_params(index=0) -> dict:
    """Retrieves a dict for XGBRegressor.

    Parameters
    ----------
    index : int
        Specifies the single-target regression subtask,
        using the Python indexing convention.

    Returns
    -------
    params : dict

    Examples
    --------
    >>> from physlearn.datasets import xgb_paper_params
    >>> xgb_paper_params(index=0)
    {'objective': 'reg:squarederror',
     'n_estimators': 80,
     'max_depth': 8,
     'booster': 'dart'}
    """
    
    # Changes to the paper's index convention.
    index += 1
    assert index >=1 and index <=5
    return xgb_paper_dict[f'target_{index}']


###########################################################
#MLPRegessor params
###########################################################


mlp_param_template = dict(activation='relu',
                          solver='lbfgs')


# Target 1
mlp_params_1 = copy.deepcopy(mlp_param_template)
mlp_params_1.update(dict(hidden_layer_sizes=(10, ),
                         alpha=15.0, max_iter=4600))

# Target 2
mlp_params_2 = copy.deepcopy(mlp_param_template)
mlp_params_2.update(dict(hidden_layer_sizes=(10, ),
                         alpha=15.0, max_iter=5020))

# Target 3
mlp_params_3 = copy.deepcopy(mlp_param_template)
mlp_params_3.update(dict(hidden_layer_sizes=(10, ),
                         alpha=13.0, max_iter=5780))

# Target 4
mlp_params_4 = copy.deepcopy(mlp_param_template)
mlp_params_4.update(dict(hidden_layer_sizes=(10, ),
                         alpha=19.0, max_iter=4160))

# Target 5
mlp_params_5 = copy.deepcopy(mlp_param_template)
mlp_params_5.update(dict(hidden_layer_sizes=(8, ),
                         alpha=19.0, max_iter=5240))


supplementary_dict = dict(target_1=mlp_params_1,
                          target_2=mlp_params_2,
                          target_3=mlp_params_3,
                          target_4=mlp_params_4,
                          target_5=mlp_params_5)

def supplementary_params(index=0) -> dict:
    """Retrieves a dict for MLPRegressor.

    Parameters
    ----------
    index : int
        Specifies the single-target regression subtask,
        using the Python indexing convention.

    Returns
    -------
    params : dict

    Examples
    --------
    >>> from physlearn.datasets import supplementary_params
    >>> supplementary_params(index=0)
    {'activation': 'relu',
     'solver': 'lbfgs',
     'hidden_layer_sizes': (10,),
     'alpha': 15.0,
     'max_iter': 4600}
    """
    
    # Changes to the paper's index convention.
    index += 1
    assert index >=1 and index <=5
    return supplementary_dict[f'target_{index}']
