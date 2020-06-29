####################################################################################
# BaseBoost params
####################################################################################

mlp_params_1 = {'hidden_layer_sizes': (3, ),
                'activation': 'tanh',
                'alpha': 17.0,
                'solver': 'lbfgs',
                'max_iter': 4390}
gbm_params_1 = {'objective': 'mean_absolute_error',
                'boosting_type': 'goss',
                'n_estimators': 1060,
                'num_leaves': 32,
                'max_depth': 20, 
                'learning_rate': 0.2,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
                'max_bin': 512,
                'subsample_for_bin': 200}
meta_params_1 = {'hidden_layer_sizes': (10, ),
                 'activation': 'tanh',
                 'alpha': 15.0,
                 'solver': 'lbfgs',
                 'max_iter': 4070}

mlp_params_2 = {'hidden_layer_sizes': (11, ),
                'activation': 'tanh',
                'alpha': 15.0,
                'solver': 'lbfgs',
                'max_iter': 4440}
gbm_params_2 = {'objective': 'mean_absolute_error',
                'boosting_type': 'goss',
                'n_estimators': 1100,
                'num_leaves': 32,
                'max_depth': 20,
                'learning_rate': 0.2,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
                'max_bin': 512,
                'subsample_for_bin': 200}
meta_params_2 = {'hidden_layer_sizes': (13, ),
                 'activation': 'tanh',
                 'alpha': 15.0,
                 'solver': 'lbfgs',
                 'max_iter': 3700}

mlp_params_3 = {'hidden_layer_sizes': (8, ),
                'activation': 'tanh',
                'alpha': 5.0,
                'solver': 'lbfgs',
                'max_iter': 3500}
gbm_params_3 = {'objective': 'mean_absolute_error',
                'boosting_type': 'goss',
                'n_estimators': 750,
                'num_leaves': 32, 
                'max_depth': 20, 
                'learning_rate': 0.2,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
                'max_bin': 512,
                'subsample_for_bin': 200}
meta_params_3 = {'hidden_layer_sizes': (8, ),
                 'activation': 'tanh',
                 'alpha': 8.0,
                 'solver': 'lbfgs',
                 'max_iter': 2000}

mlp_params_4 = {'hidden_layer_sizes': (10, ),
                'activation': 'tanh',
                'alpha': 9.0,
                'solver': 'lbfgs',
                'max_iter': 4500}
gbm_params_4 = {'objective': 'mean_absolute_error',
                'boosting_type': 'goss',
                'n_estimators': 380,
                'num_leaves': 32, 
                'max_depth': 20, 
                'learning_rate': 0.2,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
                'max_bin': 512,
                'subsample_for_bin': 200}
meta_params_4 = {'hidden_layer_sizes': (8, ),
                 'activation': 'tanh',
                 'alpha': 10.0,
                 'solver': 'lbfgs',
                 'max_iter': 4250}

mlp_params_5 = {'hidden_layer_sizes': (12, ),
                'activation': 'tanh',
                'alpha': 8.0,
                'solver': 'lbfgs',
                'max_iter': 3450}
gbm_params_5 = {'objective': 'mean_absolute_error',
                'boosting_type': 'goss',
                'n_estimators': 990,
                'num_leaves': 32, 
                'max_depth': 20, 
                'learning_rate': 0.2,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
                'max_bin': 512,
                'subsample_for_bin': 200}
meta_params_5 = {'hidden_layer_sizes': (7, ),
                 'activation': 'tanh',
                 'alpha': 4.0,
                 'solver': 'lbfgs',
                 'max_iter': 3650}

paper_params_dict = {'target_1': [[gbm_params_1, mlp_params_1], meta_params_1],
                     'target_2': [[gbm_params_2, mlp_params_2], meta_params_2],
                     'target_3': [[gbm_params_3, mlp_params_3], meta_params_3],
                     'target_4': [[gbm_params_4, mlp_params_4], meta_params_4],
                     'target_5': [[gbm_params_5, mlp_params_5], meta_params_5]}

def paper_params(index=0):
    # Change from python convention
    # to paper convention
    index += 1
    assert index >=1 and index <=5
    return paper_params_dict[f'target_{index}']


####################################################################################
#MLPRegessor params
####################################################################################

mlp_params_1 = {'hidden_layer_sizes': (10, ),
                'activation': 'relu',
                'alpha': 15.0,
                'solver': 'lbfgs',
                'max_iter': 4600}

mlp_params_2 = {'hidden_layer_sizes': (10, ),
                'activation': 'relu',
                'alpha': 15.0,
                'solver': 'lbfgs',
                'max_iter': 5020}

mlp_params_3 = {'hidden_layer_sizes': (10, ),
                'activation': 'relu',
                'alpha': 13.0,
                'solver': 'lbfgs',
                'max_iter': 5780}

mlp_params_4 = {'hidden_layer_sizes': (10, ),
                'activation': 'relu',
                'alpha': 19.0,
                'solver': 'lbfgs',
                'max_iter': 4160}

mlp_params_5 = {'hidden_layer_sizes': (8, ),
                'activation': 'relu',
                'alpha': 19.0,
                'solver': 'lbfgs',
                'max_iter': 5240}

supplementary_dict = {'target_1': mlp_params_1,
                      'target_2': mlp_params_2,
                      'target_3': mlp_params_3,
                      'target_4': mlp_params_4,
                      'target_5': mlp_params_5}

def supplementary_params(index=0):
    # Change from python convention
    # to supplementary convention
    index += 1
    assert index >=1 and index <=5
    return supplementary_dict[f'target_{index}']