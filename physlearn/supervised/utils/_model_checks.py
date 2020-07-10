import Levenshtein
import numpy as np

from ._definition import _MODEL_DICT, _SEARCH_METHOD


def _basic_autocorrect(model_choice, model_choices, model_type):
    assert isinstance(model_choice, str)
    assert isinstance(model_choices, list)
    assert isinstance(model_type, str)

    min_ = np.inf
    output_model_choice = model_choice
    for candidate_model_choice in model_choices:
        dist = Levenshtein.distance(model_choice, candidate_model_choice)
        if dist < min_:
            min_ = dist
            output_model_choice = candidate_model_choice

    if min_ > 0:
        print(model_choice, 'is not a key in the model dictionary. Instead we used',
              output_model_choice + '.')
    return output_model_choice


def _check_model_choice(model_choice, model_type):
    assert all(isinstance(arg, str) for arg in [model_choice, model_type])

    model_choices = [choice for choice in _MODEL_DICT[model_type].keys()]
    model_choice  = _basic_autocorrect(model_choice=model_choice.strip().lower(),
                                       model_choices=model_choices,
                                       model_type=model_type)
    return model_choice


def _check_stacking_layer(stacking_layer, model_type):
    try:
        assert isinstance(stacking_layer, dict)
        assert isinstance(model_type, str)
        
        model_choices = [choice for choice in _MODEL_DICT[model_type].keys()]
        if 'regressors' in stacking_layer:
            stacking_layer['regressors'] = [_basic_autocorrect(model_choice=model.strip().lower(),
                                                               model_choices=model_choices,
                                                               model_type=model_type)
                                           for model in stacking_layer['regressors']]

        if 'final_regressor' in stacking_layer:
            model = stacking_layer['final_regressor']
            stacking_layer['final_regressor'] = _basic_autocorrect(model_choice=model.strip().lower(),
                                                                   model_choices=model_choices,
                                                                   model_type=model_type)
    finally:
        return stacking_layer


def _check_bayesoptcv_parameter_type(params): 
    assert isinstance(params, dict)

    for key, param in params.items():
        if 'max_iter' in key:
            params[key] = int(param)
        elif 'n_estimators' in key:
            params[key] = int(param)
        elif 'max_depth' in key:
            params[key] = int(param)
    return params


def _parallel_search_preprocessing(raw_params):
    assert isinstance(raw_params, (dict))
    params = {}
    for raw_param, value in raw_params.items():
        search_key = 'reg__' + raw_param
        params[search_key] = value
    return params


def _sequential_search_preprocessing(raw_pbounds):
    assert isinstance(raw_pbounds, (dict))
    pbounds = {}
    for raw_pbound, value in raw_pbounds.items():
        bayesoptcv_search_key = 'reg__' + raw_pbound
        pbounds[bayesoptcv_search_key] = value
    return pbounds


def _check_search_method(search_method):
    assert isinstance(search_method, str)
    assert search_method in _SEARCH_METHOD

    search_method = search_method.strip().lower()
    if search_method in ['gridsearchcv', 'randomizedsearchcv']:
        search_taxonomy = 'parallel'
    elif search_method in ['bayesoptcv']:
        search_taxonomy = 'sequential'
    return search_method, search_taxonomy


def _prepare_best_params_filename(folder, model_choice):
    assert isinstance(folder, str)
    assert isinstance(model_choice, str)

    return folder + model_choice + '_best_params_.joblib'
    

def _convert_filename_to_csv_path(filename):
    assert isinstance(filename, str)
    
    root = os.getcwd()
    if re.search('C:', root):
        path = root + '\\' + filename + '.csv'
    else:
        path = root + '/' + filename + '.csv'
    return path