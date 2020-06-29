"""
Automatic regressor retrieval from the model dictionary with stacking support.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>


import os
import joblib
import mlxtend.regressor
import sklearn.ensemble

from abc import ABC, abstractmethod

from .utils._definition import _MODEL_DICT


_MODEL_DICT = _MODEL_DICT['regression']


class AbstractModel(ABC):
    """
    Base class for all regression model interfaces.
    """

    @abstractmethod
    def set_params(self, params):
        pass

        
class RegressionModel(AbstractModel):
    """
    Interface between model dictionary and main Regressor class.
    """

    def __init__(self, model_choice, params=None, stacking_layer=None):
        self.model_choice = model_choice
        self.params = params
        self.stacking_layer = stacking_layer

    def set_params(self, cv=5, verbose=0, random_state=0, n_jobs=-1,
                   shuffle=True, refit=True, passthrough=True,
                   meta_features=True, voting_weights=None):

        model = {}
        if self.params is not None:
            if self.stacking_layer is not None:
                if any(self.model_choice == choice for choice in ['stackingregressor', 'votingregressor']):
                    model['regressors'] = [
                        (str(index), _MODEL_DICT[choice]().set_params(**self.params[0][index]))
                        for index, choice
                        in enumerate(self.stacking_layer['regressors'])]
                else:
                    model['regressors'] = [_MODEL_DICT[choice]().set_params(**self.params[0][index])
                                          for index, choice
                                          in enumerate(self.stacking_layer['regressors'])]
                if self.model_choice != 'votingregressor':
                    final_regressor = _MODEL_DICT[self.stacking_layer['final_regressor']]()
                    model['final_regressor'] = final_regressor.set_params(**self.params[1])
            else:
                model['regressor'] = _MODEL_DICT[self.model_choice]().set_params(**self.params)
        else:
            # Retrieve default params, since
            # none were specificed
            if self.stacking_layer is not None:
                if any(self.model_choice == choice for choice in ['stackingregressor', 'votingregressor']):
                    model['regressors'] = [(str(index), _MODEL_DICT[choice]())
                                          for index, choice
                                          in enumerate(self.stacking_layer['regressors'])]
                else:
                    model['regressors'] = [_MODEL_DICT[choice]()
                                          for choice
                                          in self.stacking_layer['regressors']]
                if self.model_choice != 'votingregressor':
                    model['final_regressor'] = _MODEL_DICT[self.stacking_layer['final_regressor']]()
            else:
                model['regressor'] = _MODEL_DICT[self.model_choice]()

        if 'regressor' in model:
            out = model['regressor']
        elif 'regressors' in model:
            regressors = model['regressors']
            if 'final_regressor' in model:
                final_regressor = model['final_regressor']
                if self.model_choice == 'stackingregressor':
                    out = sklearn.ensemble.StackingRegressor(estimators=regressors, 
                                                             final_estimator=final_regressor,
                                                             cv=cv,
                                                             n_jobs=n_jobs,
                                                             passthrough=passthrough)
                elif self.model_choice == 'mlxtendstackingregressor':
                    out = mlxtend.regressor.StackingRegressor(regressors=regressors,
                                                              meta_regressor=final_regressor,
                                                              verbose=verbose,
                                                              use_features_in_secondary=passthrough,
                                                              store_train_meta_features=meta_features)
                elif self.model_choice == 'mlxtendstackingcvregressor':
                    out = mlxtend.regressor.StackingCVRegressor(regressors=regressors,
                                                                meta_regressor=final_regressor,
                                                                cv=cv,
                                                                shuffle=shuffle,
                                                                random_state=random_state,
                                                                verbose=verbose,
                                                                refit=refit,
                                                                n_jobs=n_jobs,
                                                                use_features_in_secondary=passthrough,
                                                                store_train_meta_features=meta_features)
            else:
                out = sklearn.ensemble.VotingRegressor(estimators=regressors,
                                                       weights=voting_weights,
                                                       n_jobs=n_jobs)

        return out, out.get_params()
