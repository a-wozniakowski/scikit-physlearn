"""
The :mod:`physlearn.supervised.interface` provides an interface between
:class:`physlearn.BaseRegressor` and the regressor dictionary. It includes
the :class:`physlearn.RegressorDictionaryInterface` class.
"""

# Author: Alex Wozniakowski
# License: MIT

from __future__ import annotations

import os
import joblib

import mlxtend.regressor
import sklearn.ensemble

from dataclasses import dataclass, field

from physlearn.base import AbstractEstimatorDictionaryInterface
from physlearn.supervised.utils._definition import _REGRESSOR_DICT

        
@dataclass
class RegressorDictionaryInterface(AbstractEstimatorDictionaryInterface):
    """BaseRegressor and regressor dictionary interface.

    The regressor dictionary collects key-value pairs, whereby each key
    is a lower case regressor class name that uniquely identifies the
    regressor class, e.g., ``dict('ridge': Ridge)``. As such, the interface
    manages regressor class retrieval for :class:`physlearn.BaseRegressor`
    as part of the constructor method. 

    Parameters
    ----------
    regressor_choice : str
        The dictionary key for lookup in the dictionary of regressors.
        The key must be in lower cases, e.g., the Scikit-learn
        regressor Ridge has key ``'ridge'``.

    params : dict, list, or None, optional (default=None)
        The choice of (hyper)parameters.

    stacking_options : dict or None, optional (default=None)
        A dictionary of stacking options, whereby ``layers``
        must be specified:

        layers :obj:`dict`
            A dictionary of stacking layer(s).

        shuffle :obj:`bool` or None, (default=True)
            Determines whether to shuffle the training data in
            :class:`mlxtend.regressor.StackingCVRegressor`.

        refit :obj:`bool` or None, (default=True)
            Determines whether to clone and refit the regressors in
            :class:`mlxtend.regressor.StackingCVRegressor`.

        passthrough :obj:`bool` or None, (default=True)
            Determines whether to concatenate the original features with
            the first stacking layer predictions in
            :class:`sklearn.ensemble.StackingRegressor`,
            :class:`mlxtend.regressor.StackingRegressor`, or
            :class:`mlxtend.regressor.StackingCVRegressor`.

        meta_features : :obj:`bool` or None, (default=True)
            Determines whether to make the concatenated features
            accessible through the attribute ``train_meta_features_``
            in :class:`mlxtend.regressor.StackingRegressor` and
            :class:`mlxtend.regressor.StackingCVRegressor`.

        voting_weights : :obj:`ndarray` of shape (n_regressors,) or None, (default=None)
            Sequence of weights for :class:`sklearn.ensemble.VotingRegressor`.

    Examples
    --------
    >>> from physlearn import RegressorDictionaryInterface
    >>> interface = RegressorDictionaryInterface(regressor_choice='mlpregressor',
                                                 params=dict(alpha=1))
    >>> interface.set_params()
    MLPRegressor(alpha=1)
    """

    regressor_choice: str
    params: typing.Union[dict, list] = field(default=None)
    stacking_options: dict = field(default=None)

    def get_params(self, regressor):
        """
        Retrieves the (hyper)parameters.

        Parameters
        ----------
        regressor : estimator
            A regressor that follows the Scikit-learn API.

        Notes
        -----
        The method :meth:`physlearn.RegressorDictionaryInterface.set_params`
        must be called beforehand.
        """

        if not hasattr(self, '_set_params'):
            raise AttributeError('In order to retrieve the (hyper)parameters '
                                 'call set_params beforehand.')
        else:
            return regressor.get_params()

    def set_params(self, **kwargs):
        """Sets the (hyper)parameters.

        If ``params`` is ``None``, then the default (hyper)parameters
        are set.

        Parameters
        ----------
        cv : int, cross-validation generator, an iterable, or None
            Determines the cross-validation strategy in
            :class:`sklearn.ensemble.StackingRegressor`,
            :class:`mlxtend.regressor.StackingRegressor`, or
            :class:`mlxtend.regressor.StackingCVRegressor`.

        verbose : int or None
            Determines verbosity in
            :class:`mlxtend.regressor.StackingRegressor` and
            :class:`mlxtend.regressor.StackingCVRegressor`.

        random_state : int, RandomState instance, or None
            Determines the random number generation in
            :class:`mlxtend.regressor.StackingCVRegressor`.

        n_jobs : int or None
            The number of jobs to run in parallel.

        stacking_options : dict or None, optional (default=None)
            A dictionary of stacking options, whereby ``layers``
            must be specified:

            layers :obj:`dict`
                A dictionary of stacking layer(s).

            shuffle :obj:`bool` or None, (default=True)
                Determines whether to shuffle the training data in
                :class:`mlxtend.regressor.StackingCVRegressor`.

            refit :obj:`bool` or None, (default=True)
                Determines whether to clone and refit the regressors in
                :class:`mlxtend.regressor.StackingCVRegressor`.

            passthrough :obj:`bool` or None, (default=True)
                Determines whether to concatenate the original features with
                the first stacking layer predictions in
                :class:`sklearn.ensemble.StackingRegressor`,
                :class:`mlxtend.regressor.StackingRegressor`, or
                :class:`mlxtend.regressor.StackingCVRegressor`.

            meta_features : :obj:`bool` or None, (default=True)
                Determines whether to make the concatenated features
                accessible through the attribute ``train_meta_features_``
                in :class:`mlxtend.regressor.StackingRegressor` and
                :class:`mlxtend.regressor.StackingCVRegressor`.

            voting_weights : :obj:`ndarray` of shape (n_regressors,) or None, (default=None)
                Sequence of weights for :class:`sklearn.ensemble.VotingRegressor`.
        """
        cv = kwargs.pop('cv', None)
        verbose = kwargs.pop('verbose', None)
        random_state = kwargs.pop('random_state', None)
        n_jobs = kwargs.pop('n_jobs', None)
        stacking_options = kwargs.pop('stacking_options', None)
        if isinstance(stacking_options, dict):
            # Check if the user specified the
            # various stacking options and set
            # the default behavior if unspecified.
            if 'layers' not in stacking_options:
                raise KeyError('The layers key is necessary for stacking. '
                               'Without its specification the stacking '
                               'layers are ambiguous.')
            else:
                layers = stacking_options['layers']
            if 'shuffle' in stacking_options:
                shuffle = stacking_options['shuffle']
            else:
                shuffle = True
            if 'refit' in stacking_options:
                refit = stacking_options['refit']
            else:
                refit = True
            if 'passthrough' in stacking_options:
                passthrough = stacking_options['passthrough']
            else:
                passthrough = True
            if 'meta_features' in stacking_options:
                meta_features = stacking_options['meta_features']
            else:
                meta_features = True
            if 'voting_weights' in stacking_options:
                voting_weights = stacking_options['voting_weights']
            else:
                voting_weights = None
        if kwargs:
            raise TypeError('Unknown keyword arguments: %s'
                            % (list(kwargs.keys())[0]))

        reg = {}
        if self.params is not None:
            if self.stacking_options is not None:
                if any(self.regressor_choice == choice for choice in ['stackingregressor', 'votingregressor']):
                    reg['regressors'] = [(str(index), _REGRESSOR_DICT[choice]().set_params(
                                         **self.params[0][index]))
                                        for index, choice
                                        in enumerate(layers['regressors'])]
                else:
                    reg['regressors'] = [_REGRESSOR_DICT[choice]().set_params(
                                         **self.params[0][index])
                                        for index, choice
                                        in enumerate(layers['regressors'])]
                if self.regressor_choice != 'votingregressor':
                    reg['final_regressor'] = _REGRESSOR_DICT[layers['final_regressor']]().set_params(
                                             **self.params[1])
            else:
                reg['regressor'] = _REGRESSOR_DICT[self.regressor_choice]().set_params(**self.params)
        else:
            # Retrieve default (hyper)parameters.
            if self.stacking_options is not None:
                if any(self.regressor_choice == choice for choice in ['stackingregressor', 'votingregressor']):
                    reg['regressors'] = [(str(index), _REGRESSOR_DICT[choice]())
                                        for index, choice
                                        in enumerate(layers['regressors'])]
                else:
                    reg['regressors'] = [_REGRESSOR_DICT[choice]()
                                        for choice 
                                        in layers['regressors']]
                if self.regressor_choice != 'votingregressor':
                    reg['final_regressor'] = _REGRESSOR_DICT[layers['final_regressor']]()
            else:
                reg['regressor'] = _REGRESSOR_DICT[self.regressor_choice]()

        if 'regressor' in reg:
            out = reg['regressor']
        elif 'regressors' in reg:
            if 'final_regressor' in reg:
                if self.regressor_choice == 'stackingregressor':
                    out = sklearn.ensemble.StackingRegressor(estimators=reg['regressors'], 
                                                             final_estimator=reg['final_regressor'],
                                                             cv=cv,
                                                             n_jobs=n_jobs,
                                                             passthrough=passthrough)
                elif self.regressor_choice == 'mlxtendstackingregressor':
                    out = mlxtend.regressor.StackingRegressor(regressors=reg['regressors'],
                                                              meta_regressor=reg['final_regressor'],
                                                              verbose=verbose,
                                                              use_features_in_secondary=passthrough,
                                                              store_train_meta_features=meta_features)
                elif self.regressor_choice == 'mlxtendstackingcvregressor':
                    out = mlxtend.regressor.StackingCVRegressor(regressors=reg['regressors'],
                                                                meta_regressor=reg['final_regressor'],
                                                                cv=cv,
                                                                shuffle=shuffle,
                                                                random_state=random_state,
                                                                verbose=verbose,
                                                                refit=refit,
                                                                n_jobs=n_jobs,
                                                                use_features_in_secondary=passthrough,
                                                                store_train_meta_features=meta_features)
            else:
                out = sklearn.ensemble.VotingRegressor(estimators=reg['regressors'],
                                                       weights=voting_weights,
                                                       n_jobs=n_jobs)

        # This attribute is used in get_params to check
        # if the (hyper)parameters have been set.
        setattr(self, '_set_params', True)

        return out
