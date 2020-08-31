"""
The :mod:`physlearn.supervised.interpretation.interpret_regressor` module provides
SHAP utilities for regressor interpretability. It includes the 
:class:`physlearn.ShapInterpret` class.
"""

# Author: Alex Wozniakowski
# License: MIT

import shap

import sklearn.utils.multiclass
import sklearn.utils.validation

import matplotlib.pyplot as plt

from IPython.display import display

from physlearn.supervised.regression import BaseRegressor
from physlearn.supervised.utils._data_checks import _n_targets
from physlearn.supervised.utils._definition import (_MULTI_TARGET, _SHAP_TAXONOMY,
                                                    _SHAP_SUMMARY_PLOT_CHOICE)


class ShapInterpret(BaseRegressor):
    """Interpret a regressor's output with SHAP plots."""

    def __init__(self, regressor_choice='ridge', cv=5,
                 random_state=0, verbose=0, n_jobs=-1,
                 show=True, pipeline_transform='quantilenormal',
                 pipeline_memory=None, params=None, target_index=None,
                 chain_order=None, stacking_options=None):

        super().__init__(regressor_choice=regressor_choice,
                         cv=cv,
                         random_state=random_state,
                         verbose=verbose,
                         n_jobs=n_jobs,
                         pipeline_transform=pipeline_transform,
                         pipeline_memory=pipeline_memory,
                         params=params,
                         target_index=target_index,
                         chain_order=chain_order,
                         stacking_options=stacking_options)

        self.show = show
        self.explainer_type = _SHAP_TAXONOMY[self.regressor_choice]
        self._validate_display_options()

    def _validate_display_options(self):
        assert isinstance(self.show, bool)

    def fit(self, X, y, index=None, sample_weight=None):
        """Fit regressor."""

        if index is not None and \
        sklearn.utils.multiclass.type_of_target(y) in _MULTI_TARGET:
                super().get_pipeline(y=y.iloc[:, index])
                super()._fit(regressor=self.pipe, X=X,
                             y=y.iloc[:, index].values.ravel(order='K'),
                             sample_weight=sample_weight)
        else:
            super().get_pipeline(y=y)
            super()._fit(regressor=self.pipe, X=X, y=y,
                         sample_weight=sample_weight)

    def explainer(self, X):
        """Compute the importance of each feature for the underlying regressor."""

        try:
            sklearn.utils.validation.check_is_fitted(estimator=self.pipe,
                                                     attributes='_final_estimator')
        except AttributeError:
            print('The pipeline has not been built. Please use the fit method beforehand.')
        
        if self.explainer_type == 'tree':
            explainer = shap.TreeExplainer(model=self.pipe.named_steps['reg'],
                                           feature_perturbation='interventional',
                                           data=X)
            shap_values = explainer.shap_values(X=X)
        elif self.explainer_type == 'linear':
            explainer = shap.LinearExplainer(model=self.pipe.named_steps['reg'],
                                             feature_perturbation='correlation_dependent',
                                             data=X)
            shap_values = explainer.shap_values(X=X)
        elif self.explainer_type == 'kernel':
            explainer = shap.KernelExplainer(model=self.pipe.named_steps['reg'].predict,
                                             data=X)
            shap_values = explainer.shap_values(X=X, l1_reg='aic')
        
        return explainer, shap_values

    def summary_plot(self, X, y, plot_type='dot'):
        """Visualizaion of the feature importance and feature effects."""

        assert(plot_type in _SHAP_SUMMARY_PLOT_CHOICE)

        # Automates single-target slicing
        y = super()._check_target_index(y=y)

        for index in range(_n_targets(y)):
            self.fit(X=X, y=y, index=index)
            _, shap_values = self.explainer(X=X)

            shap.summary_plot(shap_values=shap_values, features=X,
                              plot_type=plot_type, feature_names=list(X.columns),
                              show=self.show)

    def force_plot(self, X, y):
        """Interactive Javascript visualization of Shapley values."""

        shap.initjs()

        # Automates single-target slicing
        y = super()._check_target_index(y=y)

        for index in range(_n_targets(y)):
            self.fit(X=X, y=y, index=index)
            explainer, shap_values = self.explainer(X=X)
            force_plot = display(shap.force_plot(base_value=explainer.expected_value,
                                                 shap_values=shap_values,
                                                 features=X,
                                                 plot_cmap=['#52A267','#F0693B'],
                                                 feature_names=list(X.columns)))

    def dependence_plot(self, X, y, interaction_index='auto', alpha=None,
                        dot_size=None):
        """Visualization of a feature's effect on a regressor's prediction."""

        # Automates single-target slicing
        y = super()._check_target_index(y=y)

        for index in range(_n_targets(y)):
            self.fit(X=X, y=y, index=index)
            _, shap_values = self.explainer(X=X)
            shap.dependence_plot(ind='rank(0)', shap_values=shap_values,
                                 features=X, feature_names=list(X.columns),
                                 cmap=plt.get_cmap('hot'),
                                 interaction_index=interaction_index,
                                 alpha=alpha, dot_size=dot_size,
                                 show=self.show)

    def decision_plot(self, X, y):
        """Visualization of the additive feature attribution."""

        # Automates single-target slicing
        y = super()._check_target_index(y=y)

        for index in range(_n_targets(y)):
            self.fit(X=X, y=y, index=index)
            explainer, shap_values = self.explainer(X=X)
            shap.decision_plot(base_value=explainer.expected_value,
                               shap_values=shap_values,
                               feature_names=list(X.columns),
                               show=self.show)
