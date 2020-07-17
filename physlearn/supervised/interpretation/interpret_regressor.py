"""
Regressor interpretability with SHAP utilities.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import shap
import matplotlib.pyplot as plt

import sklearn.utils.multiclass

from IPython.display import display

from ..regression import BaseRegressor
from ..utils._data_checks import _n_targets
from ..utils._definition import (_MULTI_TARGET, _SHAP_TAXONOMY,
                                 _SHAP_SUMMARY_PLOT_CHOICE)


class ShapInterpret(BaseRegressor):
    """Interpret a regressor's output with SHAP plots."""

    def __init__(self, regressor_choice='ridge', cv=5, random_state=0,
                 verbose=0, n_jobs=-1, show=True, pipeline_transform='quantile_normal',
                 pipeline_memory=None, params=None, chain_order=None,
                 stacking_layer=None, stacking_cv_shuffle=True,
                 stacking_cv_refit=True, stacking_passthrough=True,
                 stacking_meta_features=True, target_index=None):

        super().__init__(regressor_choice=regressor_choice,
                         cv=cv,
                         random_state=random_state,
                         verbose=verbose,
                         n_jobs=n_jobs,
                         pipeline_transform=pipeline_transform,
                         pipeline_memory=pipeline_memory,
                         params=params,
                         chain_order=chain_order,
                         stacking_layer=stacking_layer,
                         target_index=target_index)

        assert isinstance(show, bool)

        self.show = show
        self.explainer_type = _SHAP_TAXONOMY[self.regressor_choice]

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

        if self.explainer_type == 'tree':
            explainer = shap.TreeExplainer(model=self.pipe.named_steps['reg'])
            shap_values = explainer.shap_values(X=X)
        elif self.explainer_type == 'linear':
            explainer = shap.LinearExplainer(model=self.pipe.named_steps['reg'],
                                             data=X, feature_perturbation='correlation_dependent')
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
                                                 shap_values=shap_values, features=X,
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
                                 alpha=alpha, dot_size=dot_size, show=self.show)

    def decision_plot(self, X, y):
        """Visualization of the additive feature attribution."""

        # Automates single-target slicing
        y = super()._check_target_index(y=y)

        for index in range(_n_targets(y)):
            self.fit(X=X, y=y, index=index)
            explainer, shap_values = self.explainer(X=X)
            shap.decision_plot(base_value=explainer.expected_value, shap_values=shap_values,
                               feature_names=list(X.columns), show=self.show)
