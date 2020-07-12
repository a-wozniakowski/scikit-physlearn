import shap
import matplotlib.pyplot as plt

import sklearn.utils.multiclass

from IPython.display import display

from ..base import BaseRegressor, _get_pipeline_ready, _n_targets
from ..utils._definition import _SHAP_TAXONOMY, _SHAP_SUMMARY_PLOT_CHOICE, _STACKING_LIST


class ShapInterpret(BaseRegressor):
    def __init__(self, model_choice, cv=5, random_state=0, verbose=0,
                 n_jobs=-1, show_plot=True, transform_feature='quantile_normal',
                 transform_target=None, pipeline_memory=None,
                 stacking_choice=None, params=None, chain_order=None,
                 qcensemble_n_regressors=None, 
                 qcensemble_target_index=None,
                 qcensemble_line_search_method=None):

        super().__init__(model_choice=model_choice, cv=cv, random_state=random_state,
                         verbose=verbose, n_jobs=n_jobs, transform_feature=transform_feature,
                         transform_target=transform_target, params=params,
                         pipeline_memory=pipeline_memory, stacking_choice=stacking_choice,
                         chain_order=chain_order, qcensemble_n_regressors=qcensemble_n_regressors,
                         qcensemble_target_index=qcensemble_target_index,
                         qcensemble_line_search_method=qcensemble_line_search_method)

        assert isinstance(show_plot, bool)

        self.show = show_plot
        self.explainer_type = _SHAP_TAXONOMY[self.model_choice]

    def prepare_pipeline(self, X, y):
        if self.transform_target is not None:
            if self.transform_target == 'log':
                y = _log_transform(y)

        self.pipe = _get_pipeline_ready(
            regressor=self.regressor, n_targets=_n_targets(y),
            random_state=self.random_state, transform=self.transform_feature,
            verbose=self.verbose, n_jobs=self.n_jobs, cv=self.cv,
            memory=self.pipeline_memory, chain_order=self.chain_order,
            qcensemble_n_regressors=self.qcensemble_n_regressors,
            qcensemble_target_index=self.qcensemble_target_index,
            qcensemble_line_search_method=self.qcensemble_line_search_method
        )

    def fit(self, X, y, sample_weight=None):
        if self.qcensemble_target_index is not None and \
           sklearn.utils.multiclass.type_of_target(y) == 'continuous-multioutput':
            y = y.iloc[:, self.qcensemble_target_index]

        self.prepare_pipeline(X=X, y=y)

        if sample_weight is None:
            self.pipe.fit(X, y)
        else:
            self.pipe.fit(X, y, sample_weight=sample_weight)


    def explainer(self, X):
        if self.explainer_type == 'tree':
            explainer = shap.TreeExplainer(model=self.pipe.named_steps['reg'])
            shap_values = explainer.shap_values(X=X)
        elif self.explainer_type == 'linear':
            explainer = shap.LinearExplainer(
                model=self.pipe.named_steps['reg'], data=X, feature_dependence='correlation'
            )
            shap_values = explainer.shap_values(X=X)
        elif self.explainer_type == 'kernel':
            explainer = shap.KernelExplainer(model=self.pipe.named_steps['reg'].predict, data=X)
            shap_values = explainer.shap_values(X=X, l1_reg='aic')

        return explainer, shap_values

    def summary_plot(self, X, y, plot_type='dot'):
        assert(plot_type in _SHAP_SUMMARY_PLOT_CHOICE)

        for index in range(_n_targets(y)):
            if sklearn.utils.multiclass.type_of_target(y) == 'continuous-multioutput':
                self.fit(X, y.iloc[:, index].values.ravel(order='K'))
            else:
                self.fit(X, y)
            _, shap_values = self.explainer(X=X)

            shap.summary_plot(
                shap_values=shap_values, features=X, plot_type=plot_type, 
                feature_names=list(X.columns), show=self.show
            )

    def force_plot(self, X, y):
        shap.initjs()

        for index in range(_n_targets(y)):
            self.fit(X, y.iloc[:, index].values.ravel(order='K'))
            explainer, shap_values = self.explainer(X=X)
            force_plot = display(
                shap.force_plot(
                    base_value=explainer.expected_value, shap_values=shap_values, features=X,
                    plot_cmap=['#52A267','#F0693B'], feature_names=list(X.columns)
                )
            )

    def dependence_plot(self, X, y, interaction_index='auto', alpha=None, dot_size=None):
        for index in range(_n_targets(y)):
            self.fit(X, y.iloc[:, index].values.ravel(order='K'))
            _, shap_values = self.explainer(X=X)
            shap.dependence_plot(
                ind='rank(0)', shap_values=shap_values, features=X,
                feature_names=list(X.columns), cmap=plt.get_cmap('hot'),
                interaction_index=interaction_index, alpha=alpha,
                dot_size=dot_size, show=self.show
            )

    def decision_plot(self, X, y):
        for index in range(_n_targets(y)):
            if sklearn.utils.multiclass.type_of_target(y) == 'continuous-multioutput':
                self.fit(X, y.iloc[:, index].values.ravel(order='K'))
            else:
                self.fit(X, y)
            explainer, shap_values = self.explainer(X=X)
            shap.decision_plot(
                base_value=explainer.expected_value, shap_values=shap_values, 
                feature_names=list(X.columns), show=self.show
            )

    # def multioutput_decision_plot(self, X, y, row_index):
    #     for index in range(_n_targets(y)):
    #         self.fit(X, y)
    #         explainer, shap_values = self.explainer(X=X)
    #         shap.multioutput_decision_plot(
    #             base_values=explainer.expected_value, shap_values=shap_values, 
    #             row_index=row_index, feature_names=list(X.columns)
    #         )