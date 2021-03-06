"""
The :mod:`physlearn.supervised.utils._definition` module collects the 
assortment of library definitions.
"""

# Author: Alex Wozniakowski
# License: MIT

import typing

import catboost as cat
import lightgbm as lgb
import xgboost as xgb

from mlxtend.regressor import StackingRegressor, StackingCVRegressor

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.multioutput import ClassifierChain, RegressorChain

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (AdaBoostRegressor,
                              BaggingRegressor,
                              ExtraTreesRegressor,
                              GradientBoostingRegressor,
                              RandomForestRegressor,
                              HistGradientBoostingRegressor,
                              StackingRegressor,
                              VotingRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (DotProduct, WhiteKernel,
                                              RBF)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (LinearRegression,
                                  Ridge, RidgeCV,
                                  SGDRegressor,
                                  ElasticNet, ElasticNetCV,
                                  Lars, LarsCV,
                                  Lasso, LassoCV,
                                  LassoLars, LassoLarsCV, LassoLarsIC,
                                  OrthogonalMatchingPursuit,
                                  OrthogonalMatchingPursuitCV,
                                  ARDRegression,
                                  BayesianRidge,
                                  MultiTaskElasticNet,
                                  MultiTaskElasticNetCV,
                                  MultiTaskLasso,
                                  MultiTaskLassoCV,
                                  HuberRegressor,
                                  RANSACRegressor,
                                  TheilSenRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import (PowerTransformer, QuantileTransformer,
                                   StandardScaler)


#: Collection of regressor options.
_REGRESSOR_DICT = dict(linearregression=LinearRegression,
                       ridge=Ridge,
                       ridgecv=RidgeCV,
                       sgdregressor=SGDRegressor,
                       elasticnet=ElasticNet,
                       elasticnetcv=ElasticNetCV,
                       lars=Lars,
                       larscv=LarsCV,
                       lasso=Lasso,
                       lassocv=LassoCV,
                       lassolars=LassoLars,
                       lassolarscv=LassoLarsCV,
                       lassolarsic=LassoLarsIC,
                       orthogonalmatchingpursuit=OrthogonalMatchingPursuit,
                       orthogonalmatchingpursuitcv=OrthogonalMatchingPursuitCV,
                       ardregression=ARDRegression,
                       bayesianridge=BayesianRidge,
                       multitaskelasticnet=MultiTaskElasticNet,
                       multitaskelasticnetcv=MultiTaskElasticNetCV,
                       multitasklasso=MultiTaskLasso,
                       multitasklassocv=MultiTaskLassoCV,
                       huberregressor=HuberRegressor,
                       ransacregressor=RANSACRegressor,
                       theilsenregressor=TheilSenRegressor,
                       kernelridge=KernelRidge,
                       decisiontreeregressor=DecisionTreeRegressor,
                       adaboostregressor=AdaBoostRegressor,
                       baggingregressor=BaggingRegressor,
                       extratreesregressor=ExtraTreesRegressor,
                       gradientboostingregressor=GradientBoostingRegressor,
                       randomforestregressor=RandomForestRegressor,
                       histgradientboostingregressor=HistGradientBoostingRegressor,
                       xgbregressor=xgb.XGBRegressor,
                       lgbmregressor=lgb.LGBMRegressor,
                       catboostregressor=cat.CatBoostRegressor,
                       svr=SVR,
                       gaussianprocessregressor=GaussianProcessRegressor,
                       kneighborsregressor=KNeighborsRegressor,
                       mlpregressor=MLPRegressor,
                       stackingregressor=StackingRegressor,
                       mlxtendstackingregressor=StackingRegressor,
                       mlxtendstackingcvregressor=StackingCVRegressor,
                       votingregressor=VotingRegressor)


_ESTIMATOR_DICT = dict(regression=_REGRESSOR_DICT)


_KERNEL_DICT = dict(dotproduct=DotProduct,
                    rbf=RBF,
                    whitekernel=WhiteKernel)


# We need to identify chaining as the fit method capitilizes
# the target Y, whereas other estimators conventionally write
# the target as y.
_CHAIN_FLAG = [RegressorChain(base_estimator=DummyRegressor()).__class__,
               ClassifierChain(base_estimator=DummyClassifier()).__class__]


# We need to identify XGBoost as the predict method utilizes the
# parameter data for the conventional design matrix parameter X.
_XGBOOST_FLAG = xgb.XGBRegressor().__class__


# We need to identify CatBoost as the predict method utilizes the
# parameter data for the conventional design matrix parameter X.
_CATBOOST_FLAG = cat.CatBoostRegressor().__class__


_MULTI_TARGET = ['continuous-multioutput', 'multiclass-multioutput']


_OPTIMIZE_METHOD = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
                    'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr',
                    'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov',
                    'custom']


_SCORE_CHOICE = ['mae', 'mse', 'rmse', 'r2', 'ev', 'msle']


_SCORE_MULTIOUTPUT = ['raw_values', 'uniform_average']


_PIPELINE_PARAMS = ['steps', 'memory', 'verbose', 'n_jobs',
                    'base_boosting_options']


_PIPELINE_TRANSFORM_CHOICE = ['standardscaler', 'boxcox', 'yeojohnson',
                              'quantileuniform', 'quantilenormal']


_SEARCH_METHOD = ['gridsearchcv', 'randomizedsearchcv', 'bayesoptcv']


_SHAP_TAXONOMY = dict(linearregression='linear',
                      ridge='linear',
                      ridgecv='linear',
                      sgdregressor='linear',
                      elasticnet='linear',
                      elasticnetcv='linear',
                      lars='linear',
                      larscv='linear',
                      lasso='linear',
                      lassocv='linear',
                      lassolars='linear',
                      lassolarscv='linear',
                      lassolarsic='linear',
                      orthogonalmatchingpursuit='linear',
                      orthogonalmatchingpursuitcv='linear',
                      ardregression='linear',
                      bayesianridge='linear',
                      multitaskelasticnet='linear',
                      multitaskelasticnetcv='linear',
                      multitasklasso='linear',
                      multitasklassocv='linear',
                      huberregressor='linear',
                      ransacregressor='linear',
                      theilsenregressor='linear',
                      decisiontreeregressor='tree',
                      adaboostregressor='tree',
                      extratreesregressor='tree',
                      gradientboostingregressor='tree',
                      randomforestregressor='tree',
                      histgradientboostingregressor='tree',
                      lgbmregressor='tree',
                      xgbregressor='tree',
                      catboostregressor='kernel',  # See SHAP issue #480
                      baggingregressor='kernel',
                      kernelridge='kernel',
                      svr='kernel',
                      gaussianprocessregressor='kernel',
                      kneighborsregressor='kernel',
                      mlpregressor='kernel',
                      stackingregressor='kernel',
                      mlxtendstackingregressor='kernel',
                      mlxtendstackingcvregressor='kernel',
                      votingregressor='kernel')


_SHAP_SUMMARY_PLOT_CHOICE = ['dot', 'violin', 'bar']


_BAYESOPTCV_INIT_PARAMS = ['max_iter', 'n_estimators', 'max_depth']
