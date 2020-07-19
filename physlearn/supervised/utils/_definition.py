"""
Collection of definitions used in the package.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import catboost
import lightgbm as lgb
import xgboost as xgb

import mlxtend.regressor

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, RandomForestRegressor,
                              HistGradientBoostingRegressor, StackingRegressor,
                              VotingRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (LinearRegression, Ridge, RidgeCV, SGDRegressor,
                                  ElasticNet, ElasticNetCV, Lars, LarsCV, Lasso,
                                  LassoCV, LassoLars, LassoLarsCV, LassoLarsIC,
                                  OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV,
                                  ARDRegression, BayesianRidge, MultiTaskElasticNet,
                                  MultiTaskElasticNetCV, MultiTaskLasso, MultiTaskLassoCV,
                                  HuberRegressor, RANSACRegressor, TheilSenRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler


_REGRESSION_DICT = dict(linearregression=LinearRegression,
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
                        catboostregressor=catboost.CatBoostRegressor,
                        svr=SVR,
                        gaussianprocessregressor=GaussianProcessRegressor,
                        kneighborsregressor=KNeighborsRegressor,
                        mlpregressor=MLPRegressor,
                        stackingregressor=StackingRegressor,
                        mlxtendstackingregressor=mlxtend.regressor.StackingRegressor,
                        mlxtendstackingcvregressor=mlxtend.regressor.StackingCVRegressor,
                        votingregressor=VotingRegressor)


_MODEL_DICT = dict(regression=_REGRESSION_DICT)


_KERNEL_DICT = dict(dotproduct=DotProduct,
                    rbf=RBF,
                    whitekernel=WhiteKernel)


_MULTI_TARGET = ['continuous-multioutput', 'multiclass-multioutput']


_SCORE_CHOICE = ['mae', 'mse', 'rmse', 'r2', 'ev', 'msle']


_PIPELINE_TRANSFORM_CHOICE = ['standardscaler', 'boxcox', 'yeojohnson',
                              'quantileuniform', 'quantilenormal']


_SEARCH_METHOD = ['gridsearchcv', 'randomizedsearchcv', 'bayesoptcv']


_SEARCH_TAXONOMY = ['parallel', 'sequential']


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
                      catboostregressor='kernel',  # Changed to kernel per issue #480 in SHAP
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
