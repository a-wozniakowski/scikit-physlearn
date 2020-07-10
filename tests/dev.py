import numpy as np
import pandas as pd

from physlearn import Regressor
from physlearn.datasets import load_benchmark
from physlearn.supervised import paper_params, search_params
from physlearn.supervised.utils._model_checks import _check_search_style


X_train, X_test, y_train, y_test = load_benchmark(return_split=True)

index = 2
model = 'stackingregressor'
n_regressors = 1
boosting_loss = 'huber'
line_search_regularization = 0.1
line_search_options = dict(init_guess=1, opt_method='minimize',
                           alg='Nelder-Mead', tol=1e-7,
                           options={"maxiter": 10000},
                           niter=None, T=None)

stack = dict(regressors=['mlpregressor', 'lgbmregressor'],
             final_regressor='mlpregressor')


# reg = Regressor(regressor_choice=model, n_regressors=n_regressors,
#                 boosting_loss=boosting_loss, line_search_regularization=line_search_regularization,
#                 line_search_options=line_search_options, stacking_layer=stack,
#                 params=paper_params(index), target_index=index)

reg = Regressor(n_regressors=n_regressors, boosting_loss=boosting_loss,
                line_search_regularization=line_search_regularization,
                line_search_options=line_search_options, target_index=index)

search_params = dict(alpha=np.arange(0.1, 1.5, 0.1))

reg.search(X_train, y_train, search_params=search_params)
print(reg.search_summary_)
