import pandas as pd

from physlearn import Regressor
from physlearn.datasets import load_benchmark
from physlearn.supervised import paper_params


n_qubits = 5
data = load_benchmark()
X_train, y_train = data['X_train'].iloc[:, -n_qubits:], data['y_train'].iloc[:, :n_qubits]
X_test, y_test = data['X_test'].iloc[:, -n_qubits:], data['y_test'].iloc[:, :n_qubits]

model = 'stackingregressor'
n_regressors = 1
boosting_loss = 'ls'
line_search_regularization = 0.1
line_search_options = dict(init_guess=1, opt_method='minimize',
                           alg='Nelder-Mead', tol=1e-7,
                           options={"maxiter": 10000},
                           niter=None, T=None)

stack = dict(regressors=['mlpregressor', 'lgbmregressor'],
             final_regressor='mlpregressor')

print('Building scoring DataFrame for each single-target subtask.')
test_error = []
for index in range(5):
    reg = Regressor(regressor_choice=model, n_regressors=n_regressors,
                    boosting_loss=boosting_loss, line_search_regularization=line_search_regularization,
                    line_search_options=line_search_options, stacking_layer=stack,
                    params=paper_params(index), target_index=index)

    y_pred = reg.fit(X_train, y_train).predict(X_test)
    score = reg.score(y_test, y_pred)
    test_error.append(score)

test_error = pd.concat(test_error).round(decimals=2)
print('Finished building the scoring DataFrame.')
print(test_error)
print('Finished computing the multi-target scores.')
print(test_error.mean().round(decimals=2))
