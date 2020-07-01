from physlearn.datasets.google import GoogleData
from physlearn.supervised.regression import Regressor
from physlearn.supervised.model_persistence._paper_params import paper_params


n_qubits = 5
data = GoogleData(n_qubits=n_qubits).load_benchmark()
X_train, X_test = data['X_train'].iloc[:, -n_qubits:], data['X_test'].iloc[:, -n_qubits:]
y_train, y_test = data['y_train'].iloc[:, :n_qubits], data['y_test'].iloc[:, :n_qubits]

index = 0
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

reg = Regressor(regressor_choice=model, n_regressors=1,
                boosting_loss='ls', line_search_regularization=0.1,
                line_search_options=line_search_options, stacking_layer=stack,
                params=paper_params(index), target_index=index)

reg.fit(X_train, y_train)
