import pandas as pd

from physlearn import Regressor
from physlearn.datasets import load_benchmark
from physlearn.supervised import supplementary_params


n_features = 9
n_targets = 5
data = load_benchmark()
X_train, X_test = data['X_train'].iloc[:, :n_features], data['X_test'].iloc[:, :n_features]
y_train, y_test = data['y_train'].iloc[:, :n_targets], data['y_test'].iloc[:, :n_targets]

model = 'mlpregressor'

print('Building scoring DataFrame for each single-target subtask.')
test_error = []
for index in range(5):
    reg = Regressor(regressor_choice=model, params=supplementary_params(index),
                    target_index=index)

    y_pred = reg.fit(X_train, y_train).predict(X_test)
    score = reg.score(y_test, y_pred)
    test_error.append(score)

test_error = pd.concat(test_error).round(decimals=2)
print('Finished building the scoring DataFrame.')
print(test_error)
print('Finished computing the multi-target scores.')
print(test_error.mean().round(decimals=2))
