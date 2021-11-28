import pandas as pd

from physlearn import Regressor
from physlearn.datasets import load_benchmark, xgb_paper_params


# Here we load the the training data, as well as the test data.
# The shapes of X_train and y_train are (95, 5), and the shapes
# of X_test and y_test are (41, 5).
X_train, X_test, y_train, y_test = load_benchmark(return_split=True)


# We choose XGBoost XGBRegressor.
model = 'xgbregressor'

print('Building scoring DataFrame for each single-target subtask.')
test_error = []
for index in range(5):
    # We make an instance of Regressor with our choice of
    # gradient-based one-side sampling for each single-target
    # regression subtask.
    reg = Regressor(regressor_choice=model, params=xgb_paper_params(index),
                    target_index=index)

    # We invoke the fit and predict methods, then we
    # compute the single-target test error.
    y_pred = reg.fit(X_train, y_train).predict(X_test)
    score = reg.score(y_test, y_pred)
    test_error.append(score)

test_error = pd.concat(test_error)
print('Finished building the scoring DataFrame.')
print(test_error.round(decimals=2))
print('Finished computing the multi-target scores.')
print(test_error.mean().round(decimals=2))
