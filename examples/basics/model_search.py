"""
============================
Exhaustive grid search
============================

This example introduces the search method, wherein
we perform (hyper)parameter search with the Sklearn
GridSearchCV object.
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from physlearn import Regressor


# Load the data from Sklearn
X, y = load_boston(return_X_y=True)
X, y = pd.DataFrame(X), pd.Series(y)

# Split the data, using the default test_size=0.25.
# X_train has shape (379, 13), y_train has shape (379,)
# X_test has shape (127, 13), and y_test has shape (127,).
# Namely, there are 13 features and 1 single-target regression task.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# The underlying regressor will be the Sklearn Ridge,
# which uses a special case of Tikhonov regularization. 
# Namely, the regularization is a scalar alpha times the
# identity matrix. Hence, the hyperparamter alpha controls
# the regularization strength. In the dictionary below,
# there are 3 possible choices for alpha, as well
# as the Boolean choice of fitting the intercept.
# Both alpha and fit_intercept are parameter names in Ridge.  
search_params = dict(reg__alpha=[0.1, 0.2, 0.5],
                     reg__fit_intercept=[True, False])

# The default regressor choice is ridge regression, so we do not
# need to specify regressor_choice='ridge'. The default pipeline
# transform choice is the Sklearn QuantileTransformer with the
# Gaussian output distribution. The default number of folds in
# K-fold cross-validation is 5, number of jobs to run in parallel
# is -1, and scoring choice is 'neg_mean_absolute_error', but
# we explicitly right these choices for clarity. 
reg = Regressor(cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')


# Now we perform the exhausitive search over the search_params.
# The default search method is 'gridsearchcv', which uses the
# Sklearn GridSearchCV object. Other choices include 'randomizedsearchcv',
# which uses the Sklearn RandomizedSearchCV object, and 'bayesoptcv',
# which uses the https://github.com/fmfn/BayesianOptimization 
# BayesianOptimization object.
reg.search(X_train, y_train, search_params=search_params,
           search_method='gridsearchcv')

# The aforementioned search fit 5 folds for each of the
# 6 candidates (choice of alpha, choice of fit_intercept).
# Hence, there were 30 total fits. To retrieve the exhausitive
# search results, we access the atrribute search_summary_,
# which is a DataFrame containing the best score, best choice
# of alpha and fit_intercept, and the refit time. As
# 'neg_mean_absolute_error' results in nonpositive version of
# the mean absolute error due to Sklearn conventions, we automatically
# restore nonnegativity in the search method, i.e., best score
# will be greater than or equal to 0.
print(reg.search_summary_)
