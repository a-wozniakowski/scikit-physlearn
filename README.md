# Scikit-physlearn

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-on-the-shoulders-of-giants-in/multi-target-regression-on-google-5-qubit)](https://paperswithcode.com/sota/multi-target-regression-on-google-5-qubit?p=boosting-on-the-shoulders-of-giants-in)

**Scikit-physlearn** is a Python package for single-target and multi-target regression.
It is designed to amalgamate [Scikit-learn](https://scikit-learn.org/), [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html), [XGBoost](https://xgboost.readthedocs.io/en/latest/), [CatBoost](https://catboost.ai/), and [Mlxtend](http://rasbt.github.io/mlxtend/) regressors into a unified ```Regressor```, which:

* Follows the Scikit-learn API.
* Represents data in pandas.
* Supports [*base boosting*](https://arxiv.org/abs/2005.06194).

The repository was started by Alex Wozniakowski during his graduate studies at Nanyang Technological University.

## Installation
Scikit-physlearn can be installed from [PyPI](https://pypi.org/project/scikit-physlearn/):
```
pip install scikit-physlearn
```

## Quick Start

A multi-target regression example:
```python
from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from physlearn import Regressor

# Load an example dataset from Sklearn
bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
X, y = bunch['data'], bunch['target']

# Split the data in a supervised fashion
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42)

# Select a regressor, e.g., LGBMRegressor from LightGBM, with a case-insensitive string.
reg = Regressor(regressor_choice='lgbmregressor', cv=5, n_jobs=-1,
                scoring='neg_mean_absolute_error')

# Automatically build the pipeline with final estimator MultiOutputRegressor
# from Sklearn, then exhaustively search over the (hyper)parameters.
search_params = dict(boosting_type=['gbdt', 'goss'],
                     n_estimators=[6, 8, 10, 20])
reg.search(X_train, y_train, search_params=search_params,
           search_method='gridsearchcv')

# Generate predictions with the refit regressors, then
# compute the average mean absolute error.
y_pred = reg.fit(X_train, y_train).predict(X_test)
score = reg.score(y_test, y_pred)
print(score['mae'].mean().round(decimals=2))
```
Example output:
```
8.04
```
A [SHAP](https://shap.readthedocs.io/en/latest/#) visualization example of a single-target regression subtask:
```python
from physlearn.datasets import load_benchmark
from physlearn.supervised import ShapInterpret

# Load the training data from a quantum device calibration application.
X_train, _, y_train, _ = load_benchmark(return_split=True)

# Pick the single-target regression subtask: 2, using Python indexing.
index = 1

# Select a regressor, e.g., RidgeCV from Sklearn.
interpret = ShapInterpret(regressor_choice='ridgecv', target_index=index)

# Generate a SHAP force plot, and visualize the subtask predictions.
interpret.force_plot(X_train, y_train)
```
Example output (this plot is interactive in a [notebook](https://jupyter.org/)):

<div align="center">
  <img src="https://github.com/a-wozniakowski/scikit-physlearn/blob/master/images/force_plot.png" width="500" height="250"><br><br>
</div>


For additional examples, check out the [basics](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/basics) directory.

## Base boosting

Inspired by the process of human research, wherein scientific progress derives from prior scientific knowledge, [base boosting](https://arxiv.org/abs/2005.06194) is a modification of the standard version of [gradient boosting](https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451), which is designed to emulate the paradigm of "standing on the shoulders of giants". 

<div align="center">
  <img src="https://github.com/a-wozniakowski/scikit-physlearn/blob/master/images/framework.png" width="500" height="250"><br><br>
</div>

In a multi-target regression task, a base regressor, e.g., an explict model of the domain, generates the initial multi-target predictions. Subsequently, the multi-target boosting algorithm reduces the task to independent single-target regression subtasks. For the jth single-target regression subtask, base boosting greedily fits the following additive expansion in a stagewise fashion:

<div align="center">
  <img src="http://www.sciweavers.org/tex2img.php?eq=%5Cbegin%7Balign%7D%0Ah_%7Bj%7D%28X%20%3B%20%5C%7B%20%5Calpha_%7Bj%7D%2C%20%5Ctheta_%7Bj%7D%20%5C%7D%29%20%3D%20X_%7Bj%7D%20%2B%20%5Csum_%7Bk%3D1%7D%5E%7BK_%7Bj%7D%7D%20%5Calpha_%7Bj%2Ck%7D%20b%28X%20%3B%20%5Ctheta_%7Bj%2Ck%7D%29%2C%0A%5Cend%7Balign%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" width="225" height="70"><br><br>
</div>

where the parameter alpha collects the expansion coefficients and the parameter theta collects the parameter sets, which characterize the basis function b. In contrast, the standard additive expansion:

<div align="center">
  <img src="http://www.sciweavers.org/tex2img.php?eq=%5Cbegin%7Balign%7D%0Ah_%7Bj%7D%28X%20%3B%20%5C%7B%20%5Calpha_%7Bj%7D%2C%20%5Ctheta_%7Bj%7D%5C%7D%29%20%3D%20%5Calpha_%7Bj%2C%200%7D%20%2B%20%5Csum_%7Bk%3D1%7D%5E%7BK_%7Bj%7D%7D%20%5Calpha_%7Bj%2Ck%7D%20b%28X%20%3B%20%5Ctheta_%7Bj%2Ck%7D%29%2C%0A%5Cend%7Balign%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" width="225" height="70"><br><br>
</div>

uses a constant offset value (usually determined by maximum likelihood estimation) in place of the base regressor's jth single-target prediction. In essence, this changes the initialization step in gradient boosting, and it enables base boosting to sequentially refine its predecessor's prior scientific knowledge in analogy with human scientific research.

To get started with base boosting, consider the following example, which compares [non-nested versus nested cross-validation](https://arxiv.org/abs/1809.09446) scores in a multi-target quantum device calibration application with a limited supply of [experimental data](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/physlearn/datasets/google/google_json/_5q.json):

```python
import numpy as np
from sklearn.model_selection import KFold

from physlearn import Regressor
from physlearn.datasets import load_benchmark, paper_params
from physlearn.supervised import plot_cv_comparison


# Number of random trials
n_trials = 30

# Load the training data from a quantum device calibration application, wherein
# X_train denotes the base regressor's initial predictions and y_train denotes
# the multi-target experimental observations, i.e., the extracted eigenenergies.
X_train, _, y_train, _ = load_benchmark(return_split=True)

# Select a basis function, e.g., StackingRegressor from Sklearn with first
# layer regressors: Ridge and RandomForestRegressor from Sklearn and final
# layer regressor: KNeighborsRegressor from Sklearn.
basis_fn = 'stackingregressor'
stack = dict(regressors=['ridge', 'randomforestregressor'],
             final_regressor='kneighborsregressor')

# Number of basis functions in the noise term of the additive expansion.
n_regressors = 1

# Choice of squared error loss function for the pseduo-residual computation.
boosting_loss = 'ls'

# Choice of absolute error loss function and (hyper)parameters for the line search computation.
line_search_regularization = 0.1
line_search_options = dict(init_guess=1, opt_method='minimize',
                           alg='Nelder-Mead', tol=1e-7,
                           options={"maxiter": 10000},
                           niter=None, T=None,
                           loss='lad')

# (Hyper)parameters to to exhaustively search over in the non-nested cross-valdation procedure and in
# the inner loop of the nested cross-validation procedure. Namely, the regularization strength in ridge
# regression, number of decision trees in random forest, and number of neighbors in k-nearest neighbors.
search_params = {'0__alpha': [0.5, 1.0, 1.5],
                 '1__n_estimators': [30, 50, 100],
                 'final_estimator__n_neighbors': [2, 5, 10]}

# Choose the single-target regression subtask: 5, using Python indexing.
index = 4

# Make an instance of Regressor with the aforespecified choices.
reg = Regressor(regressor_choice=basis_fn, stacking_layer=stack,
                scoring='neg_mean_absolute_error', target_index=index,
                n_regressors=n_regressors, boosting_loss=boosting_loss,
                line_search_regularization=line_search_regularization,
                line_search_options=line_search_options)

# Make arrays to store the scores.
non_nested_scores = np.zeros(n_trials)
nested_scores = np.zeros(n_trials)

# Loop through the number of random trials.
for i in range(n_trials):

    # Make two instances of k-fold cross-validation, whereby we generate the same indices
    # for non-nested cross-validation and the outer loop of nested cross-validation.
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)

    
    # Perform a non-nested cross-validation procedure with GridSearchCV from Sklearn.
    reg.search(X=X_train, y=y_train, search_params=search_params,
               search_method='gridsearchcv', cv=outer_cv)
    non_nested_scores[i] = reg.best_score_

    # Perform a 5*5-fold nested cross-validation procedure.
    outer_loop_scores = reg.nested_cross_validate(X=X_train, y=y_train,
                                                  search_params=search_params,
                                                  search_method='gridsearchcv',
                                                  outer_cv=outer_cv,
                                                  inner_cv=inner_cv,
                                                  return_inner_loop_score=False)
    nested_scores[i] = outer_loop_scores.mean()

# Illustrate the non-nested and nested mean absolute error, as well as the score difference,
# for each of the 30 random trials. Note that mean absolute error is a nonnegative score.
plot_cv_comparison(non_nested_scores=non_nested_scores, nested_scores=nested_scores,
                   n_trials=n_trials)
```

Example output:
```
Average difference of -0.038677 with standard deviation of 0.027483.
```

<div align="center">
  <img src="https://github.com/a-wozniakowski/scikit-physlearn/blob/master/images/cv_comparison.png" width="500" height="250"><br><br>
</div>

For additional examples, check out the [paper results](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results) directory:
* Generate an [augmented learning curve](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results/learning_curve.py).

<div align="center">
  <img src="https://github.com/a-wozniakowski/scikit-physlearn/blob/master/images/aug_learning_curve.png" width="500" height="250"><br><br>
</div>

* Establish a proxy of expert human-level performance on the calibration benchmark task with the [base regressor](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results/benchmark.py).
* Boost the initial predictions, generated by the base regressor, and evaulate the test error of the returned [regressor](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results/main_body.py).
* Examine the utility of the base regressor, as a data preprocessor, with a SHAP [summary plot](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results/summary_plot.py).


## Citation

If you use this package, please consider adding the corresponding citation:
```
@article{wozniakowski_2020_boosting,
  title={Boosting on the shoulders of giants in quantum device calibration},
  author={Wozniakowski, Alex and Thompson, Jayne and Gu, Mile and Binder, Felix C.},
  journal={arXiv preprint arXiv:2005.06194},
  year={2020}
}

```
