=============
Base Boosting
=============

Introduction
============

Gradient boosting is a general and effective algorithmic paradigm that
builds an additive expansion in a greedy stagewise fashion. Prior to the
first stage, gradient boosting initializes the additive expansion with
the optimal constant model, usually determined by maximum likelihood
estimation. The first stage generates residuals with the statistical
initialization, then it uses a given learning algorithm to discover
regularities in the residuals. The stage finishes by appending the learned
basis function to the statistical initialization in an additive fashion.
Subsequent stages follow suit, wherein each stage generates residuals with
the additive expansion built in the previous stage and it finishes by
appending the learned basis function to the additive expansion. As such,
gradient boosting is an entirely data-driven method that derives a model
of the domain exclusively from the statistical evidence present in the
training examples itself.

Base boosting is a modification of the algorithmic paradigm of gradient
boosting, which supplants the standard statistical initialization with
the output of any base-level regressor. As such, the output of a base-level
regressir acts as an inductive transfer mechanism in base boosting. Its role
is to incorporate prior domain knowledge into gradient boosting, as a means
of compensates for a lack of training examples.

Example
-------

Here, we consider an example that compares nested and non-nested cross-validation
procedures in a quantum computing regression task. We plan to perform thirty random
trials: 

.. code-block:: python

  n_trials = 30

wherein each random trial uses base boosting. We begin by loading the training
examples:

.. code-block:: python

    from physlearn.datasets import load_benchmark
    # Shapes of (95, 5) and (41, 5), respectively.
    X_train, _, y_train, _ = load_benchmark(return_split=True)

Next, we select a single basis function:

.. code-block:: python

    n_regressors = 1
    basis_fn = 'stackingregressor'
    stack = dict(regressors=['ridge', 'randomforestregressor'],
                 final_regressor='kneighborsregressor')

which is ``StackingRegressor`` with first layer regressors
``Ridge`` and ``RandomForestRegressor`` and final layer regressor
``KNeighborsRegressor`` from scikit-learn. 

We select the squared error loss function for the pseduo-residual computation:

.. code-block:: python

    boosting_loss = 'ls'

and we select the line search options:

.. code-block:: python

    line_search_options = dict(init_guess=1, opt_method='minimize',
                               method='Nelder-Mead', tol=1e-7,
                               options={"maxiter": 10000},
                               niter=None, T=None, loss='lad',
                               regularization=0.1)

We bundle the options into a dict:

.. code-block:: python

    base_boosting_options = dict(n_regressors=n_regressors,
                                 boosting_loss=boosting_loss,
                                 line_search_options=line_search_options)

We choose the (hyper)parameters to exhaustively search over in the
non-nested cross-valdation procedure and in the inner loop of the
nested cross-validation procedure. Namely, the regularization strength
in ridge regression, the number of decision trees in random forest, and
the number of neighbors in k-nearest neighbors:

.. code-block:: python

    search_params = {'reg__0__alpha': [0.5, 1.0, 1.5],
                     'reg__1__n_estimators': [30, 50, 100],
                     'reg__final_estimator__n_neighbors': [2, 5, 10]}

Then, we make an instance of the regressor object using the aforespecified
choices:

.. code-block:: python

    from physlearn import Regressor
    reg = Regressor(regressor_choice=basis_fn, stacking_layer=stack,
                    target_index=4, scoring='neg_mean_absolute_error',
                    base_boosting_options=base_boosting_options)

where the target index corresponds to the fifth single-target regression
subtask.

We make arrays to store the nested and non-nested cross-validation scores:

.. code-block:: python

    import numpy as np
    non_nested_scores = np.zeros(n_trials)
    nested_scores = np.zeros(n_trials)

and we also import ``KFold`` from scikit-learn:

.. code-block:: python

    from sklearn.model_selection import KFold

We start the random trials and collect the scores:

.. code-block:: python

    for i in range(n_trials):

        # Make two instances of k-fold cross-validation, whereby
        # we generate the same indices for non-nested cross-validation
        # and the outer loop of nested cross-validation.
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)

        
        # Perform a non-nested cross-validation procedure.
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

Lastly, we plot the nested and non-nested cross-validation scores,
as well as the score difference, for each of the 30 random trials:

.. code-block:: python

    from physlearn.supervised import plot_cv_comparison
    plot_cv_comparison(non_nested_scores=non_nested_scores, nested_scores=nested_scores,
                       n_trials=n_trials)

which outputs:

.. code-block:: bash

  Average difference of -0.038677 with standard deviation of 0.027483.

.. image:: https://raw.githubusercontent.com/a-wozniakowski/scikit-physlearn/master/images/cv_comparison.png
  :target: https://github.com/a-wozniakowski/scikit-physlearn/
  :width: 500px
  :height: 250px

**********
References
**********

- Alex Wozniakowski, Jayne Thompson, Mile Gu, and Felix C. Binder.
  "Boosting on the shoulders of giants in quantum device calibration",
  arXiv preprint arXiv:2005.06194 (2020).

- John Tukey. "Exploratory Data Analysis", Addison-Wesley (1977).

- Jerome Friedman. "Greedy function approximation: A gradient boosting machine,"
  Annals of Statistics, 29(5):1189–1232 (2001).

- Trevor Hastie, Robert Tibshirani, and Jerome Friedman.
  "The Elements of Statistical Learning", Springer (2009).
