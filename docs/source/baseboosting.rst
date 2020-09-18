=============
Base Boosting
=============

Introduction
============

Gradient boosting is a general and effective algorithmic paradigm that
improves the prediction accuracy of any given learning algorithm in a
stagewise fashion. The first stage generates residuals with a statistical
approach, usually based upon maximum likelihood estimation. Next, it invokes
a learning algorithm to discover regularities in the residuals. Then, it
finishes by appending the learned function to the statistical initialization
in an additive fashion. Subsequent stages follow suit, wherein each stage
generates residuals with the model from the previous stage and it finishes
by appending the learned function to the model from the previous stage in
an additive fashion. As such, the gradient boosting machine derives a model
of the domain exclusively from the statistical evidence present in the training
examples.

Base boosting is a hyrbidization of first principle approaches with the
gradient boosting machine. Namely, it supplants the standard statistical
initialization in the first stage with a first principles approach. In other
words, base boosting learns a model of the domain by building upon the 
first princples approach in a greedy stagewise fashion.

Example
-------

To get started with base boosting, consider the following example. Namely, compare
the scores between
`non-nested and nested cross-validation <https://arxiv.org/abs/1809.09446>`_ in
a multi-target quantum device calibration
`application <https://github.com/a-wozniakowski/scikit-physlearn/blob/master/physlearn/datasets/google/google_json/_5q.json>`_:

.. code-block:: python

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
    line_search_options = dict(init_guess=1, opt_method='minimize',
                               method='Nelder-Mead', tol=1e-7,
                               options={"maxiter": 10000},
                               niter=None, T=None, loss='lad',
                               regularization=0.1)

    base_boosting_options = dict(n_regressors=n_regressors,
                                 boosting_loss=boosting_loss,
                                 line_search_options=line_search_options)

    # (Hyper)parameters to to exhaustively search over in the non-nested cross-valdation procedure and in
    # the inner loop of the nested cross-validation procedure. Namely, the regularization strength in ridge
    # regression, number of decision trees in random forest, and number of neighbors in k-nearest neighbors.
    search_params = {'reg__0__alpha': [0.5, 1.0, 1.5],
                     'reg__1__n_estimators': [30, 50, 100],
                     'reg__final_estimator__n_neighbors': [2, 5, 10]}

    # Choose the single-target regression subtask: 5, using Python indexing.
    index = 4

    # Make an instance of Regressor with the aforespecified choices.
    reg = Regressor(regressor_choice=basis_fn, stacking_layer=stack,
                    target_index=index, scoring='neg_mean_absolute_error',
                    base_boosting_options=base_boosting_options)

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

Example output:

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
