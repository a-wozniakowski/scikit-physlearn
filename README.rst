.. -*- mode: rst -*-

|SOTA|_

.. |SOTA| image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-on-the-shoulders-of-giants-in/multi-target-regression-on-google-5-qubit
.. _SOTA: https://paperswithcode.com/sota/multi-target-regression-on-google-5-qubit?p=boosting-on-the-shoulders-of-giants-in


**Scikit-physlearn** is a Python package for single-target and multi-target regression.
It is designed to amalgamate
`Scikit-learn <https://scikit-learn.org/>`__,
`LightGBM <https://lightgbm.readthedocs.io/en/latest/index.html>`__,
`XGBoost <https://xgboost.readthedocs.io/en/latest/>`__,
`CatBoost <https://catboost.ai/>`__,
and `Mlxtend <http://rasbt.github.io/mlxtend/>`__ 
regressors into a unified ``Regressor``, which:

- Follows the Scikit-learn API.
- Represents data in pandas.
- Supports `base boosting <https://arxiv.org/abs/2005.06194>`__.

The repository was started by Alex Wozniakowski during his graduate studies at Nanyang Technological University.

Installation
------------

Scikit-physlearn can be installed from `PyPi <https://pypi.org/project/scikit-physlearn/>`__::

    pip install scikit-physlearn


Quick Start
-----------

A multi-target regression example:

.. code-block:: python

    from sklearn.datasets import load_linnerud
    from sklearn.model_selection import train_test_split
    from physlearn import Regressor

    # Load an example dataset from Sklearn
    bunch = load_linnerud(as_frame=True)  # returns a Bunch instance
    X, y = bunch['data'], bunch['target']

    # Split the data in a supervised fashion
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=42)

    # Select a regressor, e.g., LGBMRegressor from LightGBM,
    # with a case-insensitive string.
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

Example output:

.. code-block:: bash

    8.04


A `SHAP <https://shap.readthedocs.io/en/latest/>`__ visualization example:

.. code-block:: python

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

Example output (this plot is interactive in a `notebook <https://jupyter.org/>`_):

.. image:: https://raw.githubusercontent.com/a-wozniakowski/scikit-physlearn/master/images/force_plot.png
  :target: https://github.com/a-wozniakowski/scikit-physlearn/
  :width: 500px
  :height: 250px


For additional examples, check out the `basics <https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/basics>`__ directory.

Base boosting
-------------

Inspired by the process of human research, wherein scientific progress derives from prior scientific knowledge,
base boosting is a modification of the standard version of
`gradient boosting <https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451>`__,
which is designed to emulate the paradigm of "standing on the shoulders of giants":

.. image:: https://raw.githubusercontent.com/a-wozniakowski/scikit-physlearn/master/images/framework.png
  :target: https://github.com/a-wozniakowski/scikit-physlearn/
  :width: 500px
  :height: 250px

To evaluate its efficacy in a
superconducting quantum device calibration application with a limited supply of `experimental data <https://github.com/a-wozniakowski/scikit-physlearn/blob/master/physlearn/datasets/google/google_json/_5q.json>`_:

- Start with the `learning curve <https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results/learning_curve.py>`__ module, and use it to generate an augmented learning curve:

.. image:: https://raw.githubusercontent.com/a-wozniakowski/scikit-physlearn/master/images/aug_learning_curve.png
  :target: https://github.com/a-wozniakowski/scikit-physlearn/
  :width: 500px
  :height: 250px

- Next, run the `benchmark <https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results/benchmark.py>`__ module, and use it to obtain the base regressor's test error.
- Then, run the `main body <https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results/main_body.py>`__ module, and compare the test error of `base boosting <https://arxiv.org/abs/2005.06194>`__ with the benchmark error.


Citation
--------

If you use this package, please consider adding the corresponding citation:

.. code-block:: latex

    @article{wozniakowski_2020_boosting,
      title={Boosting on the shoulders of giants in quantum device calibration},
      author={Wozniakowski, Alex and Thompson, Jayne and Gu, Mile and Binder, Felix},
      journal={arXiv preprint arXiv:2005.06194},
      year={2020}
    }
