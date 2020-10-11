#########################
What is scikit-physlearn?
#########################

**Scikit-physlearn** is a machine learning library for single-target
and multi-target regression. It is a Python library that amalgamates
scikit-learn, LightGBM, XGBoost, CatBoost, and Mlxtend regressors into
a unified object that follows the scikit-learn API, processes pandas
data representations, interprets regressors with SHAP, and is the official
implementation of base boosting.

At the core of the library, is the ``Regressor`` object:

.. code-block:: python

    Regressor(regressor_choice='lgbmregressor')

which abstracts regressors with the concept of a key-value pair. In the example
above, the case-insensitive string ``'lgbmregressor'`` corresponds to a key,
which uniquely identifies the regressor class ``LGBMRegressor`` in the
collection of regressors. As such, the regressor interface is simple, and
it includes ``fit``, ``predict``, and ``score`` methods, as well as:

- Joblib methods for model persistence.
- A search method that bundles GridSearchCV, RandomizedSearchCV, and
  Bayesian optimization.
- Cross-validation methods such as ``cross_validate``, ``cross_val_score``,
  and ``nested_cross_validate``.
- A base boosting method with inbuilt k-fold cross-validation.

********
Citation
********

If you use this library, please consider adding the corresponding citation:

.. code-block:: latex

    @article{wozniakowski_2020_boosting,
      title={Boosting on the shoulders of giants in quantum device calibration},
      author={Wozniakowski, Alex and Thompson, Jayne and Gu, Mile and Binder, Felix C.},
      journal={arXiv preprint arXiv:2005.06194},
      year={2020}
    }

