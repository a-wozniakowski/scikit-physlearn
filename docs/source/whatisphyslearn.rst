#########################
What is scikit-physlearn?
#########################

**Scikit-physlearn** is a machine learning library that naturally handles
single-target and multi-target regression tasks. Correspondingly, the Python
library amalgamates scikit-learn, LightGBM, XGBoost, CatBoost, and Mlxtend
regressors into a unified object that follows the scikit-learn API. Furthermore,
the library contains the official implementation of base boosting.

The primary object of interest is the ``Regressor``:

.. code-block:: python

    Regressor(regressor_choice='lgbmregressor')

which uses a case-insensitive string, such as ``'lgbmregressor'``, to access
a regressor, such as ``LGBMRegressor``. The main methods include ``fit``,
``predict``, and ``score``, as well as:

- Joblib methods for model persistence.
- A search method that bundles GridSearchCV, RandomizedSearchCV, and
  Bayesian optimization.
- Cross-validation methods, such as ``cross_validate``, ``cross_val_score``,
  and ``nested_cross_validate``.
- A base boosting method with built-in model selection.

********
Citation
********

If you use this library, please consider adding the corresponding citation:

.. code-block:: latex

    @article{wozniakowski_2021_boosting,
      title={A new formulation of gradient boosting},
      author={Wozniakowski, Alex and Thompson, Jayne and Gu, Mile and Binder, Felix C.},
      journal={Machine Learning: Science and Technology},
      volume={2},
      number={4},
      year={2021}
    }

