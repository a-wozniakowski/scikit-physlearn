#######################
Scikit-physlearn Manual
#######################

**Scikit-physlearn** is a machine learning library designed to amalgamate 
`Scikit-learn <https://scikit-learn.org/>`_,
`LightGBM <https://lightgbm.readthedocs.io/en/latest/index.html>`_,
`XGBoost <https://xgboost.readthedocs.io/en/latest/>`_,
`CatBoost <https://catboost.ai/>`_,
and `Mlxtend <http://rasbt.github.io/mlxtend/>`_ 
regressors into a flexible framework that:

- Follows the Scikit-learn API.
- Processes pandas data representations.
- Solves single-target and multi-target regression tasks.
- Interprets regressors with `SHAP <https://shap.readthedocs.io/en/latest/>`_.

Additionally, the library contains the official implementation of
`base boosting <https://arxiv.org/abs/2005.06194>`_, which is an algorithmic
paradigm for building additive expansions based upon the output of any
base-level regressor. The implementation:

- Supplants the statistical initialization in gradient boosting
  with the output of any base-level regressor.
- Boosts arbitrary basis functions, i.e., it is not limited to boosting
  decision trees.
- Efficiently learns in the low data regime.

The `library <https://github.com/a-wozniakowski/scikit-physlearn>`_ was
started by Alex Wozniakowski during his graduate studies at Nanyang Technological
University.

********
Contents
********

.. toctree::
   :maxdepth: 1
   :titlesonly:

   Setting Up <setting_up>
   Quick Start <quick_start>
   Base Boosting <baseboosting>
   Python API <python_api>
   Developer Tools <developer_tools>
   License <license>


******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
