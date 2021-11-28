#######################
Scikit-physlearn Manual
#######################

**Scikit-physlearn** amalgamates
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
`the new formulation of gradient boosting <https://iopscience.iop.org/article/10.1088/2632-2153/ac1ee9>`_, which 
is known as base boosting. The implementation:

- Enables gradient boosting to improve upon prior regression predictions.
- Also, it is especially suited for the low data regime.

The `machine learning library <https://github.com/a-wozniakowski/scikit-physlearn>`_ was
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
   Datasets API <datasets_api>
   Definitions <definition>
   Developer Tools <developer_tools>
   License <license>


******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
