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
`base boosting <https://iopscience.iop.org/article/10.1088/2632-2153/ac1ee9>`_, which 
is a reformulation of gradient boosting that

- Regards predictions from any regression model as an inductive bias.
- In contrast, gradient boosting regards the prediction from a constant
  model as an inductive bias.
- Consequently, base boosting generalizes Tukey’s methods of twicing,
  thricing, and reroughing, as gradient boosting works with a variety
  of fitting criterion.

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
