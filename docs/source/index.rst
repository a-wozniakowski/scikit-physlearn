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
- Represents data with pandas.
- Solves single-target and multi-target regression tasks.
- Interprets regressors with SHAP.

Additionally, the library contains the official implementation of
`base boosting <https://arxiv.org/abs/2005.06194>`_. This modification
of the `gradient boosting machine <https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451>`_
supplants a weak learning algorithm with an explict model of the domain,
which provides gradient boosting with an inductive bias in scientific applications;
see the :doc:`base boosting <baseboosting>` page. 

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
