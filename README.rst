.. -*- mode: rst -*-

|SOTA|_ |DOCS|_ |PyPI|_

.. |SOTA| image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-on-the-shoulders-of-giants-in/multi-target-regression-on-google-5-qubit
.. _SOTA: https://paperswithcode.com/sota/multi-target-regression-on-google-5-qubit?p=boosting-on-the-shoulders-of-giants-in

.. |DOCS| image:: https://readthedocs.org/projects/scikit-physlearn/badge/?version=latest
.. _DOCS: https://scikit-physlearn.readthedocs.io/en/latest/?badge=latest

.. |PyPI| image:: https://badge.fury.io/py/scikit-physlearn.svg
.. _PyPI: https://badge.fury.io/py/scikit-physlearn

################
Scikit-physlearn
################

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

Additionally, the library contains the official implementation of *base boosting*.
This modification of the
`gradient boosting machine <https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451>`_
supplants the standard statistical initialization in the gradient boosting machine
with a first principles approach such that the gradient boosting machine learns
a general model of the domain by building upon the first principles approach in
a stagewise fashion; see the
`documentation <https://scikit-physlearn.readthedocs.io/en/latest/baseboosting.html>`_
as well as the paper results
`directory <https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results>`_.

The `library <https://github.com/a-wozniakowski/scikit-physlearn>`_ was
started by Alex Wozniakowski during his graduate studies at Nanyang Technological
University.

************
Installation
************

Scikit-physlearn can be installed from `PyPI <https://pypi.org/project/scikit-physlearn/>`__::

    pip install scikit-physlearn

To build from source, see the `installation guide <https://scikit-physlearn.readthedocs.io/en/latest/install.html>`_.

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
