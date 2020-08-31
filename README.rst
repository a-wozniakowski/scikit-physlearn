.. -*- mode: rst -*-

.. image:: https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-on-the-shoulders-of-giants-in/multi-target-regression-on-google-5-qubit
:target: https://paperswithcode.com/sota/multi-target-regression-on-google-5-qubit?p=boosting-on-the-shoulders-of-giants-in
:alt: SOTA

.. image:: https://readthedocs.org/projects/scikit-physlearn/badge/?version=latest
:target: https://scikit-physlearn.readthedocs.io/en/latest/?badge=latest
:alt: Documentation Status

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

Additionally, the library contains the official implementation of
`base boosting <https://arxiv.org/abs/2005.06194>`_. This modification
of the `gradient boosting machine <https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451>`_
supplants a weak learning algorithm with an explict model of the domain,
which provides gradient boosting with an inductive bias in scientific applications;
see the `documentation <https://scikit-physlearn.readthedocs.io/en/latest/baseboosting.html>`_
and the `paper results <https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results>`_.

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
