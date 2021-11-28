# Scikit-physlearn

[![SOTA](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-on-the-shoulders-of-giants-in/multi-target-regression-on-google-5-qubit)](https://paperswithcode.com/sota/multi-target-regression-on-google-5-qubit?p=boosting-on-the-shoulders-of-giants-in)
[![Documentation Status](https://readthedocs.org/projects/scikit-physlearn/badge/?version=latest)](https://scikit-physlearn.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://badge.fury.io/py/scikit-physlearn.svg)](https://badge.fury.io/py/scikit-physlearn)

[Documentation](https://scikit-physlearn.readthedocs.org) |
[Base boosting](https://iopscience.iop.org/article/10.1088/2632-2153/ac1ee9)

**Scikit-physlearn** amalgamates
[Scikit-learn](https://scikit-learn.org/),
[LightGBM](https://lightgbm.readthedocs.org),
[XGBoost](https://xgboost.readthedocs.org),
[CatBoost](https://catboost.ai/),
and [Mlxtend](http://rasbt.github.io/mlxtend/)
regressors into a flexible framework that:

* Follows the Scikit-learn API.
* Processes pandas data representations.
* Solves single-target and multi-target regression tasks.
* Interprets regressors with SHAP.

Additionally, the library contains the official implementation of the 
[new formulation of gradient boosting](https://iopscience.iop.org/article/10.1088/2632-2153/ac1ee9),
which is known as base boosting. The implementation:

* Enables gradient boosting to improve upon prior regression predictions.
* Also, it is especially suited for the low data regime.

The machine learning library was started by Alex Wozniakowski during his graduate studies at Nanyang Technological
University.

## Installation
Scikit-physlearn can be installed from [PyPI](https://pypi.org/project/scikit-physlearn/):
```
pip install scikit-physlearn
```

To build from source, follow the [installation guide](https://scikit-physlearn.readthedocs.io/en/latest/install.html).

## Citation

If you use this library, please consider adding the corresponding citation:
```
@article{wozniakowski_2020_boosting,
  title={A new formulation of gradient boosting},
  author={Wozniakowski, Alex and Thompson, Jayne and Gu, Mile and Binder, Felix C.},
  journal={Machine Learning: Science and Technology},
  volume={2},
  number={4},
  year={2021}
}

```
