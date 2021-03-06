# Scikit-physlearn

[![SOTA](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-on-the-shoulders-of-giants-in/multi-target-regression-on-google-5-qubit)](https://paperswithcode.com/sota/multi-target-regression-on-google-5-qubit?p=boosting-on-the-shoulders-of-giants-in)
[![Documentation Status](https://readthedocs.org/projects/scikit-physlearn/badge/?version=latest)](https://scikit-physlearn.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://badge.fury.io/py/scikit-physlearn.svg)](https://badge.fury.io/py/scikit-physlearn)

[Documentation](https://scikit-physlearn.readthedocs.org) |
[Base boosting](https://arxiv.org/abs/2005.06194)

**Scikit-physlearn** is a machine learning library designed to amalgamate 
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

Additionally, the library contains the official implementation of
[base boosting](https://arxiv.org/abs/2005.06194>), which incorporates prior
knowledge into boosting by supplanting the standard statistical initialization
with predictions from a user-specified model. The implementation:

* Enables interoperability between user-specified models and nonparametric
  statistical methods or supervised machine learning algorithms, i.e., it
  is not limited to boosting decision trees.
* Is especially suited for the low data regime.

The library was started by Alex Wozniakowski during his graduate studies at Nanyang Technological
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
  title={Boosting on the shoulders of giants in quantum device calibration},
  author={Wozniakowski, Alex and Thompson, Jayne and Gu, Mile and Binder, Felix C.},
  journal={arXiv preprint arXiv:2005.06194},
  year={2020}
}

```
