Scikit-physlearn
---

**Scikit-physlearn** is a Python package for single-target and multi-target regression.
It is designed to amalgamate 
[Scikit-learn](https://scikit-learn.org/),
[LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html),
[XGBoost](https://xgboost.readthedocs.io/en/latest/),
[CatBoost](https://catboost.ai/),
and [Mlxtend](http://rasbt.github.io/mlxtend/)
regressors into a unified ```Regressor```, which:
* Follows the Scikit-learn API.
* Represents data in pandas.
* Supports [*base boosting*](https://arxiv.org/abs/2005.06194).

The repository was started by Alex Wozniakowski during his graduate studies at Nanyang Technological University.

## Quick Start

See below for a quick tour of the Scikit-physlearn package.
* Follow the
[introduction](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/basics/introduction.py)
module to get started with single-target regression.
* Check out the
[multi-target](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/basics/multi_target.py)
module to get started with multi-target regression.
* Explore the
[model search](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/basics/model_search.py)
module to learn about (hyper)parameter optimization.

## Base boosting

<div align="center">
  <img src="https://github.com/a-wozniakowski/scikit-physlearn/blob/master/images/framework.png" width="600" height="300"><br><br>
</div>

Inspired by the process of human research, base boosting is a modification of the standard version of
[gradient boosting](https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451),
which is designed to emulate the paradigm of "standing on the shoulders of giants." To evaluate its
efficacy in a quantum device calibration application with a limited supply of experimental data:
* Run the
[main body](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results/main_body.py)
module to generate the test error results in the [benchmark task](#Citation).
* Compare these results with the scientific model's results in the
[benchmark](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results/benchmark.py)
module.
* Explore the difficulty in learning without the scientific model's inductive bias in the
[supplementary](https://github.com/a-wozniakowski/scikit-physlearn/blob/master/examples/paper_results/supplementary.py)
module.


## Citation

If you use this package, please consider adding the corresponding citation:
```
@article{wozniakowski_2020_boosting,
  title={Boosting on the shoulders of giants in quantum device calibration},
  author={Wozniakowski, Alex and Thompson, Jayne and Gu, Mile and Binder, Felix},
  journal={arXiv preprint arXiv:2005.06194},
  year={2020}
}

```
