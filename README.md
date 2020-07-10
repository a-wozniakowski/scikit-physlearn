scikit-physlearn
----------------

**scikit-physlearn** is a Python package for single-target and multi-target regression tasks. 
It is designed to follow the scikit-learn API with data representations in pandas. 
The model dictionary provides streamlined access to regressors in
[scikit-learn](https://scikit-learn.org/),
[LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html),
[XGBoost](https://xgboost.readthedocs.io/en/latest/),
[CatBoost](https://catboost.ai/),
and [Mlxtend](http://rasbt.github.io/mlxtend/).
The ```ModifiedPipeline``` object offers the ability to boost any of these regressors with a modified initialization,
as described in the paper *Boosting on the shoulders of giants in quantum device calibration*.

The repository was started by Alex Wozniakowski during his graduate studies at Nanyang Technological University.

Reference Paper
----------------
If you use this code, please consider adding the corresponding citation:
```
@article{wozniakowski2020boosting,
  title={Boosting on the shoulders of giants in quantum device calibration},
  author={Wozniakowski, Alex and Thompson, Jayne and Gu, Mile and Binder, Felix},
  journal={arXiv preprint arXiv:2005.06194},
  year={2020}
}

```

Below is the directory structure for ```paper_results```:
```
examples
|
|
|___paper_results
    |   improved_main_body.py
    |   main_body.py
    |   supplementary.py
```
