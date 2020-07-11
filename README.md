scikit-physlearn
=======

**scikit-physlearn** is a Python package for single-target and multi-target regression. 
It is designed to amalgamate regressors in
[scikit-learn](https://scikit-learn.org/),
[LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html),
[XGBoost](https://xgboost.readthedocs.io/en/latest/),
[CatBoost](https://catboost.ai/),
and [Mlxtend](http://rasbt.github.io/mlxtend/)
through a unified ```Regressor``` object, which follows the scikit-learn API and uses pandas data representations.
The regressor object supports base boosting, as introduced in the paper:
[*Boosting on the shoulders of giants in quantum device calibration*](https://arxiv.org/abs/2005.06194).

The repository was started by Alex Wozniakowski during his graduate studies at Nanyang Technological University.

Base boosting
-----------

<div align="center">
  <img src="https://github.com/a-wozniakowski/scikit-physlearn/blob/a-wozniakowski-dev/images/framework.png" width="600" height="300"><br><br>
</div>

The standard version of gradient boosting fits an additive model, whereby the first algorithmic step
initializes to the optimal constant model. In the low data regime, this choice of initialization generally
results in substandard generalization performance. Inspired by the process of human research, wherein scientific
progress derives from prior scientific knowledge, base boosting initializes with a base regressor's initial predictions.
As a greedy stagewise algorithm, base boosting sequentially appends basis functions to the base regressor's initial basis.


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

To obtain the results in ```main_body.py```, we use
[StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
as the basis function in each single-target regression subtask. Moreover, we compute the pseudo-residuals
by taking the negative gradient of the squared error loss function.

To obtain the results in ```improved_main_body.py```, we modify the aforementioned procedure. Namely, we use
[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
as the basis function in the third single-target regression subtask. Moreover, we compute the pseudo-residuals
by taking the negative gradient of the Huber loss function in each single-target regression subtask. 

To obtain the results in ```supplementary.py```, we use
[MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
for each single-target regression subtask, where the features are the raw control voltage features.


Reference
-----------

If you use this code, please consider adding the corresponding citation:
```
@article{wozniakowski_2020_boosting,
  title={Boosting on the shoulders of giants in quantum device calibration},
  author={Wozniakowski, Alex and Thompson, Jayne and Gu, Mile and Binder, Felix},
  journal={arXiv preprint arXiv:2005.06194},
  year={2020}
}

```
