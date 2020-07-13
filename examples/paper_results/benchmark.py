"""
============================
Benchmark test error
============================

This example generates the incumbent state-of-the-art model's
test error on the benchmark task in the quantum device calibration
application. 

Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>
"""

from physlearn import Regressor
from physlearn.datasets import load_benchmark


X_train, X_test, y_train, y_test = load_benchmark(return_split=True)

reg = Regressor()
test_error = reg.score(y_test, X_test)
print('The single-target benchmark results:')
print(test_error.round(decimals=2))
print('The benchmark error:')
print(test_error.mean().round(decimals=2))
