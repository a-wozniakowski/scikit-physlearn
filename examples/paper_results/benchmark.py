"""
============================
Benchmark test error
============================

This example generates the incumbent state-of-the-art's
test error on the benchmark task. 
"""

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>

from physlearn import Regressor
from physlearn.datasets import load_benchmark


# To comput the benchmark error, we only need the test data.
# We denote the initial prediction examples as X_test and
# the multi-targets as y_test. Both have the same shape,
# namely (41, 5).
_, X_test, _, y_test = load_benchmark(return_split=True)

# Here we make an instance of Regressor, so that we can
# automatically compute the test error as a DataFrame.
reg = Regressor()
test_error = reg.score(y_test, X_test)

print('The single-target test error:')
print(test_error.round(decimals=2))
print('The benchmark error:')
print(test_error.mean().round(decimals=2))
