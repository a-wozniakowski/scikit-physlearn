"""
The :mod:`physlearn._build_utils.min_dependencies` module contains all of the
minimum dependencies for scikit-physlearn.
"""

# Author: Alex Wozniakowski
# License: MIT

import platform


if platform.python_implementation() == 'PyPy':
    SCIPY_MIN_VERSION = '1.1.0'
    NUMPY_MIN_VERSION = '1.14.0'
else:
    SCIPY_MIN_VERSION = '0.19.1'
    NUMPY_MIN_VERSION = '1.13.3'

JOBLIB_MIN_VERSION = '0.11'
SCIKIT_LEARN_MIN_VERSION = '0.23.0'
PANDAS_MIN_VERSION = '1.0.0'
SHAP_MIN_VERSION = '0.35.0'
IPYTHON_MIN_VERSION = '7.11.0'
BAYESIAN_OPTIMIZATION_MIN_VERSION = '1.2.0'
CATBOOST_MIN_VERSION = '0.23.2'
XGBOOST_MIN_VERSION = '1.1.0'
LIGHTGBM_MIN_VERSION = '2.3.0'
MLXTEND_MIN_VERSION = '0.17.0'
PYTHON_LEVENSHTEIN_WHEELS_MIN_VERSION = '0.13.1'

dependent_packages = {
    'numpy': (NUMPY_MIN_VERSION, 'build, install'),
    'scipy': (SCIPY_MIN_VERSION, 'build, install'),
    'scikit-learn': (SCIKIT_LEARN_MIN_VERSION, 'build, install'),
    'pandas': (PANDAS_MIN_VERSION, 'build, install'),
    'shap': (SHAP_MIN_VERSION, 'build, install'),
    'ipython': (IPYTHON_MIN_VERSION, 'build, install'),
    'bayesian-optimization': (BAYESIAN_OPTIMIZATION_MIN_VERSION, 'build, install'),
    'catboost': (CATBOOST_MIN_VERSION, 'build, install'),
    'xgboost': (XGBOOST_MIN_VERSION, 'build, install'),
    'lightgbm': (LIGHTGBM_MIN_VERSION, 'build, install'),
    'mlxtend': (MLXTEND_MIN_VERSION, 'build, install'),
    'joblib': (JOBLIB_MIN_VERSION, 'build, install'),
    'python-levenshtein-wheels': (PYTHON_LEVENSHTEIN_WHEELS_MIN_VERSION, 'build, install'),
    'sphinx': ('3.0.3', 'docs'),
    'sphinx-gallery': ('0.7.0', 'docs')
}


tag_to_packages: dict = {
    extra: [] for extra in ['build', 'install', 'docs', 'tests']
}
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(', '):
        tag_to_packages[extra].append('{}>={}'.format(package, min_version))
