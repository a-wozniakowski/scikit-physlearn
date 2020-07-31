"""
Machine learning package for Python.
====================================

# Author: Alex Wozniakowski
# Licence: MIT
"""

from __future__ import absolute_import


__version__ = '0.1.4'


try:
    from .supervised.regression import Regressor
except ImportError:
    pass

try:
    from .pipeline import ModifiedPipeline
except ImportError:
    pass

try:
    from .loss import (LeastSquaresError, LeastAbsoluteError, HuberLossFunction,
                       QuantileLossFunction)
except ImportError:
    pass


__all__ = ['Regressor', 'ModifiedPipeline',
           'LeastSquaresError', 'LeastAbsoluteError', 'HuberLossFunction',
           'QuantileLossFunction']
