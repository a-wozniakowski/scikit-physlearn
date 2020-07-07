"""
Machine learning package for Python.
==================================

# Author: Alex Wozniakowski <wozn0001@e.ntu.edu.sg>
"""


from __future__ import absolute_import

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
