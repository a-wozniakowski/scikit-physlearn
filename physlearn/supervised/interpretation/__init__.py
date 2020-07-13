from __future__ import absolute_import


try:
    from .interpret_regressor import ShapInterpret
except ImportError:
    pass


__all__ = ['ShapInterpret']
