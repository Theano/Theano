"""
Helper functions to make theano backwards compatible with python 2.6 - 2.7
Now mostly there for compatibility as we don't support Python 2.6 anymore.
"""
from __future__ import absolute_import, print_function, division

try:
    from UserDict import DictMixin
except ImportError:
    from collections import MutableMapping as DictMixin
from collections import OrderedDict, Counter

__all__ = ['DictMixin', 'OrderedDict', 'Counter']
