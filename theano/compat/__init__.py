"""Code supporting compatibility across versions of Python.
"""

# Python 3.x compatibility
from theano.compat.six import PY3, b, BytesIO, next, get_unbound_function
from theano.compat.six.moves import configparser
from theano.compat.six.moves import reload_module as reload

__all__ = ['PY3', 'b', 'BytesIO', 'next', 'get_unbound_function',
           'configparser', 'reload']

if PY3:
    from operator import truediv as operator_div

    # In python 3.x, when an exception is reraised it saves original
    # exception in its args, therefore in order to find the actual
    # message, we need to unpack arguments recursively.
    def exc_message(e):
        msg = e.args[0]
        if isinstance(msg, Exception):
            return exc_message(msg)
        return msg

    def cmp(x, y):
        """Return -1 if x < y, 0 if x == y, 1 if x > y."""
        return (x > y) - (x < y)

    from functools import partial
    from collections import defaultdict, deque
    from sys import maxsize
    from itertools import combinations, product
    from collections import OrderedDict, MutableMapping as DictMixin

    def decode(x):
        return x.decode()

    def decode_iter(itr):
        for x in itr:
            yield x.decode()
else:

    from operator import div as operator_div

    def exc_message(e):
        return e[0]

    cmp = cmp
    from functools import partial
    from collections import defaultdict, deque

    from itertools import combinations, product
    from sys import maxsize

    # Older Python 2.x compatibility
    from theano.compat.python2x import DictMixin, OrderedDict

    def decode(x):
        return x

    def decode_iter(x):
        return x

__all__ += ['cmp', 'operator_div', 'partial', 'defaultdict', 'deque',
            'combinations', 'product', 'maxsize', 'DictMixin',
            'OrderedDict', 'decode', 'decode_iter']


class DefaultOrderedDict(OrderedDict):
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
                not callable(default_factory)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

__all__ += ['DefaultOrderedDict']
