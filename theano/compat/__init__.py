"""Code supporting compatibility across versions of Python.
"""

# Python 3.x compatibility
from six import PY3, b, BytesIO, next
from six.moves import configparser
from six.moves import reload_module as reload
import collections

__all__ = ['PY3', 'b', 'BytesIO', 'next', 'configparser', 'reload']

if PY3:
    from operator import truediv as operator_div
    izip = zip
    imap = map
    ifilter = filter

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

    def get_unbound_function(unbound):
        # Op.make_thunk isn't bound, so don't have a __func__ attr.
        # But bound method, have a __func__ method that point to the
        # not bound method. That is what we want.
        if hasattr(unbound, '__func__'):
            return unbound.__func__
        return unbound

    from collections import OrderedDict, MutableMapping as DictMixin

    def decode(x):
        return x.decode()

    def decode_iter(itr):
        for x in itr:
            yield x.decode()
else:
    from six import get_unbound_function
    from operator import div as operator_div
    from itertools import izip, imap, ifilter

    def exc_message(e):
        return e[0]

    cmp = cmp

    # Older Python 2.x compatibility
    from theano.compat.python2x import DictMixin, OrderedDict

    def decode(x):
        return x

    def decode_iter(x):
        return x

__all__ += ['cmp', 'operator_div', 'DictMixin', 'OrderedDict', 'decode',
            'decode_iter', 'get_unbound_function', 'imap', 'izip', 'ifilter']


class DefaultOrderedDict(OrderedDict):
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
                not isinstance(default_factory, collections.Callable)):
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
        return type(self), args, None, None, list(self.items())

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

__all__ += ['DefaultOrderedDict']
