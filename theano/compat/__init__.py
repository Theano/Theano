"""Code supporting compatibility across versions of Python.

"""

# Python 3.x compatibility
from theano.compat.six import PY3, b, BytesIO, next, get_unbound_function
from theano.compat.six.moves import configparser
from theano.compat.six.moves import reload_module as reload

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

    def cmp(a, b):
        """Return -1 if x < y, 0 if x == y, 1 if x > y."""
        return (a > b) - (a < b)

    all = all
    any = any
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

    # Older Python 2.x compatibility
    from theano.compat.python2x import all, any, partial, defaultdict, deque
    from theano.compat.python2x import combinations, product, maxsize
    from theano.compat.python2x import DictMixin, OrderedDict

    def decode(x):
        return x

    def decode_iter(x):
        return x
