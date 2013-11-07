"""
Utility classes and methods to pickle parts of symbolic graph.

These pickled graphs can be used, for instance, as cases for
unit tests or regression tests.
"""
__docformat__ = "restructuredtext en"
__authors__ = "Pascal Lamblin"
__copyright__ = "Copyright 2013, Universite de Montreal"
__license__ = "3-clause BSD"


import pickle
import sys
import theano
from theano.compat import PY3
from theano.compat.six import string_types


sys.setrecursionlimit(3000)
Pickler = pickle.Pickler


class StripPickler(Pickler):
    """
    Subclass of Pickler that strips unnecessary attributes from Theano objects.

    Example of use::

        fn_args = dict(inputs=inputs,
                       outputs=outputs,
                       updates=updates)
        dest_pkl = 'my_test.pkl'
        f = open(dest_pkl, 'wb')
        strip_pickler = StripPickler(f, protocol=-1)
        strip_pickler.dump(fn_args)
        f.close()
    """
    def save(self, obj):
        # Remove the tag.trace attribute from Variable and Apply nodes
        if isinstance(obj, theano.gof.utils.scratchpad):
            if hasattr(obj, 'trace'):
                del obj.trace

        # Remove manually-added docstring of Elemwise ops
        elif (isinstance(obj, theano.tensor.Elemwise)):
            if '__doc__' in obj.__dict__:
                del obj.__dict__['__doc__']

        return Pickler.save(self, obj)


# Make an unpickler that tries encoding byte streams before raising TypeError.
# This is useful with python 3, in order to unpickle files created with
# python 2.
# This code is taken from Pandas, https://github.com/pydata/pandas,
# under the same 3-clause BSD license.
def load_reduce(self):
    stack = self.stack
    args = stack.pop()
    func = stack[-1]
    try:
        value = func(*args)
    except Exception:
        # try to reencode the arguments
        if self.encoding is not None:
            new_args = []
            for arg in args:
                if isinstance(arg, string_types):
                    new_args.append(arg.encode(self.encoding))
                else:
                    new_args.append(arg)
            args = tuple(new_args)
            try:
                stack[-1] = func(*args)
                return
            except Exception:
                pass

        if self.is_verbose:
            print(sys.exc_info())
            print(func, args)

        raise

    stack[-1] = value


if PY3:
    class CompatUnpickler(pickle._Unpickler):
        pass

    # Register `load_reduce` defined above in CompatUnpickler
    CompatUnpickler.dispatch[pickle.REDUCE[0]] = load_reduce
else:
    class CompatUnpickler(pickle.Unpickler):
        pass
