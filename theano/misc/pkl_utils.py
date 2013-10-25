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
