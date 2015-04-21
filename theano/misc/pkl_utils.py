"""
Utility classes and methods to pickle parts of symbolic graph.

These pickled graphs can be used, for instance, as cases for
unit tests or regression tests.
"""
import numpy
import pickle
import sys
import tempfile
import zipfile
import warnings
from collections import defaultdict
from contextlib import closing
from pickle import HIGHEST_PROTOCOL
from six import StringIO
try:
    from pickle import DEFAULT_PROTOCOL
except ImportError:
    DEFAULT_PROTOCOL = HIGHEST_PROTOCOL

import theano
from theano import config
from theano.compat import PY3
from theano.compat.six import string_types
from theano.compile.sharedvalue import SharedVariable
try:
    from cuda_ndarray import cuda_ndarray
except ImportError:
    cuda_ndarray = None



__docformat__ = "restructuredtext en"
__authors__ = "Pascal Lamblin"
__copyright__ = "Copyright 2013, Universite de Montreal"
__license__ = "3-clause BSD"

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


class PersistentNdarrayID(object):
    """Persist ndarrays in an object by saving them to a zip file.

    Parameters
    ----------
    zip_file : :class:`zipfile.ZipFile`
        A zip file handle that the NumPy arrays will be saved to.

    Notes
    -----
    The convention for persistent ids given by this class and its derived
    classes is that the name should take the form `type.name` where `type`
    can be used by the persistent loader to determine how to load the
    object, while `name` is human-readable and as descriptive as possible.

    """
    def __init__(self, zip_file):
        self.zip_file = zip_file
        self.count = 0
        self.seen = {}

    def _resolve_name(self, obj):
        """Determine the name the object should be saved under."""
        name = 'array_{}'.format(self.count)
        self.count += 1
        return name

    def __call__(self, obj):
        if ((type(obj) is numpy.ndarray) or
                (type(obj) is cuda_ndarray.CudaNdarray)):
            if id(obj) not in self.seen:
                def write_array(f):
                    numpy.lib.format.write_array(f, numpy.asarray(obj))
                name = self._resolve_name(obj)
                zipadd(write_array, self.zip_file, name)
                if type(obj) is cuda_ndarray.CudaNdarray:
                    self.seen[id(obj)] = 'cuda_ndarray.{}'.format(name)
                else:
                    self.seen[id(obj)] = 'ndarray.{}'.format(name)
            return self.seen[id(obj)]


class PersistentSharedVariableID(PersistentNdarrayID):
    """Persist the names of shared variable arrays in the zip file.

    If a shared variable has a name, this name is used as the name of the
    NPY file inside of the zip file. NumPy arrays that aren't matched to a
    shared variable are persisted as usual (i.e. `array_0`, `array_1`,
    etc.)

    Parameters
    ----------
    allow_unnamed : bool, optional
        Allow shared variables without a name to be persisted. Defaults to
        ``True``.
    allow_duplicates : bool, optional
        Allow multiple shared variables to have the same name, in which
        case they will be numbered e.g. `x`, `x_2`, `x_3`, etc. Defaults to
        ``True``.

    Raises
    ------
    ValueError
        If an unnamed shared variable is encountered and `allow_unnamed` is
        ``False``, or if two shared variables have the same name, and
        `allow_duplicates` is ``False``.

    """
    def __init__(self, zip_file, allow_unnamed=True, allow_duplicates=True):
        super(PersistentSharedVariableID, self).__init__(zip_file)
        self.name_counter = defaultdict(int)
        self.ndarray_names = {}
        self.allow_unnamed = allow_unnamed
        self.allow_duplicates = allow_duplicates

    def _resolve_name(self, obj):
        if id(obj) in self.ndarray_names:
            name = self.ndarray_names[id(obj)]
            count = self.name_counter[name]
            if count:
                if not self.allow_duplicates:
                    raise ValueError("multiple shared variables with the name "
                                     "`{}` found".format(name))
                name = '{}_{}'.format(name, count + 1)
            self.name_counter[name] += 1
            return name
        return super(PersistentSharedVariableID, self)._resolve_name(obj)

    def __call__(self, obj):
        if isinstance(obj, SharedVariable):
            if obj.name:
                if obj.name == 'pkl':
                    ValueError("can't pickle shared variable with name `pkl`")
                self.ndarray_names[id(obj.container.storage[0])] = obj.name
            elif not self.allow_unnamed:
                raise ValueError("unnamed shared variable, {}".format(obj))
        return super(PersistentSharedVariableID, self).__call__(obj)


class PersistentNdarrayLoad(object):
    """Load NumPy arrays that were persisted to a zip file when pickling.

    Parameters
    ----------
    zip_file : :class:`zipfile.ZipFile`
        The zip file handle in which the NumPy arrays are saved.

    """
    def __init__(self, zip_file):
        self.zip_file = zip_file

    def __call__(self, persid):
        array_type, name = persid.split('.')

        array = numpy.lib.format.read_array(self.zip_file.open(name))
        if array_type == 'cuda_ndarray':
            if config.experimental.unpickle_gpu_on_cpu:
                # directly return numpy array
                warnings.warn("config.experimental.unpickle_gpu_on_cpu is set "
                              "to True. Unpickling CudaNdarray as "
                              "numpy.ndarray")
                return array
            elif cuda_ndarray:
                return cuda_ndarray.CudaNdarray(array)
            else:
                raise ImportError("Cuda not found. Cannot unpickle "
                                  "CudaNdarray")
        else:
            return array


def dump(obj, f, protocol=DEFAULT_PROTOCOL,
         persistent_id=PersistentSharedVariableID):
    """Pickles an object to a zip file using external persistence.

    Parameters
    ----------
    obj : object
        The object to pickle.
    f : file
        The file handle to save the object to.
    protocol : int, optional
        The pickling protocol to use. Unlike Python's built-in pickle, the
        default is set to `2` insstead of 0 for Python 2. The Python 3
        default (level 3) is maintained.
    persistent_id : callable
        The callable that persists certain objects in the object hierarchy
        to separate files inside of the zip file. For example,
        :class:`PersistentNdarrayID` saves any :class:`numpy.ndarray` to a
        separate NPY file inside of the zip file.

    Notes
    -----
    The final file is simply a zipped file containing at least one file,
    `pkl`, which contains the pickled object. It can contain any other
    number of external objects. Note that the zip files are compatible with
    NumPy's :func:`numpy.load` function.

    >>> import theano
    >>> foo_1 = theano.shared(0, name='foo')
    >>> foo_2 = theano.shared(1, name='foo')
    >>> with open('model.zip', 'w') as f:
    ...     dump((foo_1, foo_2, numpy.array(2)), f)
    >>> numpy.load('model.zip').keys()
    ['foo', 'foo_2', 'array_0', 'pkl']
    >>> numpy.load('model.zip')['foo']
    array(0)
    >>> with open('model.zip') as f:
    ...     foo_1, foo_2, array = load(f)
    >>> array
    array(2)

    """
    with closing(zipfile.ZipFile(f, 'w', zipfile.ZIP_DEFLATED,
                                 allowZip64=True)) as zip_file:
        def func(f):
            p = pickle.Pickler(f, protocol=protocol)
            p.persistent_id = persistent_id(zip_file)
            p.dump(obj)
        zipadd(func, zip_file, 'pkl')


def load(f, persistent_load=PersistentNdarrayLoad):
    """Load a file that was dumped to a zip file.

    Parameters
    ----------
    f : file
        The file handle to the zip file to load the object from.
    persistent_load : callable, optional
        The persistent loading function to use for unpickling. This must be
        compatible with the `persisten_id` function used when pickling.

    """
    with closing(zipfile.ZipFile(f, 'r')) as zip_file:
        p = pickle.Unpickler(StringIO(zip_file.open('pkl').read()))
        p.persistent_load = persistent_load(zip_file)
        return p.load()


def zipadd(func, zip_file, name):
    """Calls a function with a file object, saving it to a zip file.

    Parameters
    ----------
    func : callable
        The function to call.
    zip_file : :class:`zipfile.ZipFile`
        The zip file that `func` should write its data to.
    name : str
        The name of the file inside of the zipped archive that `func`
        should save its data to.

    """
    with tempfile.NamedTemporaryFile('wb', delete=False) as temp_file:
        func(temp_file)
        temp_file.close()
        zip_file.write(temp_file.name, arcname=name)
