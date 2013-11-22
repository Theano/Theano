"""A `Type` and `Op` classes to work with numpy.ndarrays symbolically."""

__docformat__ = "restructuredtext en"

import sys
import warnings
from itertools import izip

import numpy
from copy import copy as python_copy

import theano
from theano.configparser import config
from theano import gof
from theano.gof import Apply, Constant, Op, Variable

from theano.tensor import elemwise
from theano.tensor.var import (AsTensorError, TensorVariable,
                               TensorConstant,
                               _tensor_py_operators)
from theano.tensor.type import TensorType
from theano import scalar as scal
from theano.gof.python25 import partial, any, all
from theano.gof.utils import hashtype
from theano import compile, printing
from theano.printing import pprint, min_informative_str

# We use these exceptions as well.
import theano.scalar.sharedvar
from theano.gradient import grad_undefined
from theano.gradient import grad_not_implemented
from theano.gradient import DisconnectedType

### set up the external interface
from theano.tensor.elemwise import Elemwise, DimShuffle, CAReduce, Sum

import logging
_logger = logging.getLogger("theano.tensor.basic")

# This is needed as we will hide it later
python_complex = complex
python_any = any
python_all = all

# Define common subsets of dtypes (as strings).
complex_dtypes = map(str, scal.complex_types)
continuous_dtypes = map(str, scal.continuous_types)
float_dtypes = map(str, scal.float_types)
discrete_dtypes = map(str, scal.discrete_types)
all_dtypes = map(str, scal.all_types)
int_dtypes = map(str, scal.int_types)
uint_dtypes = map(str, scal.uint_types)


class ShapeError(Exception):
    """Raised when the shape cannot be computed."""
    pass


def check_equal_numpy(x, y):
    """
    Returns True iff x and y are equal (checks the dtype and
    shape if x and y are numpy.ndarray instances).
    """
    if isinstance(x, numpy.ndarray) and isinstance(y, numpy.ndarray):
        return (x.dtype == y.dtype and x.shape == y.shape and
                numpy.any(abs(x - y) < 1e-10))
    elif (isinstance(x, numpy.random.RandomState) and
          isinstance(y, numpy.random.RandomState)):
        return python_all(numpy.all(a == b) for a, b in
                          izip(x.__getstate__(), y.__getstate__()))
    else:
        return x == y

compile.register_checker(check_equal_numpy)


__oplist_constructor_list = []
"""List of functions to be listed as op constructors in the oplist
(`gen_oplist`, doc/oplist.txt)."""


def constructor(f):
    """Add `f` to :doc:`oplist`.

    Make `f` appear as a constructor in the oplist (`gen_oplist`,
    doc/oplist.txt).
    """
    __oplist_constructor_list.append(f)
    return f


def __oplist_tag(thing, tag):
    tags = getattr(thing, '__oplist_tags', [])
    tags.append(tag)
    thing.__oplist_tags = tags


if 0:
    # this starts to feel like we're enumerating all the types
    # the one place where this is used we should also allow for sparse
    # variables
    # - JB 20100226
    def as_cuda_or_tensor_variable(x, name=None, ndim=None):
        """
        This function do the same as_tensor_variable, but don't
        transfert the value on the gpu
        """
        if hasattr(x, '_as_CudaNdarrayVariable'):
            # TODO: pass name and ndim arguments
            return x._as_CudaNdarrayVariable()
        return as_tensor_variable(x, name, ndim)


def as_tensor_variable(x, name=None, ndim=None):
    """Return `x`, transformed into a `TensorType`

    This function is often used by `make_node` methods of `Op`
    subclasses to turn ndarrays, numbers, `Scalar` instances, `Apply`
    instances and `TensorType` instances into valid input list
    elements.

    :Parameters:
     - `x`: Apply instance, Variable instance, numpy.ndarray, or number
       This thing will be transformed into a `Variable` in a sensible way.  An
       ndarray argument will not be copied, but a list of numbers will be
       copied to make an ndarray.
     - `name`: str or None
       If a new `Variable` instance is created, it will be named with this
       string.
     - `ndim`: None or integer
       Return a Variable with this many dimensions.  Raise TypeError if it's
       not possible.

    :Exceptions:
     - `ValueError`: raised if an `Apply` with more then one output is fetched
     - `AsTensorError`: raised if `x` cannot be converted to a TensorType Variable

    """
    if hasattr(x, '_as_TensorVariable'):
        return x._as_TensorVariable()  # TODO: pass name and ndim arguments

    if isinstance(x, gof.Apply):
        # TODO: use Apply's default output mechanism
        if len(x.outputs) != 1:
            raise ValueError(
                "It is ambiguous which output of a multi-output Op has"
                " to be fetched.", x)
        else:
            x = x.outputs[0]
    if isinstance(x, Variable):
        if isinstance(x.type, scal.Scalar):
            x = tensor_from_scalar(x)

        if not isinstance(x.type, TensorType):
            raise AsTensorError(
                "Variable type field must be a TensorType.", x, x.type)

        if ndim is None:
            return x
        else:
            if (x.type.ndim > ndim):
                # TODO: strip off leading broadcastable dimensions
                raise ValueError(
                    'TensorType could not be cast to have %i dimensions' %
                    ndim, x.type)
            elif (x.type.ndim < ndim):
                return shape_padleft(x, n_ones=(ndim - x.type.ndim))
            else:
                return x
    if isinstance(x, (tuple, list)) and python_any(isinstance(xi, Variable)
                                                   for xi in x):
        try:
            return stack(*x)
        except (TypeError, ValueError):
            pass

    if isinstance(x, bool):
        raise AsTensorError(
            "Cannot cast True or False as a tensor variable. Please use 1 or "
            "0. This error might be caused by using the == operator on "
            "Variables. v == w does not do what you think it does, "
            "use theano.tensor.eq(v, w) instead.")

    try:
        return constant(x, name=name, ndim=ndim)
    except TypeError:
        try:
            str_x = str(x)
        except Exception:
            str_x = repr(x)
        raise AsTensorError("Cannot convert %s to TensorType" % str_x, type(x))

# this has a different name, because _as_tensor_variable is the
# function which ops use to upcast their arguments... this
# internal-use function is a good place to put debugging stuff, better
# than the global astensor.
_as_tensor_variable = as_tensor_variable

as_tensor = as_tensor_variable


class NumpyAutocaster(object):
    """
    This class is used to cast python ints and floats to numpy arrays.

    The behavior when called on scalar `x` depends on `config.cast_policy`:
        - 'numpy' will simply use the same type as found by `numpy.asarray(x)`.
        - 'numpy+floatX' will do the same, except it will use float32 instead
          of float64 if `x` is a Python float and `config.floatX` is set to
          'float32' (note that if `x` is a numpy scalar whose data type is
          float64, it is not modified since we assume the user is purposedly
          using float64).
        - 'custom' lets one define a tuple of data types such that:
            - if `x` is already a numpy scalar and its data type is in this
              tuple, then it is returned unchanged;
            - otherwise, the first data type in this tuple that can represent
              `x` without loss of precision will be used, unless `x` is a float
              and 'float32' is in the tuple (in which case `x` is cast as a
              float32);
            - if no data type can represent `x` without loss of precision, then
              the last data type in the tuple will be used.
    """
    def __init__(self, dtypes):
        """
        Constructor.

        :type dtypes: Tuple of strings.
        :param dtypes: The ordered list of preferred data types (only used when
        `config.cast_policy` is set to 'custom', see the `NumpyAutocaster` help
        for details).
        """
        self.dtypes = tuple(dtypes)

    def __call__(self, x):
        # Make sure we only deal with scalars.
        assert (isinstance(x, int) or
                isinstance(x, float) or
                (isinstance(x, numpy.ndarray) and x.ndim == 0))

        if config.cast_policy == 'numpy':
            return numpy.asarray(x)
        elif config.cast_policy == 'numpy+floatX':
            rval = numpy.asarray(x)
            if (rval.dtype == 'float64' and         # numpy wants float64
                config.floatX == 'float32' and      # but we prefer float32
                not hasattr(x, 'dtype')):           # and `x` was not typed
                rval = theano._asarray(rval, dtype='float32')
            return rval

        # The following is the original code, corresponding to the 'custom'
        # option for `config.cast_policy`.
        assert config.cast_policy == 'custom'

        try:
            # Pass through numpy scalars, since they are already typed on
            # purpose typically.
            if str(x.dtype) in self.dtypes:
                # No need to cast `x` into a new dtype. Note that we still
                # need to convert it into an array, because it may not be
                # one already (e.g. if x == numpy.float64(1.1)).
                return numpy.asarray(x)
        except AttributeError:
            # Means `x` has no 'dtype' attribute.
            pass

        # unsafe downcast of float64 variables when config.floatX == 'float32'
        # recall: float is numpy.float
        if (isinstance(x, float) and
            config.floatX in self.dtypes and
            config.floatX == 'float32'):

            return theano._asarray(x, dtype='float32')

        for dtype in self.dtypes:
            x_ = theano._asarray(x, dtype=dtype)
            if numpy.all(x == x_):
                break
        # returns either an exact x_==x, or the last cast x_
        return x_

autocast_int = NumpyAutocaster(('int8', 'int16', 'int32', 'int64'))
autocast_float = NumpyAutocaster(('float32', 'float64'))


# autocast_float dtypes might be manipulated in tensor.__init__
#
# Note: it's a bit weird for a compiler to automatically downcast
# literals like this, and it might have implications for efficiency
# when mixing types.  For example when you add 1.0 + dmatrix(), the
# 1.0 could be converted to float32, and require upcasting for the +
# operation at every position in the dmatrix.  using
# theano._asarray(1.0, dtype='float64') will circumvent this
# autocasting, and in future, our ops might be smarter about factoring
# out upcasts.  The advantage of this mechanism is to combine it with
# floatX so that 1.0 + xmatrix() will always have the same type as the
# xmatrix().
#
class autocast_float_as(object):
    """
    This class makes it possible to temporarily and locally adjust autocasting
    behavior when `config.cast_policy` is set to 'custom'.
    If `config.cast_policy` is not 'custom', an exception is raised.

    For example:
    >>> with autocast_float_as('float32') as _dummy:
    >>>    assert (fvector() + 1.1).dtype == 'float32'  # temporary downcasting
    >>> assert (fvector() + 1.1).dtype == 'float64' # back to default behaviour

    This class might be convenient in some code, but it definitely
    helps to test the autocasting mechanism.
    """
    def __init__(self, *dtypes):
        self.dtypes = dtypes
        assert config.cast_policy == 'custom'

    def __enter__(self):
        assert config.cast_policy == 'custom'
        self.old_dtypes = autocast_float.dtypes
        autocast_float.dtypes = self.dtypes

    def __exit__(self, *args):
        assert config.cast_policy == 'custom'
        autocast_float.dtypes = self.old_dtypes


def constant_or_value(x, rtype, name=None, ndim=None, dtype=None):
    """Return a symbolic `Constant` with value `x`

    :Exceptions:
     - `TypeError`: `x` could not be converted to a numpy.ndarray
     - `ValueError`: `x` could not be expanded to have ndim dimensions

    """
    if dtype is not None:
        # in this case, the semantics are that the caller is forcing the dtype
        x_ = theano._asarray(x, dtype=dtype)
    else:
        # In this case, this function should infer the dtype according to the
        # autocasting rules. See autocasting above.
        x_ = None
        if rtype is TensorConstant and isinstance(x, int):
            x_ = autocast_int(x)
        elif rtype is TensorConstant and isinstance(x, float):
            x_ = autocast_float(x)
        elif rtype is TensorConstant and isinstance(x, long):
            # We need to address the case where a long number is used in a
            # Theano graph, because on Windows 64, all shapes are expressed
            # with longs.
            # If a long fits in int64, we convert it into an int64, like
            # numpy.asarray() does up to 1.7. NumPy 1.7.1 upcasts to int64
            # if possible, but falls back to uint64 if int64 isn't possible but
            # uint64 is. We always do as NumPy 1.7.1  here.
            # If x is too big, an OverflowError will be raised by numpy.
            try:
                x_ = theano._asarray(x, dtype='int64')
            except OverflowError:
                x_ = theano._asarray(x, dtype='uint64')
        elif isinstance(x, numpy.ndarray):
            x_ = x
            # Currently we do not have a bool dtype in Theano.
            # So we upcast it to uint8 to avoid breaking our interface for
            # constant.
            if x.dtype == 'bool':
                x_ = numpy.asarray(x_, dtype='uint8')
        else:
            # Here x is probably a list or a tuple. If it contains a long,
            # we will behave like the current NumPy version: 1.7 and below,
            # it will only work if the long fits in int64. For NumPy 1.7.1+,
            # it will work if the long fits in int64 or uint64.
            x_ = numpy.asarray(x)

    assert type(x_) in [numpy.ndarray, numpy.memmap]

    bcastable = [d == 1 for d in x_.shape]
    if ndim is not None:
        if len(bcastable) < ndim:
            bcastable = [True] * (ndim - len(bcastable)) + bcastable
        elif len(bcastable) > ndim:
            # TODO: strip off dimensions of size 1
            raise ValueError(
                'ndarray could not be cast to constant with %i dimensions' %
                ndim)
        assert len(bcastable) == ndim

    try:
        if rtype is TensorConstant:
            rval = rtype(
                    TensorType(dtype=x_.dtype, broadcastable=bcastable),
                    x_.copy(),
                    name=name)
            return rval
        else:
            # leave the shape out of the type
            return rtype(TensorType(dtype=x_.dtype, broadcastable=bcastable),
                         x_, name=name)
    except Exception:
        raise TypeError("Could not convert %s to TensorType" % x, type(x))


constant_cache = {}
def constant(x, name=None, ndim=None, dtype=None):
    ret = constant_or_value(x, rtype=TensorConstant, name=name, ndim=ndim,
                             dtype=dtype)

    #We create a small cache of frequently used constant.
    #This speed up the Merge optimization for big graph.
    #We want to cache all scalar to don't merge as frequently constants.
    #But we don't want to cache too much stuff
    #So we cache integer with dtype [u]int and float where the value is between -10 and 10
    #We want to cache all broadcast pattern for scalar.
    if not constant.enable:
        return ret
    sig = ret.signature()
    if (sig not in constant_cache and ret.data.size == 1 and
        ret.data <= 10 and ret.data >= -10 and
        (ret.dtype in int_dtypes or ret.dtype in uint_dtypes or
         (ret.dtype in float_dtypes and int(ret.data) == ret.data))):
        constant_cache[sig] = ret
        # This is needed to raise a good error to the user.
        ret.cached = True

    return constant_cache.get(sig, ret)
constant.enable = True


def _obj_is_wrappable_as_tensor(x):
    try:
        constant(x)
        return True
    except TypeError:
        return False


def _wrap_tensor_into_member(x):
    return compile.module.Member(constant(x))
compile.module.register_wrapper(_obj_is_wrappable_as_tensor,
                                _wrap_tensor_into_member, no_warn=True)


if int(config.tensor.cmp_sloppy) > 1:
    # This config variable is a quick-and-dirty way to get low-precision
    # comparisons.  For a more precise setting of these tolerances set
    # them explicitly in your user code by assigning, for example,
    # "theano.tensor.basic.float32_atol = ..."

    # When config.tensor.cmp_sloppy>1 we are even more sloppy. This is
    # useful to test the GPU as they don't use extended precision and
    # this cause some difference bigger then the normal sloppy.
    float32_atol = 5e-4
    float32_rtol = 1e-3
    float64_rtol = 1e-4
    float64_atol = 1e-3
elif int(config.tensor.cmp_sloppy):
    float32_atol = 1e-4
    float32_rtol = 1e-3
    float64_rtol = 1e-4
    float64_atol = 1e-3
else:
    # If you change those value in test don't forget to put them back
    # when the test end.  Don't forget the case when the test fail.
    float32_atol = 1e-5
    float32_rtol = 1e-5

    # defaults in numpy.allclose
    float64_rtol = 1.0000000000000001e-05
    float64_atol = 1e-8
    # more strict. Atleast float32 precision.
    float64_rtol = 1.0000000000000001e-06


def _allclose(a, b, rtol=None, atol=None):
    a = numpy.asarray(a)
    b = numpy.asarray(b)
    narrow = 'float32', 'complex64'
    if (str(a.dtype) in narrow) or (str(b.dtype) in narrow):
        atol_ = float32_atol
        rtol_ = float32_rtol
    else:
        atol_ = float64_atol
        rtol_ = float64_rtol
    if rtol is not None:
        rtol_ = rtol
    if atol is not None:
        atol_ = atol

    # Work around bug in Numpy, see http://projects.scipy.org/numpy/ticket/1684
    if str(b.dtype) in int_dtypes and (numpy.absolute(b) < 0).any():
        b = theano._asarray(b, dtype='float64')

    return numpy.allclose(a, b, atol=atol_, rtol=rtol_)


class NotScalarConstantError(Exception):
    """
    Raised by get_scalar_constant_value if called on something that is
    not a scalar constant.
    """


class EmptyConstantError(NotScalarConstantError):
    """
    Raised by get_scalar_const_value if called on something that is a
    zero dimensional constant.
    """


def get_scalar_constant_value(v):
    """return the constant scalar(0-D) value underlying variable `v`

    If v is the output of dimshuffles, fills, allocs, rebroadcasts, cast
    this function digs through them.

    If `v` is not some view of constant scalar data, then raise a
    NotScalarConstantError.

    :note: There may be another function similar to this one in the
        code, but I'm not sure where it is.
    """

    if v is None:
        # None is not a scalar (and many uses of this function seem to depend
        # on passing it None)
        raise NotScalarConstantError()

    if isinstance(v, (numpy.integer, int, float)):
        return numpy.asarray(v)

    def numpy_scalar(data):
        """ Return a scalar stored in a numpy ndarray, or raise
        NotScalarConstantError if the numpy ndarray is not a scalar
        """

        # handle case where data is numpy.array([])
        if data.ndim > 0 and  (len(data.shape) == 0 or
            __builtins__['max'](data.shape) == 0):
            assert numpy.all(numpy.array([]) == data)
            raise EmptyConstantError()
        try:
            numpy.complex(data)  # works for all numeric scalars
            return data
        except Exception:
            raise NotScalarConstantError(
                'v.data is non-numeric, non-scalar, or has more than one'
                ' unique value', data)

    if isinstance(v, numpy.ndarray):
        return numpy_scalar(v)

    if isinstance(v, Constant):
        if getattr(v.tag, 'unique_value', None) is not None:
            data = v.tag.unique_value
        else:
            data = v.data
        return numpy_scalar(data)

    if v.owner:
        if isinstance(v.owner.op, (Alloc, DimShuffle, Rebroadcast,
                                   compile.ops.OutputGuard,
                                   compile.DeepCopyOp)):
            return get_scalar_constant_value(v.owner.inputs[0])
        if isinstance(v.owner.op, Elemwise) and \
                isinstance(v.owner.op.scalar_op, scal.Second):
            shape, val = v.owner.inputs
            return get_scalar_constant_value(val)
        if isinstance(v.owner.op, scal.Second):
            x, y = v.owner.inputs
            return get_scalar_constant_value(y)
        # Don't act as the constant_folding optimization here as this
        # fct is used too early in the optimization phase.  This would
        # mess with the stabilization optimization.
        if (isinstance(v.owner.op, Elemwise) and isinstance(
            v.owner.op.scalar_op, scal.Cast)) or \
            isinstance(v.owner.op, scal.Cast):
            const = get_scalar_constant_value(v.owner.inputs[0])
            ret = [[None]]
            v.owner.op.perform(v.owner, [const], ret)
            return ret[0][0]
        if isinstance(v.owner.op, theano.tensor.subtensor.Subtensor) and v.ndim == 0:
            # This condition depends on Subtensor always embedding constant
            # indices in the Op rather than making them inputs to the Apply
            # node.
            if isinstance(v.owner.inputs[0], TensorConstant) and \
                len(v.owner.inputs) == 1:
                try:
                    return v.owner.inputs[0].data.__getitem__(
                    tuple(v.owner.op.idx_list))
                except IndexError:
                    raise IndexError(
                            str(tuple(v.owner.op.idx_list)) +
                            " is not a valid index into " +
                            str(v.owner.inputs[0].data))

            # The index list 'idx_list' should have length the same
            # shape as the input.
            # TODO: implement the case where we take a scalar in a matrix
            assert len(v.owner.op.idx_list) == v.owner.inputs[0].ndim

            # Needed to make better graph in this test in theano/tensor/tests:
            # test_sharedvar.py:test_shared_options.test_specify_shape_partial
            if (v.owner.inputs[0].owner and
                isinstance(v.owner.inputs[0].owner.op, Join) and
                # Ensure the Join is joining only scalar variables (so that
                # the constant value can be found at the same index as the one
                # used in the sub-tensor).
                python_all(var.ndim == 0 for var in
                           v.owner.inputs[0].owner.inputs) and
                len(v.owner.op.idx_list) == 1):

                # Note the '+ 1' is because the first argument to Join is the
                # axis.
                ret = v.owner.inputs[0].owner.inputs[
                    v.owner.op.idx_list[0] + 1]
                ret = get_scalar_constant_value(ret)
                # join can cast implicitly its input in some case.
                return theano._asarray(ret, dtype=v.type.dtype)

            if (v.owner.inputs[0].owner and
                isinstance(v.owner.inputs[0].owner.op,
                           theano.tensor.opt.MakeVector) and
                # MakeVector normally accept only scalar as input.
                # We put this check in case there is change in the future
                python_all(var.ndim == 0 for var in
                           v.owner.inputs[0].owner.inputs) and
                len(v.owner.op.idx_list) == 1 and
                #idx_list can contain Scalar Type object.
                isinstance(v.owner.op.idx_list[0], (int, long,
                                                    numpy.integer))):

                # Python 2.4 does not support indexing with numpy.integer
                # So we cast it.
                idx = int(v.owner.op.idx_list[0])
                ret = v.owner.inputs[0].owner.inputs[idx]
                ret = get_scalar_constant_value(ret)
                # MakeVector can cast implicitly its input in some case.
                return theano._asarray(ret, dtype=v.type.dtype)

            # This is needed when we take the grad as the Shape op
            # are not already changed into MakeVector
            owner = v.owner
            leftmost_parent = owner.inputs[0]
            if (leftmost_parent.owner and
                isinstance(leftmost_parent.owner.op,
                           theano.tensor.Shape)):
                op = owner.op
                idx_list = op.idx_list
                idx = idx_list[0]
                grandparent = leftmost_parent.owner.inputs[0]
                gp_broadcastable = grandparent.type.broadcastable
                ndim = grandparent.type.ndim

                assert ndim == len(gp_broadcastable)

                if not (idx < len(gp_broadcastable)):
                    msg = "get_scalar_constant_value detected " + \
                            "deterministic IndexError: x.shape[%d] " + \
                            "when x.ndim=%d." % (ndim, idx)
                    if config.exception_verbosity == 'high':
                        msg += 'x=%s' % min_informative_str(x)
                    else:
                        msg += 'x=%s' % str(x)
                    raise ValueError(msg)

                if gp_broadcastable[idx]:
                    return numpy.asarray(1)

    raise NotScalarConstantError(v)


# Easy constructors

def tensor(*args, **kwargs):
    name = kwargs.pop('name', None)
    return TensorType(*args, **kwargs).make_variable(name=name)


def _multi(*fns):
    def f2(f, *names):
        if names and isinstance(names[0], int):
            if names == 1:
                return f()
            else:
                return [f() for i in xrange(names[0])]
        if isinstance(names, tuple):
            if len(names) == 1:
                names = names[0]
        if len(names) == 1:
            return f(names)
        else:
            return [f(name) for name in names]
    if len(fns) == 1:
        return partial(f2, fns)
    else:
        return [partial(f2, f) for f in fns]

cscalar = TensorType('complex64', ())
zscalar = TensorType('complex128', ())
fscalar = TensorType('float32', ())
dscalar = TensorType('float64', ())
bscalar = TensorType('int8', ())
wscalar = TensorType('int16', ())
iscalar = TensorType('int32', ())
lscalar = TensorType('int64', ())


def scalar(name=None, dtype=None):
    """Return a symbolic scalar variable.
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, ())
    return type(name)

scalars, fscalars, dscalars, iscalars, lscalars = _multi(
        scalar, fscalar, dscalar, iscalar, lscalar)

int_types = bscalar, wscalar, iscalar, lscalar
float_types = fscalar, dscalar
complex_types = cscalar, zscalar
int_scalar_types = int_types
float_scalar_types = float_types
complex_scalar_types = complex_types

cvector = TensorType('complex64', (False, ))
zvector = TensorType('complex128', (False, ))
fvector = TensorType('float32', (False, ))
dvector = TensorType('float64', (False, ))
bvector = TensorType('int8', (False,))
wvector = TensorType('int16', (False,))
ivector = TensorType('int32', (False, ))
lvector = TensorType('int64', (False, ))


def vector(name=None, dtype=None):
    """Return a symbolic vector variable.
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, ))
    return type(name)

vectors, fvectors, dvectors, ivectors, lvectors = _multi(
        vector, fvector, dvector, ivector, lvector)

int_vector_types = bvector, wvector, ivector, lvector
float_vector_types = fvector, dvector
complex_vector_types = cvector, zvector

cmatrix = TensorType('complex64', (False, False))
zmatrix = TensorType('complex128', (False, False))
fmatrix = TensorType('float32', (False, False))
dmatrix = TensorType('float64', (False, False))
bmatrix = TensorType('int8', (False, False))
wmatrix = TensorType('int16', (False, False))
imatrix = TensorType('int32', (False, False))
lmatrix = TensorType('int64', (False, False))


def matrix(name=None, dtype=None):
    """Return a symbolic matrix variable.
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, False))
    return type(name)

matrices, fmatrices, dmatrices, imatrices, lmatrices = _multi(
        matrix, fmatrix, dmatrix, imatrix, lmatrix)

int_matrix_types = bmatrix, wmatrix, imatrix, lmatrix
float_matrix_types = fmatrix, dmatrix
complex_matrix_types = cmatrix, zmatrix

crow = TensorType('complex64', (True, False))
zrow = TensorType('complex128', (True, False))
frow = TensorType('float32', (True, False))
drow = TensorType('float64', (True, False))
brow = TensorType('int8', (True, False))
wrow = TensorType('int16', (True, False))
irow = TensorType('int32', (True, False))
lrow = TensorType('int64', (True, False))


def row(name=None, dtype=None):
    """Return a symbolic row variable (ndim=2, broadcastable=[True,False]).
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (True, False))
    return type(name)
rows, frows, drows, irows, lrows = _multi(row, frow, drow, irow, lrow)

ccol = TensorType('complex64', (False, True))
zcol = TensorType('complex128', (False, True))
fcol = TensorType('float32', (False, True))
dcol = TensorType('float64', (False, True))
bcol = TensorType('int8', (False, True))
wcol = TensorType('int16', (False, True))
icol = TensorType('int32', (False, True))
lcol = TensorType('int64', (False, True))


def col(name=None, dtype=None):
    """Return a symbolic column variable (ndim=2, broadcastable=[False,True]).
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, True))
    return type(name)
cols, fcols, dcols, icols, lcols = _multi(col, fcol, dcol, icol, lcol)

ctensor3 = TensorType('complex64', ((False,) * 3))
ztensor3 = TensorType('complex128', ((False,) * 3))
ftensor3 = TensorType('float32', ((False,) * 3))
dtensor3 = TensorType('float64', ((False,) * 3))
btensor3 = TensorType('int8', ((False,) * 3))
wtensor3 = TensorType('int16', ((False,) * 3))
itensor3 = TensorType('int32', ((False,) * 3))
ltensor3 = TensorType('int64', ((False,) * 3))


def tensor3(name=None, dtype=None):
    """Return a symbolic 3-D variable.
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, False, False))
    return type(name)

tensor3s, ftensor3s, dtensor3s, itensor3s, ltensor3s = _multi(
        tensor3, ftensor3, dtensor3, itensor3, ltensor3)

ctensor4 = TensorType('complex64', ((False,) * 4))
ztensor4 = TensorType('complex128', ((False,) * 4))
ftensor4 = TensorType('float32', ((False,) * 4))
dtensor4 = TensorType('float64', ((False,) * 4))
btensor4 = TensorType('int8', ((False,) * 4))
wtensor4 = TensorType('int16', ((False,) * 4))
itensor4 = TensorType('int32', ((False,) * 4))
ltensor4 = TensorType('int64', ((False,) * 4))


def tensor4(name=None, dtype=None):
    """Return a symbolic 4-D variable.
    :param dtype: numeric type (None means to use theano.config.floatX)
    :param name: a name to attach to this variable
    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, False, False, False))
    return type(name)
tensor4s, ftensor4s, dtensor4s, itensor4s, ltensor4s = _multi(
        tensor4, ftensor4, dtensor4, itensor4, ltensor4)


Tensor = TensorType


# This bizarre push-import avoids a circular dependency.
elemwise.as_tensor_variable = as_tensor_variable
elemwise.TensorType = TensorType
elemwise.TensorVariable = TensorVariable
elemwise.TensorConstant = TensorConstant

#########################
# Utilities
#########################


def _scal_elemwise_with_nfunc(nfunc, nin, nout):
    """
    Replace a symbol definition with an elementwise version of the
    corresponding scalar Op.  If it is not None, the nfunc argument
    should be a string such that getattr(numpy, nfunc) implements
    a vectorized version of the elemwise operation. nin is the number
    of inputs expected by that function, and nout is the number of
    **destination** inputs it takes. That is, the function should
    take nin+nout inputs. nout == 0 means that the numpy function
    does not take a numpy array argument to put its result in.
    """
    def construct(symbol):
        symbolname = symbol.__name__
        inplace = symbolname.endswith('_inplace')
        if inplace:
            msg = "inplace"
        else:
            msg = "no_inplace"

        n = "Elemwise{%s,%s}" % (symbolname, msg)

        if inplace:
            scalar_op = getattr(scal, symbolname[:-len('_inplace')])
            inplace_scalar_op = scalar_op.__class__(scal.transfer_type(0))
            rval = elemwise.Elemwise(inplace_scalar_op, {0: 0}, name=n,
                                     nfunc_spec=(nfunc and (nfunc, nin, nout)))
        else:
            scalar_op = getattr(scal, symbolname)
            rval = elemwise.Elemwise(scalar_op, name=n,
                                     nfunc_spec=(nfunc and (nfunc, nin, nout)))

        if getattr(symbol, '__doc__', False):
            rval.__doc__ = symbol.__doc__ + '\n' + rval.__doc__

        # for the meaning of this see the ./epydoc script
        # it makes epydoc display rval as if it were a function, not an object
        rval.__epydoc_asRoutine = symbol
        rval.__module__ = 'tensor'

        pprint.assign(rval, printing.FunctionPrinter(symbolname))

        return rval
    return construct

_scal_elemwise = _scal_elemwise_with_nfunc(None, None, None)


#########################
# Casting Operations
#########################

class TensorFromScalar(Op):
    def make_node(self, s):
        assert isinstance(s.type, scal.Scalar)
        return Apply(self,
                     [s],
                     [tensor(dtype=s.type.dtype,
                             broadcastable=())])

    def perform(self, node, inp, out_):
        s, = inp
        out, = out_
        out[0] = numpy.asarray(s)

    def infer_shape(self, node, in_shapes):
        return [()]

    def grad(self, inp, grads):
        s, = inp
        dt, = grads
        if s.type.dtype in float_dtypes:
            assert dt.type.dtype in float_dtypes
            return [scalar_from_tensor(dt)]

        # If the input dtype is an integer, then so is the output dtype,
        # and the "zero" gradient can be represented in that int dtype.
        # Currently, theano.grad insists that the dtype of the returned
        # gradient has a float dtype, so we use floatX.
        if s.type.dtype in discrete_dtypes:
            return [s.zeros_like().astype(theano.config.floatX)]

        raise NotImplementedError("grad not implemented for complex dtypes")

    def __str__(self):
        return self.__class__.__name__

tensor_from_scalar = TensorFromScalar()


class ScalarFromTensor(Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, t):
        assert isinstance(t.type, TensorType)
        assert t.type.broadcastable == ()
        return Apply(self,
                     [t],
                     [scal.Scalar(dtype=t.type.dtype).make_variable()])

    def perform(self, node, inp, out_):
        s, = inp
        out, = out_
        out[0] = s.flatten()[0]

    def infer_shape(self, node, in_shapes):
        return [()]

    def grad(self, inp, grads):
        s, = inp
        dt, = grads
        return [tensor_from_scalar(dt)]

    def R_op(self, inputs, eval_points):
        if None in eval_points:
            return [None]
        return self.make_node(*eval_points).outputs

    def __str__(self):
        return self.__class__.__name__

    def c_code(self, node, name, inputs, outputs, sub):
        x, = inputs
        z, = outputs
        fail = sub['fail']
        return """
        %(z)s = ((dtype_%(x)s*)(PyArray_DATA(%(x)s)))[0];
        """ % locals()

    def c_code_cache_version(self):
        return (1,)

scalar_from_tensor = ScalarFromTensor()


# to be removed as we get the epydoc routine-documenting thing going
#-JB 20080924
def _conversion(real_value, name):
    __oplist_tag(real_value, 'casting')
    real_value.__module__ = 'tensor.basic'
    pprint.assign(real_value, printing.FunctionPrinter(name))
    return real_value


# These _conver_to_<type> functions have leading underscores to indicate that
# they should not be called directly.  They do not perform sanity checks about
# what types you are casting to what.  That logic is implemented by the
# `cast()` function below.

_convert_to_int8 = _conversion(
        elemwise.Elemwise(scal.convert_to_int8), 'int8')
"""Cast to 8-bit integer"""

_convert_to_int16 = _conversion(
        elemwise.Elemwise(scal.convert_to_int16), 'int16')
"""Cast to 16-bit integer"""

_convert_to_int32 = _conversion(
        elemwise.Elemwise(scal.convert_to_int32), 'int32')
"""Cast to 32-bit integer"""

_convert_to_int64 = _conversion(
        elemwise.Elemwise(scal.convert_to_int64), 'int64')
"""Cast to 64-bit integer"""

_convert_to_uint8 = _conversion(
        elemwise.Elemwise(scal.convert_to_uint8), 'uint8')
"""Cast to unsigned 8-bit integer"""

_convert_to_uint16 = _conversion(
        elemwise.Elemwise(scal.convert_to_uint16), 'uint16')
"""Cast to unsigned 16-bit integer"""

_convert_to_uint32 = _conversion(
        elemwise.Elemwise(scal.convert_to_uint32), 'uint32')
"""Cast to unsigned 32-bit integer"""

_convert_to_uint64 = _conversion(
        elemwise.Elemwise(scal.convert_to_uint64), 'uint64')
"""Cast to unsigned 64-bit integer"""

_convert_to_float32 = _conversion(
        elemwise.Elemwise(scal.convert_to_float32), 'float32')
"""Cast to single-precision floating point"""

_convert_to_float64 = _conversion(
        elemwise.Elemwise(scal.convert_to_float64), 'float64')
"""Cast to double-precision floating point"""

_convert_to_complex64 = _conversion(
        elemwise.Elemwise(scal.convert_to_complex64), 'complex64')
"""Cast to single-precision complex"""

_convert_to_complex128 = _conversion(
        elemwise.Elemwise(scal.convert_to_complex128), 'complex128')
"""Cast to double-precision complex"""

_cast_mapping = {
           'int8': _convert_to_int8,
           'int16': _convert_to_int16,
           'int32': _convert_to_int32,
           'int64': _convert_to_int64,
           'uint8': _convert_to_uint8,
           'uint16': _convert_to_uint16,
           'uint32': _convert_to_uint32,
           'uint64': _convert_to_uint64,
           'float32': _convert_to_float32,
           'float64': _convert_to_float64,
           'complex64': _convert_to_complex64,
           'complex128': _convert_to_complex128}


@constructor
def cast(x, dtype):
    """Symbolically cast `x` to a Tensor of type `dtype`."""
    if dtype == 'floatX':
        dtype = config.floatX

    _x = as_tensor_variable(x)
    if _x.type.dtype == dtype:
        return _x
    if _x.type.dtype.startswith('complex') and not dtype.startswith('complex'):
        raise TypeError((
            'Casting from complex to real is ambiguous: consider real(), '
            'imag(), angle() or abs()'))
    return _cast_mapping[dtype](x)

##########################
# Unary Operations
##########################


class Shape(Op):
    """
    L{Op} to return the shape of a matrix.

    @note: Non-differentiable.
    """
    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) == type(other)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x):
        # Must work for all type that have a shape attribute.
        # This will fail at execution time.
        x = as_tensor_variable(x)
        # Each type variable should implement their .shape attribute
        # and have the fct infer_shape() implemented in the op that convert
        # the type to TensorVariable to have the optimization working
        # correctly.
        return Apply(self, [x], [lvector()])

    def perform(self, node, inp, out_):
        x, = inp
        out, = out_
        out[0] = theano._asarray(x.shape, dtype='int64')

    def infer_shape(self, node, in_shapes):
        return [[len(in_shapes[0])]]

    def connection_pattern(self, node):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [[False]]

    def grad(self, inp, grads):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [DisconnectedType()()]

    def R_op(self, inputs, eval_points):
        return [None]

    def c_code(self, node, nodename, inp, out, sub):
        x, = inp
        z, = out
        if isinstance(node.inputs[0].type, TensorType):
            return """
            npy_intp shape[] = {PyArray_NDIM(%(x)s)};
            if(%(z)s == NULL || (PyArray_DIMS(%(z)s)[0] != shape[0]))
            {
                Py_XDECREF(%(z)s);
                %(z)s = (PyArrayObject*) PyArray_SimpleNew(1, shape, NPY_INT64);
            }
            for(int i=0;i<shape[0];i++)
            {
                ((npy_int64*)PyArray_GETPTR1(%(z)s, i))[0] = PyArray_DIMS(%(x)s)[i];
            }
            """ % locals()
        else:
            #TODO: if your type is not listed here, make a damn registry of
            #      shape_i ops for various types of variables.
            #      Do not continue this madness.
            return super(Shape, self).c_code(node, nodename, (x,), (out,), sub)

    def c_code_cache_version(self):
        return (1,)

@constructor
def old_shape(a):
    """
    Return the shape tuple of a TensorType Variable.

    It may be either symbolic or nonsymbolic.

    If the shape of the expression is not known at graph-construction time,
    then a symbolic lvector will be returned, corresponding to the actual
    shape at graph-execution time.
    """
    va = as_tensor_variable(a)
    # print 'HERE', va, va.type
    if None in va.type.shape:
        # Some shape components are unknown at this time
        return _shape(va)
    else:
        # all shape components are known at compile time, so we return
        # a tuple directly.  This tuple is like the numpy.ndarray.shape tuple.
        return va.type.shape

shape = Shape()
_shape = shape  # was used in the past, now use shape directly.
pprint.assign(_shape, printing.MemberPrinter('shape'))


class SpecifyShape(Op):
    """
    L{Op} that puts into the graph the user-provided shape.

    In the case where this op stays in the final graph, we assert the shape.
    For this the output of this op must be used in the graph. This is not
    the case most of the time if we only take the shape of the output.
    Maybe there are other optimizations that will mess with this.

    @note:     Maybe in the future we will never do the assert!
    @note:     We currently don't support specifying partial shape information.

    @todo:     test this op with sparse and cuda ndarray.
               Do C code for them too.
    """
    view_map = {0: [0]}

    def __hash__(self):
        return hash(type(self))

    def __eq__(self, other):
        return type(self) == type(other)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x, shape):
        if not isinstance(x, Variable):
            x = as_tensor_variable(x)
        shape = as_tensor_variable(shape)
        assert shape.ndim == 1
        assert "int" in shape.dtype
        if isinstance(shape, TensorConstant):
            assert shape.data.size == x.ndim
        return Apply(self, [x, shape], [x.type()])

    def perform(self, node, inp, out_):
        x, shape = inp
        out, = out_
        assert x.ndim == shape.size
        assert numpy.all(x.shape == shape), ("got shape", x.shape,
                                           "expected", shape)
        out[0] = x

    def infer_shape(self, node, shapes):
        xshape, sshape = shapes
        new_shape = []
        for dim in xrange(node.inputs[0].ndim):
            try:
                s = get_scalar_constant_value(node.inputs[1][dim])
                s = as_tensor_variable(s)
                new_shape.append(s)
            except NotScalarConstantError:
                new_shape.append(node.inputs[1][dim])

        assert len(new_shape) == len(xshape)
        return [new_shape]

    def connection_pattern(self, node):
        return [[True], [False]]

    def grad(self, inp, grads):
        x, s = inp
        gz, = grads
        # Should I set an SpecifyShape on gz? I think so
        # But I don't do it now as we need to make an optimization
        # to remove that op from the graph to don't block other optimization
        # Should I do an optimizer that will remove the SpecifyShape?
        # I think Yes
        return [gz, DisconnectedType()()]
        return [specify_shape(gz, s), DisconnectedType()()]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            # It means that the this op sits on top of a non-differentiable
            # path
            return [None]
        return self.make_node(eval_points[0], *inputs[1:]).outputs

    def c_code(self, node, nodename, inp, out, sub):
        if not isinstance(node.inputs[0], TensorVariable):
            # The C code below supports only Tensor.  super.c_code
            # will raise an exception to tell that there is no C code
            # for the other cases.
            return super(SpecifyShape, self).c_code(node, nodename,
                                                    inp, out, sub)
        iname, shape = inp
        oname, = out
        fail = sub['fail']

        return """
        if (PyArray_NDIM(%(iname)s) != PyArray_DIMS(%(shape)s)[0]) {
            PyErr_Format(PyExc_AssertionError,
                         "SpecifyShape: vector of shape has %%d elements,"
                         " but the input has %%d dimensions.",
                         PyArray_NDIM(%(iname)s),
                         PyArray_DIMS(%(shape)s)[0]);
            %(fail)s;
        }
        for(int i = 0; i < PyArray_NDIM(%(iname)s); i++){
            dtype_%(shape)s shp = ((dtype_%(shape)s*)PyArray_GETPTR1(%(shape)s,
                                                                     i))[0];
            if (PyArray_DIMS(%(iname)s)[i] != shp) {
                PyErr_Format(PyExc_AssertionError,
                             "SpecifyShape: dim %%d of input has shape %%d,"
                             " expected %%d.",
                             i, PyArray_DIMS(%(iname)s)[i],
                             shp);
                %(fail)s;
            }
        }
        Py_XDECREF(%(oname)s);
        %(oname)s = %(iname)s;
        Py_XINCREF(%(oname)s);
        """ % locals()

    def c_code_cache_version(self):
        return (1,)


specify_shape = SpecifyShape()


class MaxAndArgmax(Op):
    """Calculate the max and argmax over a given axis or over all axes.
    """
    nin = 2  # tensor, axis
    nout = 2  # max val, max idx
    E_axis = 'invalid axis'

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, axis=None):
        x = _as_tensor_variable(x)

        if isinstance(axis, (int, numpy.integer)):
            axis = [axis]
        elif isinstance(axis, numpy.ndarray) and axis.ndim == 0:
            axis = [int(axis)]
        elif isinstance(axis, (tuple, list)):
            axis = [int(a) for a in axis]
            if len(axis) != 1:
                axis = list(axis)
                for idx in range(len(axis)):
                    if axis[idx] < 0:
                        axis[idx] += x.type.ndim
                axis.sort()
                if axis == range(-x.type.ndim, 0, 1):
                    axis = range(x.type.ndim)
                assert axis == range(x.type.ndim), (
                    "MaxAndArgmax does not support multiple"
                    " axes. the max fct supports it.")
        elif isinstance(axis, Variable):
            if not isinstance(axis, TensorConstant):
                raise TypeError("MaxAndArgmax needs a constant axis")
            axis = axis.data
            if axis.ndim == 0:
                axis = [axis]
        # we make the axis all positive to make the infer_shape work
        # with negative axis
        if x.type.ndim > 0 and axis is not None:
            for id, a in enumerate(axis):
                if not isinstance(a, TensorVariable) and a < 0:
                    if -a > x.type.ndim:
                        raise ValueError('axis out of range')
                    axis[id] = x.type.ndim + a
        if axis is None:
            axis = _as_tensor_variable(range(x.type.ndim))
        else:
            axis = _as_tensor_variable(axis)

        # Verify that the axis is valid.
        all_axes = set()
        for ax in axis.data:
            if ax < 0 or ax >= x.type.ndim:
                raise ValueError(
                        'Invalid axis: %s (the number of dimensions of the '
                        'input is: %s)' % (axis, x.type.ndim))
            all_axes.add(ax.item())
        assert axis.ndim == 1
        inputs = [x, axis]
        # We keep the original broadcastable flags for dimensions on which
        # we do not perform the max / argmax.
        broadcastable = [b for i, b in enumerate(x.type.broadcastable)
                         if i not in all_axes]
        outputs = [tensor(x.type.dtype, broadcastable, name='max'),
                   tensor('int64', broadcastable, name='argmax')]
        return Apply(self, inputs, outputs)

    def perform(self, node, inp, outs):
        x, axis = inp
        max, max_idx = outs
        if python_all(axis == range(x.ndim)):
            axis = None
        max[0] = theano._asarray(numpy.max(x, axis),
                                 dtype=node.outputs[0].dtype)
        max_idx[0] = theano._asarray(numpy.argmax(x, axis), dtype='int64')

    def infer_shape(self, node, shapes):
        ishape, axis_shape = shapes
        axis = node.inputs[1]
        if python_all(axis.data == range(node.inputs[0].ndim)):
            return [(), ()]
        rval = tuple([ishape[i] for (i, b) in enumerate(
                    node.inputs[0].type.broadcastable) if i != axis.data])
        return [rval, rval]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None, None]
        if not isinstance(inputs[1], theano.Constant):
            raise ValueError(('R_op supported for arg_max only for '
                              'constant axis!'))
        if inputs[1].data > 1:
            raise ValueError(('R_op supported for arg_max only when '
                              ' axis is 0 or 1'))
        if inputs[0].ndim != 2:
            raise ValueError(('R_op supported for arg_max only when '
                              ' input is a matrix'))
        max_vals, max_pos = self.make_node(*inputs).outputs
        if inputs[1].data == 0:
            return [eval_points[0][max_pos,
                                   arange(eval_points[0].shape[1])], None]
        else:
            return [eval_points[0][arange(eval_points[0].shape[0]),
                                   max_pos], None]

    def grad(self, inp, grads):
        # The strict sense mathematical gradient of the maximum function is
        # not calculated here for it is not defined at every point where some
        # coordinates are identical. However, since the latter set has null
        # Lebesgue measure, the result may be interpreted as weak gradient.

        # @note: This function should work correctly for L{vector}s.
#        (x, y), (gz, gw)
#        gz*dz/dx + gw*dw/dx, gz*dz/dy + gw*dw/dy
#        gMax * dMax/dx + gArgMax * dArgMax/dx,
#                           gMax * dMax/daxis + gArgMax * dArgMax/daxis
#       g_max has one less dimension than x, so you need to complete
#        g_max to x's shape when axis=0 the broadcasting mechanism
#        does it automatically
        x, axis = inp
        g_max, g_max_idx = grads

        g_max_disconnected = isinstance(g_max.type, DisconnectedType)
        g_max_idx_disconnected = isinstance(g_max_idx.type, DisconnectedType)

        # if the op is totally disconnected, so are its inputs
        if g_max_disconnected and g_max_idx_disconnected:
            return [DisconnectedType()(), DisconnectedType()()]

        axis_grad = grad_undefined(self, 1, axis,
                "argmax is not defined for non-integer axes so"
                " argmax(x, axis+eps) is undefined")

        # if the max is disconnected but the argmax is not,
        # the gradient on its inputs is zero
        if g_max_disconnected:
            return [x.zeros_like(), axis_grad]
        xmax = max(x, axis)

        # Raise the g_max and xmax to the same number of dim as the input.
        pattern = []
        out_dim = 0
        if python_all(axis.data == range(x.ndim)):
            # We are taking the max/argmax over all dimensions.
            axis = None
        for i in range(x.ndim):
            if axis is None or i == axis.data:
                pattern.append('x')
            else:
                pattern.append(out_dim)
                out_dim += 1
        g_max_pad = DimShuffle(g_max.broadcastable, pattern)(g_max)
        xmax_pad = DimShuffle(xmax.broadcastable, pattern)(xmax)

        # Set the grad to the correct position.
        g_x = eq(xmax_pad, x) * g_max_pad
        return g_x, axis_grad

    def __str__(self):
        return self.__class__.__name__


_max_and_argmax = MaxAndArgmax()


def makeKeepDims(x, y, axis):
    """
    Reintroduces in y with length one the axes of x which have been left out
    in a prior reduction of x. With this option, the resulting tensor will
    broadcast correctly against the original tensor x.
    """
    x = as_tensor_variable(x)
    y = as_tensor_variable(y)

    if axis is None:
        axis = range(x.type.ndim)
    elif isinstance(axis, (int, numpy.integer)):
        axis = [axis]
    elif isinstance(axis, numpy.ndarray) and axis.ndim == 0:
        axis = [int(axis)]
    else:
        axis = [int(a) for a in axis]
    newaxis = []
    for a in axis:
        if not isinstance(a, int):
            raise ValueError("keepdims option can be used only with constant axis")
        if a < 0:
            a += x.type.ndim
        newaxis.append(a)
    i = 0
    new_dims = []
    for j, _ in enumerate(x.type.broadcastable):
        if j in newaxis:
            new_dims.append('x')
        else:
            new_dims.append(i)
            i += 1
    return DimShuffle(y.type.broadcastable, new_dims)(y)


@constructor
def max_and_argmax(a, axis=None, keepdims=False):
    """
    Returns maximum elements and their indices obtained by iterating over
    given axis

    When axis is None (the default value), the max is performed
    over the flattened tensor.

    keepdims: If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.
    """

    out, argout = _max_and_argmax(a, axis)

    if keepdims:
        out = makeKeepDims(a, out, axis)
        argout = makeKeepDims(a, argout, axis)
    return [out, argout]


@constructor
def max(x, axis=None, keepdims=False):
    """
    Returns maximum elements obtained by iterating over given axis

    When axis is None (the default value), the max is performed
    over the flattened tensor.

    keepdims: If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    :note: we return an error as numpy when we reduce a dim with a shape of 0
    """

    # We have a choice of implementing this call with the
    # CAReduce op or the MaxAndArgmax op.

    # MaxAndArgmax supports grad and Rop, so we prefer to use that.
    # CAReduce is faster, but optimizations will replace MaxAndArgmax[0]
    # with CAReduce at compile time, so at this stage the important
    # thing is supporting all user interface features, not speed.
    # Some cases can be implemented only with CAReduce.

    # We thus prefer to use MaxAndArgmax, if possible. It does not
    # support all axis arguments, so we may need to fall back to CAReduce.

    try:
        out = max_and_argmax(x, axis)[0]
    except Exception:
        out = CAReduce(scal.maximum, axis)(x)

    if keepdims:
        out = makeKeepDims(x, out, axis)
    return out


@constructor
def argmax(x, axis=None, keepdims=False):
    """
    Returns indices of maximum elements obtained by iterating over given axis

    When axis is None (the default value), the argmax is performed
    over the flattened tensor.

    keepdims: If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.
    """
    # In python (using MaxAndArgmax.perform()) this leads to a wasteful
    # implementation that goes through the data twice instead of once
    # but when Argmax.c_impl() is in place, it should be fine.

    argout = max_and_argmax(x, axis)[1]

    if keepdims:
        argout = makeKeepDims(x, argout, axis)
    return argout


@constructor
def min(x, axis=None, keepdims=False):
    """
    Returns minimum elements obtained by iterating over given axis

    When axis is None (the default value), the min is performed
    over the flattened tensor.

    keepdims: If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.
    """

    str_x_type = str(x.dtype)
    if str_x_type.startswith('float') or str_x_type in int_dtypes:
        return -max(-x, axis=axis, keepdims=keepdims)
    else:
        # Be careful about unsigned integers, complex
        raise NotImplementedError()


@constructor
def argmin(x, axis=None, keepdims=False):
    """
    Returns indices of minimum elements obtained by iterating over given axis

    When axis is None (the default value), the argmin is performed
    over the flattened tensor.

    keepdims: If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.
    """

    str_x_type = str(x.dtype)
    if str_x_type.startswith('float') or str_x_type in int_dtypes:
        return argmax(-x, axis=axis, keepdims=keepdims)
    else:
        # Be careful about unsigned integers, complex
        raise NotImplementedError()


@constructor
def smallest(*args):
    """
    Return the [elementwise] smallest of a variable number of arguments.

    Like python's min.
    """
    if len(args) == 2:
        a, b = args
        return switch(a < b, a, b)
    else:
        return min(stack(*args), axis=0)


@constructor
def largest(*args):
    """
    Return the [elementwise] largest of a variable number of arguments.

    Like python's max.
    """
    if len(args) == 2:
        a, b = args
        return switch(a > b, a, b)
    else:
        return max(stack(*args), axis=0)


##########################
# Comparison
##########################

@_scal_elemwise_with_nfunc('less', 2, 1)
def lt(a, b):
    """a < b"""


@_scal_elemwise_with_nfunc('greater', 2, 1)
def gt(a, b):
    """a > b"""


@_scal_elemwise_with_nfunc('less_equal', 2, 1)
def le(a, b):
    """a <= b"""


@_scal_elemwise_with_nfunc('greater_equal', 2, 1)
def ge(a, b):
    """a >= b"""


@_scal_elemwise_with_nfunc('equal', 2, 1)
def eq(a, b):
    """a == b"""


@_scal_elemwise_with_nfunc('not_equal', 2, 1)
def neq(a, b):
    """a != b"""


@_scal_elemwise_with_nfunc('isnan', 1, 1)
def isnan(a):
    """isnan(a)"""


@_scal_elemwise_with_nfunc('isinf', 1, 1)
def isinf(a):
    """isinf(a)"""


##########################
# Condition
##########################

@_scal_elemwise_with_nfunc('where', 3, 1)
def switch(cond, ift, iff):
    """if cond then ift else iff"""

where = switch
##########################
# Bit-wise
##########################


@_scal_elemwise_with_nfunc('bitwise_and', 2, 1)
def and_(a, b):
    """bitwise a & b"""
bitwise_and = and_  # numpy name for it


@_scal_elemwise_with_nfunc('bitwise_or', 2, 1)
def or_(a, b):
    """bitwise a | b"""
bitwise_or = or_  # numpy name for it


@_scal_elemwise_with_nfunc('bitwise_xor', 2, 1)
def xor(a, b):
    """bitwise a ^ b"""
bitwise_xor = xor  # numpy name for it


@_scal_elemwise_with_nfunc('invert', 1, 1)
def invert(a):
    """bitwise ~a"""
bitwise_not = invert  # numpy alias for it


##########################
# Math
##########################

@_scal_elemwise_with_nfunc('abs', 1, 1)
def abs_(a):
    """|`a`|

    TensorVariable overloads the `TensorVariable.__abs__` operator so that
    this function is called when you type abs(a).

    """

pprint.assign(abs_, printing.PatternPrinter(('|%(0)s|', -1000)))


@_scal_elemwise_with_nfunc('exp', 1, 1)
def exp(a):
    """e^`a`"""


@_scal_elemwise_with_nfunc('exp2', 1, 1)
def exp2(a):
    """2^`a`"""


@_scal_elemwise_with_nfunc('expm1', 1, 1)
def expm1(a):
    """e^`a` - 1"""


@_scal_elemwise_with_nfunc('negative', 1, 1)
def neg(a):
    """-a"""


# numpy.reciprocal does integer division on integer inputs
# (which is not very interesting)
@_scal_elemwise
def inv(a):
    """1.0/a"""


@_scal_elemwise_with_nfunc('log', 1, 1)
def log(a):
    """base e logarithm of a"""


@_scal_elemwise_with_nfunc('log2', 1, 1)
def log2(a):
    """base 2 logarithm of a"""


@_scal_elemwise_with_nfunc('log10', 1, 1)
def log10(a):
    """base 10 logarithm of a"""


@_scal_elemwise_with_nfunc('log1p', 1, 1)
def log1p(a):
    """log(1+a)"""


@_scal_elemwise_with_nfunc('sign', 1, 1)
def sgn(a):
    """sign of a"""


@_scal_elemwise_with_nfunc('ceil', 1, 1)
def ceil(a):
    """ceiling of a"""


@_scal_elemwise_with_nfunc('floor', 1, 1)
def floor(a):
    """floor of a"""


@_scal_elemwise_with_nfunc('trunc', 1, 1)
def trunc(a):
    """trunc of a"""


@constructor
def iround(a, mode="half_away_from_zero"):
    """cast(round(a,mode),'int64')"""
    return cast(round(a, mode), 'int64')


@constructor
def round(a, mode="half_away_from_zero"):
    """round_mode(a) with mode in [half_away_from_zero, half_to_even]"""
    if mode == "half_away_from_zero":
        return round_half_away_from_zero(a)
    elif mode == "half_to_even":
        return round_half_to_even(a)
    else:
        raise Exception("round mode %s is not implemented." % mode)


@_scal_elemwise_with_nfunc('around', 1, -1)
def round_half_to_even(a):
    """round_half_to_even(a)"""


@_scal_elemwise
def round_half_away_from_zero(a):
    """round_half_away_from_zero(a)"""


@_scal_elemwise_with_nfunc('square', 1, 1)
def sqr(a):
    """square of a"""


# alias to sqr, included to maintain similarity with numpy interface
square = sqr


@_scal_elemwise_with_nfunc('sqrt', 1, 1)
def sqrt(a):
    """square root of a"""


@_scal_elemwise_with_nfunc('deg2rad', 1, 1)
def deg2rad(a):
    """convert degree a to radian"""


@_scal_elemwise_with_nfunc('rad2deg', 1, 1)
def rad2deg(a):
    """convert radian a to degree"""


@_scal_elemwise_with_nfunc('cos', 1, 1)
def cos(a):
    """cosine of a"""


@_scal_elemwise_with_nfunc('arccos', 1, 1)
def arccos(a):
    """arccosine of a"""


@_scal_elemwise_with_nfunc('sin', 1, 1)
def sin(a):
    """sine of a"""


@_scal_elemwise_with_nfunc('arcsin', 1, 1)
def arcsin(a):
    """arcsine of a"""


@_scal_elemwise_with_nfunc('tan', 1, 1)
def tan(a):
    """tangent of a"""


@_scal_elemwise_with_nfunc('arctan', 1, 1)
def arctan(a):
    """arctangent of a"""


@_scal_elemwise_with_nfunc('arctan2', 1, 1)
def arctan2(a, b):
    """arctangent of a / b"""


@_scal_elemwise_with_nfunc('cosh', 1, 1)
def cosh(a):
    """hyperbolic cosine of a"""


@_scal_elemwise_with_nfunc('arccosh', 1, 1)
def arccosh(a):
    """hyperbolic arc cosine of a"""


@_scal_elemwise_with_nfunc('sinh', 1, 1)
def sinh(a):
    """hyperbolic sine of a"""


@_scal_elemwise_with_nfunc('arcsinh', 1, 1)
def arcsinh(a):
    """hyperbolic arc sine of a"""


@_scal_elemwise_with_nfunc('tanh', 1, 1)
def tanh(a):
    """hyperbolic tangent of a"""


@_scal_elemwise_with_nfunc('arctanh', 1, 1)
def arctanh(a):
    """hyperbolic arc tangent of a"""


@_scal_elemwise
def erf(a):
    """error function"""


@_scal_elemwise
def erfc(a):
    """complementary error function"""


@_scal_elemwise
def erfinv(a):
    """inverse error function"""


@_scal_elemwise
def erfcinv(a):
    """inverse complementary error function"""


@_scal_elemwise
def gamma(a):
    """gamma function"""


@_scal_elemwise
def gammaln(a):
    """log gamma function"""


@_scal_elemwise
def psi(a):
    """derivative of log gamma function"""

@_scal_elemwise
def chi2sf(x, k):
    """chi squared survival function"""



@_scal_elemwise_with_nfunc('real', 1, -1)
def real(z):
    """Return real component of complex-valued tensor `z`"""
_tensor_py_operators.real = property(real)


@_scal_elemwise_with_nfunc('imag', 1, -1)
def imag(z):
    """Return imaginary component of complex-valued tensor `z`"""
_tensor_py_operators.imag = property(imag)


@_scal_elemwise_with_nfunc('angle', 1, -1)
def angle(z):
    """Return polar-coordinate angle of complex-valued tensor `z`"""


@_scal_elemwise  # numpy.complex cannot build tensors
def complex(real, imag):
    """Return complex-valued tensor with `real` and `imag` components"""


@_scal_elemwise_with_nfunc('conj', 1, -1)
def conj(z):
    """Return the complex conjugate of `z`."""


@_scal_elemwise
def complex_from_polar(abs, angle):
    """Return complex-valued tensor from polar coordinate specification."""

##########################
# Misc
##########################


# fill, _fill_inplace = _elemwise(scal.second, 'fill',
    #"""fill WRITEME (elemwise)""")
@_scal_elemwise
def second(a, b):
    """Create a matrix by filling the shape of a with b"""

fill = second
pprint.assign(fill, printing.FunctionPrinter('fill'))


@constructor
def ones_like(model, dtype=None):
    """equivalent of numpy.ones_like"""
    if dtype is None:
        dtype = model.type.dtype
    ret = fill(model, constant(1.0, dtype=dtype))
    return ret


@constructor
def zeros_like(model, dtype=None):
    """equivalent of numpy.zeros_like"""
    if dtype is None:
        dtype = model.type.dtype
    return fill(model, constant(0.0, dtype=dtype))


def zeros(shape, dtype=None):
    """
    Create a Tensor filled with zeros, closer to Numpy's syntax than ``alloc``.
    """
    if not isinstance(shape, (list, tuple, TensorVariable)):
        shape = [shape]
    if dtype is None:
        dtype = config.floatX
    return alloc(numpy.array(0, dtype=dtype), *shape)


def ones(shape, dtype=None):
    """
    Create a Tensor filled with ones, closer to Numpy's syntax than ``alloc``.
    """
    if not isinstance(shape, (list, tuple, TensorVariable)):
        shape = [shape]
    if dtype is None:
        dtype = config.floatX
    return alloc(numpy.array(1, dtype=dtype), *shape)


class Nonzero(gof.Op):
    """
    Return the indices of the elements that are non-zero.

    Returns a matrix of shape (ndim, number of nonzero elements) such that
    element (i,j) is the index in the ith dimension of the jth non-zero
    element.

    Note this is different than NumPy, which returns a tuple of arrays, one for
    each dimension of the input array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    result : matrix
        matrix containing the indices of the non-zero elements of a.

    See Also
    --------
    nonzero_values : Return the non-zero elements of the input array
    flatnonzero : Return the indices of the non-zero elements of the
        flattened input array.

    """
    def make_node(self, a):
        a = as_tensor_variable(a)
        if a.ndim == 0:
            raise ValueError('Nonzero only supports non-scalar arrays.')
        output = [TensorType(dtype='int64', broadcastable=(False, False))()]
        return gof.Apply(self, [a], output)

    def perform(self, node, inp, out_):
        a = inp[0]
        out, = out_

        result_tuple = numpy.nonzero(a)
        if len(result_tuple[0]) > 0:
            result = numpy.vstack(result_tuple)
        else:
            result = numpy.zeros((len(result_tuple), 0))

        out[0] = result.astype('int64')

    def grad(self, inp, grads):
        return [grad_undefined(self, 0, inp[0])]


_nonzero = Nonzero()


def nonzero(a, return_matrix=False):
    """
    Returns one of the following:

        If return_matrix is False (default, same as NumPy):
            A tuple of vector arrays such that the ith element of the jth array
            is the index of the ith non-zero element of the input array in the
            jth dimension.

        If return_matrix is True (same as Theano Op):
            Returns a matrix of shape (ndim, number of nonzero elements) such
            that element (i,j) is the index in the ith dimension of the jth
            non-zero element.

    Parameters
    ----------
    a : array_like
        Input array.

    return_matrix : bool
        If True, returns a symbolic matrix. If False, returns a tuple of
        arrays. Defaults to False.

    Returns
    -------
    result : tuple of vectors or matrix

    See Also
    --------
    nonzero_values : Return the non-zero elements of the input array
    flatnonzero : Return the indices of the non-zero elements of the
        flattened input array.

    """
    matrix_result = _nonzero(a)
    if return_matrix:
        return matrix_result
    else:
        if a.ndim > 0:
            tuple_result = tuple([matrix_result[i] for i in xrange(a.ndim)])
        else:
            tuple_result = tuple([matrix_result[0]])
        return tuple_result


def flatnonzero(a):
    """
    Return a vector of indices that are non-zero in the flattened version of a.

    This is equivalent to nonzero(a.flatten(), return_matrix=True)[0]

    Parameters
    ----------
    a : tensor
        Input tensor

    Returns
    -------
    res : vector
        Output vector, containing the indices of the elements of `a.flatten()`
        that are non-zero.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    nonzero_values : Return the non-zero elements of the input array
    """
    if a.ndim == 0:
        raise ValueError('Nonzero only supports non-scalar arrays.')
    return nonzero(a.flatten(), return_matrix=True)[0]


def nonzero_values(a):
    """
    Return a vector of non-zero elements contained in the input array.

    The following behavior works to extract non-zero elements from an array
    in NumPy but is *NOT* supported by Theano:

        a[numpy.nonzero(a)]

    Instead, the nonzero_values function or method should be used:

        tensor.nonzero_values(a)
        a.nonzero_values()

    This is equivalent to the following:

        a.flatten()[tensor.flatnonzero(a)]

    Parameters
    ----------
    a : tensor
        Input tensor

    Returns
    -------
    res : vector
        Output vector, containing the non-zero elements of a.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    flatnonzero : Return the indices of the non-zero elements of the
        flattened input array.
    """
    return a.flatten()[flatnonzero(a)]


class Tri(gof.Op):
    def __init__(self, dtype=None):
        if dtype is None:
            dtype = config.floatX
        self.dtype = dtype

    def make_node(self, N, M, k):
        N = as_tensor_variable(N)
        M = as_tensor_variable(M)
        k = as_tensor_variable(k)
        return gof.Apply(self, [N, M, k],
                [TensorType(dtype=self.dtype, broadcastable=(False, False))()])

    def perform(self, node, inp, out_):
        N, M, k = inp
        out, = out_
        out[0] = numpy.tri(N, M, k, dtype=self.dtype)

    def infer_shape(self, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i]) for i in xrange(3)]

    def __eq__(self, other):
        return type(self) == type(other) and self.dtype == other.dtype

    def __hash__(self):
        return hash(self.dtype) ^ hash(type(self))


def tri(N, M=None, k=0, dtype=None):
    """
    An array with ones at and below the given diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.
        By default, `M` is taken equal to `N`.
    k : int, optional
        The sub-diagonal at and below which the array is filled.
        `k` = 0 is the main diagonal, while `k` < 0 is below it,
        and `k` > 0 is above.  The default is 0.
    dtype : dtype, optional
        Data type of the returned array.  The default is float.

    Returns
    -------
    tri : Array of shape (N, M)
        Array with its lower triangle filled with ones and zero elsewhere;
        in other words ``T[i,j] == 1`` for ``i <= j + k``, 0 otherwise.
    """
    if dtype is None:
        dtype = config.floatX
    if M is None:
        M = N
    op = Tri(dtype)
    return op(N, M, k)


def tril(m, k=0):
    """
    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.

    Parameters
    ----------
    m : array_like, shape (M, N)
        Input array.
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    tril : array, shape (M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    triu : same thing, only for the upper triangle
    """
    return m * tri(m.shape[0], m.shape[1], k=k, dtype=m.dtype)


def triu(m, k=0):
    """
    Upper triangle of an array.

    Return a copy of a matrix with the elements below the `k`-th diagonal
    zeroed.

    Please refer to the documentation for `tril` for further details.

    See Also
    --------
    tril : lower triangle of an array
    """
    return m * (1 - tri(m.shape[0], m.shape[1], k=k - 1, dtype=m.dtype))


class Eye(gof.Op):
    def __init__(self, dtype=None):
        if dtype is None:
            dtype = config.floatX
        self.dtype = dtype

    def make_node(self, n, m, k):
        n = as_tensor_variable(n)
        m = as_tensor_variable(m)
        k = as_tensor_variable(k)
        assert n.ndim == 0
        assert m.ndim == 0
        assert k.ndim == 0
        return gof.Apply(self, [n, m, k],
                [TensorType(dtype=self.dtype, broadcastable=(False, False))()])

    def perform(self, node, inp, out_):
        n, m, k = inp
        out, = out_
        out[0] = numpy.eye(n, m, k, dtype=self.dtype)

    def infer_shape(self, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i]) for i in xrange(3)]

    def __eq__(self, other):
        return type(self) == type(other) and self.dtype == other.dtype

    def __hash__(self):
        return hash(self.dtype) ^ hash(type(self))


def eye(n, m=None, k=0, dtype=None):
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : int
      Number of rows in the output.
    m : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 (the default) refers to the main diagonal,
      a positive value refers to an upper diagonal, and a negative value
      to a lower diagonal.
    dtype : data-type, optional
      Data-type of the returned array.

    Returns
    -------
    I : ndarray of shape (N,M)
      An array where all elements are equal to zero, except for the `k`-th
      diagonal, whose values are equal to one.
    """
    if dtype is None:
        dtype = config.floatX
    if m is None:
        m = n
    localop = Eye(dtype)
    return localop(n, m, k)


def identity_like(x):
    return eye(x.shape[0], x.shape[1], k=0, dtype=x.dtype)


class Alloc(gof.Op):
    """Create a Tensor from an initial value and a desired shape

    alloc(value, shape0, shape1, ..., shapeN)

    Returns an N-dimensional tensor initialized by `value` using something
    equivalent to
    >>> z = numpy.zeros(shape, value.dtype)
    >>> z += value

    The result has N dimensions, has the dtype of `value` and is obtained by
    broadcasting value over the output ndarray.

    This Op is used to replace fill() during optimizations because after shapes
    are lifted, the first argument to fill can often be pruned from the graph.
    """
    def __init__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, value, *shape):
        v = as_tensor_variable(value)
        sh = [as_tensor_variable(s) for s in shape]
        bcast = []
        if v.ndim > len(sh):
            raise TypeError("The Alloc value to use has more dimensions"
                            " than the specified dimensions",
                            v.ndim, len(sh))
        for i, s in enumerate(sh):
            if s.type.dtype[:3] not in ('int', 'uin'):
                if config.exception_verbosity == 'high':
                    s_as_str = '\n' + min_informative_str(s)
                else:
                    s_as_str = str(s)
                raise TypeError('Shape arguments to Alloc must be integers, '
                                'but argument %s is not for apply node: %s' %
                                (i, s_as_str))
            # if s is constant 1, then we're broadcastable in that dim
            try:
                const_shp = get_scalar_constant_value(s)
            except NotScalarConstantError:
                const_shp = None
            bcast.append(numpy.all(1 == const_shp))
        otype = TensorType(dtype=v.dtype, broadcastable=bcast)
        return gof.Apply(self, ([v] + sh), [otype()])

    def perform(self, node, inputs, out_):
        out, = out_
        v = inputs[0]
        sh = tuple([int(i) for i in inputs[1:]])
        if out[0] is None or out[0].shape != sh:
            if v.size == 1 and v.item() == 0:
                out[0] = numpy.zeros(sh, dtype=v.dtype)
            else:
                out[0] = numpy.empty(sh, dtype=v.dtype)
                out[0][...] = v  # broadcast v to fill us up
        else:
            # reuse the allocated memory.
            out[0][...] = v  # broadcast v to fill us up

    def c_code(self, node, name, inp, out, sub):
        vv = inp[0]
        ndim = len(inp[1:])
        zz, = out
        fail = sub['fail']

        code = """
            npy_intp shape[%(ndim)s];
            """ % dict(ndim=ndim)

        # Initialize shape
        for i, shp_i in enumerate(inp[1:]):
            code += """
                shape[%(i)s] = ((dtype_%(shp_i)s*) PyArray_DATA(%(shp_i)s))[0];
                """ % dict(i=i, shp_i=shp_i)

        code += """
            int need_new_out = (NULL == %(zz)s);
            for (int i = 0; i < %(ndim)s; i++)
                need_new_out = (need_new_out
                                || (PyArray_DIMS(%(zz)s)[i] != shape[i]));

            if (need_new_out)
            {
                Py_XDECREF(%(zz)s);
                %(zz)s = (PyArrayObject*) PyArray_SimpleNew(%(ndim)s,
                    shape, type_num_%(vv)s);
                if (!%(zz)s)
                {
                    PyErr_SetString(PyExc_MemoryError, "alloc failed");
                    %(fail)s
                }
            }

            // This function takes care of broadcasting
            PyArray_CopyInto(%(zz)s, %(vv)s);
            """ % dict(vv=vv, ndim=ndim, zz=zz, fail=fail)

        return code

    def c_code_cache_version(self):
        return (1,)

    def infer_shape(self, node, input_shapes):
        return [node.inputs[1:]]

    def connection_pattern(self, node):

        rval = [[True]]

        for ipt in node.inputs[1:]:
            rval.append([False])

        return rval

    def grad(self, inputs, grads):
        x = inputs[0]
        gz = grads[0]
        n_axes_to_sum = gz.ndim - x.ndim
        #The number of dimensions added
        axis = range(n_axes_to_sum)
        #The broadcasted dimensions
        axis_broadcasted = []
        for i, (ib, gb) in enumerate(
            zip(inputs[0].broadcastable,
                #We need the dimensions corresponding to x
                grads[0].broadcastable[-inputs[0].ndim:])):
            if ib and not gb:
                axis_broadcasted.append(i + n_axes_to_sum)
        gx = gz.sum(axis=axis + axis_broadcasted)
        if axis_broadcasted:
            new_order = list(x.broadcastable)
            idx = 0
            for i in range(x.ndim):
                if not new_order[i]:
                    new_order[i] = idx
                    idx += 1
                else:
                    new_order[i] = 'x'
            gx = gx.dimshuffle(new_order)
            #Dimshuffle to add back the broadcasted dims
        #The *elements* of the output are not connected to
        #the inputs that specify the shape. If you grow the
        #shape by epsilon, the existing elements do not
        #change.
        return [gx] + [DisconnectedType()() for i in inputs[1:]]

    def __call__(self, val, *shapes, **kwargs):
        """
        If the alloc would be useless, this function returns val.

        If this function is called outside of a graph optimization context
        (for instance, it is manually called by a user building a graph),
        then we always return an Alloc node, to allow for DebugMode to check
        for size mismatches.

        If you always want an Alloc node, call make_node.
        """
        ret = super(Alloc, self).__call__(val, *shapes, **kwargs)
        try:
            # It makes optimization difficult when useless allocs are thrown
            # into the graph at every stage of optimization.  This little logic
            # tries to help at least in some cases.
            if hasattr(val, 'fgraph') and (val.type == ret.type):
                return val
        except AttributeError:
            pass
        return ret

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(eval_points[0], *inputs[1:], **dict(return_list=True))

    def do_constant_folding(self, node):
        if not getattr(node.outputs[0], 'clients', []):
            # If there are no clients then there is no point doing constant
            # folding.
            return False
        for client in node.outputs[0].clients:
            if client[0] == 'output':
                # If the output is a constant, it will have to be deepcopied
                # each time the function is called.  So we do not fold.
                return False
            elif (not isinstance(client[0], basestring)
                    and isinstance(client[0].op, (
                        theano.tensor.subtensor.IncSubtensor,
                        theano.tensor.subtensor.AdvancedIncSubtensor1,
                        theano.tensor.subtensor.AdvancedIncSubtensor,
                        ))):
                return False
        return True


alloc = Alloc()
pprint.assign(alloc, printing.FunctionPrinter('alloc'))


"""Create a duplicate of `a` (with duplicated storage)"""
tensor_copy = elemwise.Elemwise(scal.identity)
pprint.assign(tensor_copy, printing.IgnorePrinter())


@constructor
def sum(input, axis=None, dtype=None, keepdims=False, acc_dtype=None):
    """
    Computes the sum along the given axis(es) of a tensor `input`

    When axis is None (the default value), the sum is performed
    over the flattened tensor.

    keepdims: If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    For full documentation see ``tensor.elemwise.Sum``.
    In particular please pay attention to the important warning when using
    a custom acc_dtype.
    """

    out = elemwise.Sum(axis=axis, dtype=dtype, acc_dtype=acc_dtype)(input)

    if keepdims:
        out = makeKeepDims(input, out, axis)
    return out

pprint.assign(Sum(), printing.FunctionPrinter('sum'))


@constructor
def prod(input, axis=None, dtype=None, keepdims=False, acc_dtype=None):
    """
    Computes the product along the given axis(es) of a tensor `input`

    When axis is None (the default value), the product is performed
    over the flattened tensor.

    keepdims: If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    For full documentation see ``tensor.elemwise.Prod``.
    """

    out = elemwise.Prod(axis, dtype=dtype, acc_dtype=acc_dtype)(input)

    if keepdims:
        out = makeKeepDims(input, out, axis)
    return out


class Mean(elemwise.CAReduce):
    def __init__(self, axis=None):
        elemwise.CAReduce.__init__(self, scal.add, axis)
        assert self.axis is None or len(self.axis) == 1

    def __str__(self):
        if self.axis is not None:
            return "Mean{%s}" % (", ".join(str(x) for x in self.axis))
        else:
            return "Mean"

    def _output_dtype(self, idtype):
        # we want to protect against overflow
        return 'float64'

    def perform(self, node, inp, out):
        input, = inp
        output, = out
        if self.axis is None:
            axis = None
        else:
            axis = self.axis[0]
        # numpy.asarray is needed as otherwise we can end up with a
        # numpy scalar.
        output[0] = numpy.asarray(numpy.mean(input, dtype='float64',
                                             axis=axis))

    def c_code(self, node, name, inames, onames, sub):
        if self.axis is not None:
            return super(Op, self).c_code(node, name, inames, onames, sub)
        ret = elemwise.CAReduce.c_code(self, node, name, inames, onames, sub)
        # TODO: c_code perform support only axis is None
        return ret + """
  *((double *)PyArray_DATA(%s)) /= PyArray_SIZE(%s);
  """ % (onames[0], inames[0])

# TODO: implement the grad. When done and tested, you can make this the default
# version.
#    def grad(self, (x,), (gout,)):
#      import pdb;pdb.set_trace()
#      return grad(mean(x, self.axis, op=False),[x])


@constructor
def mean(input, axis=None, dtype=None, op=False, keepdims=False,
        acc_dtype=None):
    """
    Computes the mean value along the given axis(es) of a tensor `input`

    :param axis: compute the mean along this axis of the tensor.
                 None means all axes (like numpy).
    :type axis: None or int or (list of int) (see `Sum`)

    :param dtype: dtype to cast the result of the inner summation into.
        For instance, by default, a sum of a float32 tensor will be
        done in float64 (acc_dtype would be float64 by default),
        but that result will be casted back in float32.
    :type dtype: None or string

    :param keepdims: If this is set to True, the axes which are reduced are
        left in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original tensor.

    :param acc_dtype: dtype to use for the inner summation. This will not
                  necessarily be the dtype of the output (in particular
                  if it is a discrete (int/uint) dtype, the output will
                  be in a float type).
                  If None, then we use the same rules as `sum()`.
    :type acc_dtype: None or string

    :note: for gpu, if you specify dtype=float32, everything will be done
           on the gpu.
    """

    if op:
        if dtype not in (None, 'float64'):
            raise NotImplementedError(
                    'The Mean op does not support the dtype argument, '
                    'and will always use float64. If you want to specify '
                    'the dtype, call tensor.mean(..., op=False).',
                    dtype)
        if acc_dtype not in (None, 'float64'):
            raise NotImplementedError(
                    'The Mean op does not support the acc_dtype argument, '
                    'and will always use float64. If you want to specify '
                    'acc_dtype, call tensor.mean(..., op=False).',
                    dtype)
        out = Mean(axis)(input)
        if keepdims:
            out = makeKeepDims(input, out, axis)
        return out

    if dtype is not None:
        # The summation will be done with the specified dtype.
        # sum() will complain if it is not suitable.
        sum_dtype = dtype
    else:
        # Let sum() infer the appropriate dtype.
        sum_dtype = None

    s = sum(input, axis=axis, dtype=sum_dtype, keepdims=keepdims,
            acc_dtype=acc_dtype)
    shp = shape(input)

    # Cast shp into a float type
    # TODO Once we have a consistent casting policy, we could simply
    # use true_div.
    if s.dtype in ('float32', 'complex64'):
        shp = cast(shp, 'float32')
    else:
        shp = cast(shp, 'float64')

    if axis is None:
        axis = range(input.ndim)
    elif isinstance(axis, (int, numpy.integer)):
        axis = [axis]
    elif isinstance(axis, numpy.ndarray) and axis.ndim == 0:
        axis = [int(axis)]
    else:
        axis = [int(a) for a in axis]

    # This sequential division will possibly be optimized by Theano:
    for i in axis:
        s = true_div(s, shp[i])

    return s


@constructor
def var(input, axis=None, keepdims=False):
    """
    Computes the variance along the given axis(es) of a tensor `input`.

    :param axis: Compute the variance along this axis of the tensor.
                 None means all axes (like numpy).
    :type axis: None or int or (list of int) (see `Sum`)

    :param keepdims: If this is set to True, the axes which are reduced are
        left in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original tensor.

    :note: It uses the two-pass algorithm for more stable results.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
        There exist other implementations that are even more stable, but
        probably slower.
    """

    input_ndim = input.type.ndim
    if axis is None:
        axis = range(input_ndim)
    elif isinstance(axis, (int, numpy.integer)):
        axis = [axis]
    elif isinstance(axis, numpy.ndarray) and axis.ndim == 0:
        axis = [int(axis)]
    else:
        axis = [int(a) for a in axis]

    # compute the axis-wise mean
    mean_input = mean(input, axis, keepdims=True)

    # center the input
    centered_input = input - mean_input

    # return the mean sqr
    return mean((centered_input ** 2), axis, keepdims=keepdims)


@constructor
def std(input, axis=None, keepdims=False):
    """
    Computes the standard deviation along the given axis(es)
    of a tensor `input`.

    :param axis: Compute the standard deviation along this
                axis of the tensor.
                 None means all axes (like numpy).
    :type axis: None or int or (list of int) (see `Sum`)

    :param keepdims: If this is set to True, the axes
        which are reduced are
        left in the result as dimensions with size one.
        With this option,
        the result will broadcast correctly against the
        original tensor.

    :note: It calls `var()` and `var()` uses the two-pass algorithm for more
        stable results.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
        There exist other implementations that are even more stable, but
        probably slower.
    """

    return sqrt(var(input=input, axis=axis, keepdims=keepdims))


class Default(gof.Op):
    """
    Takes an input x and a default value. If the input is not None, a
    reference to it is returned. If the input is None, a copy of the
    default value is returned instead. The input and the default must
    have exactly the same type.
    """
    view_map = {0: [0]}

    def make_node(self, x, default):
        x, default = as_tensor_variable(x), as_tensor_variable(default)
        if  x.type != default.type:
            raise TypeError('Both default() arguments must have same type',
                    x, default)
        return gof.Apply(self, [x, default], [default.type()])

    def perform(self, node, inp, out_):
        x, default = inp
        out, = out_
        if x is None:
            # why copy?  Theano can't yet understand out[0] being a view of
            # either x or y, so we can be a view of x, but only a copy of y.
            out[0] = default.copy()
        else:
            out[0] = x

default = Default()
setdefault = default  # legacy


##########################
# Arithmetics
##########################
@_scal_elemwise_with_nfunc('maximum', 2, 1)
def maximum(x, y):
    """elemwise maximum. See max for the maximum in one tensor
    """
    # see decorator for function body


@_scal_elemwise_with_nfunc('minimum', 2, 1)
def minimum(x, y):
    """elemwise minimum. See min for the minimum in one tensor
    """
    # see decorator for function body


def div_proxy(x, y):
    """Proxy for either true_div or int_div, depending on types of x, y."""
    f = eval('%s_div' % scal.int_or_true_div(
        as_tensor_variable(x).dtype in discrete_dtypes,
        as_tensor_variable(y).dtype in discrete_dtypes))
    return f(x, y)


@_scal_elemwise_with_nfunc('add', 2, 1)
def add(a, *other_terms):
    """elementwise addition"""
    # see decorator for function body


@_scal_elemwise_with_nfunc('subtract', 2, 1)
def sub(a, b):
    """elementwise subtraction"""
    # see decorator for function body


@_scal_elemwise_with_nfunc('multiply', 2, 1)
def mul(a, *other_terms):
    """elementwise multiplication"""
    # see decorator for function body


@_scal_elemwise_with_nfunc('true_divide', 2, 1)
def true_div(a, b):
    """elementwise [true] division (inverse of multiplication)"""
    # see decorator for function body


@_scal_elemwise_with_nfunc('floor_divide', 2, 1)
def floor_div(a, b):
    """elementwise [floor] division (inverse of multiplication)"""
    # see decorator for function body


# not a c/p error, floor_div and int_div are the same thing
@_scal_elemwise_with_nfunc('floor_divide', 2, 1)
def int_div(a, b):
    """elementwise integer-division"""
    # see decorator for function body


def ceil_intdiv(a, b):
    """
    Safely compute ceil(float_division(a, b)).

    Works for all dtypes, but mostly useful when a and b are int.
    """
    # If a and b are int with not many significant bits, we could
    # cast them to float to avoid doing the modulo. We do not know if this
    # is faster or not. But this is not safe for int64 as the cast will
    # lose precision.
    # e.g.: cast(cast(a, scalar.upcast(a, 'float32')) / b, scal.upcast(a, b))

    # We cast for the case when a and b are uint*. Otherwise neq will
    # force their upcast to int.
    div = int_div(a, b)
    ret = cast(neq(a % b, 0), div.dtype) + div
    assert ret.dtype == scal.upcast(div.owner.inputs[0], div.owner.inputs[1])
    return ret


def mod_check(x, y):
    """Make sure we do not try to use complex numbers."""
    if (as_tensor_variable(x).dtype in complex_dtypes or
        as_tensor_variable(y).dtype in complex_dtypes):
        # Currently forbidden.
        raise scal.Mod.complex_error
    else:
        return mod(x, y)


@_scal_elemwise_with_nfunc('mod', 2, 1)
def mod(a, b):
    """elementwise modulo"""
    # see decorator for function body


@_scal_elemwise_with_nfunc('power', 2, 1)
def pow(a, b):
    """elementwise power"""
    # see decorator for function body


# The numpy.clip don't work correctly when
# the min is bigger then the max
@_scal_elemwise  # _with_nfunc('clip', 3, 1)
def clip(x, min, max):
    """clip x to be between min and max"""
    # see decorator for function body
    # for grep: clamp, bound

pprint.assign(add, printing.OperatorPrinter('+', -2, 'either'))
pprint.assign(mul, printing.OperatorPrinter('*', -1, 'either'))
pprint.assign(sub, printing.OperatorPrinter('-', -2, 'left'))
pprint.assign(neg, printing.OperatorPrinter('-',  0, 'either'))
pprint.assign(true_div, printing.OperatorPrinter('/', -1, 'left'))
pprint.assign(int_div, printing.OperatorPrinter('//', -1, 'left'))
pprint.assign(pow, printing.OperatorPrinter('**', 1, 'right'))


##########################
# View Operations
##########################


def extract_constant(x):
    '''
     This function is basically a call to tensor.get_scalar_constant_value. The
     main difference is the behaviour in case of failure. While
     get_scalar_constant_value raises an TypeError, this function returns x,
     as a tensor if possible. If x is a ScalarVariable from a
     scalar_from_tensor, we remove the conversion. If x is just a
     ScalarVariable, we convert it to a tensor with tensor_from_scalar.
    '''
    try:
        x = get_scalar_constant_value(x)
    except NotScalarConstantError:
        pass
    if (isinstance(x, scal.ScalarVariable) or
        isinstance(x, scal.sharedvar.ScalarSharedVariable)):
        if x.owner and isinstance(x.owner.op, ScalarFromTensor):
            x = x.owner.inputs[0]
        else:
            x = tensor_from_scalar(x)
    return x


def transpose(x, axes=None):
    """
    Reorder the dimensions of x. (Default: reverse them)

    This is a macro around dimshuffle that matches the numpy.transpose
    function.

    """
    if axes is None:
        axes = range((x.ndim - 1), -1, -1)
    ret = DimShuffle(x.broadcastable, axes, inplace=False)(x)
    if x.name and axes == range((x.ndim - 1), -1, -1):
        ret.name = x.name + '.T'
    return ret


def batched_dot(x, y):
    """
    :param x: A Tensor with sizes e.g.: for  3D (dim1, dim3, dim2)
    :param y: A Tensor with sizes e.g.: for 3D (dim1, dim2, dim4)
    This function computes the dot product between the two tensors, by
    iterating over the first dimension using scan.
    Returns a tensor of size e.g. if it is 3D: (dim1, dim3, dim4)
    Example:
    >>> first = tensor.tensor3('first')
    >>> second = tensor.tensor3('second')
    >>> result = batched_dot(first, second)
    :note:  This is a subset of numpy.einsum, but we do not provide it for now.
    But numpy einsum is slower than dot or tensordot:
    http://mail.scipy.org/pipermail/numpy-discussion/2012-October/064259.html
    """
    result, updates = theano.scan(fn=lambda x_mat, y_mat:
            theano.tensor.dot(x_mat, y_mat),
            outputs_info=None,
            sequences=[x, y],
            non_sequences=None)
    return result


def split(x, splits_size, n_splits, axis=0):
    the_split = Split(n_splits)
    return the_split(x, axis, splits_size)


class Split(Op):
    """Partition a `TensorVariable` along some axis.

    .. python::

        x = vector()
        splits = lvector()
        # you have to declare right away how many split_points there will be.
        ra, rb, rc = split(x, splits, n_splits = 3, axis = 0)

        f = function([x, splits], [ra, rb, rc])

        a, b, c = f([0,1,2,3,4,5,6], [3, 2, 1])

        #a == [0,1,2]
        #b == [3, 4]
        #c == [5]

    """

    len_splits = None
    """A Split instance will have this many outputs, and require that
    the splits argument to `perform` have exactly this many elements.
    """

    def __init__(self, len_splits):
        self.len_splits = int(len_splits)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.len_splits == other.len_splits)

    def __str__(self):
        return self.__class__.__name__ + "{%s}" % self.len_splits

    def __hash__(self):
        return hash(Split) ^ self.len_splits

    def make_node(self, x, axis, splits):
        """WRITEME"""
        x = as_tensor_variable(x)
        axis = as_tensor_variable(axis)
        splits = as_tensor_variable(splits)

        if splits.type not in int_vector_types:
            raise TypeError('splits must have type tensor.lvector',
                            splits.type)
        if axis.type not in int_types:
            raise TypeError('axis must have type lscalar', axis.type)

#         # The following lines are necessary if we allow splits of zero
#         if isinstance(axis, gof.Constant):
#             x = unbroadcast(x, int(axis.data))
#         else:
#             x = unbroadcast(x, *range(x.type.ndim))

        inputs = [x, axis, splits]
        outputs = [x.type() for i in xrange(self.len_splits)]

        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        """WRITEME"""
        x, axis, splits = inputs
        # in python 2.4, x.shape[numpy.asarray(1)] don't work.
        if sys.version_info[0:2] == (2, 4) and axis.size == 1:
            axis = int(axis)

        try:
            len_along_axis = x.shape[axis]
        except:
            raise ValueError('Split.perform() with axis=(%s) is invalid'
                             ' for x.shape==(%s)'
                    % (axis, x.shape))
        if len(splits) != self.len_splits:
            raise ValueError('In Split.perform(), len(splits) != len_splits.',
                    (len(splits), self.len_splits))

        if numpy.sum(splits) != len_along_axis:
            raise ValueError('The splits sum to %s, expected %s' %
                             (numpy.sum(splits), len_along_axis))
        if not python_all(splits):
            raise ValueError('Cannot have a split of zero.')

        # Checking is done, let's roll the splitting algorithm!
        # Basically we step along the given axis of x, extracting
        # subtensors of size splits[i] as we go along.

        general_key = [slice(None, None, None) for s in x.shape]
        lower_idx = 0
        for i in xrange(self.len_splits):
            upper_idx = lower_idx + splits[i]
            general_key[axis] = slice(lower_idx, upper_idx, None)
            outputs[i][0] = x.__getitem__(general_key).copy()
            lower_idx = upper_idx

    def infer_shape(self, node, in_shapes):
        axis = node.inputs[1]
        splits = node.inputs[2]
        shp_x, shp_axis, shp_splits = in_shapes
        out_shapes = []
        for i in range(self.len_splits):
            temp = as_tensor_variable(shp_x)
            temp = theano.tensor.subtensor.set_subtensor(temp[axis], splits[i])
            temp = [temp[i] for i in range(len(shp_x))]
            out_shapes.append(temp)
        return out_shapes

    def grad(self, inputs, g_outputs):
        """Join the gradients along the axis that was used to split x."""
        _, axis, n = inputs
        return [join(axis, *g_outputs),
                grad_undefined(self, 1, axis),
                grad_undefined(self, 2, n)]

    def R_op(self, inputs, eval_points):
        if eval_points[0]  is None:
            return [None for i in self.len_splits]
        return self.make_node(eval_points[0], *inputs[1:]).outputs


class Rebroadcast(Op):
    """
    Change the input's broadcastable fields in
    some predetermined way.
    e.g.: Rebroadcast((0, True), (1, False))(x)
          would make x broadcastable in axis 0
          and not broadcastable in axis 1
    See also the unbroadcast, addbroadcast and patternbroadcast functions.

    ..note: work inplace and work for CudaNdarrayType
    """
    view_map = {0: [0]}

    def __init__(self, *axis):
        self.axis = dict(axis)
        for axis, broad in self.axis.iteritems():
            assert isinstance(axis, (numpy.integer, int)), (
                "Rebroadcast need integers axis. Got ", axis)

    def __eq__(self, other):
        return type(self) == type(other) and self.axis == other.axis

    def __hash__(self):
        items = self.axis.items()
        items.sort()  # no ambiguity because each item key is unique
        return hash(type(self)) ^ hash(tuple(items))

    def __str__(self):
        if len(self.axis) == 0:
            broadcast_pattern = []
        else:
            broadcast_pattern = ['?' for i
                                 in xrange(1 + numpy.max(self.axis.keys()))]
        for k, v in self.axis.iteritems():
            broadcast_pattern[k] = str(int(v))
        return '%s{%s}' % (self.__class__.__name__,
                           ','.join(broadcast_pattern))

    def make_node(self, x):
        if self.axis.keys() and (x.ndim <= numpy.max(self.axis.keys())):
            raise ValueError('Trying to rebroadcast nonexistant dimension')
        t = x.type.__class__(dtype=x.type.dtype,
                             broadcastable=[self.axis.get(i, b)
                                            for i, b in enumerate(
                                                x.type.broadcastable)])
        return Apply(self, [x], [t()])

    def perform(self, node, inp, out_):
        x, = inp
        out, = out_
        for axis, value in self.axis.iteritems():
            if value and x.shape[axis] != 1:
                raise ValueError('Dimension %s in Rebroadcast\'s input was'
                                 ' supposed to be 1 (got %s instead)' %
                                 (axis, x.shape[axis]))
        out[0] = x

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        # restore the broadcasting pattern of the input
        return Rebroadcast(*[(axis, x.type.broadcastable[axis])
                             for axis, value in self.axis.iteritems()])(gz),

    def infer_shape(self, node, ishapes):
        assert len(ishapes) == 1
        l = []
        one = constant(1)
        for ax in xrange(len(ishapes[0])):
            if self.axis.get(ax, False):
                l.append(one)
            else:
                l.append(ishapes[0][ax])

        return [tuple(l)]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(*eval_points, **dict(return_list=True))


def addbroadcast(x, *axes):
    """
    Make the input broadcastable in the specified axes.

    We apply the opt here not to pollute the graph especially during
    the gpu optimization
    """
    rval = Rebroadcast(*[(axis, True) for axis in axes])(x)
    return theano.tensor.opt.apply_rebroadcast_opt(rval)


def unbroadcast(x, *axes):
    """
    Make the input impossible to broadcast in the specified axes.

    We apply the opt here not to pollute the graph especially during
    the gpu optimization
    """
    rval = Rebroadcast(*[(axis, False) for axis in axes])(x)
    return theano.tensor.opt.apply_rebroadcast_opt(rval)


def patternbroadcast(x, broadcastable):
    """
    Make the input adopt a specific broadcasting pattern.

    We apply the opt here not to pollute the graph especially during the gpu
    optimization.
    """
    rval = Rebroadcast(*[(i, broadcastable[i])
                         for i in xrange(len(broadcastable))])(x)
    return theano.tensor.opt.apply_rebroadcast_opt(rval)


class Join(Op):
    """
    Concatenate multiple `TensorVariable`s along some axis.

    The axis must be given as first argument. All tensors must have the same
    shape along all dimensions other than this axis.
    Of course, TensorVariable instances do not have a shape, so this error
    cannot be caught until runtime.  See `perform()`.

    For joins involving scalar values, see @stack.

    .. python::

        x, y, z = tensor.matrix(), tensor.matrix(), tensor.matrix()
        u = tensor.vector()

        r = join(0, x, y, z)
        c = join(1, x, y, z)
        join(2, x, y, z)    # WRONG: the axis has to be an index into the shape
        join(0, x, u)       # WRONG: joined tensors must have the same rank
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return '%s' % (self.__class__.__name__)

    def make_node(self, *axis_and_tensors):
        """
        :param axis: an Int or integer-valued Variable

        :param tensors: a variable number (but not zero) of tensors to
          concatenate along the specified axis.  These tensors must have
          the same shape along all dimensions other than this axis.

        :returns: a symbolic Variable.  It has the same ndim as the
            input tensors, and the most inclusive dtype.

        """
        axis, tensors = axis_and_tensors[0], axis_and_tensors[1:]
        if not tensors:
            raise ValueError('Cannot join an empty list of tensors')
        as_tensor_variable_args = [as_tensor_variable(x) for x in tensors]

        dtypes = [x.type.dtype for x in as_tensor_variable_args]
        out_dtype = scal.upcast(*dtypes)

        output_maker = lambda bcastable: tensor(dtype=out_dtype,
                                                broadcastable=bcastable)

        return self._make_node_internal(axis, tensors,
                            as_tensor_variable_args, output_maker)

    def _make_node_internal(self, axis, tensors,
                as_tensor_variable_args, output_maker):
        if not python_all(targs.type.ndim for targs
                          in as_tensor_variable_args):
            raise TypeError('Join cannot handle arguments of dimension 0.'
                            ' For joining scalar values, see @stack')
        # Handle single-tensor joins immediately.
        if len(as_tensor_variable_args) == 1:
            bcastable = list(as_tensor_variable_args[0].type.broadcastable)
        else:
            # When the axis is fixed, a dimension should be
            # broadcastable if at least one of the inputs is
            # broadcastable on that dimension (see justification below),
            # except for the axis dimension.
            # Initialize bcastable all false, and then fill in some trues with
            # the loops.
            bcastable = [False] * len(
                as_tensor_variable_args[0].type.broadcastable)
            ndim = len(bcastable)
            # Axis can also be a constant
            if not isinstance(axis, int):
                try:
                    # Note : `get_scalar_constant_value` returns a ndarray not
                    # an int
                    axis = int(get_scalar_constant_value(axis))

                except NotScalarConstantError:
                    pass
            if isinstance(axis, int):
                # Basically, broadcastable -> length 1, but the
                # converse does not hold. So we permit e.g. T/F/T
                # joins, and if they fail at runtime they fail, but if
                # they don't then it means that the argument where
                # that broadcastable flag was False had length 1 along
                # this dimension, and therefore this dimension should
                # be broadcastable for the output.
                for x in as_tensor_variable_args:
                    for current_axis, bflag in enumerate(x.type.broadcastable):
                        # Not sure if this Op supports/supported/will support
                        # negative indices, but just to be sure...
                        if current_axis == axis % ndim:
                            continue
                        if bflag:
                            bcastable[current_axis] = True
                try:
                    bcastable[axis] = False
                except IndexError:
                    raise ValueError('Join argument "axis" is out of range'
                                     ' (given input dimensions)')
                as_tensor_variable_args = [unbroadcast(x, axis)
                                           for x in as_tensor_variable_args]
            else:
                # These unbroadcasts are for the gradient... not sure exactly
                # why...
                as_tensor_variable_args = [unbroadcast(x, *range(x.type.ndim))
                                           for x in as_tensor_variable_args]
                # When the axis may vary, no dimension can be guaranteed to be
                # broadcastable.
                bcastable = [False] * len(
                    as_tensor_variable_args[0].type.broadcastable)

        if not python_all([x.ndim == len(bcastable)
                           for x in as_tensor_variable_args[1:]]):
            raise TypeError("Join() can only join tensor with the same number of dimensions.")

        inputs = [as_tensor_variable(axis)] + list(as_tensor_variable_args)
        if inputs[0].type not in int_types:
            raise TypeError('Axis could not be cast to an integer type',
                            axis, inputs[0].type, int_types)

        outputs = [output_maker(bcastable)]

        node = Apply(self, inputs, outputs)
        return node

    def perform(self, node, axis_and_tensors, out_):
        out, = out_
        axis, tensors = axis_and_tensors[0], axis_and_tensors[1:]
        out[0] = theano._asarray(numpy.concatenate(tensors, axis=axis),
                dtype=node.outputs[0].type.dtype)

    def R_op(self, inputs, eval_points):
        if None in eval_points[1:]:
            return [None]
        return self.make_node(inputs[0], *eval_points[1:]).outputs

    def grad(self, axis_and_tensors, grads):
        """ The gradient wrt a join op is a `Split`, used to partition
        the gradient along the `axis` which was used for joining.
        """
        gz, = grads
        axis, tensors = axis_and_tensors[0], axis_and_tensors[1:]

        rval = [grad_undefined(self, 0, axis)]

        if 'float' in tensors[0].dtype or 'complex' in tensors[0].dtype:
            # assume that this is differentiable
            split = Split(len(tensors))
            split_gz = split(gz, axis, stack(*[shape(x)[axis]
                                               for x in tensors]))
            # If there is only one split, it might not be in a list.
            if not isinstance(split_gz, list):
                split_gz = [split_gz]

            rval = rval + split_gz
        else:
            # the output has integer type, so the gradient through it
            # is 0
            rval = rval + [tensor.zeros_like() for tensor in tensors]

        return rval

    def infer_shape(self, node, ishapes):
        # ishapes[0] contains the size of the axis on which we join
        # Join op should get at least one input to join
        assert len(ishapes) > 1
        n_dim = len(ishapes[1])
        for shape in ishapes[1:]:
            assert shape is not None
            assert len(shape) == n_dim

        out_shapes = []
        for dim in xrange(n_dim):
            # we have to deal with 2 possible cases in here :
            #   a) we are dealing with the dimension for which we join
            #     (called t_side from true side of the if, where the if
            #     compares current dimension with the joining dimension)
            #   b) a non joining dimension ( in which maybe a symbolic
            #      assertion can be used to make sure all tensors have
            #      the same number of elements on this non-joined dimension
            #      this is f_side
            # initialize
            t_side = ishapes[1][dim]
            f_side = ishapes[1][dim]
            # loop over tensors and sum for the joining dimension
            for shape in ishapes[2:]:
                t_side = t_side + shape[dim]
            # return the dimensions found
            out_shapes.append(switch(eq(dim, node.inputs[0]),
                              t_side, f_side))

        return [tuple(out_shapes)]


"""
    Convenience function to concatenate `TensorType`s along the given axis.

    :Parameters:
     - `tensors` : list of tensors (or list-like)
       A list of tensors to be concatenated along the given axis.
     - `axis` : int (symbolic or literal)

       On which dimension should the tensors be joined?  The `axis`
       must be a valid index into the shape of the tensors to be
       concatenated.

       The `axis` parameter may either be an integer or an object that
       can be converted to a scalar using `as_scalar`(`axis`). In the
       former case, the axis is fixed at construction, while in the
       latter it may vary over time depending on the value of the
       `axis` variable.

    The shapes of the tensors to be concatenated must be all
    identical, except in the dimension (`axis`) on which they are to
    be joined.

    """
join = Join()

pprint.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, Join),
              printing.FunctionPrinter('join'))


def roll(x, shift, axis=None):
    """
    Convenience function to roll `TensorType`s along the given axis.
    Syntax copies numpy.roll function

    Parameters
    ----------
    x : tensor_like
        Input tensor.
    shift : int (symbolic or literal)
        The number of places by which elements are shifted.
    axis : int (symbolic or literal) (optional)
        The axis along which elements are shifted. By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : tensor
        Output tensor, with the same shape as `x`.
    """
    if axis is None:
        if x.ndim > 1:
            y = x.flatten()
            return roll(y, shift, axis=0).reshape(x.shape)
        else:
            axis = 0

    # A slice of all elements in a dimension ':'
    allslice = slice(None)
    # List of slices describing the front half [:, :, shift:, :]
    front_slice = slice(-shift, None)
    front_list = ([allslice] * axis + [front_slice] +
                  [allslice] * (x.ndim - axis - 1))
    # List of slices describing the back half [:, :, :shift, :]
    end_slice = slice(0, -shift)
    end_list = ([allslice] * axis + [end_slice] +
                [allslice] * (x.ndim - axis - 1))
    return join(axis,
                x.__getitem__(tuple(front_list)),
                x.__getitem__(tuple(end_list)))


@constructor
def shape_padleft(t, n_ones=1):
    """Reshape `t` by left-padding the shape with `n_ones` 1s

    See also: `shape_padright` and `Dimshuffle`
    """
    _t = as_tensor_variable(t)

    pattern = ['x'] * n_ones + [i for i in xrange(_t.type.ndim)]
    return DimShuffle(_t.broadcastable, pattern)(_t)


@constructor
def shape_padright(t, n_ones=1):
    """Reshape `t` by right-padding the shape with `n_ones` 1s

    See also: `shape_padleft` and `Dimshuffle`
    """
    _t = as_tensor_variable(t)

    pattern = [i for i in xrange(_t.type.ndim)] + ['x'] * n_ones
    return DimShuffle(_t.broadcastable, pattern)(_t)


@constructor
def stack(*tensors):
    """Insert the arguments as slices into a tensor of 1 rank greater.

    The size in dimension 0 of the result will be equal to the number
    of tensors passed.
    """
    if len(tensors) == 0:
        raise Exception('theano.tensor.stack(*tensors) must have at least'
                        ' one parameter')
    # If all tensors are scalars of the same type, call make_vector.
    # It makes the graph simpler, by not adding DimShuffles and Rebroadcasts

    # This should be an optimization!
    # Doing it here make the graph less canonicalized
    # (more type need to be understood by all optimization)
    # And DebugMode can't detect error in this code as it is not in an
    # optimization.
    # See ticket #660
    if numpy.all([
                  # in case there is direct int in tensors.
                  isinstance(t, (numpy.number, float, int, python_complex,
                                 long)) or
                  (isinstance(t, Variable) and
                   isinstance(t.type, TensorType) and
                   t.ndim == 0)
                  for t in tensors]):
        # in case there is direct int
        tensors = map(as_tensor_variable, tensors)
        dtype = scal.upcast(*[i.dtype for i in tensors])
        return theano.tensor.opt.MakeVector(dtype)(*tensors)
    return join(0, *[shape_padleft(t, 1) for t in tensors])


@constructor
def concatenate(tensor_list, axis=0):
    """Alias for `join`(axis, *tensor_list).

    This function is similar to `join`, but uses the signature of
    numpy's concatenate function.

    This function
    :Exceptions:
     - `TypeError` : the tensor_list must be a tuple or list

    """
    # Check someone did not make the common mistake to do something like:
    #   c = concatenate(x, y)
    # instead of
    #   c = concatenate((x, y))
    if not isinstance(tensor_list, (tuple, list)):
        raise TypeError("The 'tensors' argument must be either a tuple "
                "or a list, make sure you did not forget () or [] around "
                "arguments of concatenate.", tensor_list)
    return join(axis, *tensor_list)


def get_vector_length(v):
    """Return the run-time length of a symbolic vector.

    :Parameters:
     - `v` : A rank-1 TensorType variable.

    :Exceptions:
     - `TypeError` : `v` hasn't the proper type.
     - `ValueError` : No special case applies, the length is not known.

    In general this is not possible, but for a number of special cases
    the length can be determined at compile / graph-construction time.
    This function implements these special cases.

    """
    v = as_tensor_variable(v)
    if v.ndim != 1:
        raise TypeError('argument must be symbolic vector')
    if v.type.broadcastable[0]:
        return 1
    if isinstance(v, gof.Constant) and v.type.ndim == 1:
        return len(v.data)
    if v.owner and isinstance(v.owner.op, theano.tensor.opt.MakeVector):
        return len(v.owner.inputs)
    if v.owner and isinstance(v.owner.op, Shape):
        return v.owner.inputs[0].type.ndim
    raise ValueError("length not known")


@constructor
def horizontal_stack(*args):
    """
    Horizontally stack two L{TensorType}s.
    Stack two L{TensorType}s along the second axis (column wise). These
    L{TensorType}s must have the same shape along all dimensions but the
    second.
    """
    # Note: 'horizontal_stack' and 'vertical_stack' do not behave exactly like
    # Numpy's hstack and vstack functions. This is intended, because Numpy's
    # functions have potentially confusing/incoherent behavior (try them on 1D
    # arrays). If this is fixed in a future version of Numpy, it may be worth
    # trying to get closer to Numpy's way of doing things. In the meantime,
    # better keep different names to emphasize the implementation divergences.
    assert len(args) >= 2
    for arg in args:
        assert arg.type.ndim == 2
    return concatenate(args, axis=1)


@constructor
def vertical_stack(*args):
    assert len(args) >= 2
    for arg in args:
        assert arg.type.ndim == 2
    return concatenate(args, axis=0)


class Reshape(Op):
    """Perform a reshape operation of the input x to the new shape shp.

    The number of dimensions to which to reshape to (ndim) must be
    known at graph build time."""
    view_map = {0: [0]}  # output 0 is potentially aliased to inputs [0]

    def __init__(self, ndim, name=None):
        self.ndim = ndim
        self.name = name

    def __eq__(self, other):
        # .name does not participate because it doesn't affect computations
        return (type(other) is type(self)) and (other.ndim == self.ndim)

    def __hash__(self):
        # .name does not participate because it doesn't affect computations
        return hash(type(self)) ^ hash(self.ndim)

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.ndim)

    def make_node(self, x, shp):
        x = as_tensor_variable(x)
        shp_orig = shp
        shp = as_tensor_variable(shp, ndim=1)
        if not shp.dtype.startswith('int'):
            raise TypeError("Shape must be integers", shp, shp.dtype)
        assert shp.ndim == 1
        if isinstance(shp, TensorConstant):
            bcast = [s == 1 for s in shp.data]
            return gof.Apply(self, [x, shp], [tensor(x.type.dtype, bcast)])
        else:
            bcasts = [False] * self.ndim
            shp_list = shp_orig
            if hasattr(shp_orig, "ndim") and shp_orig.ndim == 0:
                shp_list = [shp_orig]
            for index in xrange(self.ndim):
                y = shp_list[index]
                y = as_tensor_variable(y)
                # Try to see if we can infer that y has a constant value of 1.
                # If so, that dimension should be broadcastable.
                try:
                    bcasts[index] = (
                            hasattr(y, 'get_scalar_constant_value') and
                            y.get_scalar_constant_value() == 1)
                except NotScalarConstantError:
                    pass
            return gof.Apply(self, [x, shp], [tensor(x.type.dtype, bcasts)])

    def perform(self, node, inp, out_):
        x, shp = inp
        out, = out_
        if (len(shp) != self.ndim):
            raise ValueError('shape argument to Reshape.perform has incorrect'
                             ' length %i'
                             ', should be %i' % (len(shp), self.ndim), shp)
        try:
            out[0] = numpy.reshape(x, shp)
        except Exception:
            raise ValueError('Cannot reshape input of shape %s to shape %s' %
                             (x.shape, shp))
        if not out[0].flags.aligned:
            raise RuntimeError("numpy.reshape returned a not aligned tensor."
                               " NumPy versions 1.6.2, 1.7.0 and 1.7.1 have"
                               " this problem for some input shape/new shape"
                               " combinations. Use another NumPy version."
                               " Input shape: %s, input stride: %s,"
                               " new_shape: %s, new_strides: %s." % (
                                   x.shape, x.strides, shp, out[0].strides))

    def connection_pattern(self, node):
        return [[True], [False]]

    def grad(self, inp, grads):
        x, shp = inp
        g_out, = grads
        return [reshape(g_out, shape(x), ndim=x.ndim),
                DisconnectedType()()]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(eval_points[0], *inputs[1:], **dict(return_list=True))

    def infer_shape(self, node, ishapes):
        # inputs[1] can contain at most one value of '-1', meaning the actual
        # shape of the output will be automatically computed by reshape, so
        # that the total number of elements stays the same.
        # TODO: Maybe put that formula here?
        # It's not trivial, because we would have to check if the product of
        # all the non-minus-one shapes is a divisor of the product of the
        # original shapes.

        # The following expression leads to cycles in feature_shape,
        # because it tries to replace the Shape_i node by the switch
        # statement, which depends on Shape_i.
        # return [tuple([switch(eq(node.inputs[1][i], -1),
        #                      theano.tensor.opt.Shape_i(i)(node.outputs[0]),
        #                      node.inputs[1][i])
        #                    for i in xrange(self.ndim)]
        #    )]

        # Here, we only simplify if the shape (node.inputs[1]) is a constant,
        # ideally it would suffice to check that it is always non-negative.

        requ = node.inputs[1]
        if isinstance(requ, theano.tensor.TensorConstant):
            requ = list(requ.data)
            requ_part = [ele for ele in requ if ele != -1]
            crit = len(requ) - len(requ_part)
            if crit == 1 and len(requ_part) > 0:
                missing = mul(*ishapes[0]) // mul(*requ_part)
                for i, ele in enumerate(requ):
                    if ele == -1:
                        requ[i] = missing
            elif crit == 1:  # we reshape to -1
                requ = [mul(*ishapes[0])]
            elif crit > 1:
                raise ValueError('shape argument to Reshape.perform'
                    ' must have at most one entry equal to -1')
            return [requ]
        else:
            oshape = []
            for i in xrange(self.ndim):
                default_os_i = theano.tensor.opt.Shape_i(i)(node.outputs[0])
                try:
                    os_i = get_scalar_constant_value(node.inputs[1][i]).item()
                    if os_i == -1:
                        os_i = default_os_i
                except NotScalarConstantError:
                    os_i = default_os_i
                oshape.append(os_i)
            return [tuple(oshape)]

    def c_code_cache_version(self):
        return (5,)

    def c_code(self, node, name, inputs, outputs, sub):
        if isinstance(node.inputs[0], TensorVariable):
            x, shp = inputs
            z, = outputs
            new_ndim = self.ndim
            fail = sub['fail']
            return """
            assert (PyArray_NDIM(%(shp)s) == 1);
            npy_intp new_dims[%(new_ndim)s];
            PyArray_Dims newshape;
            newshape.ptr = new_dims;
            newshape.len = %(new_ndim)s;
            for (int ii = 0; ii < %(new_ndim)s; ++ii)
            {
                // -- We do not want an explicit cast here. the shp can be any
                // -- int* dtype. The compiler will explicitly upcast it, but
                // -- will err if this will downcast. This could happen if the
                // -- user pass an int64 dtype, but npy_intp endup being int32.
                new_dims[ii] = ((dtype_%(shp)s*)(
                        PyArray_BYTES(%(shp)s) +
                        ii * PyArray_STRIDES(%(shp)s)[0]))[0];
            }
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject *) PyArray_Newshape(%(x)s, &newshape,
                NPY_CORDER);
            if (!%(z)s)
            {
                //The error message should have been set by PyArray_Newshape
                %(fail)s;
            }
            if (!PyArray_ISALIGNED(%(z)s)) {
                PyErr_Format(
                    PyExc_RuntimeError,
                    "PyArray_Newshape returned an object that isn't aligned!"
                    " NumPy versions 1.6.2, 1.7.0 and 1.7.1 have"
                    " this problem for some input shape/new shape"
                    " combinations. Use another NumPy version.");
                %(fail)s;
            }
            """ % locals()
        else:
            return Op.c_code(self, node, name, inputs, outputs, sub)


def reshape(x, newshape, ndim=None, name=None):
    if ndim is None:
        try:
            ndim = get_vector_length(newshape)
        except ValueError:
            raise ValueError("The length of the provided shape (%s) cannot "
                    "be automatically determined, so Theano is not able "
                    "to know what the number of dimensions of the reshaped "
                    "variable will be. You can provide the 'ndim' keyword "
                    "argument to 'reshape' to avoid this problem." % newshape)
    op = Reshape(ndim, name)
    rval = op(x, newshape)
    return rval


class Flatten(Op):
    """
    Flattens a tensor to `outdim` dimensions by preserving the leading
    outdim - 1 shape components.
    """
    view_map = {0: [0]}

    def __init__(self, outdim=1):
        self.outdim = int(outdim)

    def __eq__(self, other):
        return type(self) == type(other) and self.outdim == other.outdim

    def __hash__(self):
        return hashtype(self) ^ hash(self.outdim)

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.outdim)

    def make_node(self, x):
        t_x = as_tensor_variable(x)
        if self.outdim < 1 or (x.ndim and self.outdim > x.ndim):
            raise ValueError('invalid output ndimensions (%i) for tensor of '
                             'rank %i' % (self.outdim, t_x.ndim))
        return gof.Apply(self, [t_x], [tensor(x.type.dtype,
                                              (False,) * self.outdim)])

    def perform(self, node, inp, out_):
        x, = inp
        out, = out_
        outdim = self.outdim
        if outdim == 1:
            try:
                out[0] = x.reshape(x.size)
            except AttributeError:
                out[0] = x.reshape((numpy.prod(x.shape),))
        elif outdim == len(x.shape):
            out[0] = x
        else:
            newshape = (x.shape[:outdim - 1] +
                        (numpy.prod(x.shape[outdim - 1:]),))
            out[0] = x.reshape(newshape)

    def infer_shape(self, node, in_shapes):
        in_shp, = in_shapes
        part1 = in_shp[:self.outdim - 1]
        part2 = in_shp[self.outdim - 1:]

        if len(part2) > 1:
            part2 = (prod(part2, dtype='int64'),)
        elif len(part2) == 1:
            # We do not want to force an upcast of part2 if its length is 1
            pass
        else:
            if len(in_shp) == 0 and self.outdim == 1:
                part2 = (1,)
            else:
                raise ValueError('invalid output ndimensions (%i) for tensor '
                                 'of rank %i' % (self.outdim, len(in_shp)))

        out_shape = (part1 + part2)
        return [out_shape]

    def grad(self, inp, grads):
        x, = inp
        g_out, = grads
        return [reshape(g_out, shape(x), x.ndim)]

    def R_op(self, inputs, eval_points):
        if None in eval_points:
            return [None]
        return self.make_node(*eval_points).outputs


def flatten(x, outdim=1):
    return Flatten(outdim)(x)


# class TileGrad(Op):
#     """
#     Calculates the gradient of the Tile Op.
#     """
#     # this is so weird, I can't think of how to make this a general thing.
#     def make_node(self, x, reps, g_out):
#         return gof.Apply(self, [x, reps, g_out], [x.type()])
#
#     def perform(self, node, inp, out):
#         x, reps, g_out = inp
#         gx, = out
#         xsh = x.shape
#         if len(reps) == 2 and reps[1] == 1 and len(x.shape) == 1:
#             gx[0] = numpy.sum(g_out, axis=0)
#         else:
#             raise NotImplementedError('x.shape, reps combination not '
#                                       'supported', (x.shape, reps))
#
# tilegrad = TileGrad()


class Tile(Op):
    """
    Construct an array by repeating the input x according to reps pattern.

    Tiles its input according to reps. The length of reps is the number of
    dimension of x and contains the number of times to tile x in each
    dimension.

    :see: `numpy.tile
    <http://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html>`_
    """
    def __init__(self, ndim):
        self.ndim = ndim

    def __eq__(self, other):
        return (type(other) is Tile) and (other.ndim == self.ndim)

    def __hash__(self):
        return hash(Tile) ^ hash(self.ndim)

    def make_node(self, x, reps):
        x = as_tensor_variable(x)
        reps = as_tensor_variable(reps)
        return gof.Apply(self, [x, reps], [tensor(x.type.dtype, [False] *
                                                  self.ndim)])

    def perform(self, node, inp, out_):
        x, reps = inp
        out, = out_
        out[0] = numpy.tile(x, reps)
        if len(out[0].shape) != self.ndim:
            raise ValueError('Tile.perform produced incorrect shape')

    def infer_shape(self, node, in_shapes):
        # Note: in contrast with numpy, it is assumed that x.shape and reps
        # have equal length;  see also tile function below

        # Note: if reps were to be allowed not to be a constant and x.shape
        # and reps to be unequal, the following block of code could be used:
        ## prepend 1 to x.shape if needed
        # if self.ndim > x.ndim:
        # shp = concatenate(ones(self.ndim - x.ndim), shp)
        ## prepend 1 to reps if needed
        # reps = concatenate(ones(self.ndim - reps.shape[0]), reps)

        x, reps = node.inputs
        shp = in_shapes[0]
        tiled_shp = shp * reps
        out_shape = []
        for i in range(self.ndim):
            out_shape.append(tiled_shp[i])
        return [out_shape]

    def grad(self, inp, grads):
        x, reps = inp
        g_out, = grads
        # return [tilegrad(x, reps, g_out), None]
        raise NotImplementedError()


def tile(x, reps, ndim=None):
    """
    Tile input array `x` according to `reps`. See the docstring of `numpy.tile`
    for details.

    Currently, `reps` must be a constant, x.ndim and len(reps) must be equal
    and, if specified, 'ndim' must be equal to both.

    TODO: expand this.
    """

    try:
        assert python_all([int(i) == i for i in iter(reps)])
    except (TypeError, AssertionError):
        raise ValueError("reps argument to tile must be a constant (e.g. "
        "tuple, list of integers)")
    if len(reps) != x.ndim:
        raise ValueError("len(reps) != x.ndim not currently supported")
    elif (ndim is not None) and ndim != x.ndim:
        raise ValueError("if specified, ndim must be equal to both x.ndim and "
                         "len(reps)")

    if not hasattr(tile, 'op'):
        tile.op = {}

    if ndim is None:
        ndim = len(reps)

    # backport
    # ndim = len(reps) if ndim is None else ndim
    # not sure if len(shp) is going to work.
    if ndim not in tile.op:
        tile.op[ndim] = Tile(ndim)
    return tile.op[ndim](x, reps)


class ARange(Op):
    """Create an array containing evenly spaced values within a given interval.

    Parameters and behaviour are the same as numpy.arange().
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def __eq__(self, other):
        return type(self) == type(other) and self.dtype == other.dtype

    def __hash__(self):
        return hash(self.dtype)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, start, stop, step):
        start, stop, step = map(as_tensor_variable, (start, stop, step))
        assert start.ndim == 0
        assert stop.ndim == 0
        assert step.ndim == 0

        inputs = [start, stop, step]
        outputs = [tensor(self.dtype, (False,))]
        return Apply(self, inputs, outputs)

    def infer_shape(self, node, i_shapes):
        start, stop, step = node.inputs

        def is_constant_value(var, value):
            try:
                v = get_scalar_constant_value(var)
                return numpy.all(v == value)
            except NotScalarConstantError:
                pass
            return False

        if is_constant_value(step, 1):
            if is_constant_value(start, 0):
                return [(cast(stop, 'int64'),)]
            else:
                return [(maximum(cast(stop - start, 'int64'), 0),)]
        else:
            return [(maximum(cast(ceil(cast((stop - start), 'float64')
                                       / step), 'int64'), 0),)]

    def perform(self, node, inp, out_):
        start, stop, step = inp
        out, = out_
        start = start.item()
        stop = stop.item()
        step = step.item()
        out[0] = numpy.arange(start, stop, step, dtype=self.dtype)

    def connection_pattern(self, node):

        return [[True], [False], [True]]

    def grad(self, inputs, grads):
        start, stop, step = inputs
        gz, = grads
        # start and step affect the output values
        # but the outputs are integers so there's
        # no gradient through them
        # stop does not affect the output values,
        # just the output shape, so it is disconnected
        return [start.zeros_like(),
                DisconnectedType()(),
                step.zeros_like()]

    def R_op(self, inputs, eval_points):
        return [None]
_arange = {}


def arange(start, stop=None, step=1, dtype=None):
    # If only one argument is provided, it is in fact the "stop" argument,
    # and start is 0.
    if stop is None:
        start, stop = 0, start

    start, stop, step = map(as_tensor_variable, (start, stop, step))
    # If dtype is not provided, infer it from the other arguments
    if dtype is None:
        dtype = scal.upcast(start.type.dtype, stop.type.dtype, step.type.dtype)
        if config.cast_policy in ('numpy', 'numpy+floatX'):
            # We enforce numpy semantics, except in the special case where
            # `config.cast_policy` is 'numpy+floatX' and we want to use float32
            # rather than float64.
            # As an example, if `start`, `stop` and `step` are all int32,
            # `numpy.arange` returns an int64 array (on 64-bit platforms),
            # while the upcast above returns int32.
            numpy_dtype = numpy.arange(
                    start=numpy.array(0, dtype=start.dtype),
                    stop=numpy.array(1, dtype=stop.dtype),
                    step=numpy.array(1, dtype=step.dtype)).dtype
            if numpy_dtype != dtype:
                if (config.cast_policy == 'numpy+floatX' and
                    config.floatX == 'float32' and
                    numpy_dtype == 'float64' and
                    # No explicit float64 in the three arguments?
                    python_all(dt != 'float64'
                        for dt in [s.dtype for s in (start, stop, step)])):
                    # We use float32 instead.
                    assert dtype != 'float64'
                    dtype = 'float32'
                else:
                    # We use the same dtype as numpy instead of the result of
                    # the upcast.
                    dtype = str(numpy_dtype)

    if dtype not in _arange:
        _arange[dtype] = ARange(dtype)
    return _arange[dtype](start, stop, step)


class PermuteRowElements(Op):
    """Permute the elements of each row (inner-most dim) of a tensor.

    A permutation will be applied to every row (vector) of the input tensor x.
    Depending on the dimensionality of x and the permutation tensor y,
    different cases are possible.
    If y.ndim = 1, y is a single permutation, that will be applied to every
    vector of x. For instance, if x is a matrix, the same permutation will be
    applied to each row of x.
    If x.ndim = y.ndim, each row of x corresponds to a row of y, containing
    a permutation that will be applied to that row. For instance, if x and y
    are two matrices, a different permutation will be applied to each row of x.
    If x.ndim > y.ndim, y will be broadcasted to fit x, then each row (vector)
    of x will be reordered according to the corresponding row of y. (This is
    a generalization of the first case).
    If x.ndim = 1, every permutation in y will be applied to x, and the output
    will contain all the results.
    If x.ndim < y.ndim, x will be broadcasted to fit y, and different
    permutations contained in y will be applied to each vector in x. (This is
    a generalization of the previous case).

    If the "inverse" argument is True, the Op will perform the inverse
    permutation instead.
    """

    def make_node(self, x, y, inverse):
        x = as_tensor_variable(x)
        y = as_tensor_variable(y)
        if inverse:  # as_tensor_variable does not accept booleans
            inverse = as_tensor_variable(1)
        else:
            inverse = as_tensor_variable(0)

        # y should contain integers
        assert (y.type.dtype.startswith('int') or
                y.type.dtype.startswith('uint'))
        # Inverse should be an integer scalar
        assert inverse.type.ndim == 0 and\
                (inverse.type.dtype.startswith('int') or\
                 inverse.type.dtype.startswith('uint'))

        # Match shapes of x and y
        x_dim = x.type.ndim
        y_dim = y.type.ndim

        if x_dim > y_dim:
            y = shape_padleft(y, n_ones=(x_dim - y_dim))
        elif x_dim < y_dim:
            x = shape_padleft(x, n_ones=(y_dim - x_dim))

        # Compute the broadcastable pattern of the output
        out_broadcastable = [xb and yb for xb, yb in
                             izip(x.type.broadcastable, y.type.broadcastable)]
        out_type = tensor(dtype=x.type.dtype, broadcastable=out_broadcastable)

        inputlist = [x, y, inverse]
        outputlist = [out_type]
        return Apply(self, inputlist, outputlist)

    def _rec_perform(self, node, x, y, inverse, out, curdim):
        """Perform the permutation by doing a recursion over the input
        dimensions.

        For every dimension, starting with the leftmost, the right set of
        indices is determined (depending if broadcasting or not), then
        the function is recursively called on the appropriate subtensors.

        The terminal case is reached when the current tensors are vector,
        then the permutation contained in y is applied to x.

        :param x: The input tensor, on which the permutation is applied
        :param y: Tensor containing the permutations to apply
        :param out: Tensor storing the output result
        :param curdim: Counter of the current depth of recursion
        :param inverse: Wether to apply permutations or their inverse
        """
        if len(x.shape) == 1:
            # Numpy advanced indexing works in this case
            if inverse:
                out[y] = x[:]
            else:
                out[:] = x[y]
            if (numpy.__version__ <= '1.6.1' and
                    out.size != numpy.uint32(out.size)):
                warnings.warn(
                        'Numpy versions 1.6.1 and below have a bug preventing '
                        'advanced indexing from correctly filling arrays that '
                        'are too big (>= 2^32 elements). It is possible that '
                        'out (%s), with shape %s, is not correctly filled.'
                        % (out, out.shape))
        else:
            xs0 = x.shape[0]
            ys0 = y.shape[0]
            if xs0 == ys0:
                for i in xrange(xs0):
                    self._rec_perform(node, x[i], y[i], inverse, out[i],
                                      curdim + 1)
            elif ys0 == 1 and node.inputs[1].type.broadcastable[curdim]:
                # Broadcast y
                for i in xrange(xs0):
                    self._rec_perform(node, x[i], y[0], inverse, out[i],
                                      curdim + 1)
            elif xs0 == 1 and node.inputs[0].type.broadcastable[curdim]:
                # Broadcast x
                for i in xrange(ys0):
                    self._rec_perform(node, x[0], y[i], inverse, out[i],
                                      curdim + 1)
            else:
                raise ValueError('Dimension mismatch: %s, %s' % (xs0, ys0))

    def perform(self, node, inp, out):
        x, y, inverse = inp
        outs, = out
        x_s = x.shape
        y_s = y.shape
        assert len(x_s) == len(y_s)

        # Make sure the output is big enough
        out_s = []
        for xdim, ydim in izip(x_s, y_s):
            if xdim == ydim:
                outdim = xdim
            elif xdim == 1:
                outdim = ydim
            elif ydim == 1:
                outdim = xdim
            else:
                raise ValueError('Dimension mismatch: %s, %s' % (xdim, ydim))
            out_s.append(outdim)

        if outs[0] is None or outs[0].shape != out_s:
            outs[0] = numpy.empty(out_s, dtype=x.dtype)

        self._rec_perform(node, x, y, inverse, outs[0], curdim=0)

    def infer_shape(self, node, in_shapes):
        shp_x = in_shapes[0]
        shp_y = in_shapes[1]
        assert len(shp_x) == len(shp_y)
        out_shape = []
        for i in range(len(shp_x)):
            out_shape.append(maximum(shp_x[i], shp_y[i]))
        return [out_shape]

    def grad(self, inp, grads):
        x, y, inverse = inp
        gz, = grads
        # First, compute the gradient wrt the broadcasted x.
        # If 'inverse' is False (0), apply the inverse of y on gz.
        # Else, apply y on gz.
        gx = permute_row_elements(gz, y, eq(inverse, 0))

        # If x has been broadcasted along some axes, we need to sum
        # the gradient over these axes, but keep the dimension (as
        # broadcastable)
        broadcasted_dims = [dim for dim in xrange(gz.type.ndim)
                            if x.type.broadcastable[dim]
                            and not gz.type.broadcastable[dim]]
        gx = Sum(axis=broadcasted_dims)(gx)

        # Sum(...) removed the dimensions in broadcasted_dims,
        # so we need to put them back.
        newdims = []
        i = 0
        for dim in xrange(gz.type.ndim):
            if dim in broadcasted_dims:
                newdims.append('x')
            else:
                newdims.append(i)
                i += 1

        gx = DimShuffle(gx.type.broadcastable, newdims)(gx)
        assert gx.type.broadcastable == x.type.broadcastable

        # if x is an integer type, then so is the output.
        # this means f(x+eps) = f(x) so the gradient with respect
        # to x is zero
        if x.type.dtype.find('int') != -1:
            gx = x.zeros_like()

        # The elements of y and of inverse both affect the output,
        # so they are connected to the output,
        # and the transformation isn't defined if their values
        # are non-integer, so the gradient with respect to them is
        # undefined

        return [gx, grad_undefined(self, 1, y),
                grad_undefined(self, 1, inverse)]

_permute_row_elements = PermuteRowElements()


def permute_row_elements(x, y, inverse=0):
    return _permute_row_elements(x, y, inverse)


def inverse_permutation(perm):
    """Computes the inverse of permutations.
    Each row of input should contain a permutation of the first integers.
    """
    return permute_row_elements(
            arange(perm.shape[-1], dtype=perm.dtype),
            perm,
            inverse=True)


#########################
# Linalg : Dot
#########################
#
# For BLAS-related ops see blas.py
#
# TODO: Dotinv should go here, Eigs, Svd, etc.


class Dot(Op):
    """
    Computes the dot product of two variables. For two matrices, this is
    equivalent to matrix multiplication. For two vectors, this is the inner
    product.

    :note: matrix-matrix products are sometimes optimized to Dot22 or Gemm ops.
    (see tensor.blas)

    :note: vector-vector products are sometimes optimized to Ger or CGer.  (see
    tensor.blas)

    :note: matrix-vector products are sometimes optimized to Gemv, CGemv (see
    tensor.blas)

    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    # the rationale for Dot22 is related to getting GEMM Ops into the
    # graph.  See Dot22 in tensor.blas for details.

    def make_node(self, *inputs):
        inputs = map(as_tensor_variable, inputs)

        if len(inputs) != 2:
            raise TypeError(
                'theano.tensor.Dot: 2 arguments required, %d given ' %
                len(inputs))
        if inputs[0].ndim not in (1, 2):
            raise TypeError(
                'theano.tensor.Dot: input 0 (0-indexed) must have ndim of '
                '1 or 2, %d given. Consider calling theano.tensor.dot '
                'instead.' % inputs[0].ndim)
        if inputs[1].ndim not in (1, 2):
            raise TypeError(
                'theano.tensor.Dot: input 1 (0-indexed) must have ndim of '
                '1 or 2, %d given. Consider calling theano.tensor.dot '
                'instead.' % inputs[1].ndim)

        i_broadcastables = [input.type.broadcastable for input in inputs]
        bx, by = i_broadcastables
        if len(by) == 2:  # y is a matrix
            bz = bx[:-1] + by[-1:]
        elif len(by) == 1:  # y is vector
            bz = bx[:-1]

        i_dtypes = [input.type.dtype for input in inputs]
        outputs = [tensor(scal.upcast(*i_dtypes), bz)]
        return Apply(self, inputs, outputs)

    def perform(self, node, inp, out):
        x, y = inp
        z, = out

        # the asarray is here because dot between two vectors
        # gives a numpy float object but we need to return a 0d
        # ndarray
        z[0] = numpy.asarray(numpy.dot(x, y))

    def grad(self, inp, grads):

        x, y = inp
        gz, = grads
        xdim, ydim, gdim = x.type.ndim, y.type.ndim, gz.type.ndim

        #grad is scalar, so x is vector and y is vector
        if gdim == 0:
            xgrad = gz * y
            ygrad = gz * x

        #x is vector, y is matrix, grad is vector
        elif xdim == 1 and ydim == 2:
            xgrad = dot(gz, y.T)
            ygrad = outer(x.T, gz)

        #x is matrix, y is vector, grad is vector
        elif xdim == 2 and ydim == 1:
            xgrad = outer(gz, y.T)
            ygrad = dot(x.T, gz)

        #x is matrix, y is matrix, grad is matrix
        elif xdim == ydim == 2:
            xgrad = dot(gz, y.T)
            ygrad = dot(x.T, gz)

        # If x or y contain broadcastable dimensions but only one of
        # them know that a matching dimensions is broadcastable, the
        # above code don't always return the right broadcast pattern.
        # This cause problem down the road. See gh-1461.
        if xgrad.broadcastable != x.broadcastable:
            xgrad = patternbroadcast(xgrad, x.broadcastable)
        if ygrad.broadcastable != y.broadcastable:
            ygrad = patternbroadcast(ygrad, y.broadcastable)

        rval = xgrad, ygrad

        for elem in rval:
            assert elem.dtype.find('float') != -1

        return rval

    def R_op(self, inputs, eval_points):
        # R_op for a \dot b evaluted at c for a and d for b is
        # simply c \dot b + a \dot d


        assert len(inputs) == 2
        assert len(eval_points) == 2
        if eval_points[0] is None and eval_points[1] is None:
            return [None]

        debugger_available = config.compute_test_value != 'off'

        if debugger_available:
            try:
                iv0 = gof.op.get_test_value(inputs[0])
            except AttributeError:
                gof.op.missing_test_message(
                    'first input passed to Dot.R_op has no test value')
                debugger_available = False

            try:
                iv1 = gof.op.get_test_value(inputs[1])
            except AttributeError:
                gof.op.missing_test_message(
                    'second input passed to Dot.R_op has no test value')
                debugger_available = False

            if eval_points[0]:
                try:
                    ev0 = gof.op.get_test_value(eval_points[0])
                except AttributeError:
                    gof.op.missing_test_message(
                        'first eval point passed to Dot.R_op has no test value')
                    debugger_available = False
            if eval_points[1]:
                try:
                    ev1 = gof.op.get_test_value(eval_points[1])
                except AttributeError:
                    gof.op.missing_test_message(
                        'second eval point passed to Dot.R_op has no test value')
                    debugger_available = False

        if debugger_available:
            input_values = [iv0, iv1]
            eval_point_values = [ev0, ev1]

            for i in xrange(2):
                if eval_point_values[i] is not None and \
                   input_values[i].shape != eval_point_values[i].shape:
                    raise ValueError('input ' + str(i) + ' and eval_point ' +
                                     str(i) + ' to Dot.R_op '
                                     'should have the '
                                     'same shape, but their shapes are'
                                     ' %s and %s, respectively' % (
                            str(input_values[i].shape),
                            str(eval_point_values[i].shape)))
        if eval_points[0]:
            t1 = self(eval_points[0], inputs[1])
        if eval_points[1]:
            t2 = self(inputs[0], eval_points[1])

        if eval_points[0] and eval_points[1]:
            return [t1 + t2]
        elif eval_points[0]:
            return [t1]
        else:
            return [t2]

    def infer_shape(self, node, shapes):
        xshp, yshp = shapes
        x, y = node.inputs

        # vector / vector
        if x.ndim == 1 and y.ndim == 1:
            return [()]
        # matrix / vector
        if x.ndim == 2 and y.ndim == 1:
            return [xshp[:-1]]
        # vector / matrix
        if x.ndim == 1 and y.ndim == 2:
            return [yshp[-1:]]
        # matrix / matrix
        if x.ndim == 2 and y.ndim == 2:
            return [xshp[:-1] + yshp[-1:]]
        raise NotImplementedError()

    def __str__(self):
        return "dot"

_dot = Dot()
pprint.assign(_dot, printing.OperatorPrinter(printing.special['middle_dot'],
                                            -1, 'left'))


def dot(a, b):
    """
    Computes the dot product of two variables. For two matrices, this is
    equivalent to matrix multiplication. For two vectors, this is the inner
    product. When one variable is a scalar, this is like elementwise
    multiplication.  For N dimensions, this is a sum product over the last axis
    of the first array and the second-to-last axis of the second array:

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Note that this dot function does one of three things, in the following
    sequence:

        1.  If either a or b is scalar, it returns the elementwise product
            without calling the Theano Dot op.

        2.  If either a or b has more than 2 dimensions, it calls Theano's
            tensordot function with appropriate axes. The tensordot function
            expresses high-dimensional dot products in terms of 2D matrix
            multiplications, so it may be possible to futherize optimize for
            performance.

        3.  If both a and b have either 1 or 2 dimensions, it calls Theano's
            Dot op on a and b.

    :note: matrix-matrix products are sometimes optimized to Dot22 or Gemm ops.
    (see tensor.blas)

    :note: vector-vector products are sometimes optimized to Ger or CGer.  (see
    tensor.blas)

    :note: matrix-vector products are sometimes optimized to Gemv, CGemv (see
    tensor.blas)

    """
    a, b = as_tensor_variable(a), as_tensor_variable(b)

    if a.ndim == 0 or b.ndim == 0:
        return a * b
    elif a.ndim > 2 or b.ndim > 2:
        return tensordot(a, b, [[a.ndim - 1], [numpy.maximum(0, b.ndim - 2)]])
    else:
        return _dot(a, b)


#########################
# Linalg : TensorDot
#########################

def tensordot(a, b, axes=2):
    """
    Given two tensors a and b,tensordot computes a generalized dot product over
    the provided axes. Theano's implementation reduces all expressions to
    matrix or vector dot products and is based on code from Tijmen Tieleman's
    gnumpy (http://www.cs.toronto.edu/~tijmen/gnumpy.html).

    :param a: the first tensor variable
    :type a: symbolic tensor

    :param b: the second tensor variable
    :type b: symbolic tensor

    :param axes: an integer or array. If an integer, the number of axes
                 to sum over. If an array, it must have two array
                 elements containing the axes to sum over in each tensor.

                 Note that the default value of 2 is not guaranteed to work
                 for all values of a and b, and an error will be raised if
                 that is the case. The reason for keeping the default is to
                 maintain the same signature as numpy's tensordot function
                 (and np.tensordot raises analogous errors for non-compatible
                 inputs).

                 If an integer i, it is converted to an array containing
                 the last i dimensions of the first tensor and the first
                 i dimensions of the second tensor:
                     axes = [range(a.ndim - i, b.ndim), range(i)]

                 If an array, its two elements must contain compatible axes
                 of the two tensors. For example, [[1, 2], [2, 0]] means sum
                 over the 2nd and 3rd axes of a and the 3rd and 1st axes of b.
                 (Remember axes are zero-indexed!) The 2nd axis of a and the
                 3rd axis of b must have the same shape; the same is true for
                 the 3rd axis of a and the 1st axis of b.
    :type axes: int or array-like of length 2

    :returns: a tensor with shape equal to the concatenation of a's shape
              (less any dimensions that were summed over) and b's shape
              (less any dimensions that were summed over).
    :rtype: symbolic tensor

    It may be helpful to consider an example to see what tensordot does.
    Theano's implementation is identical to NumPy's. Here a has shape (2, 3, 4)
    and b has shape (5, 6, 4, 3). The axes to sum over are [[1, 2], [3, 2]] --
    note that a.shape[1] == b.shape[3] and a.shape[2] == b.shape[2]; these axes
    are compatible. The resulting tensor will have shape (2, 5, 6) -- the
    dimensions that are not being summed:

        a = np.random.random((2,3,4))
        b = np.random.random((5,6,4,3))

        #tensordot
        c = np.tensordot(a, b, [[1,2],[3,2]])

        #loop replicating tensordot
        a0, a1, a2 = a.shape
        b0, b1, _, _ = b.shape
        cloop = np.zeros((a0,b0,b1))

        #loop over non-summed indices -- these exist
        #in the tensor product.
        for i in range(a0):
            for j in range(b0):
                for k in range(b1):
                    #loop over summed indices -- these don't exist
                    #in the tensor product.
                    for l in range(a1):
                        for m in range(a2):
                            cloop[i,j,k] += a[i,l,m] * b[j,k,m,l]

        np.allclose(c, cloop) #true

    This specific implementation avoids a loop by transposing a and b such that
    the summed axes of a are last and the summed axes of b are first. The
    resulting arrays are reshaped to 2 dimensions (or left as vectors, if
    appropriate) and a matrix or vector dot product is taken. The result is
    reshaped back to the required output dimensions.

    In an extreme case, no axes may be specified. The resulting tensor
    will have shape equal to the concatenation of the shapes of a and b:

        c = np.tensordot(a, b, 0)
        print(a.shape) #(2,3,4)
        print(b.shape) #(5,6,4,3)
        print(c.shape) #(2,3,4,5,6,4,3)

    See the documentation of numpy.tensordot for more examples.
    """
    a, b = as_tensor_variable(a), as_tensor_variable(b)

    # axes must be a scalar or list/tuple of length 2
    if not numpy.isscalar(axes) and len(axes) != 2:
        raise ValueError('Axes should be an integer or a '
                         'list/tuple of len 2 (%s was provided)' % repr(axes))

    # if 'axes' is a number of axes to multiply and sum over (trailing axes
    # of a, leading axes of b), we can just reshape and use dot.
    elif numpy.isscalar(axes):
        axes = int(axes)

        # check if axes is valid given the dimension of a and b
        if axes > a.ndim:
            raise ValueError('axes can not be larger than the dimension of '
                    'a (a.ndim=%i, axes=%i)' % (a.ndim, axes))
        if axes > b.ndim:
            raise ValueError('axes can not be larger than than the dimension '
                             'of b (b.ndim=%i, axes=%i)' % (b.ndim, axes))

        outshape = concatenate([a.shape[:a.ndim - axes], b.shape[axes:]])
        outndim = a.ndim + b.ndim - (2 * axes)

        a_shape_0 = b_shape_0 = a_shape_1 = b_shape_1 = 1
        for s0 in range(a.ndim - axes):
            a_shape_0 *= a.shape[s0]
        for s0 in range(axes):
            b_shape_0 *= b.shape[s0]
        for s1 in range(a.ndim - axes, a.ndim):
            a_shape_1 *= a.shape[s1]
        for s1 in range(axes, b.ndim):
            b_shape_1 *= b.shape[s1]

        a_reshaped = a.reshape((a_shape_0, a_shape_1), ndim=2)
        b_reshaped = b.reshape((b_shape_0, b_shape_1), ndim=2)

        return _dot(a_reshaped, b_reshaped).reshape(outshape, outndim)

    # if 'axes' is a list, transpose a and b such that the summed axes of a
    # are last and the summed axes of b are first.
    else:
        #get first axis element as a tuple
        try:
            a_axes = tuple(axes[0])
        except TypeError:
            a_axes = tuple([axes[0]])

        #get second axis element as a tuple
        try:
            b_axes = tuple(axes[1])
        except TypeError:
            b_axes = tuple([axes[1]])

        # the two axes lists must have the same length
        if len(a_axes) != len(b_axes):
            raise ValueError('Axes elements must have the same length.')

        # check that there aren't more axes than a has dimensions
        if len(a_axes) > a.ndim:
            raise ValueError('axes[0] should be array_like with length '
                             'less than the dimensions of a '
                             '(a.ndim=%i, len(axes[0])=%i).' %
                             (a.ndim, len(a_axes)))

        # check that a_axes doesn't contain an axis greater than or equal to
        # a's dimensions. also check if len > 0 so numpy.max won't raise an
        # error.
        if len(a_axes) > 0 and numpy.max(numpy.array(a_axes)) >= a.ndim:
            raise ValueError('axes[0] contains dimensions greater than or '
                             'equal to a.ndim (a.ndim=%i, max(axes[0])=%i).' %
                             (a.ndim, numpy.max(numpy.array(a_axes))))

        # check that there aren't more axes than b has dimensions
        if len(b_axes) > b.ndim:
            raise ValueError('axes[1] should be array_like, of length '
                             'smaller than the dimension of b '
                             '(a.ndim=%i, len(axes[0])=%i).' %
                             (b.ndim, len(b_axes)))

        # check that b_axes doesn't contain an axis greater than or equal to
        # b's dimensions. also check if len > 0 so numpy.max won't raise an
        # error.
        if len(b_axes) > 0 and numpy.max(numpy.array(b_axes)) >= b.ndim:
            raise ValueError('axes[1] contains dimensions greater than or '
                             'equal to b.ndim (b.ndim=%i, max(axes[1])=%i).' %
                             (b.ndim, numpy.max(numpy.array(b_axes))))

        a_order = (tuple(x for x in tuple(xrange(a.ndim)) if x not in a_axes)
                + a_axes)
        b_order = (b_axes
                + tuple(x for x in tuple(xrange(b.ndim)) if x not in b_axes))

        a_shuffled = a.dimshuffle(a_order)
        b_shuffled = b.dimshuffle(b_order)

        # now that a and b are in the right order, call tensordot recursively
        return tensordot(a_shuffled, b_shuffled, len(a_axes))


def outer(x, y):
    """Return vector-vector outer product.

    If an input isn't a vector, we flatten it first.
    """
    if x.ndim != 1:
        x = x.flatten()
    if y.ndim != 1:
        y = y.flatten()
    return dot(
            x.dimshuffle(0, 'x'),
            y.dimshuffle('x', 0))


def any(x, axis=None, keepdims=False):
    out = elemwise.Any(axis)(x)

    if keepdims:
        out = makeKeepDims(x, out, axis)
    return out


def all(x, axis=None, keepdims=False):
    out = elemwise.All(axis)(x)

    if keepdims:
        out = makeKeepDims(x, out, axis)
    return out


class Diagonal(Op):
    """Return specified diagonals.

    :param x: A tensor variable with x.ndim >= 2.

    :return: A vector representing the diagonal elements.
    """

    def __init__(self, offset=0, axis1=0, axis2=1):
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.offset == other.offset and
                self.axis1 == other.axis1 and
                self.axis2 == other.axis2)

    def __hash__(self):
        return (hash(type(self)) ^ hash(self.offset) ^
                hash(self.axis1) ^ hash(self.axis2))

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim >= 2
        return Apply(self, [x], [tensor(dtype=x.dtype,
                                        broadcastable=[False] * (x.ndim - 1))])

    def perform(self, node, (x,), (z,)):
        z[0] = x.diagonal(self.offset, self.axis1, self.axis2)

    def grad(self, (x,), (gz,)):
        return [grad_not_implemented(self, 0, x)]

    def infer_shape(self, node, shapes):
        in_shape, = shapes
        dim1 = in_shape[self.axis1]
        dim2 = in_shape[self.axis2]
        out_shape = [d for i, d in enumerate(in_shape)
                     if i not in (self.axis1, self.axis2)]
        # The following logic is inspired by C code of PyArray_Diagonal().
        offset = self.offset
        if offset > 0:
            diag_size = clip(dim2 - offset, 0, dim1)
        elif offset < 0:
            diag_size = clip(dim1 + offset, 0, dim2)
        else:
            diag_size = minimum(dim1, dim2)
        out_shape.append(diag_size)
        return [tuple(out_shape)]

    def __str__(self):
        return self.__class__.__name__


def diagonal(a, offset=0, axis1=0, axis2=1):
    if (offset, axis1, axis2) == (0, 0, 1):
        from theano.sandbox.linalg import extract_diag
        return extract_diag(a)
    return Diagonal(offset, axis1, axis2)(a)


class Diag(Op):

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, diag):
        diag = as_tensor_variable(diag)
        if diag.type.ndim != 1:
            raise TypeError('data argument must be a vector', diag.type)

        return Apply(self, [diag], [matrix(dtype=diag.dtype)])

    def perform(self, node, inputs, (z,)):
        z[0] = numpy.diag(inputs[0])

    def grad(self, inputs, (gz,)):
        return [diagonal(gz)]

    def infer_shape(self, nodes, shapes):
        return [(shapes[0][0],) * 2]

    def __str__(self):
        return self.__class__.__name__


def diag(v, k=0):
    if v.ndim == 1:
        assert k == 0, "diagonals other than main are not implemented"
        return Diag()(v)
    elif v.ndim == 2:
        return diagonal(v, k)
    else:
        raise ValueError("Input must be 1- or 2-d.")


def stacklists(arg):
    """
    Recursively stack lists of tensors to maintain similar structure.

    This function can create a tensor from a shaped list of scalars:

    >>> from theano.tensor import stacklists, scalars, matrices
    >>> from theano import function
    >>> a, b, c, d = scalars('abcd')
    >>> X = stacklists([[a, b], [c, d]])
    >>> f = function([a, b, c, d], X)
    >>> f(1, 2, 3, 4)
    array([[ 1.,  2.],
           [ 3.,  4.]], dtype=float32)

    We can also stack arbitrarily shaped tensors. Here we stack matrices into
    a 2 by 2 grid:

    >>> from numpy import ones
    >>> a, b, c, d = matrices('abcd')
    >>> X = stacklists([[a, b], [c, d]])
    >>> f = function([a, b, c, d], X)
    >>> x = ones((4, 4), 'float32')
    >>> f(x, x, x, x).shape
    (2, 2, 4, 4)
    """
    if isinstance(arg, (tuple, list)):
        return stack(*map(stacklists, arg))
    else:
        return arg
