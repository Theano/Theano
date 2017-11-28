"""A `Type` and `Op` classes to work with numpy.ndarrays symbolically."""
from __future__ import absolute_import, print_function, division

from six.moves import builtins
import sys
import warnings

import numpy as np
from six import integer_types
from six.moves import xrange
import numbers

import theano
from theano.compat import izip
from theano import config
from theano import gof
from theano.gof import Apply, Constant, Op, Variable, ParamsType
from theano.gof.type import Generic

from theano.scalar import int32 as int32_t
from theano.tensor import elemwise
from theano.tensor.var import (AsTensorError, TensorVariable,
                               TensorConstant, TensorConstantSignature,
                               _tensor_py_operators)
from theano.tensor.type import TensorType, values_eq_approx_always_true
from theano.tensor.type_other import NoneConst
from theano import scalar as scal
from functools import partial
from theano import compile, printing
from theano.printing import pprint, min_informative_str
# For history
from theano.compile import Rebroadcast, Shape, shape
from theano.scalar import int32


# We use these exceptions as well.
import theano.scalar.sharedvar
from theano.gradient import grad_undefined
from theano.gradient import grad_not_implemented
from theano.gradient import DisconnectedType

# set up the external interface
from theano.tensor.elemwise import Elemwise, DimShuffle, CAReduce, Sum

import logging
_logger = logging.getLogger("theano.tensor.basic")

__docformat__ = "restructuredtext en"

# This is needed as we will hide it later
python_complex = complex
python_any = any
python_all = all

# Define common subsets of dtypes (as strings).
complex_dtypes = list(map(str, scal.complex_types))
continuous_dtypes = list(map(str, scal.continuous_types))
float_dtypes = list(map(str, scal.float_types))
integer_dtypes = list(map(str, scal.integer_types))
discrete_dtypes = list(map(str, scal.discrete_types))
all_dtypes = list(map(str, scal.all_types))
int_dtypes = list(map(str, scal.int_types))
uint_dtypes = list(map(str, scal.uint_types))


class ShapeError(Exception):
    """Raised when the shape cannot be computed."""
    pass


def check_equal_numpy(x, y):
    """
    Return True iff x and y are equal.

    Checks the dtype and shape if x and y are numpy.ndarray instances.

    """
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return (x.dtype == y.dtype and x.shape == y.shape and
                np.all(abs(x - y) < 1e-10))
    elif (isinstance(x, np.random.RandomState) and
          isinstance(y, np.random.RandomState)):
        return python_all(np.all(a == b) for a, b in
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


def as_tensor_variable(x, name=None, ndim=None):
    """Return `x`, transformed into a `TensorType`.

    This function is often used by `make_node` methods of `Op` subclasses
    to turn ndarrays, numbers, `Scalar` instances, `Apply` instances and
    `TensorType` instances into valid input list elements.

    Parameters
    ----------
    x : Apply instance, Variable instance, numpy.ndarray, or number
        This thing will be transformed into a `Variable` in a sensible way. An
        ndarray argument will not be copied, but a list of numbers will be
        copied to make an ndarray.
    name : str or None
        If a new `Variable` instance is created, it will be named with this
        string.
    ndim : None or integer
        Return a Variable with this many dimensions.

    Raises
    ------
    ValueError
        If an `Apply` with more than one output is fetched or
        if `x` cannot be made into a Variable with `ndim` dimensions.
    AsTensorError
        If `x` cannot be converted to a TensorType Variable.

    """
    if hasattr(x, '_as_TensorVariable'):
        return x._as_TensorVariable()  # TODO: pass name and ndim arguments

    if isinstance(x, gof.Apply):
        # use Apply's default output mechanism
        if (x.op.default_output is None) and (len(x.outputs) != 1):
            raise ValueError(
                "It is ambiguous which output of a multi-output Op has"
                " to be fetched.", x)

        x = x.default_output()
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
                # strip off leading broadcastable dimensions
                first_non_broadcastable = [idx for idx in xrange(x.ndim)
                                           if not x.broadcastable[idx]][0]
                x = x.dimshuffle(list(range(x.ndim))[first_non_broadcastable:])
                if x.ndim > ndim:
                    raise ValueError(
                        'TensorType could not be cast to have %i dimensions'
                        % ndim, x.type
                    )
                return x
            elif (x.type.ndim < ndim):
                return shape_padleft(x, n_ones=(ndim - x.type.ndim))
            else:
                return x
    if isinstance(x, (tuple, list)) and python_any(isinstance(xi, Variable)
                                                   for xi in x):
        try:
            return stack(x)
        except (TypeError, ValueError):
            pass

    if isinstance(x, bool):
        raise AsTensorError(
            "Cannot cast True or False as a tensor variable. Please use "
            "np.array(True) or np.array(False) if you need these constants. "
            "This error might be caused by using the == operator on "
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


def constant(x, name=None, ndim=None, dtype=None):
    """Return a symbolic `Constant` with value `x`.

    Raises
    ------
    TypeError
        `x` could not be converted to a numpy.ndarray.
    ValueError
        `x` could not be expanded to have ndim dimensions.

    Note
    ----
    We create a small cache of frequently used constant.
    This speed up the Merge optimization for big graph.
    We want to cache all scalar to don't merge as frequently constants.
    But we don't want to cache too much stuff.
    So we cache integer with dtype [u]int and float where the value is
    between -10 and 10.
    We cache all broadcast pattern for scalar.

    """
    x_ = scal.convert(x, dtype=dtype)

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
        ttype = TensorType(dtype=x_.dtype, broadcastable=bcastable)
        if not constant.enable:
            return TensorConstant(ttype, x_, name=name)

        sig = TensorConstantSignature((ttype, x_))
        if sig in constant_cache:
            return constant_cache[sig]

        ret = TensorConstant(ttype, x_, name=name)
        if (x_.size == 1 and
            (-10) <= x_ <= 10 and
            (x_.dtype in int_dtypes or x_.dtype in uint_dtypes or
             (x_.dtype in float_dtypes and
              # Limit the size of the cache.
              len(constant_cache) < 10000))):
            constant_cache[sig] = ret
            # This is needed to raise a good error to the user.
            ret.cached = True
        return ret
    except Exception:
        raise TypeError("Could not convert %s to TensorType" % x, type(x))


constant.enable = True
constant_cache = {}


def _obj_is_wrappable_as_tensor(x):
    try:
        constant(x)
        return True
    except TypeError:
        return False


if int(config.tensor.cmp_sloppy) > 1:
    # This config variable is a quick-and-dirty way to get low-precision
    # comparisons.  For a more precise setting of these tolerances set
    # them explicitly in your user code by assigning, for example,
    # "theano.tensor.basic.float32_atol = ..."

    # When config.tensor.cmp_sloppy>1 we are even more sloppy. This is
    # useful to test the GPU as they don't use extended precision and
    # this cause some difference bigger then the normal sloppy.
    float16_atol = 1e-2
    float16_rtol = 5e-2

    float32_atol = 5e-4
    float32_rtol = 1e-3

    float64_rtol = 1e-4
    float64_atol = 1e-3
elif int(config.tensor.cmp_sloppy):
    float16_atol = 5e-3
    float16_rtol = 1e-2

    float32_atol = 1e-4
    float32_rtol = 1e-3

    float64_rtol = 1e-4
    float64_atol = 1e-3
else:
    # If you change those value in test don't forget to put them back
    # when the test end.  Don't forget the case when the test fail.
    float16_atol = 1e-3
    float16_rtol = 1e-3

    float32_atol = 1e-5
    float32_rtol = 1e-5

    # defaults in numpy.allclose
    # Don't be more strict then numpy rtol
    # It cause useless error.
    float64_rtol = 1.0000000000000001e-05
    float64_atol = 1e-8


def _get_atol_rtol(a, b):
    tiny = ('float16',)
    narrow = ('float32', 'complex64')
    if (str(a.dtype) in tiny) or (str(b.dtype) in tiny):
        atol = float16_atol
        rtol = float16_rtol
    elif (str(a.dtype) in narrow) or (str(b.dtype) in narrow):
        atol = float32_atol
        rtol = float32_rtol
    else:
        atol = float64_atol
        rtol = float64_rtol
    return atol, rtol


def _allclose(a, b, rtol=None, atol=None):
    a = np.asarray(a)
    b = np.asarray(b)
    atol_, rtol_ = _get_atol_rtol(a, b)
    if rtol is not None:
        rtol_ = rtol
    if atol is not None:
        atol_ = atol

    return np.allclose(a, b, atol=atol_, rtol=rtol_)


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


def numpy_scalar(data):
    """ Return a scalar stored in a numpy ndarray.

    Raises
    ------
     NotScalarConstantError
        If the numpy ndarray is not a scalar.

    """

    # handle case where data is numpy.array([])
    if (data.ndim > 0 and
        (len(data.shape) == 0 or
         builtins.max(data.shape) == 0)):
        assert np.all(np.array([]) == data)
        raise EmptyConstantError()
    try:
        np.complex(data)  # works for all numeric scalars
        return data
    except Exception:
        raise NotScalarConstantError(
            'v.data is non-numeric, non-scalar, or has more than one'
            ' unique value', data)


get_scalar_constant_value_elemwises = (
    scal.Cast, scal.Switch,
    scal.NEQ, scal.EQ,
    scal.LT, scal.GT, scal.LE, scal.GE,
    scal.Sub, scal.Add, scal.Mod, scal.Mul,
    scal.IntDiv, scal.TrueDiv, scal.Minimum, scal.Maximum)


def get_scalar_constant_value(orig_v, elemwise=True,
                              only_process_constants=False,
                              max_recur=10):
    """Return the constant scalar(0-D) value underlying variable `v`.

    If `v` is the output of dimshuffles, fills, allocs, rebroadcasts,
    cast, OutputGuard, DeepCopyOp, ScalarFromTensor, ScalarOp, Elemwise
    and some pattern with Subtensor, this function digs through them.

    If `v` is not some view of constant scalar data, then raise a
    NotScalarConstantError.

    Parameters
    ----------
    elemwise : bool
        If False, we won't try to go into elemwise. So this call is faster.
        But we still investigate in Second Elemwise (as this is a substitute
        for Alloc)
    only_process_constants : bool
        If True, we only attempt to obtain the value of `orig_v` if it's
        directly constant and don't try to dig through dimshuffles, fills,
        allocs, and other to figure out its value.
    max_recur : int
        The maximum number of recursion.

    Notes
    -----
        There may be another function similar to this one in the code,
        but I'm not sure where it is.

    """
    v = orig_v
    while True:
        if v is None:
            # None is not a scalar (and many uses of this function seem
            # to depend on passing it None)
            raise NotScalarConstantError()

        if isinstance(v, (np.integer, integer_types, float)):
            return np.asarray(v)

        if isinstance(v, np.ndarray):
            return numpy_scalar(v).copy()

        if isinstance(v, Constant):
            if getattr(v.tag, 'unique_value', None) is not None:
                data = v.tag.unique_value
            else:
                data = v.data
            return numpy_scalar(data).copy()

        if (not only_process_constants and
                getattr(v, 'owner', None) and
                max_recur > 0):
            max_recur -= 1
            if isinstance(v.owner.op, (Alloc, DimShuffle, Rebroadcast,
                                       # outputguard is only used in debugmode but we
                                       # keep it here to avoid problems with old pickels.
                                       compile.ops.OutputGuard,
                                       compile.DeepCopyOp)):
                v = v.owner.inputs[0]
                continue
            elif isinstance(v.owner.op, theano.compile.ops.Shape_i):
                i = v.owner.op.i
                inp = v.owner.inputs[0]
                if isinstance(inp, Constant):
                    return np.asarray(inp.data.shape[i])
                # The shape of a broadcastable dimension is 1
                if (hasattr(inp.type, 'broadcastable') and
                        inp.type.broadcastable[i]):
                    return np.asarray(1)

            # Don't act as the constant_folding optimization here as this
            # fct is used too early in the optimization phase.  This would
            # mess with the stabilization optimization and be too slow.
            # We put all the scalar Ops used by get_canonical_form_slice()
            # to allow it to determine the broadcast pattern correctly.
            elif isinstance(v.owner.op, (ScalarFromTensor, TensorFromScalar)):
                v = v.owner.inputs[0]
                continue
            elif isinstance(v.owner.op, theano.tensor.opt.Assert):
                # check if all conditions are constant and true
                cond = [get_scalar_constant_value(c, max_recur=max_recur)
                        for c in v.owner.inputs[1:]]
                if builtins.all([0 == c.ndim and c != 0 for c in cond]):
                    v = v.owner.inputs[0]
                    continue
            elif isinstance(v.owner.op, scal.ScalarOp):
                if isinstance(v.owner.op, scal.Second):
                    # We don't need both input to be constant for second
                    shp, val = v.owner.inputs
                    v = val
                    continue
                if isinstance(v.owner.op, get_scalar_constant_value_elemwises):
                    const = [get_scalar_constant_value(i, max_recur=max_recur)
                             for i in v.owner.inputs]
                    ret = [[None]]
                    v.owner.op.perform(v.owner, const, ret)
                    return ret[0][0].copy()
            # In fast_compile, we don't enable local_fill_to_alloc, so
            # we need to investigate Second as Alloc. So elemwise
            # don't disable the check for Second.
            elif isinstance(v.owner.op, Elemwise):
                if isinstance(v.owner.op.scalar_op, scal.Second):
                    # We don't need both input to be constant for second
                    shp, val = v.owner.inputs
                    v = val
                    continue
                elif elemwise and isinstance(
                        v.owner.op.scalar_op,
                        get_scalar_constant_value_elemwises):
                    const = [get_scalar_constant_value(i, max_recur=max_recur)
                             for i in v.owner.inputs]
                    ret = [[None]]
                    v.owner.op.perform(v.owner, const, ret)
                    return ret[0][0].copy()
            elif (isinstance(v.owner.op, theano.tensor.subtensor.Subtensor) and
                  v.ndim == 0):
                if isinstance(v.owner.inputs[0], TensorConstant):
                    cdata = tuple(v.owner.op.get_constant_idx(v.owner.inputs))
                    try:
                        return v.owner.inputs[0].data.__getitem__(cdata).copy()
                    except IndexError:
                        raise IndexError(
                            str(tuple(v.owner.op.idx_list)) +
                            " is not a valid index into " +
                            str(v.owner.inputs[0].data))

                # The index list 'idx_list' should have length the same
                # shape as the input.
                # TODO: implement the case where we take a scalar in a matrix
                assert len(v.owner.op.idx_list) == v.owner.inputs[0].ndim

                # Needed to make better graph in this test in
                # theano/tensor/tests/test_sharedvar.py:
                # test_shared_options.test_specify_shape_partial
                if ((v.owner.inputs[0].owner and
                     isinstance(v.owner.inputs[0].owner.op, Join) and
                     len(v.owner.op.idx_list) == 1)):
                    # Ensure the Join is joining only scalar variables (so that
                    # the constant value can be found at the same index as the
                    # one used in the sub-tensor).
                    if python_all(var.ndim == 0 for var in
                                  v.owner.inputs[0].owner.inputs[1:]):
                        idx = v.owner.op.idx_list[0]
                        if isinstance(idx, gof.Type):
                            idx = get_scalar_constant_value(v.owner.inputs[1],
                                                            max_recur=max_recur)
                        # Note the '+ 1' is because the first argument to Join
                        # is the axis.
                        ret = v.owner.inputs[0].owner.inputs[idx + 1]
                        ret = get_scalar_constant_value(ret, max_recur=max_recur)
                        # join can cast implicitly its input in some case.
                        return theano._asarray(ret, dtype=v.type.dtype)
                    if python_all(var.ndim == 1 for var in
                                  v.owner.inputs[0].owner.inputs[1:]):
                        idx = v.owner.op.idx_list[0]
                        if isinstance(idx, gof.Type):
                            idx = get_scalar_constant_value(v.owner.inputs[1],
                                                            max_recur=max_recur)
                        try:
                            # TODO: assert joined axis is 0.
                            length = 0
                            loop = False
                            for joined in v.owner.inputs[0].owner.inputs[1:]:
                                ll = get_vector_length(joined)
                                if idx < length + ll:
                                    v = joined[idx - length]
                                    loop = True
                                    break
                                length += ll
                            if loop:
                                continue
                        except TypeError:
                            pass
                        except ValueError:
                            pass

                elif (v.owner.inputs[0].owner and
                      isinstance(v.owner.inputs[0].owner.op,
                                 theano.tensor.opt.MakeVector) and
                      # MakeVector normally accept only scalar as input.
                      # We put this check in case there is change in the future
                      python_all(var.ndim == 0 for var in
                                 v.owner.inputs[0].owner.inputs) and
                      len(v.owner.op.idx_list) == 1):

                    idx = v.owner.op.idx_list[0]
                    if isinstance(idx, gof.Type):
                        idx = get_scalar_constant_value(v.owner.inputs[1],
                                                        max_recur=max_recur)
                    # Python 2.4 does not support indexing with numpy.integer
                    # So we cast it.
                    idx = int(idx)
                    ret = v.owner.inputs[0].owner.inputs[idx]
                    ret = get_scalar_constant_value(ret, max_recur=max_recur)
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
                    if isinstance(idx, gof.Type):
                        idx = get_scalar_constant_value(owner.inputs[1],
                                                        max_recur=max_recur)
                    grandparent = leftmost_parent.owner.inputs[0]
                    gp_broadcastable = grandparent.type.broadcastable
                    ndim = grandparent.type.ndim
                    if grandparent.owner and isinstance(grandparent.owner.op,
                                                        Rebroadcast):
                        ggp_broadcastable = grandparent.owner.inputs[0].broadcastable
                        l = [b1 or b2 for b1, b2 in zip(ggp_broadcastable,
                                                        gp_broadcastable)]
                        gp_broadcastable = tuple(l)

                    assert ndim == len(gp_broadcastable)

                    if not (idx < len(gp_broadcastable)):
                        msg = ("get_scalar_constant_value detected " +
                               "deterministic IndexError: x.shape[%d] " +
                               "when x.ndim=%d.") % (idx, ndim)
                        if config.exception_verbosity == 'high':
                            msg += ' x=%s' % min_informative_str(v)
                        else:
                            msg += ' x=%s' % str(v)
                        raise ValueError(msg)

                    if gp_broadcastable[idx]:
                        return np.asarray(1)

        raise NotScalarConstantError(v)


# Easy constructors

def tensor(*args, **kwargs):
    name = kwargs.pop('name', None)
    return TensorType(*args, **kwargs)(name=name)


def _multi(*fns):
    def f2(f, *names):
        if names and isinstance(names[0], integer_types):
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

    Parameters
    ----------
    dtype: numeric
        None means to use theano.config.floatX.
    name
        A name to attach to this variable.

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

    Parameters
    ----------
    dtype: numeric
        None means to use theano.config.floatX.
    name
        A name to attach to this variable

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

    Parameters
    ----------
    dtype: numeric
        None means to use theano.config.floatX.
    name
        A name to attach to this variable.

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

    Parameters
    ----------
    dtype: numeric type
        None means to use theano.config.floatX.
    name
        A name to attach to this variable.

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

    Parameters
    ----------
    dtype : numeric
        None means to use theano.config.floatX.
    name
        A name to attach to this variable.

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

    Parameters
    ----------
    dtype: numeric type
        None means to use theano.config.floatX.
    name
        A name to attach to this variable.

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

    Parameters
    ----------
    dtype: numeric type
        None means to use theano.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, False, False, False))
    return type(name)
tensor4s, ftensor4s, dtensor4s, itensor4s, ltensor4s = _multi(
    tensor4, ftensor4, dtensor4, itensor4, ltensor4)

ctensor5 = TensorType('complex64', ((False,) * 5))
ztensor5 = TensorType('complex128', ((False,) * 5))
ftensor5 = TensorType('float32', ((False,) * 5))
dtensor5 = TensorType('float64', ((False,) * 5))
btensor5 = TensorType('int8', ((False,) * 5))
wtensor5 = TensorType('int16', ((False,) * 5))
itensor5 = TensorType('int32', ((False,) * 5))
ltensor5 = TensorType('int64', ((False,) * 5))


def tensor5(name=None, dtype=None):
    """Return a symbolic 5-D variable.

    Parameters
    ----------
    dtype: numeric type
        None means to use theano.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, False, False, False, False))
    return type(name)
tensor5s, ftensor5s, dtensor5s, itensor5s, ltensor5s = _multi(
    tensor5, ftensor5, dtensor5, itensor5, ltensor5)

ctensor6 = TensorType('complex64', ((False,) * 6))
ztensor6 = TensorType('complex128', ((False,) * 6))
ftensor6 = TensorType('float32', ((False,) * 6))
dtensor6 = TensorType('float64', ((False,) * 6))
btensor6 = TensorType('int8', ((False,) * 6))
wtensor6 = TensorType('int16', ((False,) * 6))
itensor6 = TensorType('int32', ((False,) * 6))
ltensor6 = TensorType('int64', ((False,) * 6))


def tensor6(name=None, dtype=None):
    """Return a symbolic 6-D variable.

    Parameters
    ----------
    dtype: numeric type
        None means to use theano.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False,) * 6)
    return type(name)
tensor6s, ftensor6s, dtensor6s, itensor6s, ltensor6s = _multi(
    tensor6, ftensor6, dtensor6, itensor6, ltensor6)

ctensor7 = TensorType('complex64', ((False,) * 7))
ztensor7 = TensorType('complex128', ((False,) * 7))
ftensor7 = TensorType('float32', ((False,) * 7))
dtensor7 = TensorType('float64', ((False,) * 7))
btensor7 = TensorType('int8', ((False,) * 7))
wtensor7 = TensorType('int16', ((False,) * 7))
itensor7 = TensorType('int32', ((False,) * 7))
ltensor7 = TensorType('int64', ((False,) * 7))


def tensor7(name=None, dtype=None):
    """Return a symbolic 7-D variable.

    Parameters
    ----------
    dtype: numeric type
        None means to use theano.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False,) * 7)
    return type(name)
tensor7s, ftensor7s, dtensor7s, itensor7s, ltensor7s = _multi(
    tensor7, ftensor7, dtensor7, itensor7, ltensor7)


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


def _pack(x):
    """
    Convert x to a list if it is an iterable, otherwise wrap it in a list.
    """
    try:
        return list(x)
    except TypeError:
        return [x]


def check_and_normalize_axes(x, axis):
    """
    Check axes, normalize and convert them to a Python list of integers.
    Return an empty list if argument is None.

    Parameters
    ----------
    x: Tensor variable
    axis = Integer, tuple or list of integers

    Returns
    -------
    axis: list of integers
    """
    x = as_tensor_variable(x)
    if axis is None:
        axis = []
    elif (isinstance(axis, (integer_types, np.integer)) or
            (isinstance(axis, np.ndarray) and axis.ndim == 0)):
                axis = [int(axis)]
    elif isinstance(axis, (tuple, list, np.ndarray)):
        axis = [int(i) for i in axis]
    elif isinstance(axis, Variable):
        if NoneConst.equals(axis):
            axis = []
        elif not isinstance(axis, TensorConstant):
            raise TypeError("Computation needs a constant axis. Got %s" % axis)
        else:
            assert axis.dtype in integer_dtypes
            if (isinstance(axis.data, (integer_types, np.integer)) or
                    (isinstance(axis.data, np.ndarray) and axis.data.ndim == 0)):
                        axis = [int(axis.data)]
            elif isinstance(axis.data, (list, np.ndarray)):
                axis = [int(i) for i in axis.data]
    else:
        raise TypeError("Axis must be an integer, tuple, list of integers or a TensorVariable. Got %s" % axis)
    if len(axis) > 0:
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] += x.type.ndim
            if axis[i] < 0 or axis[i] >= x.type.ndim:
                raise ValueError("Computation needs a valid axis number for %d-D tensor. Got %d" % (x.type.ndim, axis[i]))
        axis = list(set(axis))
        axis.sort()
    return axis


#########################
# Casting Operations
#########################

class TensorFromScalar(Op):

    __props__ = ()

    def make_node(self, s):
        assert isinstance(s.type, scal.Scalar)
        return Apply(self,
                     [s],
                     [tensor(dtype=s.type.dtype,
                             broadcastable=())])

    def perform(self, node, inp, out_):
        s, = inp
        out, = out_
        out[0] = np.asarray(s)

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

tensor_from_scalar = TensorFromScalar()


class ScalarFromTensor(Op):

    __props__ = ()

    def make_node(self, t):
        assert isinstance(t.type, TensorType)
        assert t.type.broadcastable == ()
        return Apply(self,
                     [t],
                     [scal.get_scalar_type(dtype=t.type.dtype).make_variable()]
                     )

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
# -JB 20080924
def _conversion(real_value, name):
    __oplist_tag(real_value, 'casting')
    real_value.__module__ = 'tensor.basic'
    pprint.assign(real_value, printing.FunctionPrinter(name))
    return real_value


# These _conver_to_<type> functions have leading underscores to indicate that
# they should not be called directly.  They do not perform sanity checks about
# what types you are casting to what.  That logic is implemented by the
# `cast()` function below.

_convert_to_bool = _conversion(
    elemwise.Elemwise(scal.convert_to_bool), 'bool')
"""Cast to boolean"""

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

_convert_to_float16 = _conversion(
    elemwise.Elemwise(scal.convert_to_float16), 'float16')
"""Cast to half-precision floating point"""

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
    'bool': _convert_to_bool,
    'int8': _convert_to_int8,
    'int16': _convert_to_int16,
    'int32': _convert_to_int32,
    'int64': _convert_to_int64,
    'uint8': _convert_to_uint8,
    'uint16': _convert_to_uint16,
    'uint32': _convert_to_uint32,
    'uint64': _convert_to_uint64,
    'float16': _convert_to_float16,
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


class MaxAndArgmax(Op):
    """
    Calculate the max and argmax over a given axis or over all axes.

    """
    nin = 2  # tensor, axis
    nout = 2  # max val, max idx
    E_axis = 'invalid axis'
    params_type = Generic()
    __props__ = ('axis',)
    _f16_ok = True

    def __init__(self, axis):
        assert isinstance(axis, list)
        self.axis = tuple(axis)

    def get_params(self, node):
        return self.axis

    def make_node(self, x):
        x = _as_tensor_variable(x)

        # We keep the original broadcastable flags for dimensions on which
        # we do not perform the max / argmax.
        all_axes = set(self.axis)
        broadcastable = [b for i, b in enumerate(x.type.broadcastable)
                         if i not in all_axes]
        inputs = [x]
        outputs = [tensor(x.type.dtype, broadcastable, name='max'),
                   tensor('int64', broadcastable, name='argmax')]
        return Apply(self, inputs, outputs)

    def perform(self, node, inp, outs, params):
        x = inp[0]
        axes = params
        max, max_idx = outs
        if axes is None:
            axes = tuple(range(x.ndim))
        else:
            axes = tuple(int(ax) for ax in axes)
        max[0] = theano._asarray(np.max(x, axes),
                                 dtype=node.outputs[0].dtype)
        # Numpy does not support multiple axes for argmax
        # Work around
        keep_axes = np.array([i for i in range(x.ndim) if i not in axes],
                             dtype='int64')
        # Not-reduced axes in front
        transposed_x = np.transpose(x, np.concatenate((keep_axes, axes)))
        kept_shape = transposed_x.shape[:len(keep_axes)]
        reduced_shape = transposed_x.shape[len(keep_axes):]

        # Numpy.prod returns 1.0 when arg is empty, so we cast it to int64
        # Otherwise reshape would complain citing float arg
        new_shape = kept_shape + (np.prod(reduced_shape, dtype='int64'),)
        reshaped_x = transposed_x.reshape(new_shape)

        max_idx[0] = theano._asarray(np.argmax(reshaped_x, axis=-1),
                                     dtype='int64')

    def c_code(self, node, name, inp, out, sub):
        if len(self.axis) != 1 and len(self.axis) != node.inputs[0].ndim:
            raise NotImplementedError("NumPy C-API can compute max and argmax only for 1 axis or for all axes.")
        x = inp[0]
        axis = sub['params']
        max, argmax = out
        fail = sub["fail"]
        ret = """
        #if PY_MAJOR_VERSION >= 3
            #ifndef PyInt_AS_LONG
                #define PyInt_AS_LONG PyLong_AS_LONG
            #endif
        #endif

        int axis;

        if (PyTuple_GET_SIZE(%(axis)s) == PyArray_NDIM(%(x)s)) {
            axis = NPY_MAXDIMS;
        } else if(PyTuple_GET_SIZE(%(axis)s) == 1) {
            PyObject* axis_object = PyTuple_GET_ITEM(%(axis)s, 0);
            axis = (int)PyInt_AS_LONG(axis_object);
            if (axis > PyArray_NDIM(%(x)s)-1 || axis < -PyArray_NDIM(%(x)s)) {
                PyErr_SetString(PyExc_ValueError,
                "MaxAndArgmax: bad axis argument");
                %(fail)s
            }
        } else {
            PyErr_SetString(PyExc_NotImplementedError,
            "MaxAndArgmax: NumPy C-API can compute max and argmax only for 1 axis or for all axes.");
            %(fail)s
        }

        Py_CLEAR(%(max)s);
        Py_CLEAR(%(argmax)s);//todo pass them as out parameter.

        %(max)s = (PyArrayObject*)PyArray_Max(%(x)s, axis, NULL);
        if (%(max)s == NULL) {
            %(fail)s;
        }
        if (!PyArray_CheckExact(%(max)s)) {
            %(max)s = (PyArrayObject*)PyArray_FromAny((PyObject*)%(max)s, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
            if(%(max)s == NULL){
                %(fail)s;
            }
        }

        %(argmax)s = (PyArrayObject*)PyArray_ArgMax(%(x)s, axis, NULL);
        if (%(argmax)s == NULL) {
            Py_CLEAR(%(max)s);
            %(fail)s;
        }
        if (!PyArray_CheckExact(%(argmax)s)) {
            %(argmax)s = (PyArrayObject*)PyArray_FromAny((PyObject*)%(argmax)s, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
            if(%(argmax)s == NULL){
                %(fail)s;
            }
        }
        if (PyArray_TYPE(%(argmax)s) != NPY_INT64) {
            PyObject * tmp = PyArray_Cast(%(argmax)s, NPY_INT64);
            if (NULL == tmp){
                %(fail)s;
            }
            Py_DECREF(%(argmax)s);
            %(argmax)s = (PyArrayObject*)tmp;
        }
        """
        return ret % locals()

    def c_code_cache_version(self):
        return (5,)

    def infer_shape(self, node, shapes):
        ishape = shapes[0]
        rval = tuple(ishape[i] for (i, b) in enumerate(
            node.inputs[0].type.broadcastable) if i not in self.axis)
        return [rval, rval]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None, None]
        if len(self.axis) != 1:
            raise ValueError(('R_op supported for arg_max only for '
                              'one axis!'))
        if self.axis[0] > 1:
            raise ValueError(('R_op supported for arg_max only when '
                              ' axis is 0 or 1'))
        if inputs[0].ndim != 2:
            raise ValueError(('R_op supported for arg_max only when '
                              ' input is a matrix'))
        max_vals, max_pos = self.make_node(*inputs).outputs
        if self.axis[0] == 0:
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
        # (x, y), (gz, gw)
        # gz*dz/dx + gw*dw/dx, gz*dz/dy + gw*dw/dy
        # gMax * dMax/dx + gArgMax * dArgMax/dx,
        # gMax * dMax/daxis + gArgMax * dArgMax/daxis
        # g_max has one less dimension than x, so you need to complete
        # g_max to x's shape when axis=0 the broadcasting mechanism
        # does it automatically
        x = inp[0]
        axis = _as_tensor_variable(self.axis)
        g_max, g_max_idx = grads

        g_max_disconnected = isinstance(g_max.type, DisconnectedType)
        g_max_idx_disconnected = isinstance(g_max_idx.type, DisconnectedType)

        # if the op is totally disconnected, so are its inputs
        if g_max_disconnected and g_max_idx_disconnected:
            return [DisconnectedType()(), DisconnectedType()()]

        # if the max is disconnected but the argmax is not,
        # the gradient on its inputs is zero
        if g_max_disconnected:
            return [x.zeros_like()]
        if NoneConst.equals(axis):
            axis_ = list(range(x.ndim))
        else:
            axis_ = axis
        xmax = max(x, axis_)

        # Raise the g_max and xmax to the same number of dim as the input.
        pattern = []
        out_dim = 0
        if NoneConst.equals(axis):
            # We are taking the max/argmax over all dimensions.
            axis = None
        for i in xrange(x.ndim):
            if axis is None or i in axis.data:
                pattern.append('x')
            else:
                pattern.append(out_dim)
                out_dim += 1
        g_max_pad = DimShuffle(g_max.broadcastable, pattern)(g_max)
        xmax_pad = DimShuffle(xmax.broadcastable, pattern)(xmax)

        # Set the grad to the correct position.
        g_x = eq(xmax_pad, x) * g_max_pad
        return g_x,


class Argmax(Op):
    """
    Calculate the argmax over a given axis or over all axes.
    """
    nin = 2  # tensor, axis
    nout = 1
    E_axis = 'invalid axis'
    __props__ = ('axis',)
    _f16_ok = True

    params_type = ParamsType(c_axis=scal.int64)

    def __init__(self, axis):
        if axis is not None:
            axis = tuple(axis)
        self.axis = tuple(axis)

    def get_params(self, node):
        if self.axis is not None and len(self.axis) == 1:
            c_axis = np.int64(self.axis[0])
        else:
            # The value here doesn't matter, it won't be used
            c_axis = np.int64(-1)
        return self.params_type.get_params(c_axis=c_axis)

    def make_node(self, x, axis=None):
        x = _as_tensor_variable(x)
        if self.axis is None:
            all_axes = list(range(x.ndim))
        else:
            all_axes = self.axis
        inputs = [x]

        # We keep the original broadcastable flags for dimensions on which
        # we do not perform the argmax.
        broadcastable = [b for i, b in enumerate(x.type.broadcastable)
                         if i not in all_axes]
        outputs = [tensor('int64', broadcastable, name='argmax')]
        return Apply(self, inputs, outputs)

    def prepare_node(self, node, storage_map, compute_map, impl):
        if len(node.inputs) == 2:
            raise ValueError('You are trying to compile a graph with an old Argmax node.  Either reoptimize your graph or rebuild it to get the new node format.')

    def perform(self, node, inp, outs, params):
        x, = inp
        axes = self.axis
        max_idx, = outs
        if axes is None:
            axes = tuple(range(x.ndim))

        # Numpy does not support multiple axes for argmax
        # Work around
        keep_axes = np.array([i for i in range(x.ndim) if i not in axes],
                             dtype='int64')
        # Not-reduced axes in front
        transposed_x = np.transpose(x, np.concatenate((keep_axes,
                                                       axes)))
        kept_shape = transposed_x.shape[:len(keep_axes)]
        reduced_shape = transposed_x.shape[len(keep_axes):]
        new_shape = kept_shape + (np.prod(reduced_shape),)
        reshaped_x = transposed_x.reshape(new_shape)

        max_idx[0] = theano._asarray(np.argmax(reshaped_x, axis=-1),
                                     dtype='int64')

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        argmax, = out
        fail = sub["fail"]
        params = sub["params"]
        if self.axis is None:
            axis_code = "axis = NPY_MAXDIMS;"
        else:
            if len(self.axis) > 1:
                raise NotImplementedError()
            # params is only used here for now
            axis_code = """
            axis = %(params)s->c_axis;
            if(axis > PyArray_NDIM(%(x)s)-1 || axis < -PyArray_NDIM(%(x)s)){
                PyErr_SetString(PyExc_ValueError,
                "Argmax, bad axis argument");
                %(fail)s
            }
            """ % locals()
        ret = """
        int axis;

        Py_CLEAR(%(argmax)s);//todo pass them as out parameter.
        %(axis_code)s

        %(argmax)s = (PyArrayObject*)PyArray_ArgMax(%(x)s, axis, NULL);
        if(%(argmax)s == NULL){
            %(fail)s;
        }
        if(!PyArray_CheckExact(%(argmax)s)){
            %(argmax)s = (PyArrayObject*)PyArray_FromAny((PyObject*)%(argmax)s, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
            if(%(argmax)s == NULL){
                %(fail)s;
            }
        }
        if(PyArray_TYPE(%(argmax)s) != NPY_INT64){
            PyObject * tmp = PyArray_Cast(%(argmax)s, NPY_INT64);
            if (NULL == tmp){
                %(fail)s;
            }
            Py_DECREF(%(argmax)s);
            %(argmax)s = (PyArrayObject*)tmp;
        }
        """
        return ret % locals()

    def c_code_cache_version(self):
        return (1,)

    def infer_shape(self, node, shapes):
        ishape, = shapes
        if self.axis is None:
            return [()]
        rval = tuple([ishape[i] for (i, b) in enumerate(
            node.inputs[0].type.broadcastable) if i not in self.axis])
        return [rval]

    def grad(self, inp, grads):
        x, = inp

        return [x.zeros_like()]


def makeKeepDims(x, y, axis):
    """
    Reintroduces in y with length one the axes of x which have been left out
    in a prior reduction of x. With this option, the resulting tensor will
    broadcast correctly against the original tensor x.

    """
    x = as_tensor_variable(x)
    y = as_tensor_variable(y)

    if axis is None:
        axis = list(range(x.type.ndim))
    elif isinstance(axis, (integer_types, np.integer)):
        axis = [axis]
    elif isinstance(axis, np.ndarray) and axis.ndim == 0:
        axis = [int(axis)]
    else:
        axis = [int(a) for a in axis]
    newaxis = []
    for a in axis:
        if not isinstance(a, integer_types):
            raise ValueError(
                "keepdims option can be used only with constant axis")
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
    given axis.

    When axis is None (the default value), the max is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims : bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """
    # Check axis and convert it to a Python list of integers.
    # Axis will be used as an op param of MaxAndArgmax.
    a = as_tensor_variable(a)
    axis = check_and_normalize_axes(a, axis)
    if len(axis) == 0:
        axis = list(range(a.type.ndim))
    out, argout = MaxAndArgmax(axis)(a)

    if keepdims:
        out = makeKeepDims(a, out, axis)
        argout = makeKeepDims(a, argout, axis)
    return [out, argout]


@constructor
def max(x, axis=None, keepdims=False):
    """
    Returns maximum elements obtained by iterating over given axis.

    When axis is None (the default value), the max is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    Notes
    -----
    We return an error as numpy when we reduce a dim with a shape of 0.

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
    Returns indices of maximum elements obtained by iterating over given axis.

    When axis is None (the default value), the argmax is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims : bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """
    argout = max_and_argmax(x, axis)[1]

    if keepdims:
        argout = makeKeepDims(x, argout, axis)
    return argout


@constructor
def min(x, axis=None, keepdims=False):
    """
    Returns minimum elements obtained by iterating over given axis.

    When axis is None (the default value), the min is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """
    x = as_tensor_variable(x)
    str_x_type = str(x.dtype)
    if str_x_type.startswith('float') or str_x_type in int_dtypes:
        return -max(-x, axis=axis, keepdims=keepdims)
    elif str_x_type in uint_dtypes:
        itype = np.iinfo(x.dtype)
        max_val = np.array(itype.max, dtype=itype.dtype)
        return max_val - max(max_val - x, axis=axis, keepdims=keepdims)
    elif str_x_type == 'bool':
        return ~max(~x, axis=axis, keepdims=keepdims)
    else:
        # Be careful about unsigned integers, complex
        raise NotImplementedError()


@constructor
def argmin(x, axis=None, keepdims=False):
    """
    Returns indices of minimum elements obtained by iterating over given axis.

    When axis is None (the default value), the argmin is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """
    x = as_tensor_variable(x)
    str_x_type = str(x.dtype)
    if str_x_type.startswith('float') or str_x_type in int_dtypes:
        return argmax(-x, axis=axis, keepdims=keepdims)
    elif str_x_type in uint_dtypes:
        itype = np.iinfo(x.dtype)
        return argmax(itype.max - x, axis=axis, keepdims=keepdims)
    elif str_x_type == 'bool':
        return argmax(~x, axis=axis, keepdims=keepdims)
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
        return min(stack(args), axis=0)


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
        return max(stack(args), axis=0)


##########################
# Comparison
##########################

@_scal_elemwise
def lt(a, b):
    """a < b"""


@_scal_elemwise
def gt(a, b):
    """a > b"""


@_scal_elemwise
def le(a, b):
    """a <= b"""


@_scal_elemwise
def ge(a, b):
    """a >= b"""


@_scal_elemwise
def eq(a, b):
    """a == b"""


@_scal_elemwise
def neq(a, b):
    """a != b"""


@_scal_elemwise
def isnan(a):
    """isnan(a)"""

# Rename isnan to isnan_ to allow to bypass it when not needed.
# glibc 2.23 don't allow isnan on int, so we remove it from the graph.
isnan_ = isnan


def isnan(a):
    """isnan(a)"""
    a = as_tensor_variable(a)
    if a.dtype in discrete_dtypes:
        return alloc(np.asarray(False, dtype="bool"),
                     *[a.shape[i] for i in range(a.ndim)])
    return isnan_(a)


@_scal_elemwise
def isinf(a):
    """isinf(a)"""


# Rename isnan to isnan_ to allow to bypass it when not needed.
# glibc 2.23 don't allow isnan on int, so we remove it from the graph.
isinf_ = isinf


def isinf(a):
    """isinf(a)"""
    a = as_tensor_variable(a)
    if a.dtype in discrete_dtypes:
        return alloc(np.asarray(False, dtype="bool"),
                     *[a.shape[i] for i in range(a.ndim)])
    return isinf_(a)


def allclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """
    Implement Numpy's ``allclose`` on tensors.

    ``absolute(a - b) <= (atol + rtol * absolute(b))``

    Parameters
    ----------
    a : tensor
        Input to compare.
    b : tensor
        Input to compare.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.
    equal_nan: bool
        Whether to consider nan's in the same place to be close.

    Returns
    -------
    bool
        A boolean value (of type int8 returned by the tensor elementwise `all`
        function) whether all elements in a and b are in the tolerance range
        defined above.

    Notes
    -----
    Not a symmetric equation. See Numpy's documentation.

    """
    return all(isclose(a, b, rtol, atol, equal_nan))


def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """
    Implements Numpy's ``isclose`` on tensors.

    The tolerance values are positive, typically very small numbers. The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    ``absolute(a - b) <= (atol + rtol * absolute(b))``

    Parameters
    ----------
    a : tensor
        Input to compare.
    b : tensor
        Input to compare.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.
    equal_nan : bool
        Whether to consider nan's in the same place to be close

    Returns
    -------
    int8
        A boolean (int8) array where two arrays are element-wise equal
        within a tolerance.

    Notes
    -----
    Not a symmetric equation. See Numpy's documentation.

    Examples
    --------
    >>> import theano
    >>> import numpy as np
    >>> a = theano._asarray([1e10, 1e-7], dtype="float64")
    >>> b = theano._asarray([1.00001e10, 1e-8], dtype="float64")
    >>> theano.tensor.isclose(a, b).eval()
    array([1, 0], dtype=int8)
    >>> a = theano._asarray([1e10, 1e-8], dtype="float64")
    >>> b = theano._asarray([1.00001e10, 1e-9], dtype="float64")
    >>> theano.tensor.isclose(a, b).eval()
    array([1, 1], dtype=int8)
    >>> a = theano._asarray([1e10, 1e-8], dtype="float64")
    >>> b = theano._asarray([1.0001e10, 1e-9], dtype="float64")
    >>> theano.tensor.isclose(a, b).eval()
    array([0, 1], dtype=int8)
    >>> a = theano._asarray([1.0, np.nan], dtype="float64")
    >>> b = theano._asarray([1.0, np.nan], dtype="float64")
    >>> theano.tensor.isclose(a, b).eval()
    array([1, 0], dtype==int8)
    >>> a = theano._asarray([1.0, np.nan], dtype="float64")
    >>> b = theano._asarray([1.0, np.nan], dtype="float64")
    >>> theano.tensor.isclose(a, b, equal_nan=True).eval()
    array([1, 1], dtype==int8)
    >>> a = theano._asarray([1.0, np.inf], dtype="float64")
    >>> b = theano._asarray([1.0, -np.inf], dtype="float64")
    >>> theano.tensor.isclose(a, b).eval()
    array([1, 0], dtype==int8)
    >>> a = theano._asarray([1.0, np.inf], dtype="float64")
    >>> b = theano._asarray([1.0, np.inf], dtype="float64")
    >>> theano.tensor.isclose(a, b).eval()
    array([1, 1], dtype==int8)

    """
    # close will be an int8 array of 1 where within tolerance
    # and 0 where not within tolerance or there was a nan or inf value.
    diff = abs(a - b)
    tolerance = atol + rtol * abs(b)
    close_prelim = le(diff, tolerance)

    a_nan = isnan(a)
    b_nan = isnan(b)
    nans = bitwise_or(a_nan, b_nan)

    a_inf = isinf(a)
    b_inf = isinf(b)
    infs = bitwise_or(a_inf, b_inf)

    nans_or_infs = bitwise_or(nans, infs)

    # close is now an array of 0's except where elements are not nan or inf
    # and are within the tolerance.
    close = bitwise_and(close_prelim, bitwise_not(nans_or_infs))

    # deal with signed inf values. this will make an array inf_eq of 0's
    # except where inf values have the same sign.
    both_infs = bitwise_and(a_inf, b_inf)
    inf_signs_eq = eq(a_inf * sgn(a), b_inf * sgn(b))
    inf_eq = bitwise_and(both_infs, inf_signs_eq)

    # now create the potential result combining close and inf_eq
    close_with_infs = bitwise_or(close, inf_eq)

    # deal with comparing nan's.
    if equal_nan:
        both_nans = bitwise_and(a_nan, b_nan)
        return bitwise_or(close_with_infs, both_nans)
    # otherwise nan's aren't considered close.
    else:
        return close_with_infs


##########################
# Condition
##########################

@_scal_elemwise
def switch(cond, ift, iff):
    """if cond then ift else iff"""

where = switch
##########################
# Bit-wise
##########################


@_scal_elemwise
def and_(a, b):
    """bitwise a & b"""
bitwise_and = and_  # numpy name for it


@_scal_elemwise
def or_(a, b):
    """bitwise a | b"""
bitwise_or = or_  # numpy name for it


@_scal_elemwise
def xor(a, b):
    """bitwise a ^ b"""
bitwise_xor = xor  # numpy name for it


@_scal_elemwise
def invert(a):
    """bitwise ~a"""
bitwise_not = invert  # numpy alias for it


##########################
# Math
##########################

@_scal_elemwise
def abs_(a):
    """|`a`|

    TensorVariable overloads the `TensorVariable.__abs__` operator so that
    this function is called when you type abs(a).

    """

pprint.assign(abs_, printing.PatternPrinter(('|%(0)s|', -1000)))


@_scal_elemwise
def exp(a):
    """e^`a`"""


@_scal_elemwise
def exp2(a):
    """2^`a`"""


@_scal_elemwise
def expm1(a):
    """e^`a` - 1"""


@_scal_elemwise
def neg(a):
    """-a"""


# numpy.reciprocal does integer division on integer inputs
# (which is not very interesting)
@_scal_elemwise
def inv(a):
    """1.0/a"""


@_scal_elemwise
def log(a):
    """base e logarithm of a"""


@_scal_elemwise
def log2(a):
    """base 2 logarithm of a"""


@_scal_elemwise
def log10(a):
    """base 10 logarithm of a"""


@_scal_elemwise
def log1p(a):
    """log(1+a)"""


@_scal_elemwise
def sgn(a):
    """sign of a"""


@_scal_elemwise
def ceil(a):
    """ceiling of a"""


@_scal_elemwise
def floor(a):
    """floor of a"""


@_scal_elemwise
def trunc(a):
    """trunc of a"""


@constructor
def iround(a, mode=None):
    """cast(round(a,mode),'int64')"""
    return cast(round(a, mode), 'int64')


@constructor
def round(a, mode=None):
    """round_mode(a) with mode in [half_away_from_zero, half_to_even].
    Default to half_to_even."""
    if mode is None:
        mode = "half_to_even"
        if config.warn.round:
            warnings.warn(
                "theano.tensor.round() changed its default from"
                " `half_away_from_zero` to `half_to_even` to have"
                " the same default as NumPy. Use the Theano flag"
                " `warn.round=False` to disable this warning.")
    if mode == "half_away_from_zero":
        return round_half_away_from_zero(a)
    elif mode == "half_to_even":
        return round_half_to_even(a)
    else:
        raise Exception("round mode %s is not implemented." % mode)


@_scal_elemwise
def round_half_to_even(a):
    """round_half_to_even(a)"""


@_scal_elemwise
def round_half_away_from_zero(a):
    """round_half_away_from_zero(a)"""


@_scal_elemwise
def sqr(a):
    """square of a"""


# alias to sqr, included to maintain similarity with numpy interface
square = sqr


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    """Calculate the covariance matrix.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`m = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`. Code and docstring ported from numpy.
    ----------
    m : array_like
        A 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column is
        observations of all those variables.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same form
        as that of `m`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True, then
        normalization is by ``N``. These values can be overridden by using the
        keyword ``ddof``.
    ddof : int, optional
        If not ``None`` the default value implied by `bias` is overridden.
        The default value is ``None``.
    Returns
    -------
    out : The covariance matrix of the variables.
    """

    if fweights is not None:
        raise NotImplementedError('fweights are not implemented')
    if aweights is not None:
        raise NotImplementedError('aweights are not implemented')

    if not rowvar and m.shape[0] != 1:
        m = m.T

    if y is not None:
        if not rowvar and y.shape[0] != 1:
            y = y.T
        m = theano.tensor.concatenate((m, y), axis=0)

    if ddof is None:
        if not bias:
            ddof = 1
        else:
            ddof = 0

    # Determine the normalization
    fact = m.shape[1] - ddof

    m -= m.mean(axis=1, keepdims=1)
    c = m.dot(m.T)
    c *= theano.tensor.constant(1) / fact
    return c.squeeze()


@_scal_elemwise
def sqrt(a):
    """square root of a"""


@_scal_elemwise
def deg2rad(a):
    """convert degree a to radian"""


@_scal_elemwise
def rad2deg(a):
    """convert radian a to degree"""


@_scal_elemwise
def cos(a):
    """cosine of a"""


@_scal_elemwise
def arccos(a):
    """arccosine of a"""


@_scal_elemwise
def sin(a):
    """sine of a"""


@_scal_elemwise
def arcsin(a):
    """arcsine of a"""


@_scal_elemwise
def tan(a):
    """tangent of a"""


@_scal_elemwise
def arctan(a):
    """arctangent of a"""


@_scal_elemwise
def arctan2(a, b):
    """arctangent of a / b"""


@_scal_elemwise
def cosh(a):
    """hyperbolic cosine of a"""


@_scal_elemwise
def arccosh(a):
    """hyperbolic arc cosine of a"""


@_scal_elemwise
def sinh(a):
    """hyperbolic sine of a"""


@_scal_elemwise
def arcsinh(a):
    """hyperbolic arc sine of a"""


@_scal_elemwise
def tanh(a):
    """hyperbolic tangent of a"""


@_scal_elemwise
def arctanh(a):
    """hyperbolic arc tangent of a"""


@_scal_elemwise
def erf(a):
    """error function"""


@_scal_elemwise
def erfc(a):
    """complementary error function"""


@_scal_elemwise
def erfcx(a):
    """scaled complementary error function"""


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
def tri_gamma(a):
    """second derivative of the log gamma function"""


@_scal_elemwise
def chi2sf(x, k):
    """chi squared survival function"""


@_scal_elemwise
def j0(x):
    """Bessel function of the first kind of order 0."""


@_scal_elemwise
def j1(x):
    """Bessel function of the first kind of order 1."""


@_scal_elemwise
def jv(v, x):
    """Bessel function of the first kind of order v (real)."""


@_scal_elemwise
def i0(x):
    """Modified Bessel function of the first kind of order 0."""


@_scal_elemwise
def i1(x):
    """Modified Bessel function of the first kind of order 1."""


@_scal_elemwise
def iv(v, x):
    """Modified Bessel function of the first kind of order v (real)."""


@_scal_elemwise
def real(z):
    """Return real component of complex-valued tensor `z`"""
_tensor_py_operators.real = property(real)


@_scal_elemwise
def imag(z):
    """Return imaginary component of complex-valued tensor `z`"""
_tensor_py_operators.imag = property(imag)


@_scal_elemwise
def angle(z):
    """Return polar-coordinate angle of complex-valued tensor `z`"""


@_scal_elemwise  # numpy.complex cannot build tensors
def complex(real, imag):
    """Return complex-valued tensor with `real` and `imag` components"""


@_scal_elemwise
def conj(z):
    """Return the complex conjugate of `z`."""


@_scal_elemwise
def complex_from_polar(abs, angle):
    """Return complex-valued tensor from polar coordinate specification."""

##########################
# Misc
##########################


# fill, _fill_inplace = _elemwise(scal.second, 'fill',
# """fill WRITEME (elemwise)""")
@_scal_elemwise
def second(a, b):
    """Create a matrix by filling the shape of a with b"""

fill = second
pprint.assign(fill, printing.FunctionPrinter('fill'))


@constructor
def ones_like(model, dtype=None, opt=False):
    """equivalent of numpy.ones_like
    Parameters
    ----------
    model : tensor
    dtype : data-type, optional
    opt : If True, we will return a constant instead of a graph when possible.
          Useful for Theano optimization, not for user building a graph as this
          have the consequence that model isn't always in the graph.

    Returns
    -------
    tensor
        tensor the shape of model containing ones of the type of dtype.
    """
    if dtype is None:
        dtype = model.type.dtype
    ret = constant(1.0, dtype=dtype)
    if opt and ret.type == model.type:
        return ret
    return fill(model, ret)


@constructor
def zeros_like(model, dtype=None, opt=False):
    """equivalent of numpy.zeros_like
    Parameters
    ----------
    model : tensor
    dtype : data-type, optional
    opt : If True, we will return a constant instead of a graph when possible.
          Useful for Theano optimization, not for user building a graph as this
          have the consequence that model isn't always in the graph.

    Returns
    -------
    tensor
        tensor the shape of model containing zeros of the type of dtype.
    """

    if dtype is None:
        dtype = model.type.dtype
    ret = constant(0.0, dtype=dtype)
    if opt and ret.type == model.type:
        return ret
    return fill(model, ret)


def zeros(shape, dtype=None):
    """
    Create a Tensor filled with zeros, closer to Numpy's syntax than ``alloc``.
    """
    if not isinstance(shape, (list, tuple, TensorVariable)):
        shape = [shape]
    if dtype is None:
        dtype = config.floatX
    return alloc(np.array(0, dtype=dtype), *shape)


def ones(shape, dtype=None):
    """
    Create a Tensor filled with ones, closer to Numpy's syntax than ``alloc``.
    """
    if not isinstance(shape, (list, tuple, TensorVariable)):
        shape = [shape]
    if dtype is None:
        dtype = config.floatX
    return alloc(np.array(1, dtype=dtype), *shape)


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
    matrix
        Matrix containing the indices of the non-zero elements of a.

    See Also
    --------
    nonzero_values : Return the non-zero elements of the input array
    flatnonzero : Return the indices of the non-zero elements of the
        flattened input array.

    """
    __props__ = ()

    def make_node(self, a):
        a = as_tensor_variable(a)
        if a.ndim == 0:
            raise ValueError('Nonzero only supports non-scalar arrays.')
        output = [TensorType(dtype='int64', broadcastable=(False, False))()]
        return gof.Apply(self, [a], output)

    def perform(self, node, inp, out_):
        a = inp[0]
        out, = out_

        result_tuple = np.nonzero(a)
        if len(result_tuple[0]) > 0:
            result = np.vstack(result_tuple)
        else:
            result = np.zeros((len(result_tuple), 0))

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
    tuple of vectors or matrix

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
    vector
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
    vector
        Output vector, containing the non-zero elements of a.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    flatnonzero : Return the indices of the non-zero elements of the
        flattened input array.

    """
    return a.flatten()[flatnonzero(a)]


class Tri(gof.Op):

    __props__ = ("dtype",)

    def __init__(self, dtype=None):
        if dtype is None:
            dtype = config.floatX
        self.dtype = dtype

    def make_node(self, N, M, k):
        N = as_tensor_variable(N)
        M = as_tensor_variable(M)
        k = as_tensor_variable(k)
        return gof.Apply(
            self,
            [N, M, k],
            [TensorType(dtype=self.dtype, broadcastable=(False, False))()])

    def perform(self, node, inp, out_):
        N, M, k = inp
        out, = out_
        out[0] = np.tri(N, M, k, dtype=self.dtype)

    def infer_shape(self, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i]) for i in xrange(3)]


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
    Array of shape (N, M)
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
    array, shape (M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    triu : Same thing, only for the upper triangle.

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
    tril : Lower triangle of an array.

    """
    return m * (1 - tri(m.shape[0], m.shape[1], k=k - 1, dtype=m.dtype))


class Eye(gof.Op):

    __props__ = ("dtype", )

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
        return gof.Apply(
            self,
            [n, m, k],
            [TensorType(dtype=self.dtype, broadcastable=(False, False))()])

    def perform(self, node, inp, out_):
        n, m, k = inp
        out, = out_
        out[0] = np.eye(n, m, k, dtype=self.dtype)

    def infer_shape(self, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i]) for i in xrange(3)]


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
    ndarray of shape (N,M)
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


def alloc_validate_shape(shape):
    sh = [as_tensor_variable(s) for s in shape]
    bcast = []
    for i, s in enumerate(sh):
        def err_str():
            if config.exception_verbosity == 'high':
                return '\n' + min_informative_str(s)
            else:
                return str(s)
        if s.type.dtype not in integer_dtypes:
            s_as_str = err_str()
            raise TypeError('Shape arguments to Alloc must be integers, '
                            'but argument %s is not for apply node: %s' %
                            (i, s_as_str))
        if s.ndim != 0:
            s_as_str = err_str()
            raise TypeError(
                "Each shape dimension to Alloc must be a scalar, ",
                'but dimension %s have %d dimensions for apply node: %s' %
                (i, s.ndim, s_as_str))

        # if s is constant 1, then we're broadcastable in that dim
        try:
            const_shp = get_scalar_constant_value(s)
        except NotScalarConstantError:
            const_shp = None
        bcast.append(1 == const_shp)
    return sh, bcast


class Alloc(gof.Op):
    """Create a Tensor from an initial value and a desired shape.

    alloc(value, shape0, shape1, ..., shapeN)

    Returns an N-dimensional tensor initialized by `value` using something
    equivalent to

        z = numpy.zeros(shape, value.dtype)
        z += value

    The result has N dimensions, has the dtype of `value` and is obtained by
    broadcasting value over the output ndarray.

    This Op is used to replace fill() during optimizations because after shapes
    are lifted, the first argument to fill can often be pruned from the graph.

    """
    _f16_ok = True
    __props__ = ()

    def validate_shape(self, shape):
        return alloc_validate_shape(shape)

    def make_node(self, value, *shape):
        v = as_tensor_variable(value)
        sh, bcast = alloc_validate_shape(shape)
        if v.ndim > len(sh):
            raise TypeError("The Alloc value to use has more dimensions"
                            " than the specified dimensions",
                            v.ndim, len(sh))
        otype = TensorType(dtype=v.dtype, broadcastable=bcast)
        return gof.Apply(self, [v] + sh, [otype()])

    def perform(self, node, inputs, out_):
        out, = out_
        v = inputs[0]
        sh = tuple([int(i) for i in inputs[1:]])
        if out[0] is None or out[0].shape != sh:
            if v.size == 1 and v.item() == 0:
                out[0] = np.zeros(sh, dtype=v.dtype)
            else:
                out[0] = np.empty(sh, dtype=v.dtype)
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
                    shape, PyArray_TYPE((PyArrayObject*) py_%(vv)s));
                if (!%(zz)s)
                {
                    PyErr_SetString(PyExc_MemoryError, "alloc failed");
                    %(fail)s
                }
            }

            // This function takes care of broadcasting
            if (PyArray_CopyInto(%(zz)s, %(vv)s) == -1)
              %(fail)s
            """ % dict(vv=vv, ndim=ndim, zz=zz, fail=fail)

        return code

    def c_code_cache_version(self):
        return (2,)

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
        # The number of dimensions added
        axis = list(range(n_axes_to_sum))
        # The broadcasted dimensions
        axis_broadcasted = []
        axis_kept = []
        for i, (ib, gb) in enumerate(
            zip(inputs[0].broadcastable,
                # We need the dimensions corresponding to x
                grads[0].broadcastable[-inputs[0].ndim:])):
            if ib and not gb:
                axis_broadcasted.append(i + n_axes_to_sum)
            else:
                axis_kept.append(i)
        gx = gz.sum(axis=axis + axis_broadcasted)
        if axis_broadcasted:
            new_order = ['x'] * x.ndim
            for idx, axis in enumerate(axis_kept):
                new_order[axis] = idx
            gx = gx.dimshuffle(new_order)
            # Dimshuffle to add back the broadcasted dims
        # The *elements* of the output are not connected to
        # the inputs that specify the shape. If you grow the
        # shape by epsilon, the existing elements do not
        # change.
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
            elif (
                # The following ops work inplace of their input id 0.
                client[1] == 0 and
                isinstance(client[0].op, (
                    # Ops that will work inplace on the Alloc. So if they
                    # get constant_folded, they would copy the
                    # constant and this is less efficients.

                    # Not doing the constant folding could also lower
                    # the peak memory usage, as we the "constant" won't
                    # always exists.
                    theano.tensor.subtensor.IncSubtensor,
                    theano.tensor.subtensor.AdvancedIncSubtensor1,
                    theano.tensor.subtensor.AdvancedIncSubtensor,
                    theano.tensor.blas.Gemv,
                    theano.tensor.blas_c.CGemv,
                    theano.tensor.blas.Ger,
                    theano.tensor.blas_c.CGer,
                    theano.tensor.blas_scipy.ScipyGer))):
                return False
            # If the clients is a transfer to the GPU, we don't want to
            # fold. We let the Alloc being moved to the GPU, then we
            # let the GPU algo decide if it need to fold it or not.
            elif client[0].op.__class__.__name__.lower().startswith("gpu"):
                return False
        return True

alloc = Alloc()
pprint.assign(alloc, printing.FunctionPrinter('alloc'))


def transfer(var, target):
    """
    Return a version of `var` transferred to `target`.

    `cpu` mean a TensorType (on the CPU).  Other types may define
    additional targets.

    Parameters
    ----------
    var : variable
        A theano variable
    target : str
        The target of the transfer
    """
    if target == 'cpu':
        return as_tensor_variable(var)
    else:
        for trans in transfer._others:
            res = trans(var, target)
            if res is not None:
                return res
    raise ValueError("Can't transfer to target %s" % (target,))

transfer._others = []


def register_transfer(fn):
    """
    Register a transfer function for alternative targets.

    Parameters
    ----------
    fn : callable
    """
    transfer._others.append(fn)

"""Create a duplicate of `a` (with duplicated storage)"""
tensor_copy = elemwise.Elemwise(scal.identity)
pprint.assign(tensor_copy, printing.IgnorePrinter())


@constructor
def sum(input, axis=None, dtype=None, keepdims=False, acc_dtype=None):
    """
    Computes the sum along the given axis(es) of a tensor `input`.

    When axis is None (the default value), the sum is performed
    over the flattened tensor.

    For full documentation see ``tensor.elemwise.Sum``.
    In particular please pay attention to the important warning when using
    a custom acc_dtype.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """

    out = elemwise.Sum(axis=axis, dtype=dtype, acc_dtype=acc_dtype)(input)

    if keepdims:
        out = makeKeepDims(input, out, axis)
    return out

pprint.assign(Sum(), printing.FunctionPrinter('sum'))


@constructor
def prod(input, axis=None, dtype=None, keepdims=False, acc_dtype=None,
         no_zeros_in_input=False):
    """
    Computes the product along the given axis(es) of a tensor `input`.

    When axis is None (the default value), the product is performed
    over the flattened tensor.

    For full documentation see ``tensor.elemwise.Prod``.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """

    out = elemwise.Prod(axis, dtype=dtype, acc_dtype=acc_dtype,
                        no_zeros_in_input=no_zeros_in_input)(input)

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
        output[0] = np.asarray(np.mean(input, dtype='float64',
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
    Computes the mean value along the given axis(es) of a tensor `input`.

    Parameters
    ----------
    axis : None or int or (list of int) (see `Sum`)
        Compute the mean along this axis of the tensor.
        None means all axes (like numpy).
    dtype: None or string
        Dtype to cast the result of the inner summation into.
        For instance, by default, a sum of a float32 tensor will be
        done in float64 (acc_dtype would be float64 by default),
        but that result will be casted back in float32.
    keepdims: bool
        If this is set to True, the axes which are reduced are
        left in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original tensor.
    acc_dtype: None or string
        Dtype to use for the inner summation. This will not
        necessarily be the dtype of the output (in particular
        if it is a discrete (int/uint) dtype, the output will
        be in a float type). If None, then we use the same rules as `sum()`.

    Notes
    -----
    For gpu, if you specify dtype=float32, everything will be done on the gpu.

    """
    input = as_tensor_variable(input)
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
        sum_dtype = None
        # float16 overflows on the cast way too often
        if input.dtype == 'float16':
            sum_dtype = 'float32'

    s = sum(input, axis=axis, dtype=sum_dtype, keepdims=keepdims,
            acc_dtype=acc_dtype)
    shp = shape(input)

    # Cast shp into a float type
    # TODO Once we have a consistent casting policy, we could simply
    # use true_div.
    if s.dtype in ('float16', 'float32', 'complex64'):
        shp = cast(shp, 'float32')
    else:
        shp = cast(shp, 'float64')

    if axis is None:
        axis = list(range(input.ndim))
    elif isinstance(axis, (integer_types, np.integer)):
        axis = [axis]
    elif isinstance(axis, np.ndarray) and axis.ndim == 0:
        axis = [int(axis)]
    else:
        axis = [int(a) for a in axis]

    # This sequential division will possibly be optimized by Theano:
    for i in axis:
        s = true_div(s, shp[i])

    # This can happen when axis is an empty list/tuple
    if s.dtype != shp.dtype and s.dtype in discrete_dtypes:
        s = cast(s, shp.dtype)

    if dtype == 'float16' or (dtype is None and input.dtype == 'float16'):
        s = cast(s, 'float16')
    s.name = 'mean'
    return s


@constructor
def var(input, axis=None, ddof=0, keepdims=False, corrected=False):
    """
    Computes the variance along the given axis(es) of a tensor `input`.

    Parameters
    ----------
    axis: None or int or (list of int) (see `Sum`)
        Compute the variance along this axis of the tensor.
        None means all axes (like numpy).
    ddof: Degrees of freedom; 0 would compute the ML estimate, 1 would compute
        the unbiased estimate.
    keepdims : bool
        If this is set to True, the axes which are reduced are
        left in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original tensor.
    corrected : bool
        If this is set to True, the 'corrected_two_pass' algorithm is
        used to compute the variance.
        Refer : http://www.cs.yale.edu/publications/techreports/tr222.pdf

    Notes
    -----
    Default uses the two-pass algorithm (reference below).
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
    Also supports 'corrected_two_pass' algorithm (using the 'corrected' flag)
    which is numerically more stable. There exist other implementations that
    offer better stability, but probably slower.

    """

    if isinstance(ddof, (bool)):
        raise ValueError('Parameter keepdims is now at index 3: (input, \
                          axis=None, ddof=0, keepdims=False, corrected=False)')

    input_ndim = input.type.ndim
    if axis is None:
        axis = list(range(input_ndim))
    elif isinstance(axis, (integer_types, np.integer)):
        axis = [axis]
    elif isinstance(axis, np.ndarray) and axis.ndim == 0:
        axis = [int(axis)]
    else:
        axis = [int(a) for a in axis]

    # compute the axis-wise mean
    mean_input = mean(input, axis, keepdims=True)

    # center the input
    centered_input = input - mean_input

    # return the mean sqr
    two = constant(2, dtype=centered_input.dtype)
    if ddof == 0:
        v = mean((centered_input ** two), axis, keepdims=keepdims)
    else:
        shp = shape(input) - ddof
        v = sum((centered_input ** two), axis=axis, keepdims=keepdims)
        for i in axis:
            v = true_div(v, shp[i])

    # use 'corrected_two_pass' algorithm
    if corrected:
        if ddof == 0:
            error = mean(centered_input, axis, keepdims=keepdims) ** 2
        else:
            shp = shape(input) - ddof
            shp_inp = shape(input)
            error = sum(centered_input, axis=axis, keepdims=keepdims) ** 2
            for i in axis:
                error = true_div(error, shp[i] * shp_inp[i])
        v = v - error

    v.name = 'var'
    return v


@constructor
def std(input, axis=None, ddof=0, keepdims=False, corrected=False):
    """
    Computes the standard deviation along the given axis(es) of a tensor `input`.

    Parameters
    ----------
    axis: None or int or (list of int) (see `Sum`)
        Compute the variance along this axis of the tensor.
        None means all axes (like numpy).
    ddof: Degrees of freedom; 0 would compute the ML estimate, 1 would compute
        the unbiased estimate.
    keepdims : bool
        If this is set to True, the axes which are reduced are
        left in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original tensor.
    corrected : bool
        If this is set to True, the 'corrected_two_pass' algorithm is
        used to compute the variance.
        Refer : http://www.cs.yale.edu/publications/techreports/tr222.pdf

    Notes
    -----
    It calls 'var()' and 'var()' uses the two-pass algorithm (reference below).
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
    Function 'var()' also supports 'corrected_two_pass' algorithm (using the
    'corrected' flag) which is numerically more stable. There exist other
    implementations that offer better stability, but probably slower.

    """

    if isinstance(ddof, (bool)):
        raise ValueError('Parameter keepdims is now at index 3: (input, \
                          axis=None, ddof=0, keepdims=False, corrected=False)')

    ret = sqrt(var(input=input, axis=axis, ddof=ddof,
                   keepdims=keepdims, corrected=corrected))
    ret.name = 'std'
    return ret


class Default(gof.Op):
    """
    Takes an input x and a default value.

    If the input is not None, a reference to it is returned.
    If the input is None, a copy of the default value is returned instead.
    The input and the default must have exactly the same type.

    """
    view_map = {0: [0]}
    __props__ = ()

    def make_node(self, x, default):
        x, default = as_tensor_variable(x), as_tensor_variable(default)
        if x.type != default.type:
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
@_scal_elemwise
def maximum(x, y):
    """elemwise maximum. See max for the maximum in one tensor"""
    # see decorator for function body


@_scal_elemwise
def minimum(x, y):
    """elemwise minimum. See min for the minimum in one tensor"""
    # see decorator for function body


def div_proxy(x, y):
    """Proxy for either true_div or int_div, depending on types of x, y."""
    f = scal.int_or_true_div(
        as_tensor_variable(x).dtype in discrete_dtypes,
        as_tensor_variable(y).dtype in discrete_dtypes)
    if f is scal.int_div:
        return int_div(x, y)
    else:
        return true_div(x, y)


def divmod(x, y):
    """elementvise divmod, using floor_div and mod_check"""
    return floor_div(x, y), mod_check(x, y)


@_scal_elemwise
def add(a, *other_terms):
    """elementwise addition"""
    # see decorator for function body


@_scal_elemwise
def sub(a, b):
    """elementwise subtraction"""
    # see decorator for function body


@_scal_elemwise
def mul(a, *other_terms):
    """elementwise multiplication"""
    # see decorator for function body


@_scal_elemwise
def true_div(a, b):
    """elementwise [true] division (inverse of multiplication)"""
    # see decorator for function body


@_scal_elemwise
def int_div(a, b):
    """elementwise [floor] division (inverse of multiplication)"""
    # see decorator for function body


# floor_div and int_div are the same thing
floor_div = int_div


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
    if ((as_tensor_variable(x).dtype in complex_dtypes or
         as_tensor_variable(y).dtype in complex_dtypes)):
        # Currently forbidden.
        raise scal.Mod.complex_error
    else:
        return mod(x, y)


@_scal_elemwise
def mod(a, b):
    """elementwise modulo"""
    # see decorator for function body


@_scal_elemwise
def pow(a, b):
    """elementwise power"""
    # see decorator for function body


@_scal_elemwise
def clip(x, min, max):
    """
    Clip x to be between min and max.

    Notes
    -----
    When `x` is equal to the boundaries, the output is considered
    to be `x`, so at these points, the gradient of the cost wrt the output
    will be propagated to `x`, not to `min` nor `max`. In other words,
    on these points, the gradient wrt `x` will be equal to the gradient wrt
    the output, and the gradient wrt `min` and `max` will be zero.

    """
    # see decorator for function body
    # for grep: clamp, bound

pprint.assign(add, printing.OperatorPrinter('+', -2, 'either'))
pprint.assign(mul, printing.OperatorPrinter('*', -1, 'either'))
pprint.assign(sub, printing.OperatorPrinter('-', -2, 'left'))
pprint.assign(neg, printing.OperatorPrinter('-', 0, 'either'))
pprint.assign(true_div, printing.OperatorPrinter('/', -1, 'left'))
pprint.assign(int_div, printing.OperatorPrinter('//', -1, 'left'))
pprint.assign(pow, printing.OperatorPrinter('**', 1, 'right'))


##########################
# View Operations
##########################


def extract_constant(x, elemwise=True, only_process_constants=False):
    """
    This function is basically a call to tensor.get_scalar_constant_value.

    The main difference is the behaviour in case of failure. While
    get_scalar_constant_value raises an TypeError, this function returns x,
    as a tensor if possible. If x is a ScalarVariable from a
    scalar_from_tensor, we remove the conversion. If x is just a
    ScalarVariable, we convert it to a tensor with tensor_from_scalar.

    """
    try:
        x = get_scalar_constant_value(x,
                                      elemwise,
                                      only_process_constants)
    except NotScalarConstantError:
        pass
    if ((isinstance(x, scal.ScalarVariable) or
         isinstance(x, scal.sharedvar.ScalarSharedVariable))):
        if x.owner and isinstance(x.owner.op, ScalarFromTensor):
            x = x.owner.inputs[0]
        else:
            x = tensor_from_scalar(x)
    return x


def transpose(x, axes=None):
    """
    Reorder the dimensions of x. (Default: reverse them)

    This is a macro around dimshuffle that matches the numpy.transpose function.

    """
    if axes is None:
        axes = list(range((x.ndim - 1), -1, -1))
    ret = DimShuffle(x.broadcastable, axes)(x)
    if x.name and axes == list(range((x.ndim - 1), -1, -1)):
        ret.name = x.name + '.T'
    return ret


def batched_dot(a, b):
    """
    Compute the batched dot product of two variables:

        batched_dot(a, b)[i] = dot(a[i], b[i])

    Note that this batched_dot function does one of three things, in the
    following sequence:

        1.  If either a or b is a vector, it returns the batched elementwise
            product without calling the Theano BatchedDot op.

        2.  If both a and b have either 2 or 3 dimensions, it calls Theano's
            BatchedDot op on a and b.

        3.  If either a or b has more than 3 dimensions, it calls Theano's
            batched_tensordot function with appropriate axes. The
            batched_tensordot function expresses high-dimensional batched
            dot products in terms of batched matrix-matrix dot products, so
            it may be possible to futherize optimize for performance.
    """
    a, b = as_tensor_variable(a), as_tensor_variable(b)

    if a.ndim == 0:
        raise TypeError("a must have at least one (batch) axis")
    elif b.ndim == 0:
        raise TypeError("b must have at least one (batch) axis")
    elif a.ndim == 1:
        return a.dimshuffle(*([0] + ["x"] * (b.ndim - 1))) * b
    elif b.ndim == 1:
        return a * b.dimshuffle(*([0] + ["x"] * (a.ndim - 1)))
    elif a.ndim > 3 or b.ndim > 3:
        return batched_tensordot(
            a, b, [[a.ndim - 1], [np.maximum(1, b.ndim - 2)]])
    else:
        # avoid circular import
        return theano.tensor.blas.BatchedDot()(a, b)


def batched_tensordot(x, y, axes=2):
    """
    Compute a batched tensordot product.

    A hybrid of batched_dot and tensordot, this function computes the
    tensordot product between the two tensors, by iterating over the
    first dimension to perform a sequence of tensordots.

    Parameters
    ----------
    x : tensor
        A Tensor with sizes e.g.: for 3D (dim1, dim3, dim2)
    y : tensor
        A Tensor with sizes e.g.: for 3D (dim1, dim2, dim4)
    axes: int or array-like of length 2
        If an integer, the number of axes to sum over.
        If an array, it must have two array elements containing the axes to sum
        over in each tensor.

        If an integer i, it is converted to an array containing
        the last i dimensions of the first tensor and the first
        i dimensions of the second tensor (excluding the first
        (batch) dimension):
            axes = [list(range(a.ndim - i, b.ndim)), list(range(1,i+1))]

        If an array, its two elements must contain compatible axes
        of the two tensors. For example, [[1, 2], [2, 4]] means sum
        over the 2nd and 3rd axes of a and the 3rd and 5th axes of b.
        (Remember axes are zero-indexed!) The 2nd axis of a and the
        3rd axis of b must have the same shape; the same is true for
        the 3rd axis of a and the 5th axis of b.

    Like tensordot, this function uses a series of dimshuffles and
    reshapes to reduce the tensor dot product to a matrix or vector
    dot product.  Finally, it calls batched_dot to compute the result.
    """
    return _tensordot_as_dot(x, y, axes, dot=batched_dot, batched=True)


def split(x, splits_size, n_splits, axis=0):
    the_split = Split(n_splits)
    return the_split(x, axis, splits_size)


class Split(Op):
    """Partition a `TensorVariable` along some axis.

    Examples
    --------
    >>> x = vector()
    >>> splits = lvector()
    You have to declare right away how many split_points there will be.
    >>> ra, rb, rc = split(x, splits, n_splits = 3, axis = 0)
    >>> f = function([x, splits], [ra, rb, rc])
    >>> a, b, c = f([0,1,2,3,4,5], [3, 2, 1])
    a == [0,1,2]
    b == [3, 4]
    c == [5]

    """

    len_splits = None
    """A Split instance will have this many outputs, and require that
    the splits argument to `perform` have exactly this many elements.
    """
    __props__ = ("len_splits",)

    def __init__(self, len_splits):
        self.len_splits = int(len_splits)

    def __str__(self):
        return self.__class__.__name__ + "{%s}" % self.len_splits

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

        if np.sum(splits) != len_along_axis:
            raise ValueError('The splits sum to %s, expected %s' %
                             (np.sum(splits), len_along_axis))
        if python_any([nb < 0 for nb in splits]):
            raise ValueError('Split: you tried to make an ndarray with a '
                             'negative number of elements.')

        # Checking is done, let's roll the splitting algorithm!
        # Basically we step along the given axis of x, extracting
        # subtensors of size splits[i] as we go along.

        general_key = [slice(None, None, None) for s in x.shape]
        lower_idx = 0
        for i in xrange(self.len_splits):
            upper_idx = lower_idx + splits[i]
            general_key[axis] = slice(lower_idx, upper_idx, None)
            outputs[i][0] = x.__getitem__(tuple(general_key)).copy()
            lower_idx = upper_idx

    def infer_shape(self, node, in_shapes):
        axis = node.inputs[1]
        splits = node.inputs[2]
        shp_x, shp_axis, shp_splits = in_shapes
        out_shapes = []
        for i in xrange(self.len_splits):
            temp = as_tensor_variable(shp_x)
            temp = theano.tensor.subtensor.set_subtensor(temp[axis], splits[i])
            temp = [temp[i] for i in xrange(len(shp_x))]
            out_shapes.append(temp)
        return out_shapes

    def grad(self, inputs, g_outputs):
        """Join the gradients along the axis that was used to split x."""
        x, axis, n = inputs
        outputs = self(*inputs, **dict(return_list=True))
        # If all the output gradients are disconnected, then so are the inputs
        if python_all([isinstance(g.type, DisconnectedType)
                       for g in g_outputs]):
            return [DisconnectedType()(),
                    grad_undefined(self, 1, axis),
                    grad_undefined(self, 2, n)]
        # Else, we have to make them zeros before joining them
        new_g_outputs = []
        for o, g in zip(outputs, g_outputs):
            if isinstance(g.type, DisconnectedType):
                new_g_outputs.append(o.zeros_like())
            else:
                new_g_outputs.append(g)

        return [join(axis, *new_g_outputs),
                grad_undefined(self, 1, axis),
                grad_undefined(self, 2, n)]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None for i in self.len_splits]
        return self.make_node(eval_points[0], *inputs[1:]).outputs

    def c_code_cache_version(self):
        return (2,)

    def c_support_code(self):
        return """
        /* Return 1 if output has the correct shape. */
        int split_output_shape_is_correct (
            PyArrayObject* output, PyArrayObject* array_to_split, int axis_to_split, npy_intp split_size
        ) {
            return
                PyArray_NDIM(output) == PyArray_NDIM(array_to_split)
                && memcmp(
                    PyArray_DIMS(output),
                    PyArray_DIMS(array_to_split),
                    axis_to_split * sizeof(npy_intp)
                ) == 0
                && memcmp(
                    PyArray_DIMS(output) + axis_to_split + 1,
                    PyArray_DIMS(array_to_split) + axis_to_split + 1,
                    (PyArray_NDIM(array_to_split) - axis_to_split - 1) * sizeof(npy_intp)
                ) == 0
                && split_size == PyArray_DIM(output, axis_to_split);
        }
        """

    def c_code(self, node, name, inputs, outputs, sub):
        if self.len_splits == 0:
            # There are no outputs, then nothing to do.
            return ''

        # outputs_pointers lists the addresses of the pointers to the outputs.
        outputs_pointers = '&' + (', &'.join(outputs))
        x, axis, splits = inputs
        fail = sub['fail']
        x_typenum = np.dtype(node.inputs[0].dtype).num
        x_itemsize = np.dtype(node.inputs[0].dtype).itemsize
        axis_dtype = node.inputs[1].type.dtype_specs()[1]
        splits_dtype = node.inputs[2].type.dtype_specs()[1]
        expected_splits_count = self.len_splits

        return """
        int ndim = PyArray_NDIM(%(x)s);
        int axis = (int)(*(%(axis_dtype)s*)PyArray_GETPTR1(%(axis)s, 0));
        int splits_count = PyArray_DIM(%(splits)s, 0);
        npy_intp len_along_axis, sum_of_splits = 0, current_split_length = 0, current_split_start = 0;
        npy_intp* split_dims = NULL;
        PyObject* split_view = NULL;
        npy_intp data_offset;
        int i;
        PyArrayObject** outputs[] = {%(outputs_pointers)s};

        /* Check inputs. */

        if (splits_count != %(expected_splits_count)s) {
            PyErr_Format(PyExc_ValueError,
                "Split: splits count (%%d) != expected count (%%d).", splits_count, %(expected_splits_count)s);
            %(fail)s
        }

        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis >= ndim) {
            PyErr_Format(PyExc_IndexError, "Split: invalid axis %%d for a %%d-D array.", axis, ndim);
            %(fail)s
        }
        len_along_axis = PyArray_DIM(%(x)s, axis);

        for (i = 0; i < splits_count; ++i) {
            current_split_length = (npy_intp)(*(%(splits_dtype)s*)PyArray_GETPTR1(%(splits)s, i));
            if (current_split_length < 0) {
                PyErr_Format(PyExc_ValueError,
                    "Split: you try to take a negative number (%%ld) of elements.", current_split_length);
                %(fail)s
            }
            sum_of_splits += current_split_length;
        }
        if (sum_of_splits != len_along_axis) {
            PyErr_Format(PyExc_ValueError, "Split: the splits sums to %%ld, expected %%ld.", sum_of_splits, len_along_axis);
            %(fail)s
        }

        /* Check outputs. */

        split_dims = (npy_intp*) malloc(ndim * sizeof(npy_intp));
        if (split_dims == NULL) {
            PyErr_NoMemory();
            %(fail)s
        }

        memcpy(split_dims, PyArray_DIMS(%(x)s), ndim * sizeof(npy_intp));

        for (i = 0; i < splits_count; ++i) {
            PyArrayObject** output = outputs[i];
            current_split_length = (npy_intp) (* (%(splits_dtype)s*) PyArray_GETPTR1(%(splits)s, i));
            if (*output == NULL || !split_output_shape_is_correct(*output, %(x)s, axis, current_split_length)) {
                Py_XDECREF(*output);
                split_dims[axis] = current_split_length;
                *output = (PyArrayObject*)PyArray_EMPTY(ndim, split_dims, %(x_typenum)s, PyArray_IS_F_CONTIGUOUS(%(x)s));
                if (outputs == NULL) {
                    PyErr_SetString(PyExc_RuntimeError, "Split: unable to allocate an output.");
                    free(split_dims);
                    %(fail)s
                }
            }
        }

        /* Compute split. */

        for (i = 0; i < splits_count; ++i) {
            current_split_length = (npy_intp) (* (%(splits_dtype)s*) PyArray_GETPTR1(%(splits)s, i));
            data_offset = PyArray_STRIDE(%(x)s, axis) * current_split_start;
            split_dims[axis] = current_split_length;
            split_view = PyArray_New(&PyArray_Type,
                                    ndim, split_dims,
                                    %(x_typenum)s,
                                    PyArray_STRIDES(%(x)s),
                                    PyArray_BYTES(%(x)s) + data_offset,
                                    %(x_itemsize)s,
                                    PyArray_FLAGS(%(x)s),
                                    NULL);
            if (split_view == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Split: unable to create a view for a split.");
                free(split_dims);
                %(fail)s
            }
            if (PyArray_CopyInto(*outputs[i], (PyArrayObject*)split_view) != 0) {
                PyErr_SetString(PyExc_RuntimeError, "Split: unable to copy a split view into the output.");
                Py_XDECREF(split_view);
                free(split_dims);
                %(fail)s
            }
            Py_XDECREF(split_view);
            current_split_start += current_split_length;
        }

        free(split_dims);
        """ % locals()


def addbroadcast(x, *axes):
    """
    Make the input broadcastable in the specified axes.

    For example, addbroadcast(x, 0) will make the first dimension of
    x broadcastable. When performing the function, if the length of
    x along that dimension is not 1, a ValueError will be raised.

    We apply the opt here not to pollute the graph especially during
    the gpu optimization

    Parameters
    ----------
    x : tensor_like
        Input theano tensor.
    axis : an int or an iterable object such as list or tuple of int values
        The dimension along which the tensor x should be broadcastable.
        If the length of x along these dimensions is not 1, a ValueError will
        be raised.

    Returns
    -------
    tensor
        A theano tensor, which is broadcastable along the specified dimensions.

    """
    rval = Rebroadcast(*[(axis, True) for axis in axes])(x)
    return theano.tensor.opt.apply_rebroadcast_opt(rval)


def unbroadcast(x, *axes):
    """
    Make the input impossible to broadcast in the specified axes.

    For example, addbroadcast(x, 0) will make the first dimension
    of x broadcastable. When performing the function, if the length
    of x along that dimension is not 1, a ValueError will be raised.

    We apply the opt here not to pollute the graph especially during
    the gpu optimization

    Parameters
    ----------
    x : tensor_like
        Input theano tensor.
    axis : an int or an iterable object such as list or tuple of int values
        The dimension along which the tensor x should be unbroadcastable.
        If the length of x along these dimensions is not 1, a ValueError will
        be raised.

    Returns
    -------
    tensor
        A theano tensor, which is unbroadcastable along the specified dimensions.

    """
    rval = Rebroadcast(*[(axis, False) for axis in axes])(x)
    return theano.tensor.opt.apply_rebroadcast_opt(rval)


def patternbroadcast(x, broadcastable):
    """
    Make the input adopt a specific broadcasting pattern.

    Broadcastable must be iterable. For example,
    patternbroadcast(x, (True, False)) will make the first
    dimension of x broadcastable and the second dimension
    not broadcastable, so x will now be a row.

    We apply the opt here not to pollute the graph especially during the gpu
    optimization.

    Parameters
    ----------
    x : tensor_like
        Input theano tensor.
    broadcastable : an iterable object such as list or tuple of bool values
        A set of boolean values indicating whether a dimension should be
        broadcastable or not. If the length of x along these dimensions is
        not 1, a ValueError will be raised.

    Returns
    -------
    tensor
        A theano tensor, which is unbroadcastable along the specified dimensions.

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

    See Also
    --------
    stack : For joins involving scalar values

    Examples
    --------
    >>> x, y, z = tensor.matrix(), tensor.matrix(), tensor.matrix()
    >>> u = tensor.vector()

    >>> r = join(0, x, y, z)
    >>> c = join(1, x, y, z)
    >>> join(2, x, y, z)    # WRONG: the axis has to be an index into the shape
    >>> join(0, x, u)       # WRONG: joined tensors must have the same rank

    """
    check_input = False
    __props__ = ("view",)

    def __init__(self, view=-1):
        self.view = view
        if view != -1:
            # since the first input is always the axis, the tensors
            # start from index 1.
            self.view_map = {0: [1 + view]}

    def __str__(self):
        if self.view == -1:
            return self.__class__.__name__
        else:
            return "%s{%s}" % (
                self.__class__.__name__,
                ", ".join("%s=%r" % (p, getattr(self, p))
                          for p in self.__props__))

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "view"):
            self.view = -1

    def make_node(self, *axis_and_tensors):
        """
        Parameters
        ----------
        axis: an Int or integer-valued Variable
        tensors
            A variable number (but not zero) of tensors to
            concatenate along the specified axis.  These tensors must have
            the same shape along all dimensions other than this axis.

        Returns
        -------
        A symbolic Variable
            It has the same ndim as the input tensors, and the most inclusive
            dtype.

        """
        axis, tensors = axis_and_tensors[0], axis_and_tensors[1:]
        if not tensors:
            raise ValueError('Cannot join an empty list of tensors')
        as_tensor_variable_args = [as_tensor_variable(x) for x in tensors]

        dtypes = [x.type.dtype for x in as_tensor_variable_args]
        out_dtype = scal.upcast(*dtypes)

        def output_maker(bcastable):
            return tensor(dtype=out_dtype, broadcastable=bcastable)

        return self._make_node_internal(
            axis, tensors, as_tensor_variable_args, output_maker)

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
            if not isinstance(axis, integer_types):
                try:
                    # Note : `get_scalar_constant_value` returns a ndarray not
                    # an int
                    axis = int(get_scalar_constant_value(axis))

                except NotScalarConstantError:
                    pass
            if isinstance(axis, integer_types):
                # Basically, broadcastable -> length 1, but the
                # converse does not hold. So we permit e.g. T/F/T
                # joins, and if they fail at runtime they fail, but if
                # they don't then it means that the argument where
                # that broadcastable flag was False had length 1 along
                # this dimension, and therefore this dimension should
                # be broadcastable for the output.

                if axis < -ndim:
                    raise IndexError("Join axis %d out of bounds [0, %d)" %
                                     (axis, ndim))
                if axis < 0:
                    axis += ndim

                for x in as_tensor_variable_args:
                    for current_axis, bflag in enumerate(x.type.broadcastable):
                        # Constant negative axis can no longer be negative at
                        # this point. It safe to compare this way.
                        if current_axis == axis:
                            continue
                        if bflag:
                            bcastable[current_axis] = True
                try:
                    bcastable[axis] = False
                except IndexError:
                    raise ValueError('Join argument "axis" is out of range'
                                     ' (given input dimensions)')
            else:
                # When the axis may vary, no dimension can be guaranteed to be
                # broadcastable.
                bcastable = [False] * len(
                    as_tensor_variable_args[0].type.broadcastable)

        if not python_all([x.ndim == len(bcastable)
                           for x in as_tensor_variable_args[1:]]):
            raise TypeError("Join() can only join tensors with the same "
                            "number of dimensions.")

        inputs = [as_tensor_variable(axis)] + list(as_tensor_variable_args)
        if inputs[0].type not in int_types:
            raise TypeError('Axis could not be cast to an integer type',
                            axis, inputs[0].type, int_types)

        outputs = [output_maker(bcastable)]

        node = Apply(self, inputs, outputs)
        return node

    def perform(self, node, axis_and_tensors, out_):
        out, = out_
        view = self.view
        axis, tensors = axis_and_tensors[0], axis_and_tensors[1:]
        # we check these tensors for being empty.
        if (view != -1) and np.all(
                [tensor.shape[axis] == 0 for tensor in
                 tensors[0:view] + tensors[view + 1:]]):
            out[0] = tensors[view]

        else:
            ndim = tensors[0].ndim
            if axis < -ndim:
                raise IndexError("Join axis %d out of bounds [0, %d)" %
                                 (axis, ndim))

            out[0] = theano._asarray(np.concatenate(tensors, axis=axis),
                                     dtype=node.outputs[0].type.dtype)

    def c_code_cache_version(self):
        return (5,)

    def c_code(self, node, name, inputs, outputs, sub):
        axis, tensors = inputs[0], inputs[1:]
        view = self.view
        non_empty_tensor = tensors[view]
        input_1 = tensors[0]
        l = len(tensors)
        out, = outputs
        fail = sub['fail']
        adtype = node.inputs[0].type.dtype_specs()[1]
        copy_to_list = []

        for i, inp in enumerate(tensors):
            copy_to_list.append(
                """Py_INCREF(%s);
                   PyList_SetItem(list, %s, (PyObject*)%s);"""
                % (inp, i, inp))

        copy_inputs_to_list = '\n'.join(copy_to_list)
        n = len(tensors)

        code = """
        int axis = ((%(adtype)s *)PyArray_DATA(%(axis)s))[0];
        PyObject* list = PyList_New(%(l)s);
        %(copy_inputs_to_list)s
        int tensors_lens_sum;
        if(%(view)s != -1) {
            tensors_lens_sum = 0;

            for(int i=0; i < %(n)s; i++){
                tensors_lens_sum += PyArray_DIM((PyArrayObject *)(PyList_GetItem(list, i)), axis);
            }
            tensors_lens_sum -= PyArray_DIM(%(non_empty_tensor)s, axis);
        }
        if(%(view)s != -1 && tensors_lens_sum == 0) {
            Py_XDECREF(%(out)s);
            Py_INCREF(%(non_empty_tensor)s);
            %(out)s = %(non_empty_tensor)s;
        }else{
            //PyObject* PyArray_Concatenate(PyObject* obj, int axis)
            int ndim = PyArray_NDIM(%(input_1)s);
            if( axis < -ndim ){
                PyErr_Format(PyExc_IndexError,
                             "Join axis %%d out of bounds [0, %%d)", axis, ndim);
                %(fail)s
            }
            Py_XDECREF(%(out)s);
            %(out)s = (PyArrayObject *)PyArray_Concatenate(list, axis);
            Py_DECREF(list);
            if(!%(out)s){
                %(fail)s
            }
        }
        """ % locals()
        return code

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

        dtypes = [as_tensor_variable(x).type.dtype for x in tensors]
        out_dtype = scal.upcast(*dtypes)

        if 'float' in out_dtype or 'complex' in out_dtype:
            # assume that this is differentiable
            split = Split(len(tensors))
            split_gz = split(gz, axis, stack([shape(x)[axis]
                                              for x in tensors]))
            # If there is only one split, it might not be in a list.
            if not isinstance(split_gz, list):
                split_gz = [split_gz]
            # Split.make_node isn't always able to infer the right
            # broadcast. As the grad need to keep the information,
            # read it if needed.
            split_gz = [patternbroadcast(g, t.broadcastable)
                        for t, g in zip(tensors, split_gz)]
            rval = rval + split_gz
        else:
            # the output has integer type, so the gradient through it
            # is 0
            rval = rval + [tensor.zeros_like(dtype=config.floatX)
                           for tensor in tensors]

        return rval

    def infer_shape(self, node, ishapes):
        # ishapes[0] contains the size of the axis on which we join
        # Join op should get at least one input to join
        assert len(ishapes) > 1
        n_dim = len(ishapes[1])
        for shp in ishapes[1:]:
            assert shp is not None
            assert len(shp) == n_dim

        # The joining dimension could be negative, but we need it to be
        # in [0, n_dim) in the loop below.
        # An axis < -n_dim or >= ndim would be invalid, but this is
        # not checked here. An Assert op would be a way of addressing that,
        # but it may disrupt optimizations.
        join_dim = switch(ge(node.inputs[0], 0),
                          node.inputs[0],
                          node.inputs[0] + n_dim)
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
            for shp in ishapes[2:]:
                t_side = t_side + shp[dim]
            # return the dimensions found
            out_shapes.append(switch(eq(dim, join_dim),
                              t_side, f_side))

        return [tuple(out_shapes)]


join_ = Join()
pprint.assign(Join, printing.FunctionPrinter('join'))


def join(axis, *tensors_list):
    """
    Convenience function to concatenate `TensorType`s along the given axis.

    This function will not add the op in the graph when it is not useful.
    For example, in the case that the list of tensors to be concatenated
    is one, it will just return the tensor.

    Parameters
    ----------
    tensors : list of tensors (or list-like)
        A list of tensors to be concatenated along the given axis.
        The shapes of the tensors to be concatenated must be all
        identical, except in the dimension (`axis`) on which they are to
        be joined.
    axis : int (symbolic or literal)
        On which dimension should the tensors be joined?  The `axis`
        must be a valid index into the shape of the tensors to be
        concatenated.
        The `axis` parameter may either be an integer or an object that
        can be converted to a scalar using `as_scalar`(`axis`). In the
        former case, the axis is fixed at construction, while in the
        latter it may vary over time depending on the value of the
        `axis` variable.
    """
    if len(tensors_list) == 1:
        return tensors_list[0]
    else:
        return join_(axis, *tensors_list)


def roll(x, shift, axis=None):
    """
    Convenience function to roll TensorTypes along the given axis.

    Syntax copies numpy.roll function.

    Parameters
    ----------
    x : tensor_like
        Input tensor.
    shift : int (symbolic or literal)
        The number of places by which elements are shifted.
    axis : int (symbolic or literal), optional
        The axis along which elements are shifted. By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    tensor
        Output tensor, with the same shape as ``x``.

    """
    if axis is None:
        if x.ndim > 1:
            y = x.flatten()
            return roll(y, shift, axis=0).reshape(x.shape)
        else:
            axis = 0

    if axis < 0:
        axis += x.ndim

    # Shift may be larger than the size of the axis. If so, since the
    # roll operation is cyclic, we can take the shift modulo the size
    # of the axis
    shift = shift % x.shape[axis]

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
    """Reshape `t` by left-padding the shape with `n_ones` 1s.

    See Also
    --------
    shape_padaxis
    shape_padright
    Dimshuffle

    """
    _t = as_tensor_variable(t)

    pattern = ['x'] * n_ones + [i for i in xrange(_t.type.ndim)]
    return DimShuffle(_t.broadcastable, pattern)(_t)


@constructor
def shape_padright(t, n_ones=1):
    """Reshape `t` by right-padding the shape with `n_ones` 1s.

    See Also
    --------
    shape_padaxis
    shape_padleft
    Dimshuffle

    """
    _t = as_tensor_variable(t)

    pattern = [i for i in xrange(_t.type.ndim)] + ['x'] * n_ones
    return DimShuffle(_t.broadcastable, pattern)(_t)


@constructor
def shape_padaxis(t, axis):
    """Reshape `t` by inserting 1 at the dimension `axis`.

    Example
    -------
    >>> tensor = theano.tensor.tensor3()
    >>> theano.tensor.shape_padaxis(tensor, axis=0)
    DimShuffle{x,0,1,2}.0
    >>> theano.tensor.shape_padaxis(tensor, axis=1)
    DimShuffle{0,x,1,2}.0
    >>> theano.tensor.shape_padaxis(tensor, axis=3)
    DimShuffle{0,1,2,x}.0
    >>> theano.tensor.shape_padaxis(tensor, axis=-1)
    DimShuffle{0,1,2,x}.0

    See Also
    --------
    shape_padleft
    shape_padright
    Dimshuffle

    """
    _t = as_tensor_variable(t)

    ndim = _t.ndim + 1
    if not -ndim <= axis < ndim:
        msg = 'axis {0} is out of bounds [-{1}, {1})'.format(axis, ndim)
        raise IndexError(msg)
    if axis < 0:
        axis += ndim

    pattern = [i for i in xrange(_t.type.ndim)]
    pattern.insert(axis, 'x')
    return DimShuffle(_t.broadcastable, pattern)(_t)


@constructor
def stack(*tensors, **kwargs):
    """Stack tensors in sequence on given axis (default is 0).

    Take a sequence of tensors and stack them on given axis to make a single
    tensor. The size in dimension `axis` of the result will be equal to the number
    of tensors passed.

    Note: The interface stack(*tensors) is deprecated, you should use
    stack(tensors, axis=0) insted.

    Parameters
    ----------
    tensors : list or tuple of tensors
        A list of tensors to be stacked.
    axis : int
        The index of the new axis. Default value is 0.

    Examples
    --------
    >>> a = theano.tensor.scalar()
    >>> b = theano.tensor.scalar()
    >>> c = theano.tensor.scalar()
    >>> x = theano.tensor.stack([a, b, c])
    >>> x.ndim # x is a vector of length 3.
    1
    >>> a = theano.tensor.tensor4()
    >>> b = theano.tensor.tensor4()
    >>> c = theano.tensor.tensor4()
    >>> x = theano.tensor.stack([a, b, c])
    >>> x.ndim # x is a 5d tensor.
    5
    >>> rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
    >>> rval.shape # 3 tensors are stacked on axis 0
    (3, 2, 2, 2, 2)
    >>> x = theano.tensor.stack([a, b, c], axis=3)
    >>> x.ndim
    5
    >>> rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
    >>> rval.shape # 3 tensors are stacked on axis 3
    (2, 2, 2, 3, 2)
    >>> x = theano.tensor.stack([a, b, c], axis=-2)
    >>> x.ndim
    5
    >>> rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
    >>> rval.shape # 3 tensors are stacked on axis -2
    (2, 2, 2, 3, 2)
    """
    # ---> Remove this when moving to the new interface:
    if not tensors and not kwargs:
        raise Exception('theano.tensor.stack(tensors, axis) must have at least'
                        ' one parameter')

    if not kwargs and not isinstance(tensors[0], (list, tuple)):
        warnings.warn('stack(*tensors) interface is deprecated, use'
                      ' stack(tensors, axis=0) instead.', DeprecationWarning,
                      stacklevel=3)
        axis = 0
    elif 'tensors' in kwargs:
        tensors = kwargs['tensors']
        if 'axis' in kwargs:
            axis = kwargs['axis']
        else:
            axis = 0
    else:
        if len(tensors) == 2:
            axis = tensors[1]
        elif 'axis' in kwargs:
            axis = kwargs['axis']
        else:
            axis = 0
        tensors = tensors[0]
    # <--- Until here.

    if len(tensors) == 0:
        raise Exception('tensors is empty. You should at least provide one'
                        ' tensor to theano.tensor.stack(tensors, axis).')

    # If all tensors are scalars of the same type, call make_vector.
    # It makes the graph simpler, by not adding DimShuffles and Rebroadcasts

    # This should be an optimization!
    # Doing it here make the graph less canonicalized
    # (more type need to be understood by all optimization)
    # And DebugMode can't detect error in this code as it is not in an
    # optimization.
    # See ticket #660
    if np.all(
        [  # in case there is direct int in tensors.
            isinstance(t, (np.number, float, integer_types,
                           python_complex)) or
            (isinstance(t, Variable) and
             isinstance(t.type, TensorType) and
             t.ndim == 0)
            for t in tensors]):
        # in case there is direct int
        tensors = list(map(as_tensor_variable, tensors))
        dtype = scal.upcast(*[i.dtype for i in tensors])
        return theano.tensor.opt.MakeVector(dtype)(*tensors)
    return join(axis, *[shape_padaxis(t, axis) for t in tensors])


@constructor
def concatenate(tensor_list, axis=0):
    """Alias for `join`(axis, *tensor_list).

    This function is similar to `join`, but uses the signature of
    numpy's concatenate function.

    Raises
    ------
    TypeError
        The tensor_list must be a tuple or list.

    """
    # Check someone did not make the common mistake to do something like:
    #   c = concatenate(x, y)
    # instead of
    #   c = concatenate((x, y))
    if not isinstance(tensor_list, (tuple, list)):
        raise TypeError(
            "The 'tensors' argument must be either a tuple "
            "or a list, make sure you did not forget () or [] around "
            "arguments of concatenate.", tensor_list)
    return join(axis, *tensor_list)


def get_vector_length(v):
    """Return the run-time length of a symbolic vector.

    Parameters
    ----------
    v
        A rank-1 TensorType variable.

    Raises
    ------
    TypeError
        `v` hasn't the proper type.
    ValueError
        No special case applies, the length is not known.
        In general this is not possible, but for a number of special cases
        the length can be determined at compile / graph-construction time.
        This function implements these special cases.

    """
    v = as_tensor_variable(v)
    if v.ndim != 1:
        raise TypeError("argument must be symbolic vector, got '%s'" %
                        v)
    if v.type.broadcastable[0]:
        return 1
    if isinstance(v, gof.Constant) and v.type.ndim == 1:
        return len(v.data)
    if v.owner and isinstance(v.owner.op, theano.tensor.opt.MakeVector):
        return len(v.owner.inputs)
    if v.owner and isinstance(v.owner.op, Shape):
        return v.owner.inputs[0].type.ndim
    # If we take a slice, we know how many elements it will result in
    if ((v.owner and
         isinstance(v.owner.op, theano.tensor.subtensor.Subtensor) and
         isinstance(v.owner.op.idx_list[0], slice) and
         v.owner.inputs[0].owner and
         isinstance(v.owner.inputs[0].owner.op, theano.compile.ops.Shape))):
        start = extract_constant(theano.tensor.subtensor.get_idx_list(
            v.owner.inputs, v.owner.op.idx_list)[0].start)
        stop = extract_constant(theano.tensor.subtensor.get_idx_list(
            v.owner.inputs, v.owner.op.idx_list)[0].stop)
        step = extract_constant(theano.tensor.subtensor.get_idx_list(
            v.owner.inputs, v.owner.op.idx_list)[0].step)

        ndim = v.owner.inputs[0].owner.inputs[0].ndim
        types = (numbers.Integral, np.integer)
        if start is None:
            start = 0
        elif isinstance(start, types) and start < 0:
            start += ndim
            if start < 0:
                start = 0
        if stop is None:
            stop = ndim
        elif isinstance(stop, types):
            if stop > ndim:
                stop = ndim
            elif stop < 0:
                stop += ndim
        if step is None:
            step = 1

        if (isinstance(stop, types) and
                isinstance(start, types) and
                isinstance(step, types) and
                start >= 0 and stop >= 0 and
                step > 0 and stop >= start):
            return (stop - start - 1) // step + 1
    if isinstance(v, Variable):
        msg = theano.printing.debugprint(v, file='str')
    else:
        msg = str(v)
    raise ValueError("length not known: %s" % msg)


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
    known at graph build time.
    """
    view_map = {0: [0]}  # output 0 is potentially aliased to inputs [0]
    _f16_ok = True

    check_input = False
    __props__ = ("ndim",)
    params_type = ParamsType(ndim=int32)
    # name does not participate because it doesn't affect computations

    def __init__(self, ndim, name=None):
        self.ndim = int(ndim)
        if ndim < 0:
            raise ValueError("The output dimensions after reshape must be 0 or greater")
        assert name is None, 'name attribute for Reshape has been deprecated'

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.ndim)

    def make_node(self, x, shp):
        x = as_tensor_variable(x)
        shp_orig = shp
        shp = as_tensor_variable(shp, ndim=1)
        if not (shp.dtype in int_dtypes or
                (isinstance(shp, TensorConstant) and shp.data.size == 0)):
            # It raises an error if shp is not of integer type,
            # except when shp is constant and empty
            # (in this case, shp.dtype does not matter anymore).
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

    def perform(self, node, inp, out_, params):
        x, shp = inp
        out, = out_
        if (len(shp) != self.ndim):
            raise ValueError('shape argument to Reshape.perform has incorrect'
                             ' length %i'
                             ', should be %i' % (len(shp), self.ndim), shp)
        try:
            out[0] = np.reshape(x, shp)
        except Exception:
            raise ValueError('Cannot reshape input of shape %s to shape %s' %
                             (x.shape, shp))

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

        # If current variable is a scalar and its dimensionality should
        # change to self.ndim, then use size 1 for all new dimensions.
        if len(ishapes[0]) == 0:
            return [(1,) * self.ndim]

        requ = node.inputs[1]
        input_size = mul(*ishapes[0])
        if isinstance(requ, theano.tensor.TensorConstant):
            requ = list(requ.data)
            requ_part = [ele for ele in requ if ele != -1]
            crit = len(requ) - len(requ_part)
            if crit == 1 and len(requ_part) > 0:
                # If there are both 0 and -1 in requ_size, it is impossible
                # to determine a right output, but we can at least prevent
                # a division by 0. We do not want to keep a negative
                # size here as it could lead to further weird errors
                # after other optimizations.
                requ_size = mul(*requ_part)
                missing = input_size // (1 if requ_size == 0 else requ_size)
                for i, ele in enumerate(requ):
                    if ele == -1:
                        requ[i] = missing
            elif crit == 1:  # we reshape to -1
                requ = [input_size] if ishapes[0] else [1]
            elif crit > 1:
                raise ValueError('shape argument to Reshape.perform'
                                 ' must have at most one entry equal to -1')
            return [requ]
        else:
            requ = [requ[i] for i in xrange(self.ndim)]
            # since new_dims can have negative value (-1), the
            # multiplication of all values should be negated
            # to give a positive value.
            # To avoid optimization complexity, we avoid checking
            # for the case when there are two or more '-1' values.
            if self.ndim:
                requ_size = -mul(*requ)
                # If there are both 0 and -1 in requ_size, it is impossible
                # to determine a right output, but we can at least prevent
                # a division by 0. We do not want to keep a negative
                # size here as it could lead to further weird errors
                # after other optimizations.
                rest_size = input_size // maximum(requ_size, 1)
            return [tuple([switch(eq(requ[i], -1),
                                  rest_size,
                                  requ[i])
                           for i in xrange(self.ndim)])]

    def c_code_cache_version(self):
        return (8,)

    def c_code(self, node, name, inputs, outputs, sub):
        if isinstance(node.inputs[0], TensorVariable):
            x, shp = inputs
            z, = outputs
            sdtype = node.inputs[1].type.dtype_specs()[1]
            fail = sub['fail']
            params = sub['params']
            return """
            assert (PyArray_NDIM(%(shp)s) == 1);
            npy_intp new_dims[%(params)s->ndim];
            PyArray_Dims newshape;
            newshape.ptr = new_dims;
            newshape.len = %(params)s->ndim;
            for (int ii = 0; ii < %(params)s->ndim; ++ii)
            {
                // -- We do not want an explicit cast here. the shp can be any
                // -- int* dtype. The compiler will explicitly upcast it, but
                // -- will err if this will downcast. This could happen if the
                // -- user pass an int64 dtype, but npy_intp endup being int32.
                new_dims[ii] = ((%(sdtype)s*)(
                        PyArray_BYTES(%(shp)s) +
                        ii * PyArray_STRIDES(%(shp)s)[0]))[0];
            }
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject *) PyArray_Newshape(%(x)s, &newshape, NPY_CORDER);
            if (!%(z)s)
            {
                //The error message should have been set by PyArray_Newshape
                %(fail)s;
            }
            """ % locals()
        else:
            return Op.c_code(self, node, name, inputs, outputs, sub)


def reshape(x, newshape, ndim=None):
    if ndim is None:
        newshape = as_tensor_variable(newshape)
        if newshape.ndim != 1:
            raise TypeError(
                "New shape in reshape must be a vector or a list/tuple of"
                " scalar. Got %s after conversion to a vector." % newshape)
        try:
            ndim = get_vector_length(newshape)
        except ValueError:
            raise ValueError(
                "The length of the provided shape (%s) cannot "
                "be automatically determined, so Theano is not able "
                "to know what the number of dimensions of the reshaped "
                "variable will be. You can provide the 'ndim' keyword "
                "argument to 'reshape' to avoid this problem." % newshape)
    op = Reshape(ndim)
    rval = op(x, newshape)
    return rval


class Flatten(Op):
    """
    Flatten a tensor.

    Flattens a tensor to `outdim` dimensions by preserving the leading
    outdim - 1 shape components.

    .. note:: The interface Flatten(Op) is deprecated, you should use flatten.
    """
    view_map = {0: [0]}

    check_input = False
    __props__ = ("outdim",)

    def __init__(self, outdim=1):
        warnings.warn(
            "Flatten class is deprecated, "
            "please use flatten method instead.",
            DeprecationWarning,
            stacklevel=4)
        self.outdim = int(outdim)

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.outdim)

    def make_node(self, x):
        t_x = as_tensor_variable(x)
        if self.outdim < 1 or (x.ndim and self.outdim > x.ndim):
            raise ValueError('invalid output ndimensions (%i) for tensor of '
                             'rank %i' % (self.outdim, t_x.ndim))

        # Infer the broadcastable pattern of the output. For every dimension
        # unaffected by the flatten, the broadcast flag should be unchanged.
        # For the dimension resulting from the collapse of other dimensions,
        # it should be broadcastable iff all the collapsed dimensions were
        # broadcastable.
        bcast_kept_dims = x.broadcastable[:self.outdim - 1]
        bcast_new_dim = python_all(x.broadcastable[self.outdim - 1:])
        broadcastable = bcast_kept_dims + (bcast_new_dim,)

        return gof.Apply(self, [t_x], [tensor(x.type.dtype,
                                              broadcastable)])

    def perform(self, node, inp, out_):
        x, = inp
        out, = out_
        outdim = self.outdim
        if outdim == 1:
            try:
                out[0] = x.reshape(x.size)
            except AttributeError:
                out[0] = x.reshape((np.prod(x.shape),))
        elif outdim == len(x.shape):
            out[0] = x
        else:
            newshape = (x.shape[:outdim - 1] +
                        (np.prod(x.shape[outdim - 1:]),))
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

    def c_code_cache_version(self):
        return (1, 1)

    def c_code(self, node, name, inputs, outputs, sub):
        x, = inputs
        out, = outputs
        outdim = self.outdim
        fail = sub['fail']
        return """
        if (%(outdim)s == PyArray_NDIM(%(x)s))
        {
            Py_XDECREF(%(out)s);
            Py_XINCREF(%(x)s);
            %(out)s = %(x)s;
        }
        else
        {
            Py_XDECREF(%(out)s);

            if (%(outdim)s == 1)
            {
                npy_intp size = PyArray_SIZE(%(x)s);
                PyArray_Dims newshape;
                newshape.ptr = &size;
                newshape.len = 1;
                %(out)s = (PyArrayObject*)PyArray_Newshape(%(x)s,
                                                           &newshape,
                                                           NPY_CORDER);
            }
            else
            {
                npy_intp *oldshape = PyArray_DIMS(%(x)s);
                npy_intp newshape_dims[%(outdim)s];

                int i;
                for (i = 0; i < %(outdim)s - 1; ++i)
                    newshape_dims[i] = oldshape[i];

                newshape_dims[i] = 1;

                for (int j = %(outdim)s - 1; j < PyArray_NDIM(%(x)s); ++j)
                    newshape_dims[i] *= oldshape[j];

                PyArray_Dims newshape;
                newshape.ptr = newshape_dims;
                newshape.len = %(outdim)s;
                %(out)s = (PyArrayObject*)PyArray_Newshape(%(x)s,
                                                           &newshape,
                                                           NPY_CORDER);
            }
        }
        if (!%(out)s)
        {
            //The error message should have been set by
            // PyArray_Newshape
            %(fail)s;
        }
        """ % locals()


def is_flat(var, ndim=None, outdim=None):
    """
    Verifies the dimensionality of the var is equal to
    outdim. This method is usually called after flatten method on a
    variable, where the first outdim-1 dimension size(s) of the variable
    is kept intact, and the last dimension size of the variable is made
    equal to the multiplication of its remaining dimension size(s), such that
    the variable would end up with as many dimension as outdim.

    Parameters
    ----------
        var : theano.tensor.var.TensorVariable
            the theano var on which the dimensionality is checked.

        outdim : int
            the expected dimensionality of var.

    Returns
    -------
    bool
        the comparison result of var's dim
        and the expected outdim.
    """
    if outdim is None and ndim is None:
        ndim = 1
    elif outdim is not None and ndim is not None:
        raise ValueError("You should only specify ndim")
    elif outdim is not None:
        warnings.warn(
            "flatten outdim parameter is deprecated, use ndim instead.")
        ndim = outdim
    return var.ndim == ndim


def flatten(x, ndim=None, outdim=None):
    """
    Reshapes the variable x by keeping
    the first outdim-1 dimension size(s) of x the same,
    and making the last dimension size of x equal to
    the multiplication of its remaining dimension size(s).

    Parameters
    ----------
        x : theano.tensor.var.TensorVariable
            the variable that should be reshaped.

        ndim : int
            the number of dimensions of the returned variable
            Default 1.
        outdim : int
            DEPRECATED synonym for ndim
    Returns
    -------
    theano.tensor.var.TensorVariable
        the flattend variable with dimensionality of outdim
    """
    if outdim is None and ndim is None:
        ndim = 1
    elif outdim is not None and ndim is not None:
        raise ValueError("You should only specify ndim")
    elif outdim is not None:
        warnings.warn(
            "flatten outdim parameter is deprecated, use ndim instead.")

        ndim = outdim
    # Any input variable can be flattened to have ndim of 1,
    # even if it's a scalar. Otherwise, ndim must be positive
    # and smaller than x.ndim.
    if ndim < 1 or (ndim > 1 and ndim > x.ndim):
        raise ValueError('ndim %s out of bound [1, %d)'
                         % (ndim, x.ndim + 1))

    if ndim > 1:
        dims = tuple(x.shape[:ndim - 1]) + (-1,)
    else:
        dims = (-1,)
    x_reshaped = x.reshape(dims)
    bcast_kept_dims = x.broadcastable[:ndim - 1]
    bcast_new_dim = python_all(x.broadcastable[ndim - 1:])
    broadcastable = bcast_kept_dims + (bcast_new_dim,)
    x_reshaped = theano.tensor.addbroadcast(
        x_reshaped, *filter(lambda i: broadcastable[i], range(ndim)))
    return x_reshaped


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

    .. note:: Deprecated
              Use tile() instead.

    Tiles its input according to reps. The length of reps is the number of
    dimension of x and contains the number of times to tile x in each
    dimension.

    See Also
    --------
    numpy.tile : http://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html

    """
    __props__ = ("ndim",)

    def __init__(self, ndim):
        self.ndim = ndim

    def __str__(self):
        return self.__class__.__name__ + "{ndim=%d}" % self.ndim

    def make_node(self, x, reps):
        warnings.warn((
            "Tile op is deprecated, use tile function instead."), stacklevel=3)
        x = as_tensor_variable(x)
        reps = as_tensor_variable(reps)
        return gof.Apply(self, [x, reps], [tensor(x.type.dtype, [False] *
                                                  self.ndim)])

    def perform(self, node, inp, out_):
        x, reps = inp
        out, = out_
        res = np.tile(x, reps)
        if res.ndim != self.ndim:
            raise ValueError(
                'Tile.perform produced incorrect number of dimensions')

        if (np.asarray(reps) == 1).all():
            # In that case, some NumPy version return a view!  As this
            # op isn't declared as inplace, we need to check that and
            # copy the data.
            if np.may_share_memory(res, x):
                res = res.copy()
        out[0] = res

    def infer_shape(self, node, in_shapes):
        # Note: in contrast with numpy, it is assumed that x.shape and reps
        # have equal length;  see also tile function below

        # Note: if reps were to be allowed not to be a constant and x.shape
        # and reps to be unequal, the following block of code could be used:
        # prepend 1 to x.shape if needed
        # if self.ndim > x.ndim:
        # shp = concatenate(ones(self.ndim - x.ndim), shp)
        # prepend 1 to reps if needed
        # reps = concatenate(ones(self.ndim - reps.shape[0]), reps)

        x, reps = node.inputs
        shp = in_shapes[0]
        tiled_shp = shp * reps
        out_shape = []
        for i in xrange(self.ndim):
            out_shape.append(tiled_shp[i])
        return [out_shape]

    def grad(self, inp, grads):
        x, reps = inp
        g_out, = grads
        # return [tilegrad(x, reps, g_out), None]
        raise NotImplementedError()


def tile(x, reps, ndim=None):
    """
    Tile input array `x` according to `reps`.

    See the docstring of `numpy.tile` for details.

    'reps' can be constant integer (e.g. 3), constant vector(e.g. [2 3]),
    symbolic scalar (e.g. tensor.iscalar()), symbolic vector (e.g. tensor.ivector())
    or a list of symbolic scalar (e.g. [tensor.iscalar(), tensor.iscalar()]).

    ndim is the number of the dimensions of the output, if it is provided, ndim
    should be equal or larger than x.ndim and len(reps), otherwise, we will use
    max(x.ndim, len(reps)) as ndim. If reps is symbolic vector, the ndim has to
    be provided.

    """

    if ndim is not None and ndim < x.ndim:
        raise ValueError("ndim should be equal or larger than x.ndim")

    # if reps is tensor.scalar, integer or tensor.vector, we convert it to a list.
    if not isinstance(reps, (list, tuple)):
        reps_astensor = as_tensor_variable(reps)
        ndim_check = reps_astensor.ndim
        if reps_astensor.dtype not in theano.tensor.discrete_dtypes:
            raise ValueError("elements of reps must be integer dtype")

        # tensor.scalar/integer case
        if ndim_check == 0:
            reps = [reps]

        # tensor.vector case
        elif ndim_check == 1:
            if ndim is None:
                raise ValueError("if reps is tensor.vector, you should specify "
                                 "the ndim")
            else:
                offset = ndim - reps.shape[0]

                # assert that reps.shape[0] does not exceed ndim
                offset = theano.tensor.opt.assert_(offset, ge(offset, 0))

                # if reps.ndim is less than x.ndim, we pad the reps with
                # "1" so that reps will have the same ndim as x.
                reps_ = [switch(i < offset, 1, reps[i - offset]) for i in range(ndim)]
                reps = reps_

        # other raise error
        else:
            raise ValueError("the dimension of reps should not exceed 1")
    else:
        if ndim is not None and len(reps) > ndim:
            raise ValueError("len(reps) should be equal or less than ndim")
        if not np.all([isinstance(r, integer_types) or
                       (isinstance(r, TensorVariable) and
                        r.dtype in theano.tensor.discrete_dtypes) for r in reps]):
            raise ValueError("elements of reps must be scalars of integer dtype")

    # if reps.ndim is less than x.ndim, we pad the reps with
    # "1" so that reps will have the same ndim as x.
    reps = list(reps)
    if ndim is None:
        ndim = builtins.max(len(reps), x.ndim)
    if len(reps) < ndim:
        reps = [1] * (ndim - len(reps)) + reps

    shape = [1] * (ndim - x.ndim) + [x.shape[i] for i in xrange(x.ndim)]
    alloc_shape = reps + shape
    y = alloc(x, *alloc_shape)
    shuffle_ind = np.arange(ndim * 2).reshape(2, ndim)
    shuffle_ind = shuffle_ind.transpose().flatten()
    y = y.dimshuffle(*shuffle_ind)
    new_shapes = [sh * reps[i] for i, sh in enumerate(shape)]
    y = y.reshape(new_shapes)

    return y


class ARange(Op):
    """Create an array containing evenly spaced values within a given interval.

    Parameters and behaviour are the same as numpy.arange().

    """
    __props__ = ("dtype",)

    def __init__(self, dtype):
        self.dtype = dtype

    def make_node(self, start, stop, step):
        start, stop, step = map(as_tensor_variable, (start, stop, step))
        assert start.ndim == 0
        assert stop.ndim == 0
        assert step.ndim == 0

        inputs = [start, stop, step]
        outputs = [tensor(self.dtype, (False,))]

        return Apply(self, inputs, outputs)

    @theano.configparser.change_flags(warn_float64='ignore')
    def infer_shape(self, node, i_shapes):
        # Note start, stop and step can be float numbers.
        start, stop, step = node.inputs

        def is_constant_value(var, value):
            try:
                v = get_scalar_constant_value(var)
                return np.all(v == value)
            except NotScalarConstantError:
                pass
            return False

        def upcast(var):
            if (var.dtype in integer_dtypes and
                    # We do not want to cast uint64 to int64 as this can
                    # loose information. If we upcast uint64 with int64,
                    # this give float64. This is safer then checking for
                    # uint64 in case we support [u]int128 or other in the
                    # future.
                    scal.upcast(var.dtype, 'int64') == 'int64'):
                return cast(var, 'int64')
            return var

        if is_constant_value(step, 1):
            if is_constant_value(start, 0):
                return [(cast(stop, 'int64'),)]
            else:
                stop = upcast(stop)
                start = upcast(start)
                return [(maximum(cast(stop - start, 'int64'), 0),)]
        else:
            stop = upcast(stop)
            start = upcast(start)
            return [(maximum(cast(ceil(cast((stop - start), 'float64') / step),
                    'int64'), 0),)]

    def perform(self, node, inp, out_):
        start, stop, step = inp
        out, = out_
        start = start.item()
        stop = stop.item()
        step = step.item()
        out[0] = np.arange(start, stop, step, dtype=self.dtype)

    def connection_pattern(self, node):

        return [[True], [False], [True]]

    def L_op(self, inputs, outputs, grads):
        start, stop, step = inputs
        gz, = grads
        # `start` and `step` affect the output values
        # but the outputs are integers so there's
        # no gradient through them.
        # When they are not integers, the gradients are
        # as expressed below.
        # `stop` does not affect the output values,
        # just the output shape, so it is disconnected.

        if self.dtype in discrete_dtypes:
            return [start.zeros_like(dtype=config.floatX),
                    DisconnectedType()(),
                    step.zeros_like(dtype=config.floatX)]
        else:
            num_steps_taken = outputs[0].shape[0]
            return [gz.sum(),
                    DisconnectedType()(),
                    (gz * arange(num_steps_taken, dtype=self.dtype)).sum()]

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
        # don't try to be stingy and byte-optimize, this leads to
        # overflow problems.
        if dtype in int_dtypes:
            dtype = 'int64'
        if dtype in uint_dtypes:
            dtype = 'uint64'
        if config.cast_policy in ('numpy', 'numpy+floatX'):
            # We enforce numpy semantics, except in the special case where
            # `config.cast_policy` is 'numpy+floatX' and we want to use float32
            # rather than float64.
            # As an example, if `start`, `stop` and `step` are all int32,
            # `numpy.arange` returns an int64 array (on 64-bit platforms),
            # while the upcast above returns int32.
            numpy_dtype = np.arange(
                start=np.array(0, dtype=start.dtype),
                stop=np.array(1, dtype=stop.dtype),
                step=np.array(1, dtype=step.dtype)).dtype
            if numpy_dtype != dtype:
                if (config.cast_policy == 'numpy+floatX' and
                    config.floatX == 'float32' and
                    numpy_dtype == 'float64' and
                    # No explicit float64 in the three arguments?
                    python_all(
                        dt != 'float64'
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


class _nd_grid(object):
    """Create a dense n-dimensional 'meshgrid' with equally spaced points.

    Used to create the instance ``mgrid`` and ``ogrid`` which act similarly
    to their numpy equivalents.

    Parameters
    ----------
    sparse : boolean, optional, default=True
        Specifying False leads to the equivalent of numpy's mgrid functionality.
        Specifying True leads to the equivalent of ogrid.

    Examples
    --------
    >>> a = T.mgrid[0:5, 0:3]
    >>> a[0].eval()
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2],
           [3, 3, 3],
           [4, 4, 4]], dtype=int8)
    >>> a[1].eval()
    array([[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]], dtype=int8)
    >>> b = T.ogrid[0:5, 0:3]
    >>> b[0].eval()
    array([[0],
           [1],
           [2],
           [3],
           [4]], dtype=int8)
    >>> b[1].eval()
    array([[0, 1, 2, 3]], dtype=int8)

    """

    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, *args):

        ndim = len(args[0])
        for sl in args[0]:
            if isinstance(sl.step, python_complex):
                raise NotImplementedError("Not implemented for slices "
                                          "whose step is complex")
        ranges = [arange(sl.start or 0,
                         sl.stop,
                         sl.step or 1) for sl in args[0]]
        shapes = [tuple([1] * j + [r.shape[0]] + [1] * (ndim - 1 - j))
                  for j, r in enumerate(ranges)]
        ranges = [r.reshape(shape) for r, shape in zip(ranges, shapes)]
        if self.sparse:
            grids = ranges
        else:
            grids = []
            ones = [ones_like(r) for r in ranges]
            for i in range(ndim):
                grid = 1
                for j in range(ndim):
                    if j == i:
                        grid = grid * ranges[j]
                    else:
                        grid = grid * ones[j]
                grids.append(grid)
        return grids


mgrid = _nd_grid()
ogrid = _nd_grid(sparse=True)


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
    __props__ = ()

    def make_node(self, x, y, inverse):
        x = as_tensor_variable(x)
        y = as_tensor_variable(y)
        if inverse:  # as_tensor_variable does not accept booleans
            inverse = as_tensor_variable(1)
        else:
            inverse = as_tensor_variable(0)

        # y should contain integers
        assert y.type.dtype in integer_dtypes
        # Inverse should be an integer scalar
        assert (inverse.type.ndim == 0 and inverse.type.dtype in integer_dtypes)

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

        Parameters
        ----------
        x : tensor
            The input tensor, on which the permutation is applied.
        y : tensor
            Tensor containing the permutations to apply.
        out : tensor
            Tensor storing the output result.
        curdim : int
            Counter of the current depth of recursion.
        inverse
            Wether to apply permutations or their inverse.

        """
        if len(x.shape) == 1:
            # Numpy advanced indexing works in this case
            if inverse:
                out[y] = x[:]
            else:
                out[:] = x[y]
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
            outs[0] = np.empty(out_s, dtype=x.dtype)

        self._rec_perform(node, x, y, inverse, outs[0], curdim=0)

    def infer_shape(self, node, in_shapes):
        shp_x = in_shapes[0]
        shp_y = in_shapes[1]
        assert len(shp_x) == len(shp_y)
        out_shape = []
        for i in xrange(len(shp_x)):
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
                            if x.type.broadcastable[dim] and
                            not gz.type.broadcastable[dim]]
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
        if x.type.dtype in discrete_dtypes:
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

    Notes
    -----
    Matrix-matrix products are sometimes optimized to Dot22 or Gemm ops
    (see tensor.blas).
    Vector-vector products are sometimes optimized to Ger or CGer (see
    tensor.blas).
    Matrix-vector products are sometimes optimized to Gemv, CGemv (see
    tensor.blas).

    """
    __props__ = ()

    # the rationale for Dot22 is related to getting GEMM Ops into the
    # graph.  See Dot22 in tensor.blas for details.

    def make_node(self, *inputs):
        inputs = list(map(as_tensor_variable, inputs))

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
        z[0] = np.asarray(np.dot(x, y))

    def grad(self, inp, grads):

        x, y = inp
        gz, = grads
        xdim, ydim, gdim = x.type.ndim, y.type.ndim, gz.type.ndim

        # grad is scalar, so x is vector and y is vector
        if gdim == 0:
            xgrad = gz * y
            ygrad = gz * x

        # x is vector, y is matrix, grad is vector
        elif xdim == 1 and ydim == 2:
            xgrad = dot(gz, y.T)
            ygrad = outer(x.T, gz)

        # x is matrix, y is vector, grad is vector
        elif xdim == 2 and ydim == 1:
            xgrad = outer(gz, y.T)
            ygrad = dot(x.T, gz)

        # x is matrix, y is matrix, grad is matrix
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
    Computes the dot product of two variables.

    For two matrices, this is equivalent to matrix multiplication.
    For two vectors, this is the inner product.
    When one variable is a scalar, this is like elementwise multiplication.
    For N dimensions, this is a sum product over the last axis
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

    Notes
    -----
    Matrix-matrix products are sometimes optimized to Dot22 or Gemm ops
    (see tensor.blas).
    Vector-vector products are sometimes optimized to Ger or CGer (see
    tensor.blas).
    Matrix-vector products are sometimes optimized to Gemv, CGemv (see
    tensor.blas).

    """
    a, b = as_tensor_variable(a), as_tensor_variable(b)

    if a.ndim == 0 or b.ndim == 0:
        return a * b
    elif a.ndim > 2 or b.ndim > 2:
        return tensordot(a, b, [[a.ndim - 1], [np.maximum(0, b.ndim - 2)]])
    else:
        return _dot(a, b)


#########################
# Linalg : TensorDot
#########################

def _tensordot_as_dot(a, b, axes, dot, batched):
    """
    Reduces a tensor dot product to a matrix or vector dot product. Based
    on code from Tijmen Tieleman's gnumpy
    (http://www.cs.toronto.edu/~tijmen/gnumpy.html).

    Please see the documentation of tensordot for the meaning of the a, b
    and axes arguments.

    :param dot: a function that accepts two symbolic variables and computes
                the appropriate dot product (e.g. dot, batched_dot)
    :type dot: function

    :param batched: whether to treat the first axis of a and b as a batch
                    axis.  If so, this axis will be preserved in the output,
                    allowing this function to be used also for batched
                    tensor dot products.
    :type batched: boolean

    :returns: a tensor with shape equal to the concatenation of a's shape
              (less any dimensions that were summed over) and b's shape
              (less the first dimension and any dimensions that were summed
              over).
    :rtype: symbolic tensor
    """
    a, b = as_tensor_variable(a), as_tensor_variable(b)

    if not np.isscalar(axes) and len(axes) != 2:
        raise ValueError('Axes should be an integer or a '
                         'list/tuple of len 2 (%s was provided)'
                         % str(axes))

    # if 'axes' is a number of axes to multiply and sum over (trailing axes
    # of a, leading axes of b), we can just reshape and use dot.
    elif np.isscalar(axes):
        axes = int(axes)

        for operand_name, operand in (("a", a), ("b", b)):
            if axes > operand.ndim:
                raise ValueError(
                    'axes can not be larger than the dimension of %s '
                    '(%s.ndim=%i, axes=%i)'
                    % (operand_name, operand_name, operand.ndim, axes))
            if batched and axes == operand.ndim:
                raise ValueError(
                    'axes to sum over must not include the batch axis '
                    'of %s (%s.ndim=%i, axes=%i)'
                    % (operand_name, operand_name, operand.ndim, axes))

        batch_axes = 1 if batched else 0
        a_outaxes = slice(0, a.ndim - axes)
        b_outaxes = slice(batch_axes + axes, b.ndim)
        outshape = concatenate([a.shape[a_outaxes], b.shape[b_outaxes]])
        outbcast = a.broadcastable[a_outaxes] + b.broadcastable[b_outaxes]
        outndim = len(outbcast)

        a_shape = [1] * 2
        b_shape = [1] * 2

        # compute total size of summed axes
        for i in xrange(0, axes):
            a_shape[1] *= a.shape[-(i + 1)]
            b_shape[0] *= b.shape[batch_axes + i]
        # compute total size of other axes
        for i in xrange(0, a.ndim - axes - batch_axes):
            a_shape[0] *= a.shape[batch_axes + i]
        for i in xrange(0, b.ndim - axes - batch_axes):
            b_shape[1] *= b.shape[-(i + 1)]

        if batched:
            a_shape.insert(0, a.shape[0])
            b_shape.insert(0, b.shape[0])

        a_reshaped = a.reshape(a_shape)
        b_reshaped = b.reshape(b_shape)

        out_reshaped = dot(a_reshaped, b_reshaped)
        out = out_reshaped.reshape(outshape, outndim)
        # Make sure the broadcastable pattern of the result is correct,
        # since some shape information can be lost in the reshapes.
        return patternbroadcast(out, outbcast)

    # if 'axes' is a list, transpose a and b such that the summed axes of a
    # are last and the summed axes of b are first.
    else:
        axes = [_pack(axes_) for axes_ in axes]

        if len(axes[0]) != len(axes[1]):
            raise ValueError('Axes elements must have the same length.')

        for i, (operand_name, operand) in enumerate((("a", a),
                                                     ("b", b))):
            if len(axes[i]) > operand.ndim:
                raise ValueError(
                    'axes[%i] should be array_like with length less than '
                    'the dimensions of %s (%s.ndim=%i, len(axes[0])=%i).' %
                    (i, operand_name, operand_name, operand.ndim,
                     len(axes[i])))
            if len(axes[i]) > 0 and np.max(axes[i]) >= operand.ndim:
                raise ValueError(
                    'axes[%i] contains dimensions greater than or equal '
                    'to %s.ndim (%s.ndim=%i, max(axes[0])=%i).' %
                    (i, operand_name, operand_name, operand.ndim,
                     np.max(np.array(axes[i]))))
            if batched and 0 in axes[i]:
                raise ValueError(
                    'axes to sum over must not contain the batch axis '
                    '(axes[%i]=%s)' %
                    (i, axes[i]))

        batch_axes = [0] if batched else []
        other_axes = [[x for x in xrange(operand.ndim)
                       if x not in axes[i] and x not in batch_axes]
                      for i, operand in enumerate((a, b))]

        a_shuffled = a.dimshuffle(batch_axes + other_axes[0] + axes[0])
        b_shuffled = b.dimshuffle(batch_axes + axes[1] + other_axes[1])

        # now that a and b are in the right order, recur with integer axes
        return _tensordot_as_dot(a_shuffled, b_shuffled, len(axes[0]),
                                 dot=dot, batched=batched)


def tensordot(a, b, axes=2):
    """
    Compute a generalized dot product over provided axes.

    Given two tensors a and b, tensordot computes a generalized dot product over
    the provided axes. Theano's implementation reduces all expressions to
    matrix or vector dot products and is based on code from Tijmen Tieleman's
    gnumpy (http://www.cs.toronto.edu/~tijmen/gnumpy.html).

    Parameters
    ----------
    a: symbolic tensor
        The first tensor variable.
    b: symbolic tensor
        The second tensor variable
    axes: int or array-like of length 2
        If an integer, the number of axes to sum over.
        If an array, it must have two array elements containing the axes
        to sum over in each tensor.

        Note that the default value of 2 is not guaranteed to work
        for all values of a and b, and an error will be raised if
        that is the case. The reason for keeping the default is to
        maintain the same signature as numpy's tensordot function
        (and np.tensordot raises analogous errors for non-compatible
        inputs).

        If an integer i, it is converted to an array containing
        the last i dimensions of the first tensor and the first
        i dimensions of the second tensor:
            axes = [list(range(a.ndim - i, b.ndim)), list(range(i))]

        If an array, its two elements must contain compatible axes
        of the two tensors. For example, [[1, 2], [2, 0]] means sum
        over the 2nd and 3rd axes of a and the 3rd and 1st axes of b.
        (Remember axes are zero-indexed!) The 2nd axis of a and the
        3rd axis of b must have the same shape; the same is true for
        the 3rd axis of a and the 1st axis of b.

    Returns
    -------
    symbolic tensor
        A tensor with shape equal to the concatenation of a's shape
        (less any dimensions that were summed over) and b's shape
        (less any dimensions that were summed over).

    Examples
    --------
    It may be helpful to consider an example to see what tensordot does.
    Theano's implementation is identical to NumPy's. Here a has shape (2, 3, 4)
    and b has shape (5, 6, 4, 3). The axes to sum over are [[1, 2], [3, 2]] --
    note that a.shape[1] == b.shape[3] and a.shape[2] == b.shape[2]; these axes
    are compatible. The resulting tensor will have shape (2, 5, 6) -- the
    dimensions that are not being summed:

    >>> a = np.random.random((2,3,4))
    >>> b = np.random.random((5,6,4,3))

    #tensordot
    >>> c = np.tensordot(a, b, [[1,2],[3,2]])

    #loop replicating tensordot
    >>> a0, a1, a2 = a.shape
    >>> b0, b1, _, _ = b.shape
    >>> cloop = np.zeros((a0,b0,b1))

    #loop over non-summed indices -- these exist
    #in the tensor product.
    >>> for i in range(a0):
    ...     for j in range(b0):
    ...         for k in range(b1):
    ...             #loop over summed indices -- these don't exist
    ...             #in the tensor product.
    ...             for l in range(a1):
    ...                 for m in range(a2):
    ...                     cloop[i,j,k] += a[i,l,m] * b[j,k,m,l]

    >>> np.allclose(c, cloop)
    true

    This specific implementation avoids a loop by transposing a and b such that
    the summed axes of a are last and the summed axes of b are first. The
    resulting arrays are reshaped to 2 dimensions (or left as vectors, if
    appropriate) and a matrix or vector dot product is taken. The result is
    reshaped back to the required output dimensions.

    In an extreme case, no axes may be specified. The resulting tensor
    will have shape equal to the concatenation of the shapes of a and b:

    >>> c = np.tensordot(a, b, 0)
    >>> print(a.shape)
    (2,3,4)
    >>> print(b.shape)
    (5,6,4,3)
    >>> print(c.shape)
    (2,3,4,5,6,4,3)

    See the documentation of numpy.tensordot for more examples.

    """
    return _tensordot_as_dot(a, b, axes, dot=dot, batched=False)


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


# Some NumPy version like 1.9.2 return a view for numpy.diagonal
x = np.zeros((4, 4))
numpy_diagonal_return_view = np.may_share_memory(np.diagonal(x), x)
del x


class ExtractDiag(Op):
    """
    Return specified diagonals.

    If x is 2-D, returns the diagonal of x with the given offset,
    i.e., the collection of elements of the form x[i, i+offset].
    If x has more than two dimensions, then the axes specified by
    axis1 and axis2 are used to determine the 2-D sub-array whose
    diagonal is returned. The shape of the resulting array can be
    determined by removing axis1 and axis2 and appending an index
    to the right equal to the size of the resulting diagonals.

    Parameters
    ----------
    x: A tensor variable with x.ndim >= 2.

    offset: Offset of the diagonal from the main diagonal.
        Can be positive or negative.
        Defaults to main diagonal (0).

    axis1: Axis to be used as the first axis of the 2-D
        sub-arrays from which the diagonals should be taken.
        Defaults to first axis (0).

    axis2: Axis to be used as the second axis of the 2-D
        sub-arrays from which the diagonals should be taken.
        Defaults to second axis (1).



    Returns
    -------
    array_of_diagonals:
        If x is 2-D, a 1-D array of the same type as a
        containing the diagonal is returned.
        If the dimension of x is greater than two, then an
        array of diagonals is returned, "packed" from left-most
        dimension to right-most (e.g., if x is 3-D, then the
        diagonals are "packed" along rows).



    Raises
    ------
    ValueError
        If the dimension of x is less than 2.


    See Also
    --------
    numpy.diagonal:
        https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.diagonal.html
    """
    __props__ = ("offset", "axis1", "axis2", "view")

    def __init__(self, offset=0, axis1=0, axis2=1, view=False):
        self.view = view
        if self.view and not numpy_diagonal_return_view:
            warnings.warn("View will forced to False. ExtractDiag property view is "
                          "set to True but numpy version %s and prior versions of "
                          "numpy.diagonal() do not return a view. Update "
                          "numpy to use ExtractDiag(view=True)" %
                          np.version.version)
            self.view = False
        if self.view:
            self.view_map = {0: [0]}
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def make_node(self, x):
        x = as_tensor_variable(x)

        if x.ndim < 2:
            raise ValueError('ExtractDiag needs an input with 2 or more '
                             'dimensions', x)
        return Apply(self, [x], [x.type.__class__(
            dtype=x.dtype,
            broadcastable=[False] * (x.ndim - 1))()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = x.diagonal(self.offset, self.axis1, self.axis2)
        if not self.view:
            z[0] = z[0].copy()

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout

        if x.ndim == 2:
            x = theano.tensor.zeros_like(x)
            xdiag = theano.tensor.AllocDiag(offset=self.offset)(gz)
            return [theano.tensor.set_subtensor(
                x[:xdiag.shape[0], :xdiag.shape[1]], xdiag)]
        else:
            warnings.warn("gradient of theano.tensor.basic.ExtractDiag only"
                          "works for matrices.")
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

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.view and not numpy_diagonal_return_view:
            warnings.warn("View will forced to False. ExtractDiag property view is "
                          "set to True but numpy version %s and prior versions of "
                          "numpy.diagonal() do not return a view. Update "
                          "numpy to use ExtractDiag(view=True)" %
                          np.version.version)
            self.view = False

        if self.view:
            self.view_map = {0: [0]}

        if "offset" not in state:
            self.offset = 0
        if "axis1" not in state:
            self.axis1 = 0
        if "axis2" not in state:
            self.axis2 = 1


def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    A helper function for `theano.tensor.ExtractDiag`. It accepts tensor with
    `ndim >= 2` as input. The name `diagonal` is just meant to keep it
    consistent with numpy.

    Parameters
    ----------
    a : symbolic tensor
    offset : int
        offset
    axis1 : int
    axis2 : int

    Returns
    -------
    tensor : symbolic tensor

    """
    return ExtractDiag(offset, axis1, axis2)(a)


class AllocDiag(Op):
    """
    An op that copies a vector to the diagonal of an empty matrix. It does the
    inverse of ExtractDiag.

    Usage: T.AllocDiag()(x)

    `x` should be a tensor vector. The parenthesis in the front should indicate
    which main diagonal the vector value goes into. By default it is set to
    `0`, which corresponds to setting the values of x to the main diagonal in
    the returned matrix.

    Parameters
    ----------
    axis1: Axis to be used as the first axis of the 2-D
        sub-arrays to which the diagonals will be allocated.
        Defaults to first axis (0).

    axis2: Axis to be used as the second axis of the 2-D
        sub-arrays to which the diagonals will be allocated.
        Defaults to second axis (1).

    offset: Offset of the diagonal from the main diagonal defined by `axis1`
        and `axis2`.
        Can be positive or negative.
        Defaults to main diagonal (0).

    x: symbolic vector
        A tensor vector consists of diagonal values.

    Returns
    -------
    tensor : symbolic tenstor
        A tensor with passed tensor values at their corresponding diagonals.

    """

    __props__ = ("offset", "axis1", "axis2")

    def __init__(self, offset=0, axis1=0, axis2=1):
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def make_node(self, diag):
        diag = as_tensor_variable(diag)
        if diag.type.ndim < 1:
            raise ValueError('AllocDiag needs an input with 1 or more '
                             'dimensions', diag.type)
        return Apply(
            self, [diag],
            [diag.type.__class__(
                dtype=diag.dtype,
                broadcastable=[False] * (diag.ndim + 1))()]
        )

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs

        axis1 = np.minimum(self.axis1, self.axis2)
        axis2 = np.maximum(self.axis1, self.axis2)
        offset = self.offset

        # Create array with one extra dimension for resulting matrix
        result_shape = x.shape[:-1] + (x.shape[-1] + abs(offset),) * 2
        result = np.zeros(result_shape, dtype=x.dtype)

        # Create slice for diagonal in final 2 axes
        idxs = np.arange(x.shape[-1])
        diagonal_slice = ((len(result_shape) - 2) * [slice(None)] +
                          [idxs + np.maximum(0, -offset),
                           idxs + np.maximum(0, offset)])

        # Fill in final 2 axes with x
        result[diagonal_slice] = x

        if len(x.shape) > 1:
            # Re-order axes so they correspond to diagonals at axis1, axis2
            axes = list(range(len(x.shape[:-1])))
            last_idx = axes[-1]
            axes = axes[:axis1] + [last_idx + 1] + axes[axis1:]
            axes = axes[:axis2] + [last_idx + 2] + axes[axis2:]
            result = result.transpose(axes)

        z[0] = result

    def grad(self, inputs, gout):
        (gz,) = gout
        return [diagonal(
            gz,
            offset=self.offset,
            axis1=self.axis1,
            axis2=self.axis2
        )]

    def infer_shape(self, nodes, shapes):
        (x_shape,) = shapes
        axis1 = np.minimum(self.axis1, self.axis2)
        axis2 = np.maximum(self.axis1, self.axis2)

        result_shape = list(x_shape[:-1])
        diag_shape = x_shape[-1] + abs(self.offset)
        result_shape = result_shape[:axis1] + [diag_shape] + result_shape[axis1:]
        result_shape = result_shape[:axis2] + [diag_shape] + result_shape[axis2:]
        return [tuple(result_shape)]

    def __setstate__(self, state):
        if "view_map" in state:
            del state["view_map"]

        self.__dict__.update(state)

        if "offset" not in state:
            self.offset = 0
        if "axis1" not in state:
            self.axis1 = 0
        if "axis2" not in state:
            self.axis2 = 1


def diag(v, k=0):
    """
    A helper function for two ops: `theano.tensor.ExtractDiag` and
    `theano.tensor.AllocDiag`. The name `diag` is meant to keep it consistent
    with numpy. It both accepts tensor vector and tensor matrix.
    While the passed tensor variable `v` has `v.ndim>=2`, it builds a
    `ExtractDiag` instance, and returns a vector with its entries equal to
    `v`'s main diagonal; otherwise if `v.ndim` is `1`, it builds an `AllocDiag`
    instance, and returns a matrix with `v` at its k-th diaogonal.

    Parameters
    ----------
    v : symbolic tensor
    k : int
        offset

    Returns
    -------
    tensor : symbolic tensor

    """

    if v.ndim == 1:
        return AllocDiag(k)(v)
    elif v.ndim >= 2:
        return diagonal(v, offset=k)
    else:
        raise ValueError("Input must has v.ndim >= 1.")


def stacklists(arg):
    """
    Recursively stack lists of tensors to maintain similar structure.

    This function can create a tensor from a shaped list of scalars:

    Examples
    --------
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
        return stack(list(map(stacklists, arg)))
    else:
        return arg


def ptp(a, axis=None):
    """
    Range of values (maximum - minimum) along an axis.

    The name of the function comes from the acronym for peak to peak.

    Parameters
    ----------
    a
        Input tensor.
    axis
        Axis along which to find the peaks. By default, flatten the array.

    Returns
    -------
    array
        A new array holding the result.

    """

    a = as_tensor_variable(a)

    out = max(a, axis) - min(a, axis)

    return out


def power(x, y):
    return x ** y


def swapaxes(y, axis1, axis2):
    "swap axes of inputted tensor"
    y = as_tensor_variable(y)
    ndim = y.ndim
    li = list(range(0, ndim))
    li[axis1], li[axis2] = li[axis2], li[axis1]
    return y.dimshuffle(li)


def choose(a, choices, out=None, mode='raise'):
    """
    Construct an array from an index array and a set of arrays to choose from.

    First of all, if confused or uncertain, definitely look at the Examples -
    in its full generality, this function is less simple than it might seem
    from the following code description (below ndi = numpy.lib.index_tricks):

    np.choose(a,c) == np.array([c[a[I]][I] for I in ndi.ndindex(a.shape)]).

    But this omits some subtleties. Here is a fully general summary:

    Given an ``index`` array (a) of integers and a sequence of n arrays
    (choices), a and each choice array are first broadcast, as necessary,
    to arrays of a common shape; calling these Ba and
    Bchoices[i], i = 0,...,n-1 we have that, necessarily,
    Ba.shape == Bchoices[i].shape for each i.
    Then, a new array with shape Ba.shape is created as follows:

    - if mode=raise (the default), then, first of all, each element of a
      (and thus Ba) must be in the range [0, n-1]; now, suppose that
      i (in that range) is the value at the (j0, j1, ..., jm) position in Ba -
      then the value at the same position in the new array is the value in
      Bchoices[i] at that same position;

    - if mode=wrap, values in a (and thus Ba) may be any (signed) integer;
      modular arithmetic is used to map integers outside the range [0, n-1]
      back into that range; and then the new array is constructed as above;

    - if mode=clip, values in a (and thus Ba) may be any (signed) integer;
      negative integers are mapped to 0; values greater than n-1 are mapped
      to n-1; and then the new array is constructed as above.

    Parameters
    ----------
    a : int array
        This array must contain integers in [0, n-1], where n is the number of
        choices, unless mode=wrap or mode=clip, in which cases any integers
        are permissible.
    choices : sequence of arrays
        Choice arrays. a and all of the choices must be broadcastable to
        the same shape. If choices is itself an array (not recommended),
        then its outermost dimension (i.e., the one corresponding to
        choices.shape[0]) is taken as defining the ``sequence``.
    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.
    mode : {``raise`` (default), ``wrap``, ``clip``}, optional
        Specifies how indices outside [0, n-1] will be treated:
        ``raise`` : an exception is raised
        ``wrap`` : value becomes value mod n
        ``clip`` : values < 0 are mapped to 0, values > n-1 are mapped to n-1

    Returns
    -------
    merged_array - array
        The merged result.

    Raises
    ------
    ValueError - shape mismatch
        If a and each choice array are not all broadcastable to the same shape.

    """
    # This is done to keep the same function signature then NumPy.
    assert out is None
    return Choose(mode)(a, choices)


class Choose(Op):
    __props__ = ('mode',)

    def __init__(self, mode):
        assert mode in ("raise", "wrap", "clip")
        self.mode = mode

    def infer_shape(self, node, shapes):

        if isinstance(node.inputs[1], TensorVariable):
            # We have padded node.inputs[0] to the right number of
            # dimensions for the output
            l = []
            for sh1, sh2, b1 in zip(shapes[0],
                                    shapes[1][1:],
                                    node.inputs[0].broadcastable):
                if b1:
                    l.append(sh2)
                else:
                    l.append(sh1)
            return [tuple(l)]
        else:
            import theano.typed_list
            assert isinstance(node.inputs[1],
                              theano.typed_list.TypedListVariable)
            raise ShapeError("Case not implemented")
            shape = shapes[0]
            for i in xrange(len(shapes[0]) - 1):
                shape[i] = shapes[1][i]
            return [(shape)]

    def make_node(self, a, choices):
        # Import here as it isn't imported by default and we can't
        # import at the top as it would cause circular import.
        import theano.typed_list
        a = as_tensor_variable(a)
        if a.dtype not in theano.tensor.discrete_dtypes:
            raise TypeError(
                'choose first argument must have an [u]int* dtype. Got %s.'
                % a.dtype)

        if isinstance(choices, (tuple, list,
                                theano.typed_list.TypedListVariable)):
            choice = theano.typed_list.make_list(choices)
            choice_ndim = choice.ttype.ndim
            choice_bcast = choice.ttype.broadcastable
        else:
            choice = as_tensor_variable(choices)
            choice_ndim = choice.ndim - 1
            choice_bcast = choice.broadcastable[1:]
        out_ndim = np.max([a.ndim, choice_ndim])

        # Make explicit all added broadcastable dimensions.
        a = shape_padleft(a, out_ndim - a.ndim)
        if len(choice_bcast) != out_ndim:
            if isinstance(choice.type, TensorType):
                choice = choice.dimshuffle(0,
                                           *(('x',) * (out_ndim - choice_ndim) +
                                             tuple(range(1, choice.ndim))))
                choice_ndim = choice.ndim - 1
                choice_bcast = choice.broadcastable[1:]
            else:
                raise NotImplementedError(
                    "We currently didn't implemented that case. "
                    "To make it work, explicitly add dimensions "
                    "of size one for dimensions that will be broadcasted")

        bcast = [False] * out_ndim
        for idx, (b1, b2) in enumerate(
            zip(a.broadcastable,
                (True,) * (out_ndim - choice_ndim) + choice_bcast)):
            if b1 and b2:
                bcast[idx] = True
        o = TensorType(choice.dtype, bcast)
        return Apply(self, [a, choice], [o()])

    def perform(self, node, inputs, outputs):
        (z,) = outputs
        a = inputs[0]
        choice = inputs[1]
        # TODO reuse out?
        z[0] = np.choose(a, choice, mode=self.mode)


class AllocEmpty(gof.Op):
    """Implement Alloc on the cpu, but without initializing memory."""

    __props__ = ("dtype", )
    params_type = ParamsType(typecode=int32_t)

    # specify the type of the data
    def __init__(self, dtype):
        assert isinstance(dtype, str), dtype
        self.dtype = dtype.lower()

    @property
    def typecode(self):
        return np.dtype(self.dtype).num

    def make_node(self, *shape):
        shape, bcast = alloc_validate_shape(shape)
        otype = TensorType(dtype=self.dtype, broadcastable=bcast)
        output = otype()

        output.tag.values_eq_approx = values_eq_approx_always_true
        # The outut can contain nan/inf.  output.type is a new
        # instance, so we can do this only for that variable.
        output.type.filter_checks_isfinite = False

        # We can't reuse filter_checks_isfinite as by default it is
        # False and it is set to true only in DebugMode.
        # We can't set it in the type as other make_node can reuse the type.
        # We can't set it in the variable as it isn't copied when we copy
        # the variale. So we set it in the tag.
        output.tag.nan_guard_mode_check = False
        return Apply(self, shape, [output])

    def debug_perform(self, node, inputs, out_, params):
        self.perform(node, inputs, out_, params)
        out_[0][0].fill(-123456789)

    def perform(self, node, inputs, out_, params):
        out, = out_
        sh = tuple([int(i) for i in inputs])
        if out[0] is None or out[0].shape != sh:
            out[0] = np.empty(sh, dtype=self.dtype)

    def c_code(self, node, name, inputs, out_, sub):
        out, = out_
        fail = sub['fail']
        shps = inputs
        nd = len(shps)
        params = sub['params']
        str = "npy_intp dims[%(nd)s];\n" % locals()
        for idx, sh in enumerate(shps):
            str += "dims[%(idx)s] =" \
                   "((npy_intp)((dtype_%(sh)s*)" \
                   " PyArray_DATA(%(sh)s))[0]);\n" % locals()

        # Validate that the output storage exists
        str += "if(%(out)s==NULL\n" % locals()
        for idx, sh in enumerate(shps):
            str += "||PyArray_DIMS(%(out)s)[%(idx)s]!=dims[%(idx)s]" % locals()

        str += """){
            /* Reference received to invalid output variable.
            Decrease received reference's ref count and allocate new
            output variable */
            Py_XDECREF(%(out)s);
            %(out)s = (PyArrayObject*)PyArray_EMPTY(%(nd)s,
                                                    dims,
                                                    %(params)s->typecode,
                                                    0);
            if (!%(out)s)
            {
                PyErr_SetString(PyExc_MemoryError, "alloc failed");
                %(fail)s;
            }
        }
        """ % locals()
        return str

    def infer_shape(self, node, input_shapes):
        return [node.inputs]

    def c_code_cache_version(self):
        return (4,)

    def do_constant_folding(self, node):
        return False

    def connection_pattern(self, node):
        return [[False] for i in node.inputs]

    def grad(self, inputs, grads):
        return [DisconnectedType()() for i in inputs]

    def R_op(self, inputs, eval_points):
        return [zeros(inputs, self.dtype)]
