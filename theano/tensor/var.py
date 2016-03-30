from __future__ import absolute_import, print_function, division
import copy
import traceback as tb
import warnings

import numpy
from six import integer_types
from six.moves import xrange

import theano
from theano.compat import PY3
from theano.scalar import ComplexError, IntegerDivisionError
from theano.gof import Constant, Variable
from theano.gof.utils import hashtype
from theano.tensor.utils import hash_from_ndarray
from theano.tensor.type import TensorType
from theano.configparser import config


def equal_slices(s1, s2):
    return (s1.start == s2.start and
            s1.stop == s2.stop and
            s1.step == s2.step)


class AsTensorError(TypeError):
    """
    Raised when as_tensor_variable isn't able to create a TensorVariable.

    """
    pass


class _tensor_py_operators(object):
    # UNARY
    def __abs__(self):
        return theano.tensor.basic.abs_(self)

    def __neg__(self):
        return theano.tensor.basic.neg(self)

    # CASTS
    # REMOVED THESE BECAUSE PYTHON appears to require __int__ to return
    # an int. -JB 20081112
    # def __int__(self): return convert_to_int32(self)
    # def __float__(self): return convert_to_float64(self)
    # def __complex__(self): return convert_to_complex128(self)

    # COMPARISONS
    _is_nonzero = True

    def __lt__(self, other):
        rval = theano.tensor.basic.lt(self, other)
        rval._is_nonzero = False
        return rval

    def __le__(self, other):
        rval = theano.tensor.basic.le(self, other)
        rval._is_nonzero = False
        return rval

    def __gt__(self, other):
        rval = theano.tensor.basic.gt(self, other)
        rval._is_nonzero = False
        return rval

    def __ge__(self, other):
        rval = theano.tensor.basic.ge(self, other)
        rval._is_nonzero = False
        return rval

    def __nonzero__(self):
        # Python 2.x
        return self.__bool__()

    def __bool__(self):
        # This is meant to prohibit stuff like a < b < c, which is internally
        # implemented as (a < b) and (b < c). The trouble with this is the
        # side-effect that checking for a non-NULL a by typing "if a: ..."
        # uses the same __nonzero__ method.  We want these both to work, but
        # it seems impossible.  Currently, all vars evaluate to nonzero except
        # the return values of comparison operators, which raise this
        # exception.  If you can think of a better solution, go for it!
        #
        # __bool__ is Python 3.x data model. __nonzero__ is Python 2.x.
        if self._is_nonzero:
            return True
        else:
            raise TypeError(
                "Variables do not support boolean operations. This "
                "can happen if you do a logical operation (<, <=, >, <=, "
                "==, !=) between a numpy.ndarray and a Theano tensor"
                "variable. Due to NumPy implementation before NumPy 1.8, "
                "we cannot make the Python syntax work when the ndarray "
                "is on the left, and this results in this error. To work "
                "around that, either call "
                "theano.tensor.{lt,le,eq,ne,gt,ge}(ndarray, tensor), or "
                "use the Python syntax with the Theano tensor on the "
                "left. Or update to NumPy 1.8 or above."
            )

    # BITWISE
    def __invert__(self):
        return theano.tensor.basic.invert(self)

    def __and__(self, other):
        return theano.tensor.basic.and_(self, other)

    def __or__(self, other):
        return theano.tensor.basic.or_(self, other)

    def __xor__(self, other):
        return theano.tensor.basic.xor(self, other)

    def __rand__(self, other):
        return theano.tensor.basic.and_(other, self)

    def __ror__(self, other):
        return theano.tensor.basic.or_(other, self)

    def __rxor__(self, other):
        return theano.tensor.basic.xor(other, self)

    # def __iand__(self, other):
    #    return _and_inplace(self, other)
    #
    # def __ior__(self, other):
    #    return _or_inplace(self, other)
    #
    # def __ixor__(self, other):
    #    return _xor_inplace(self, other)

    # ARITHMETIC - NORMAL
    def __add__(self, other):
        try:
            return theano.tensor.basic.add(self, other)
        # We should catch the minimum number of exception here.
        # Otherwise this will convert error when Theano flags
        # compute_test_value is used
        # Evidently, we need to catch NotImplementedError
        # TypeError from as_tensor_variable are caught in Elemwise.make_node
        # Oterwise TensorVariable * SparseVariable won't work!
        except (NotImplementedError, AsTensorError):
            # We must return NotImplemented and not an
            # NotImplementedError or raise an NotImplementedError.
            # That way python will give a good error message like this
            # `TypeError: unsupported operand type(s) for +:
            # 'TensorVariable' and 'TensorVariable'`
            return NotImplemented

    def __sub__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return theano.tensor.basic.sub(self, other)
        except (NotImplementedError, AsTensorError):
            return NotImplemented

    def __mul__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return theano.tensor.mul(self, other)
        except (NotImplementedError, AsTensorError):
            return NotImplemented

    def __div__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return theano.tensor.basic.div_proxy(self, other)
        except IntegerDivisionError:
            # This is to raise the exception that occurs when trying to divide
            # two integer arrays (currently forbidden).
            raise
        except (NotImplementedError, AsTensorError):
            return NotImplemented
    if PY3:
        __truediv__ = __div__

    def __pow__(self, other):
        # See explanation in __add__ for the error catched
        # adn the return value in that case
        try:
            return theano.tensor.basic.pow(self, other)
        except (NotImplementedError, AsTensorError):
            return NotImplemented

    def __mod__(self, other):
        # See explanation in __add__ for the error catched
        # adn the return value in that case
        try:
            return theano.tensor.basic.mod_check(self, other)
        except ComplexError:
            # This is to raise the exception that occurs when trying to compute
            # x % y with either x or y a complex number.
            raise
        except (NotImplementedError, AsTensorError):
            return NotImplemented

    def __divmod__(self, other):
        return theano.tensor.basic.divmod(self, other)

    def __truediv__(self, other):
        return theano.tensor.basic.true_div(self, other)

    def __floordiv__(self, other):
        return theano.tensor.basic.floor_div(self, other)

    def __rtruediv__(self, other):
        return theano.tensor.basic.true_div(other, self)

    def __rfloordiv__(self, other):
        return theano.tensor.basic.floor_div(other, self)

    # DO NOT USE THESE BECAUSE INPLACE OPS SHOULD BE INSERTED
    # BY OPTIMIZATIONS ONLY
    # ARITHMETIC - INPLACE
    # def __iadd__(self, other):
    #    return _add_inplace(self, other)
    # def __isub__(self, other):
    #    return _sub_inplace(self, other)
    #
    # def __imul__(self, other):
    #    return _mul_inplace(self, other)
    #
    # def __idiv__(self, other):
    #    return _div_inplace(self, other)
    #
    # def __ipow__(self, other):
    #    return _pow_inplace(self, other)

    # ARITHMETIC - RIGHT-OPERAND
    def __radd__(self, other):
        return theano.tensor.basic.add(other, self)

    def __rsub__(self, other):
        return theano.tensor.basic.sub(other, self)

    def __rmul__(self, other):
        return theano.tensor.basic.mul(other, self)

    def __rdiv__(self, other):
        return theano.tensor.basic.div_proxy(other, self)

    def __rmod__(self, other):
        return theano.tensor.basic.mod(other, self)

    def __rdivmod__(self, other):
        return theano.tensor.basic.divmod(other, self)

    def __rpow__(self, other):
        return theano.tensor.basic.pow(other, self)

    # TRANSPOSE
    T = property(lambda self: theano.tensor.basic.transpose(self))

    def transpose(self, *axes):
        """

        Returns
        -------
        object
            `tensor.transpose(self, axes)` or `tensor.transpose(self, axes[0])`.

        If only one `axes` argument is provided and it is iterable, then it is
        assumed to be the entire axes tuple, and passed intact to
        tensor.transpose.

        """
        if len(axes) == 0:
            return theano.tensor.basic.transpose(self)
        try:
            iter(axes[0])
            iterable = True
        except TypeError:
            iterable = False
        if len(axes) == 1 and iterable:
            return theano.tensor.basic.transpose(self, axes[0])
        else:
            return theano.tensor.basic.transpose(self, axes)

    shape = property(lambda self: theano.tensor.basic.shape(self))

    size = property(lambda self: self.shape[0] if self.ndim == 1 else
                    theano.tensor.basic.prod(self.shape))

    # We can't implement __len__ to provide a better error message.
    def any(self, axis=None, keepdims=False):
        return theano.tensor.basic.any(self, axis=axis, keepdims=keepdims)

    def all(self, axis=None, keepdims=False):
        return theano.tensor.basic.all(self, axis=axis, keepdims=keepdims)

    # Otherwise TensorVariable[:-1] does not work as Python 2.5.1 calls
    # __len__ before calling __getitem__. It also does not catch the raised
    # Exception!
    # def __len__(self):
    #     # We can't implement __len__ as Python requests that this
    #     # function returns an integer >=0
    #     raise Exception("Theano Variables can't work with len(Theano "
    #                     "Variable) due to Python restriction. You can use "
    #                     "TheanoVariable.shape[0] instead.")

    def reshape(self, shape, ndim=None):
        """Return a reshaped view/copy of this variable.

        Parameters
        ----------
        shape
            Something that can be converted to a symbolic vector of integers.
        ndim
            The length of the shape. Passing None here means for
            Theano to try and guess the length of `shape`.

        .. warning:: This has a different signature than numpy's
                     ndarray.reshape!
                     In numpy you do not need to wrap the shape arguments
                     in a tuple, in theano you do need to.

        """

        if ndim is not None:
            if not isinstance(ndim, integer_types):
                raise ValueError("Expected ndim to be an integer, is " +
                                 str(type(ndim)))

        return theano.tensor.basic.reshape(self, shape, ndim=ndim)

    def dimshuffle(self, *pattern):
        """
        Reorder the dimensions of this variable, optionally inserting
        broadcasted dimensions.

        Parameters
        ----------
        pattern
            List/tuple of int mixed with 'x' for broadcastable dimensions.

        Examples
        --------
        For example, to create a 3D view of a [2D] matrix, call
        ``dimshuffle([0,'x',1])``.  This will create a 3D view such that the
        middle dimension is an implicit broadcasted dimension.  To do the same
        thing on the transpose of that matrix, call ``dimshuffle([1, 'x', 0])``.

        Notes
        -----
        This function supports the pattern passed as a tuple, or as a
        variable-length argument (e.g. ``a.dimshuffle(pattern)`` is equivalent
        to ``a.dimshuffle(*pattern)`` where ``pattern`` is a list/tuple of ints
        mixed with 'x' characters).

        See Also
        --------
        DimShuffle

        """
        if (len(pattern) == 1) and (isinstance(pattern[0], (list, tuple))):
            pattern = pattern[0]
        op = theano.tensor.basic.DimShuffle(list(self.type.broadcastable),
                                            pattern)
        return op(self)

    def flatten(self, ndim=1):
        return theano.tensor.basic.flatten(self, ndim)

    def ravel(self):
        return theano.tensor.basic.flatten(self)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return theano.tensor.basic.diagonal(self, offset, axis1, axis2)

    # Transfer the data to another device
    def transfer(self, target):
        """
        If `target` is `'cpu'` this will transfer to a TensorType (if
        not already one).  Other types may define additional targets.

        Parameters
        ----------
        target : str
            The desired location of the output variable
        """
        return theano.tensor.transfer(self, target)

    # Elemwise
    def arccos(self):
        return theano.tensor.arccos(self)

    def arccosh(self):
        return theano.tensor.arccosh(self)

    def arcsin(self):
        return theano.tensor.arcsin(self)

    def arcsinh(self):
        return theano.tensor.arcsinh(self)

    def arctan(self):
        return theano.tensor.arctan(self)

    def arctanh(self):
        return theano.tensor.arctanh(self)

    def ceil(self):
        return theano.tensor.ceil(self)

    def cos(self):
        return theano.tensor.cos(self)

    def cosh(self):
        return theano.tensor.cosh(self)

    def deg2rad(self):
        return theano.tensor.deg2rad(self)

    def exp(self):
        return theano.tensor.exp(self)

    def exp2(self):
        return theano.tensor.exp2(self)

    def expm1(self):
        return theano.tensor.expm1(self)

    def floor(self):
        return theano.tensor.floor(self)

    def log(self):
        return theano.tensor.log(self)

    def log10(self):
        return theano.tensor.log10(self)

    def log1p(self):
        return theano.tensor.log1p(self)

    def log2(self):
        return theano.tensor.log2(self)

    def rad2deg(self):
        return theano.tensor.rad2deg(self)

    def sin(self):
        return theano.tensor.sin(self)

    def sinh(self):
        return theano.tensor.sinh(self)

    def sqrt(self):
        return theano.tensor.sqrt(self)

    def tan(self):
        return theano.tensor.tan(self)

    def tanh(self):
        return theano.tensor.tanh(self)

    def trunc(self):
        return theano.tensor.trunc(self)

    # CASTING
    def astype(self, dtype):
        return theano.tensor.cast(self, dtype)

    # SLICING/INDEXING
    def __getitem__(self, args):
        if (isinstance(args, list) and
                any([isinstance(a, slice) for a in args])):
            pass
        elif not isinstance(args, tuple):
            args = args,
        # Convert python literals to theano constants
        args = theano.tensor.subtensor.make_constant(args)
        # Determine if advanced indexing is needed or not
        # The logic is already in Subtensor.convert: if it succeeds,
        # standard indexing is used; if it fails with
        # AdvancedIndexingError, advanced indexing
        advanced = False
        axis = None
        for i, arg in enumerate(args):
            try:
                if arg is not numpy.newaxis:
                    theano.tensor.subtensor.Subtensor.convert(arg)
            except theano.tensor.subtensor.AdvancedIndexingError:
                if advanced:
                    axis = None
                    break
                else:
                    advanced = True
                    axis = i

        if advanced:
            if (axis is not None and
                all(isinstance(a, slice) and
                    equal_slices(a, slice(None)) for a in args[:axis]) and
                all(isinstance(a, slice) and
                    equal_slices(a, slice(None)) for a in args[axis + 1:]) and
                isinstance(args[axis],
                           (numpy.ndarray, list,
                            TensorVariable, TensorConstant,
                            theano.tensor.sharedvar.TensorSharedVariable))):
                return self.take(args[axis], axis)
            else:
                return theano.tensor.subtensor.advanced_subtensor(self, *args)
        else:
            if numpy.newaxis in args:
                # None (aka np.newaxis) in numpy indexing means to add a
                # broadcastable dimension, which theano traditionally did with
                # the dimshuffle op.  The following code converts numpy-style
                # indexing on self to traditional [read: implemented] theano
                # indexing on a dimshuffled view of self.

                counter = 0
                pattern = []
                new_args = []
                for arg in args:
                    if arg == numpy.newaxis:
                        pattern.append('x')
                        new_args.append(slice(None, None, None))
                    else:
                        pattern.append(counter)
                        counter += 1
                        new_args.append(arg)
                view = self.dimshuffle(pattern)
                rval = view.__getitem__(tuple(new_args))
                return rval
            else:
                return theano.tensor.subtensor.Subtensor(args)(
                    self, *theano.tensor.subtensor.Subtensor.collapse(
                        args,
                        lambda entry: isinstance(entry, Variable)))

    def take(self, indices, axis=None, mode='raise'):
        return theano.tensor.subtensor.take(self, indices, axis, mode)

    # COPYING
    def copy(self, name=None):
        """Return a symbolic copy and optionally assign a name.

        Does not copy the tags.
        """
        copied_variable = theano.tensor.basic.tensor_copy(self)
        copied_variable.name = name
        return copied_variable

    def __iter__(self):
        try:
            for i in xrange(theano.tensor.basic.get_vector_length(self)):
                yield self[i]
        except TypeError:
            # This prevents accidental iteration via builtin.sum(self)
            raise TypeError(('TensorType does not support iteration. '
                             'Maybe you are using builtin.sum instead of '
                             'theano.tensor.sum? (Maybe .max?)'))

    # CONVENIENT ACCESS TO TYPE PROPERTIES
    ndim = property(lambda self: self.type.ndim)
    """The rank of this tensor."""

    broadcastable = property(lambda self: self.type.broadcastable)
    """
    The broadcastable signature of this tensor.

    See Also
    --------
    broadcasting

    """

    dtype = property(lambda self: self.type.dtype)
    """The dtype of this tensor."""

    # extra pseudo-operator symbols
    def __dot__(left, right):
        return theano.tensor.basic.dot(left, right)

    def __rdot__(right, left):
        return theano.tensor.basic.dot(left, right)

    dot = __dot__

    def sum(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See `theano.tensor.sum`."""
        return theano.tensor.basic.sum(self, axis=axis,
                                       dtype=dtype, keepdims=keepdims,
                                       acc_dtype=acc_dtype)

    def prod(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See `theano.tensor.prod`."""
        return theano.tensor.basic.prod(self, axis=axis,
                                        dtype=dtype, keepdims=keepdims,
                                        acc_dtype=acc_dtype)

    def norm(self, L, axis=None):
        if L == 0:
            raise NotImplementedError()
        if numpy.isinf(L):
            raise NotImplementedError()
        # optimizations will/should catch cases like L=1, L=2
        return theano.tensor.basic.pow(
            theano.tensor.basic.pow(
                theano.tensor.basic.abs_(self), L).sum(axis=axis), 1.0 / L)

    def mean(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See `theano.tensor.mean`."""
        return theano.tensor.basic.mean(self, axis=axis,
                                        dtype=dtype, keepdims=keepdims,
                                        acc_dtype=acc_dtype)

    def var(self, axis=None, keepdims=False):
        """See `theano.tensor.var`."""
        return theano.tensor.basic.var(self, axis, keepdims=keepdims)

    def std(self, axis=None, keepdims=False):
        """See `theano.tensor.std`."""
        return theano.tensor.basic.std(self, axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        """See `theano.tensor.min`."""
        return theano.tensor.basic.min(self, axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """See `theano.tensor.max`."""
        return theano.tensor.basic.max(self, axis, keepdims=keepdims)

    def argmin(self, axis=None, keepdims=False):
        """See `theano.tensor.argmin`."""
        return theano.tensor.basic.argmin(self, axis, keepdims=keepdims)

    def argmax(self, axis=None, keepdims=False):
        """See `theano.tensor.argmax`."""
        return theano.tensor.basic.argmax(self, axis, keepdims=keepdims)

    def nonzero(self, return_matrix=False):
        """See `theano.tensor.nonzero`."""
        return theano.tensor.basic.nonzero(self, return_matrix=return_matrix)

    def nonzero_values(self):
        """See `theano.tensor.nonzero_values`."""
        return theano.tensor.basic.nonzero_values(self)

    def sort(self, axis=-1, kind='quicksort', order=None):
        """See `theano.tensor.sort`."""
        return theano.tensor.sort(self, axis, kind, order)

    def argsort(self, axis=-1, kind='quicksort', order=None):
        """See `theano.tensor.argsort`."""
        return theano.tensor.argsort(self, axis, kind, order)

    def clip(self, a_min, a_max):
        "Clip (limit) the values in an array."
        return theano.tensor.basic.clip(self, a_min, a_max)

    def conj(self):
        """See `theano.tensor.conj`."""
        return theano.tensor.basic.conj(self)

    conjugate = conj

    def repeat(self, repeats, axis=None):
        """See `theano.tensor.repeat`."""
        return theano.tensor.extra_ops.repeat(self, repeats, axis)

    def round(self, mode="half_away_from_zero"):
        """See `theano.tensor.round`."""
        return theano.tensor.basic.round(self, mode)

    def trace(self):
        return theano.tensor.nlinalg.trace(self)

    # TO TRUMP NUMPY OPERATORS
    __array_priority__ = 1000

    def get_scalar_constant_value(self):
        return theano.tensor.basic.get_scalar_constant_value(self)

    def zeros_like(model, dtype=None):
        return theano.tensor.basic.zeros_like(model, dtype=dtype)

    def cumsum(self, axis=None):
        return theano.tensor.extra_ops.cumsum(self, axis)

    def cumprod(self, axis=None):
        return theano.tensor.extra_ops.cumprod(self, axis)

    def ptp(self, axis=None):
        """See 'theano.tensor.ptp'."""

        return theano.tensor.ptp(self, axis)

    def swapaxes(self, axis1, axis2):
        """
        Return 'tensor.swapaxes(self, axis1, axis2).

        If a matrix is provided with the right axes, its transpose
        will be returned.

        """
        return theano.tensor.basic.swapaxes(self, axis1, axis2)

    def fill(self, value):
        """Fill inputted tensor with the assigned value."""
        return theano.tensor.basic.fill(self, value)

    def choose(self, a, choices, out=None, mode='raise'):
        """
        Construct an array from an index array and a set of arrays to choose
        from.

        """
        return theano.tensor.basic.choose(self, a, choices, out=None,
                                          mode='raise')

    def squeeze(self):
        """
        Remove broadcastable dimensions from the shape of an array.

        It returns the input array, but with the broadcastable dimensions
        removed. This is always `x` itself or a view into `x`.

        """
        return theano.tensor.extra_ops.squeeze(self)

    def compress(self, a, axis=None):
        """Return selected slices only."""
        return theano.tensor.extra_ops.compress(self, a, axis=axis)


class TensorVariable(_tensor_py_operators, Variable):
    """
    Subclass to add the tensor operators to the basic `Variable` class.

    """

    def __init__(self, type, owner=None, index=None, name=None):
        super(TensorVariable, self).__init__(type, owner=owner,
                                             index=index, name=name)
        if (config.warn_float64 != 'ignore' and type.dtype == 'float64'):
            msg = ('You are creating a TensorVariable '
                   'with float64 dtype. You requested an action via '
                   'the Theano flag warn_float64={ignore,warn,raise,pdb}.')
            if config.warn_float64 == "warn":
                # Get the user stack. We don't want function inside the
                # tensor and gof directory to be shown to the user.
                x = tb.extract_stack()
                nb_rm = 0
                while x:
                    file_path = x[-1][0]
                    rm = False
                    for p in ["theano/tensor/", "theano\\tensor\\",
                              "theano/gof/", "theano\\tensor\\"]:
                        if p in file_path:
                            x = x[:-1]
                            nb_rm += 1
                            rm = True
                            break
                    if not rm:
                        break
                warnings.warn(msg, stacklevel=1 + nb_rm)
            elif config.warn_float64 == "raise":
                raise Exception(msg)
            elif config.warn_float64 == 'pdb':
                import pdb
                pdb.set_trace()
TensorType.Variable = TensorVariable


class TensorConstantSignature(tuple):
    """
    A Signature object for comparing TensorConstant instances.

    An instance is a pair: (Type instance, ndarray).

    """
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        try:
            (t0, d0), (t1, d1) = self, other
        except Exception:
            return False

        # N.B. compare shape to ensure no broadcasting in ==
        if t0 != t1 or d0.shape != d1.shape:
            return False

        self.no_nan  # Ensure has_nan is computed.
        # Note that in the comparisons below, the elementwise comparisons
        # come last because they are the most expensive checks.
        if self.has_nan:
            other.no_nan  # Ensure has_nan is computed.
            return (other.has_nan and
                    self.sum == other.sum and
                    (self.no_nan.mask == other.no_nan.mask).all() and
                    # Note that the second test below (==) may crash e.g. for
                    # a single scalar NaN value, so we do not run it when all
                    # values are missing.
                    (self.no_nan.mask.all() or
                     (self.no_nan == other.no_nan).all()))
        else:
            # Simple case where we do not need to worry about NaN values.
            # (note that if there are NaN values in d1, this will return
            # False, which is why we do not bother with testing `other.has_nan`
            # here).
            return (self.sum == other.sum) and numpy.all(d0 == d1)

    def __hash__(self):
        t, d = self
        return hashtype(self) ^ hash(t) ^ hash(d.shape) ^ hash(self.sum)

    def theano_hash(self):
        _, d = self
        return hash_from_ndarray(d)

    def _get_sum(self):
        """Compute sum of non NaN / Inf values in the array."""
        try:
            return self._sum
        except AttributeError:
            self._sum = self.no_nan.sum()
            # The following 2 lines are needede as in Python 3.3 with NumPy
            # 1.7.1, numpy.ndarray and numpy.memmap aren't hashable.
            if type(self._sum) is numpy.memmap:
                self._sum = numpy.asarray(self._sum).item()
            if self.has_nan and self.no_nan.mask.all():
                # In this case the sum is not properly computed by numpy.
                self._sum = 0
            if numpy.isinf(self._sum) or numpy.isnan(self._sum):
                # NaN may happen when there are both -inf and +inf values.
                if self.has_nan:
                    # Filter both NaN and Inf values.
                    mask = self.no_nan.mask + numpy.isinf(self[1])
                else:
                    # Filter only Inf values.
                    mask = numpy.isinf(self[1])
                if mask.all():
                    self._sum = 0
                else:
                    self._sum = numpy.ma.masked_array(self[1], mask).sum()
                # At this point there should be no more NaN.
                assert not numpy.isnan(self._sum)
        return self._sum
    sum = property(_get_sum)

    def _get_no_nan(self):
        try:
            return self._no_nan
        except AttributeError:
            nan_mask = numpy.isnan(self[1])
            if nan_mask.any():
                self._no_nan = numpy.ma.masked_array(self[1], nan_mask)
                self.has_nan = True
            else:
                self._no_nan = self[1]
                self.has_nan = False
        return self._no_nan
    no_nan = property(_get_no_nan)


class TensorConstant(_tensor_py_operators, Constant):
    """Subclass to add the tensor operators to the basic `Constant` class.

    To create a TensorConstant, use the `constant` function in this module.

    """
    def __init__(self, type, data, name=None):
        Constant.__init__(self, type, data, name)
        self.tag.unique_value = None
        if isinstance(data, numpy.ndarray) and data.ndim > 0:
            flat_data = data.ravel()
            if flat_data.shape[0]:
                if (flat_data == flat_data[0]).all():
                    self.tag.unique_value = flat_data[0]

    def __str__(self):
        if self.tag.unique_value is not None:
            name = "%s of %s" % (str(self.data.shape),
                                 str(self.tag.unique_value))
        else:
            name = "%s" % self.data
        if len(name) > 20:
            name = name[:10] + ".." + name[-10:]

        return "TensorConstant{%s}" % name

    def signature(self):
        return TensorConstantSignature((self.type, self.data))

    def equals(self, other):
        # Override Contant.equals to allow to compare with numpy.ndarray
        if isinstance(other, numpy.ndarray):
            # Make a TensorConstant to be able to compare
            other = theano.tensor.basic.constant(other)
        return (isinstance(other, TensorConstant) and
                self.signature() == other.signature())

    def __copy__(self):
        # We need to do this to remove the cached attribute
        return type(self)(self.type, self.data, self.name)

    def __deepcopy__(self, memo):
        # We need to do this to remove the cached attribute
        return type(self)(copy.deepcopy(self.type, memo),
                          copy.deepcopy(self.data, memo),
                          copy.deepcopy(self.name, memo))

TensorType.Constant = TensorConstant
