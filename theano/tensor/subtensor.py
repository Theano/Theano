from __future__ import absolute_import, print_function, division
import sys
from textwrap import dedent
import warnings
import logging

import numpy
from six import integer_types
from six.moves import xrange

import theano
from theano.compat import izip
from theano.gradient import DisconnectedType
from theano import gof
from theano.gof import Apply, hashtype, Op, Type, MethodNotDefined
from theano.printing import pprint
from theano import scalar as scal
from theano.tensor.basic import alloc
from theano.tensor.basic import (addbroadcast, clip, get_scalar_constant_value,
                                 TensorType, NotScalarConstantError)
from theano.tensor.elemwise import DimShuffle
from theano.tensor.type_other import NoneConst, SliceType, NoneTypeT, make_slice
from theano import config

if config.cxx:
    import theano.gof.cutils  # needed to import cutils_ext
    from cutils_ext.cutils_ext import inplace_increment

_logger = logging.getLogger("theano.tensor.subtensor")

# Do a lazy import of the sparse module
sparse_module_ref = None


class AdvancedIndexingError(TypeError):
    """
    Raised when Subtensor is asked to perform advanced indexing.

    """

    def __init__(self, *args):
        TypeError.__init__(self, *args)


##########
# Helpful functions to deal with Subtensor and IncSubtensor
##########

def make_constant(args):
    """
    Convert python litterals to theano constants in subtensor arguments.

    """
    def conv(a):
            if a is None:
                return a
            elif isinstance(a, slice):
                return slice(conv(a.start),
                             conv(a.stop),
                             conv(a.step))
            elif isinstance(a, (integer_types, numpy.integer)):
                return scal.ScalarConstant(scal.int64, a)
            else:
                return a
    return tuple(map(conv, args))


def get_idx_list(inputs, idx_list, get_count=False):
    """
    Given a list of inputs to the subtensor and its idx_list reorders
    the inputs according to the idx list to get the right values.

    If get_counts=True, instead returns the number of inputs consumed
    during this process.

    """

    # The number of indices
    n = len(inputs) - 1

    # The subtensor (or idx_list) does not depend on the inputs.
    if n == 0:
        return tuple(idx_list)
    indices = list(reversed(list(inputs[1:])))

    # General case
    def convert(entry):
        if isinstance(entry, gof.Type):
            return indices.pop()
        elif isinstance(entry, slice):
            return slice(convert(entry.start),
                         convert(entry.stop),
                         convert(entry.step))
        else:
            return entry
    cdata = tuple(map(convert, idx_list))
    if get_count:
        return n - len(indices)
    else:
        return cdata


def get_canonical_form_slice(theslice, length):
    """
    Given a slice [start:stop:step] transform it into a canonical form
    that respects the conventions imposed by python and numpy.

    In a canonical form a slice is represented by a canonical form slice,
    in which 0 <= start <= stop <= length and step > 0, and a flag which says
    if the resulting set of numbers needs to be reversed or not.

    """
    from theano.tensor import switch, lt, ge, sgn
    if isinstance(theslice, slice):

        def analyze(x):
            try:
                x_constant = get_scalar_constant_value(x)
                is_constant = True
            except theano.tensor.NotScalarConstantError:
                x_constant = theano.tensor.extract_constant(x)
                is_constant = False
            return x_constant, is_constant

        start, is_start_constant = analyze(theslice.start)
        stop, is_stop_constant = analyze(theslice.stop)
        step, is_step_constant = analyze(theslice.step)
        length, is_length_constant = analyze(length)

        if step is None:
            step = 1
            is_step_constant = True

        # First handle the easier and common case where `step` is 1 and
        # either `start` or `stop` is a range boundary. More specializations
        # could be added later. This makes the resulting graph smaller than
        # in the generic case below.
        if step == 1:
            is_start_0 = (
                start is None or start == 0 or
                (is_start_constant and is_length_constant and
                 start < 0 and start + length <= 0))
            is_stop_length = (
                stop is None or stop in [length, sys.maxsize] or
                (is_stop_constant and is_length_constant and
                 stop >= length))
            if is_start_0:
                # 0:stop:1
                if is_stop_length:
                    # Full slice.
                    return slice(0, length, 1), 1
                if is_stop_constant and stop >= 0:
                    return (slice(0, switch(lt(stop, length), stop, length),
                                  1), 1)
                stop_plus_len = stop + length
                stop = switch(
                    lt(stop, 0),
                    # stop < 0
                    switch(
                        lt(stop_plus_len, 0),
                        # stop + len < 0
                        0,
                        # stop + len >= 0
                        stop_plus_len),
                    # stop >= 0: use min(stop, length)
                    switch(lt(stop, length), stop, length))
                return slice(0, stop, 1), 1
            elif is_stop_length:
                # start:length:1
                if is_start_constant and start >= 0:
                    return slice(switch(lt(start, length), start, length),
                                 length, 1), 1
                start_plus_len = start + length
                start = switch(
                    lt(start, 0),
                    # start < 0
                    switch(
                        lt(start_plus_len, 0),
                        # start + len < 0
                        0,
                        # start + len >= 0
                        start_plus_len),
                    # start >= 0: use min(start, length)
                    switch(lt(start, length), start, length))
                return slice(start, length, 1), 1

        # This is the generic case.

        if is_step_constant:
            # When we know the sign of `step`, the graph can be made simpler.
            assert step != 0
            if step > 0:
                def switch_neg_step(a, b):
                    return b
                abs_step = step
                sgn_step = 1
            else:
                def switch_neg_step(a, b):
                    return a
                abs_step = -step
                sgn_step = -1
        else:
            is_step_neg = lt(step, 0)

            def switch_neg_step(a, b):
                return switch(is_step_neg, a, b)
            abs_step = abs(step)
            sgn_step = sgn(step)

        defstart = switch_neg_step(length - 1, 0)
        defstop = switch_neg_step(-1, length)
        if start is None:
            start = defstart
        else:
            start = switch(lt(start, 0), start + length, start)
            start = switch(lt(start, 0), switch_neg_step(-1, 0), start)
            start = switch(ge(start, length),
                           switch_neg_step(length - 1, length),
                           start)
        if stop is None or stop == sys.maxsize:
            # The special "maxsize" case is probably not needed here,
            # as slices containing maxsize are not generated by
            # __getslice__ anymore.
            stop = defstop
        else:
            stop = switch(lt(stop, 0), stop + length, stop)
            stop = switch(lt(stop, 0), -1, stop)
            stop = switch(ge(stop, length), length, stop)

        nw_stop = switch_neg_step(start + 1, stop)
        slice_len = (start - stop - 1) // abs_step + 1
        slice_len = switch(lt(slice_len, 0), 0, slice_len)
        neg_start = nw_stop - (slice_len - 1) * abs_step - 1
        neg_start = switch(lt(neg_start, 0), (nw_stop - 1), neg_start)
        nw_start = switch_neg_step(neg_start, start)
        nw_start = switch(lt(nw_start, 0), 0, nw_start)
        nw_stop = switch(lt(nw_stop, 0), 0, nw_stop)
        # Ensure start <= stop.
        nw_start = switch(lt(nw_start, nw_stop), nw_start, nw_stop)

        nw_step = abs_step
        if step != 1:
            reverse = sgn_step
            return slice(nw_start, nw_stop, nw_step), reverse
        else:
            return slice(nw_start, nw_stop, nw_step), 1
    else:
        value = theano.tensor.extract_constant(theslice)
        value = switch(lt(value, 0), (value + length), value)

        return value, 1


class Subtensor(Op):
    """
    Return a subtensor view.

    The inputs array is the tensor x, followed by scalar integer types.
    TODO: WRITEME: how are the scalar integer variables formatted?

    This class uses a relatively complex internal representation of the inputs
    to remember how the input tensor x should be sliced.

    idx_list: instance variable TODO: WRITEME: is this a list or a tuple?
                                        (old docstring gives two conflicting
                                        descriptions)
              elements are either integers, theano scalar types, or slices.
              one element per "explicitly named dimension"
                TODO: WRITEME: what is an "explicitly named dimension" ?

              if integer:
                  indexes into the inputs array
              if slice:
                  start/stop/step members of each slice are integer indices
                  into the inputs array or None
                  integer indices be actual integers or theano scalar types

    Note that the idx_list defines the Op, so two Subtensor instances are
    considered to be different Ops if they have different idx_list fields.
    This means that the entries in it are theano Types, not theano Variables.

    @todo: add support for advanced tensor indexing (in Subtensor_dx too).

    """
    e_invalid = ('The index list is longer (size %d) than the number of '
                 'dimensions of the tensor(namely %d). You are asking for '
                 'a dimension of the tensor that does not exist! You might '
                 'need to use dimshuffle to add extra dimension to your '
                 'tensor.')
    e_subslice = 'nested slicing is not supported'
    e_indextype = "Invalid index type or slice for Subtensor"
    debug = 0
    check_input = False
    view_map = {0: [0]}
    _f16_ok = True
    __props__ = ("idx_list",)

    @staticmethod
    def collapse(idxs, cond):
        """
        Parameters
        ----------
        idxs : a list of indices or slices.
        cond : a callable that returns a bool

        Returns
        -------
        list
            idxs, with the slices flattened out into a list.
            If cond is true for an entry, does not flatten it.

        """
        ret = []

        def helper(entry):
            if cond(entry):
                ret.append(entry)
            elif isinstance(entry, slice):
                helper(entry.start)
                helper(entry.stop)
                helper(entry.step)

        for idx in idxs:
            helper(idx)

        return ret

    @staticmethod
    def convert(entry, slice_ok=True):
        """
        Change references to Variables into references to Types.

        The "idx_list" field is unique to each Subtensor instance.
        It is not unique to each Apply node, so it should not refer to
        specific Variables.
        TODO: WRITEME: This method also accepts "entry" already being a Type;
            when would that happen?

        """
        invalid_scal_types = [scal.float64, scal.float32, scal.float16]
        scal_types = [scal.int64, scal.int32, scal.int16, scal.int8]
        tensor_types = [theano.tensor.lscalar, theano.tensor.iscalar,
                        theano.tensor.wscalar, theano.tensor.bscalar]
        invalid_tensor_types = [theano.tensor.fscalar, theano.tensor.dscalar,
                                theano.tensor.cscalar, theano.tensor.zscalar]
        if (isinstance(entry, gof.Variable) and
            (entry.type in invalid_scal_types or
             entry.type in invalid_tensor_types)):
            raise TypeError("Expected an integer")

        if isinstance(entry, gof.Variable) and entry.type in scal_types:
            return entry.type
        elif isinstance(entry, gof.Type) and entry in scal_types:
            return entry

        if (isinstance(entry, gof.Variable) and
                entry.type in tensor_types and
                numpy.all(entry.type.broadcastable)):
            return scal.get_scalar_type(entry.type.dtype)
        elif (isinstance(entry, gof.Type) and
              entry in tensor_types and
              numpy.all(entry.broadcastable)):
            return scal.get_scalar_type(entry.dtype)
        elif slice_ok and isinstance(entry, slice):
            a = entry.start
            b = entry.stop
            c = entry.step

            if a is not None:
                slice_a = Subtensor.convert(a, False)
            else:
                slice_a = None

            if b is not None and b != sys.maxsize:
                # The special "maxsize" case is probably not needed here,
                # as slices containing maxsize are not generated by
                # __getslice__ anymore.
                slice_b = Subtensor.convert(b, False)
            else:
                slice_b = None

            if c is not None:
                slice_c = Subtensor.convert(c, False)
            else:
                slice_c = None

            return slice(slice_a, slice_b, slice_c)
        elif isinstance(entry, (integer_types, numpy.integer)):
            # Disallow the use of python scalars in idx_list
            raise TypeError("Python scalar in idx_list."
                            "Please report this error to theano-dev.")
        else:
            raise AdvancedIndexingError(Subtensor.e_indextype, entry)

    def get_constant_idx(self, inputs, allow_partial=False,
                         only_process_constants=False, elemwise=True):
        """
        Return the idx_list with constant inputs replaced by their
        python scalar equivalent.
        May raise `theano.tensor.NotScalarConstantError` if the idx contains
        non-constant entries.

        If allow_partial is True, then entries that are not constant will
        stay as their input variable rather than raising an exception.

        None entries are always left as-is.

        Parameters
        ----------
        only_process_constants
            If True, we only attempt to obtain the value of an index/slice if
            it's directly constant and don't try to dig through dimshuffles,
            fills, allocs, and other to figure out its value.

        Examples
        --------
        Example usage where v, a are appropriately typed theano variables :
        >>> b = a[v, 1:3]
        >>> b.owner.op.idx_list
        (Scalar(int64), slice(Scalar(int64), Scalar(int64), None))
        >>> b.owner.op.get_constant_idx(b.owner.inputs, allow_partial=True)
        [v, slice(1, 3, None)]
        >>> b.owner.op.get_constant_idx(b.owner.inputs)
        NotScalarConstantError: v

        """
        real_idx = get_idx_list(inputs, self.idx_list)

        def conv(val):
            if val is None:
                return None
            elif isinstance(val, slice):
                return slice(conv(val.start),
                             conv(val.stop),
                             conv(val.step))
            else:
                try:
                    return get_scalar_constant_value(
                        val,
                        only_process_constants=only_process_constants,
                        elemwise=elemwise)
                except theano.tensor.NotScalarConstantError:
                    if allow_partial:
                        return val
                    else:
                        raise

        return list(map(conv, real_idx))

    def __init__(self, idx_list):
        self.idx_list = tuple(map(self.convert, idx_list))

    @staticmethod
    def my_as_scalar(a):
        # Since scal.as_scalar does not know about tensor types (it would
        # create a circular import) , this method converts either a
        # TensorVariable or a ScalarVariable to a scalar.
        if isinstance(a, gof.Variable) and isinstance(a.type, TensorType):
            return theano.tensor.scalar_from_tensor(a)
        else:
            return scal.as_scalar(a)

    def make_node(self, x, *inputs):
        """
        Parameters
        ----------
        x
            The tensor to take a subtensor of.
        inputs
            A list of theano Scalars.

        """
        x = theano.tensor.as_tensor_variable(x)
        inputs = tuple(self.my_as_scalar(a) for a in inputs)

        idx_list = list(self.idx_list)
        if len(idx_list) > x.type.ndim:
            exception = ValueError(Subtensor.e_invalid % (
                len(idx_list), x.type.ndim))
            exception.subtensor_invalid = True
            raise exception

        input_types = Subtensor.collapse(idx_list,
                                         lambda entry: isinstance(entry,
                                                                  gof.Type))
        if len(inputs) != len(input_types):
            raise IndexError(
                "Not enough inputs to fill in the Subtensor template.",
                inputs, idx_list)
        for input, expected_type in izip(inputs, input_types):
            if input.type != expected_type:
                raise TypeError(
                    "Wrong type for Subtensor template. Expected %s, got %s."
                    % (input.type, expected_type))

        # infer the broadcasting pattern
        padded = (self.get_constant_idx((None,) + inputs, allow_partial=True) +
                  [slice(None, None, None)] * (x.type.ndim - len(idx_list)))
        broadcastable = []
        for i, (p, bc) in enumerate(izip(padded, x.type.broadcastable)):
            if isinstance(p, slice):
                if bc:
                    start = p.start
                    try:
                        start = get_scalar_constant_value(start)
                    except NotScalarConstantError:
                        pass
                    if start is None or start == 0:
                        start = p.start
                        if start is None:
                            start = 0
                        if (p.stop is None or
                            (isinstance(p.stop, (integer_types, numpy.integer,
                                                 numpy.ndarray)) and
                             p.stop > start)):
                            broadcastable.append(True)
                            continue

                broadcastable.append(False)

        return gof.Apply(self,
                         (x, ) + inputs,
                         [theano.tensor.tensor(dtype=x.type.dtype,
                                               broadcastable=broadcastable)])

    def perform(self, node, inputs, out_):
        out, = out_
        x = inputs[0]

        cdata = get_idx_list(inputs, self.idx_list)
        if len(cdata) == 1:
            cdata = cdata[0]

        out[0] = numpy.asarray(x.__getitem__(cdata))

    def infer_shape(self, node, shapes):
        xshp = shapes[0]
        assert len(xshp) == node.inputs[0].ndim
        outshp = []
        actual_idx_list = list(get_idx_list(node.inputs, self.idx_list))
        padded = (actual_idx_list +
                  [slice(None, None, None)] * (len(xshp) - len(self.idx_list)))
        i = 0
        for idx, xl in izip(padded, xshp):
            if isinstance(idx, slice):
                # If it is the default (None, None, None) slice, or a variant,
                # the shape will be xl
                if ((idx.start in [None, 0]) and
                        (idx.stop in [None, sys.maxsize]) and
                        (idx.step is None or idx.step == 1)):
                    outshp.append(xl)
                else:
                    cnf = get_canonical_form_slice(idx, xl)[0]
                    if cnf.step == 1:
                        length = cnf.stop - cnf.start
                    else:
                        length = (cnf.stop - cnf.start - 1) // cnf.step + 1
                    outshp.append(length)
                i += 1
            else:
                # That dimension is dropped
                pass
        assert i == node.outputs[0].ndim
        assert len(outshp) == node.outputs[0].ndim
        return [outshp]

    def grad(self, inputs, grads):
        gz, = grads
        x = inputs[0]
        rest = inputs[1:]
        if x.dtype in theano.tensor.discrete_dtypes:
            first = x.zeros_like().astype(theano.config.floatX)
        else:
            # For best optimization, we let this as an inc.
            # This allow the opt local_IncSubtensor_serialize to apply first.
            # We need to implement an optimization that will convert this to a
            # set subtensor.
            first = IncSubtensor(self.idx_list)(x.zeros_like(),
                                                gz, *rest)
        return ([first] + [DisconnectedType()()] * len(rest))

    def connection_pattern(self, node):

        rval = [[True]]

        for ipt in node.inputs[1:]:
            rval.append([False])

        return rval

    def __hash__(self):
        # TODO: optimize by cache this hash value
        msg = []
        for entry in self.idx_list:
            if isinstance(entry, slice):
                msg += [(entry.start, entry.stop, entry.step)]
            else:
                msg += [entry]

        idx_list = tuple(msg)
        # backport
        # idx_list = tuple((entry.start, entry.stop, entry.step)
        #                 if isinstance(entry, slice)
        #                 else entry
        #                 for entry in self.idx_list)
        return hash(idx_list)

    @staticmethod
    def str_from_slice(entry):
        msg = []
        for x in [entry.start, entry.stop, entry.step]:
            if x is None:
                msg.append("")
            else:
                msg.append(str(x))
        return ":".join(msg)

    def __str__(self):
        indices = []
        for entry in self.idx_list:
            if isinstance(entry, slice):
                indices.append(self.str_from_slice(entry))
            else:
                indices.append(str(entry))
        return "%s{%s}" % (self.__class__.__name__, ", ".join(indices))

    @staticmethod
    def default_helper_c_code_args():
        """
        Returns a dictionary of default arguments to helper_c_code.

        """

        return {"c_prefix": "PyArray",
                "strides_mul": 1}

    @staticmethod
    def helper_c_code(node, name, inputs, outputs, sub, idx_list, view_ndim,
                      c_prefix=None,
                      strides_mul=None):
        """
        The parameters c_prefix are there to allow reusing this
        function on PyArray and CudaNdarray object.

        This fct take as input the x.

        """

        default_args = Subtensor.default_helper_c_code_args()

        if strides_mul is None:
            strides_mul = default_args['strides_mul']

        if c_prefix is None:
            c_prefix = default_args['c_prefix']

        #
        # two arrays are created in C code:
        # is_slice: len == ndim, 0 means int, 1 means slice
        # subtensor_spec: len = n_ints + 3 * n_slices
        #
        fail = sub['fail']
        init_cmds = []  # initialization for subtensor_spec
        is_slice = []
        # TODO: change that, it might lead to unexpected results,
        # see assembla-#767
        NONE_CODE = sys.maxsize - 1

        pos = [0, 1]  # annoying version of global variable for init_entry

        def inc_spec_pos(amt):
            pos[0] += amt

        def inc_input_pos(amt):
            pos[1] += amt

        def spec_pos():
            return pos[0]

        def input_pos():
            return pos[1]

        def init_entry(entry, depth=0):
            if isinstance(entry, (numpy.integer, integer_types)):
                init_cmds.append(
                    "subtensor_spec[%i] = %i;" % (spec_pos(),
                                                  entry))
                inc_spec_pos(1)
                if depth == 0:
                    is_slice.append(0)
            elif isinstance(entry, Type):
                init_cmds.append(
                    "subtensor_spec[%i] = %s;" % (spec_pos(),
                                                  inputs[input_pos()]))
                inc_spec_pos(1)
                inc_input_pos(1)
                if depth == 0:
                    is_slice.append(0)
            elif entry is None:
                init_cmds.append(
                    "subtensor_spec[%i] = %i;" % (spec_pos(),
                                                  NONE_CODE))
                inc_spec_pos(1)
                if depth == 0:
                    is_slice.append(0)
            elif depth == 0 and isinstance(entry, slice):
                init_entry(entry.start, depth + 1)
                init_entry(entry.stop, depth + 1)
                init_entry(entry.step, depth + 1)
                is_slice.append(1)
            else:
                assert 0, entry

        for entry in idx_list:
            init_entry(entry)
        # make sure we used all inputs
        assert input_pos() == len(inputs), input_pos()
        assert len(is_slice) <= node.inputs[0].ndim, node.inputs[0].ndim

        len_is_slice = len(is_slice)

        len_subtensor_spec = spec_pos()
        subensor_spec = "npy_intp subtensor_spec[%(len_subtensor_spec)s];" % locals()
        if len_subtensor_spec == 0:
            subensor_spec = "npy_intp * subtensor_spec = NULL;"

        if is_slice:
            is_slice_init = "int is_slice[] = {" + ",".join([str(s) for s in
                                                             is_slice]) + "};"
        else:
            is_slice_init = "int* is_slice = NULL;"
        subtensor_init = "\n".join(init_cmds)

        x, = inputs[:1]
        z, = outputs

        if view_ndim:
            rval = """
        // Argument of the view
        npy_intp xview_dims[%(view_ndim)s];
        npy_intp xview_strides[%(view_ndim)s];

        """ % locals()
        else:
            rval = """
        // Argument of the view
        npy_intp* xview_dims = NULL;
        npy_intp* xview_strides = NULL;

        """

        rval += """
        // One more argument of the view
        npy_intp xview_offset = 0;

        // The subtensor is created by iterating over the dimensions
        // and updating stride, shape, and data pointers

        %(is_slice_init)s
        %(subensor_spec)s
        %(subtensor_init)s;
        int spec_pos = 0; //position in subtensor_spec
        int inner_ii = 0; // the current dimension of zview
        int outer_ii = 0; // current dimension of z


        for (; outer_ii < %(len_is_slice)s; ++outer_ii)
        {
            if (is_slice[outer_ii])
            {
                npy_intp length = %(c_prefix)s_DIMS(%(x)s)[outer_ii];
                npy_intp slicelength;
                npy_intp start = subtensor_spec[spec_pos+0];
                npy_intp stop  = subtensor_spec[spec_pos+1];
                npy_intp step  = subtensor_spec[spec_pos+2];
                if (step == %(NONE_CODE)s) step = 1;

                npy_intp defstart = step < 0 ? length-1 : 0;
                npy_intp defstop = step < 0 ? -1 : length;

                // logic adapted from
                // PySlice_GetIndicesEx in python source
                if (!step)
                {
                    PyErr_Format(PyExc_ValueError,
                                 "slice step cannot be zero");
                    %(fail)s;
                }

                if (start == %(NONE_CODE)s)
                {
                    start = defstart;
                }
                else
                {
                    if (start < 0) start += length;
                    if (start < 0) start = (step < 0) ? -1 : 0;
                    if (start >= length)
                        start = (step < 0) ? length - 1 : length;
                }

                if (stop == %(NONE_CODE)s)
                {
                    stop = defstop;
                }
                else
                {
                    if (stop < 0) stop += length;
                    if (stop < 0) stop = (step < 0) ? -1 : 0;
                    if (stop >= length)
                        stop = (step < 0) ? length - 1 : length;
                }

                if ((step < 0 && stop >= start)
                    || (step > 0 && start >= stop)) {
                    slicelength = 0;
                }
                else if (step < 0) {
                    slicelength = (stop-start+1)/step+1;
                }
                else {
                    slicelength = (stop-start-1)/step+1;
                }

                if (0){
                    fprintf(stdout, "start %%zi\\n", start);
                    fprintf(stdout, "stop %%zi\\n", stop);
                    fprintf(stdout, "step %%zi\\n", step);
                    fprintf(stdout, "length %%zi\\n", length);
                    fprintf(stdout, "slicelength %%zi\\n", slicelength);
                }

                assert (slicelength <= length);

                xview_offset += (npy_intp)%(c_prefix)s_STRIDES(%(x)s)[outer_ii]
                    * start * %(strides_mul)s;
                xview_dims[inner_ii] = slicelength;
                xview_strides[inner_ii] = (npy_intp)%(c_prefix)s_STRIDES(%(x)s)[outer_ii] * step;

                inner_ii += 1;
                spec_pos += 3;
            }
            else // tuple coord `outer_ii` is an int
            {
                int idx = subtensor_spec[spec_pos];
                if (idx < 0) idx += %(c_prefix)s_DIMS(%(x)s)[outer_ii];
                if (idx >= 0)
                {
                    if (idx < %(c_prefix)s_DIMS(%(x)s)[outer_ii])
                    {
                        xview_offset += (npy_intp)%(c_prefix)s_STRIDES(%(x)s)[outer_ii] * idx *
                               %(strides_mul)s;
                    }
                    else
                    {
                        PyErr_Format(PyExc_IndexError,"index out of bounds");
                        %(fail)s;
                    }
                }
                else
                {
                    PyErr_Format(PyExc_IndexError,"index out of bounds");
                    %(fail)s;
                }

                spec_pos += 1;
            }
        }
        assert (inner_ii <= %(view_ndim)s);
        while (inner_ii < %(view_ndim)s)
        {
            assert (outer_ii < %(c_prefix)s_NDIM(%(x)s));
            xview_dims[inner_ii] = %(c_prefix)s_DIMS(%(x)s)[outer_ii];
            xview_strides[inner_ii] = %(c_prefix)s_STRIDES(%(x)s)[outer_ii];

            inner_ii += 1;
            outer_ii += 1;
        }
        """ % locals()
        # print rval
        return rval

    @staticmethod
    def helper_c_code_cache_version():
        return (9,)

    def c_code(self, node, name, inputs, outputs, sub):  # DEBUG
        if not isinstance(node.inputs[0].type, theano.tensor.TensorType):
            raise NotImplementedError()

        x = inputs[0]
        z, = outputs
        ndim = node.inputs[0].ndim
        view_ndim = node.outputs[0].ndim
        fail = sub['fail']

        decl = "PyArrayObject * xview = NULL;"

        checkNDim = """
        if (PyArray_NDIM(%(x)s) != %(ndim)s){
            PyErr_SetString(PyExc_ValueError,
                                     "Expected %(ndim)s dimensions input"
                                        );
            %(fail)s
        }
        """ % locals()

        get_xview = self.helper_c_code(node, name, inputs, outputs, sub,
                                       self.idx_list, view_ndim)
        build_view = """
        //TODO: give this Op a second output so that this view can be cached
        //TODO: alternatively, fix the memory leak on failure
        Py_INCREF(PyArray_DESCR(%(x)s));
        xview = (PyArrayObject*)PyArray_NewFromDescr(
                &PyArray_Type,
                PyArray_DESCR(%(x)s),
                %(view_ndim)s,
                xview_dims,
                xview_strides,
                PyArray_BYTES(%(x)s) + xview_offset,
                PyArray_FLAGS(%(x)s),
                NULL);
        assert (PyArray_NDIM(xview) == %(view_ndim)s);
        if (!xview)
        {
            %(fail)s;
        }
        """ % locals()

        finish_view = """
        Py_XDECREF(%(z)s);
        Py_INCREF(py_%(x)s);
        PyArray_SetBaseObject(xview, py_%(x)s);
        assert(py_%(x)s == (PyObject*)%(x)s);
        %(z)s = xview;
        """ % locals()

        return (decl + checkNDim +
                "{" + get_xview + build_view + finish_view + "}")

    def c_code_cache_version(self):
        hv = self.helper_c_code_cache_version()
        # If `helper_c_code_cache_version` is not versioned we do not want to
        # have a versioned version of this op's C code.
        if len(hv) == 0:
            return ()
        return (4, hv)

    def R_op(self, inputs, eval_points):
        # Subtensor is not differentiable wrt to its indices, therefore we
        # do not even need to consider the eval_points provided for those
        # (they should be defaulted to zeros_like by the global R_op)
        if eval_points[0] is None:
            return [None]
        return self(eval_points[0], *inputs[1:], **dict(return_list=True))


class SubtensorPrinter:

    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print Subtensor.")
        elif isinstance(r.owner.op, Subtensor):
            idxs = r.owner.op.idx_list
            inputs = list(r.owner.inputs)
            input = inputs.pop(0)
            sidxs = []
            old_precedence = getattr(pstate, 'precedence', None)
            try:
                pstate.precedence = -1000

                for entry in idxs:
                    if isinstance(entry, integer_types):
                        sidxs.append(str(entry))
                    elif isinstance(entry, scal.Scalar):
                        sidxs.append(pstate.pprinter.process(inputs.pop()))
                    elif isinstance(entry, slice):
                        if entry.start is None or entry.start == 0:
                            msg1 = ""
                        else:
                            msg1 = entry.start

                        if entry.stop is None or entry.stop == sys.maxsize:
                            msg2 = ""
                        else:
                            msg2 = entry.stop

                        if entry.step is None:
                            msg3 = ""
                        else:
                            msg3 = ":%s" % entry.step

                        sidxs.append("%s:%s%s" % (msg1, msg2, msg3))
            finally:
                pstate.precedence = old_precedence

            try:
                pstate.precedence = 1000
                sub = pstate.pprinter.process(input, pstate)
            finally:
                pstate.precedence = old_precedence
            return "%s[%s]" % (sub, ", ".join(sidxs))
        else:
            raise TypeError("Can only print Subtensor.")

pprint.assign(Subtensor, SubtensorPrinter())


def set_subtensor(x, y, inplace=False,
                  tolerate_inplace_aliasing=False):
    """
    Return x with the given subtensor overwritten by y.

    Parameters
    ----------
    x
        Symbolic variable for the lvalue of = operation.
    y
        Symbolic variable for the rvalue of = operation.
    tolerate_inplace_aliasing
        See inc_subtensor for documentation.

    Examples
    --------
    To replicate the numpy expression "r[10:] = 5", type

    >>> r = ivector()
    >>> new_r = set_subtensor(r[10:], 5)

    """
    return inc_subtensor(x, y, inplace, set_instead_of_inc=True,
                         tolerate_inplace_aliasing=tolerate_inplace_aliasing)


def inc_subtensor(x, y, inplace=False, set_instead_of_inc=False,
                  tolerate_inplace_aliasing=False):
    """
    Return x with the given subtensor incremented by y.

    Parameters
    ----------
    x
        The symbolic result of a Subtensor operation.
    y
        The amount by which to increment the subtensor in question.
    inplace
        Don't use. Theano will do it when possible.
    set_instead_of_inc
        If True, do a set_subtensor instead.
    tolerate_inplace_aliasing:
        Allow x and y to be views of a single underlying array even while
        working inplace. For correct results, x and y must not be overlapping
        views; if they overlap, the result of this Op will generally be
        incorrect. This value has no effect if inplace=False.

    Examples
    --------
    To replicate the numpy expression "r[10:] += 5", type

    >>> r = ivector()
    >>> new_r = inc_subtensor(r[10:], 5)

    """
    # First of all, y cannot have a higher dimension than x,
    # nor have non-broadcastable dimensions where x is broadcastable.

    x = theano.tensor.as_tensor_variable(x)
    y = theano.tensor.as_tensor_variable(y)

    if y.ndim > x.ndim:
        raise TypeError(("Trying to increment a %d-dimensional "
                         "subtensor with a %d-dimensional value.") % (x.ndim,
                                                                      y.ndim))

    dim_offset = x.ndim - y.ndim
    for dim in xrange(y.ndim):
        if (x.broadcastable[dim + dim_offset] and not y.broadcastable[dim]):
            # It is acceptable to try to increment a subtensor with a
            # broadcastable dim with a tensor that is not broadcastable
            # on that dimension. However, its length must then be 1.
            # We insert a Rebroadcast Op to make sure it is the case.
            y = addbroadcast(y, dim)

    if not x.owner:
        raise TypeError('x must be the result of a subtensor operation')

    # retrieve idx_list from x.owner
    if isinstance(x.owner.op, Subtensor):
        if tolerate_inplace_aliasing:
            destroyhandler_tolerate_aliased = [[0, 1]]
        else:
            destroyhandler_tolerate_aliased = []
        the_op = IncSubtensor(
            x.owner.op.idx_list, inplace, set_instead_of_inc,
            destroyhandler_tolerate_aliased=destroyhandler_tolerate_aliased)
        real_x = x.owner.inputs[0]
        real_idxargs = x.owner.inputs[1:]
        return the_op(real_x, y, *real_idxargs)
    elif isinstance(x.owner.op, AdvancedSubtensor1):
        real_x = x.owner.inputs[0]
        ilist = x.owner.inputs[1]
        the_op = AdvancedIncSubtensor1(inplace,
                                       set_instead_of_inc=set_instead_of_inc)
        return the_op(real_x, y, ilist)
    elif isinstance(x.owner.op, AdvancedSubtensor):
        real_x = x.owner.inputs[0]
        ilist = x.owner.inputs[1:]

        the_op = AdvancedIncSubtensor(inplace,
                                      set_instead_of_inc=set_instead_of_inc)
        return the_op(real_x, y, *ilist)
    elif isinstance(x.owner.op, DimShuffle):
        inner_x = x.owner.inputs[0]
        # In the dimshuffle case, there are in fact two dimshuffles:
        # one to make the indexed dimension the last one,
        # and one to put it back where it was. So, in the case where we have
        # inc_subtensor(x[:,i], y), the graph is actually
        # inc_subtensor((x.T)[i].T, y).
        # We could get all the way to x, and then get rid of the dimshuffles
        # completely, but the problem is that advanced_inc_subtensor1 can only
        # work on the first (outer-most, left-most) dimension of x,
        # just like advanced_subtensor1.
        # So we call advanced_inc_subtensor1(x.T, i, y.T) (as we also need to
        # transpose y if it is not a scalar or a vector), but then we need to
        # return something that has the same shape as x, not as x.T (inner_x).
        # So re-apply the outer dimshuffle on the new inc_subtensor,
        # and return advanced_inc_subtensor1(x.T, i, y.T).T.

        # Get the dimshuffle pattern to apply to y.
        x_order = x.owner.op.new_order
        y_order = ['x'] * x.ndim
        for i, v in enumerate(x_order):
            if v != 'x' and (v - dim_offset) >= 0:
                y_order[v - dim_offset] = i

        # Warn if this code path would have produced wrong results in the past
        if config.warn.inc_set_subtensor1:
            # Dimshuffle pattern for y that would be equivalent to past code
            prev_y_order = ['x'] * (dim_offset) + list(range(y.ndim))
            if y_order != prev_y_order:
                warnings.warn(
                    'Although your current code is fine, please note that '
                    'earlier versions prior to 0.7 (or this development '
                    'version) may have yielded an incorrect result in '
                    'this `inc_subtensor` or `set_subtensor` operation. '
                    'To remove this warning, you can either set the '
                    '`warn.inc_set_subtensor1` config option to `False`, '
                    'or `warn.ignore_bug_before` to at least "0.7".',
                    stacklevel=2)

        inner_incsubtensor = inc_subtensor(
            inner_x,
            y.dimshuffle(y_order),
            inplace=inplace,
            set_instead_of_inc=set_instead_of_inc,
            tolerate_inplace_aliasing=tolerate_inplace_aliasing)
        # The broadcastable pattern of inner_x may not be the same as
        # the one of x, so we have to build a new dimshuffle here,
        # instead of reusing x.owner.op().
        return inner_incsubtensor.dimshuffle(x.owner.op.new_order)

    elif isinstance(x.owner.op, theano.tensor.Reshape):
        # This case happens when the indices are not arranged as a vector, but
        # as a higher-dimensional array. This is handled by the subtensor
        # by flattening this list, taking the subtensor, then reshaping the
        # result.
        inner_x = x.owner.inputs[0]
        # Try to apply inc_subtensor on inner_x.
        # If it works, there is no need to reshape, as the inc_subtensor
        # will have the same shape as inner_x, which is what we want.
        # We also explicitly duplicate y to its broadcasted shape
        # before we partially flatten it to inner_x dimension. This is
        # not strictly needed in all cases, but it is easier this way.
        if y.ndim > 0:
            # This if is needed to prevent some useless warning about
            # old code bug.
            expanded_y = alloc(y, *[x.shape[i] for i in xrange(x.ndim)])
            flattened_y = expanded_y.reshape(inner_x.shape)
        else:
            flattened_y = y

        # Warn if this code path would have produced wrong results in the past
        if config.warn.inc_set_subtensor1:
            if inner_x.ndim > 1 and sum(y.broadcastable) > 0:
                warnings.warn(
                    'Although your current code is fine, please note that '
                    'earlier versions prior to 0.7 (or this development '
                    'version) may have yielded an incorrect result in '
                    'this `inc_subtensor` or `set_subtensor` operation. '
                    'To remove this warning, you can either set the '
                    '`warn.inc_set_subtensor1` config option to `False`, '
                    'or `warn.ignore_bug_before` to at least "0.7".',
                    stacklevel=2)

        inner_incsubtensor = inc_subtensor(
            inner_x,
            flattened_y,
            inplace=inplace,
            set_instead_of_inc=set_instead_of_inc,
            tolerate_inplace_aliasing=tolerate_inplace_aliasing)
        return inner_incsubtensor
    else:
        raise TypeError('x must be the result of a subtensor operation')


class IncSubtensor(Op):
    """
    Increment a subtensor.

    This is like numpy's

        x[i,j,k] += y

    It is used internally to implement the gradient on SubTensor.

    Parameters
    ----------
    set_instead_of_inc
        If True set the subtensor to the value instead of incrementing it by
        that value.

    """

    check_input = False
    __props__ = ("idx_list", "inplace", "set_instead_of_inc")

    def __init__(self, idx_list, inplace=False, set_instead_of_inc=False,
                 destroyhandler_tolerate_aliased=None):
        if destroyhandler_tolerate_aliased is None:
            destroyhandler_tolerate_aliased = []
        self.idx_list = list(map(Subtensor.convert, idx_list))
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [0]}
        self.destroyhandler_tolerate_aliased = list(
            destroyhandler_tolerate_aliased)
        self.set_instead_of_inc = set_instead_of_inc

    def __hash__(self):
        msg = []
        for entry in self.idx_list:
            if isinstance(entry, slice):
                msg += [(entry.start, entry.stop, entry.step)]
            else:
                msg += [entry]

        idx_list = tuple(msg)
        # backport
        # idx_list = tuple((entry.start, entry.stop, entry.step)
        #                 if isinstance(entry, slice)
        #                 else entry
        #                 for entry in self.idx_list)
        return (hashtype(self) ^ hash(idx_list) ^ hash(self.inplace) ^
                hash(self.set_instead_of_inc))

    def __str__(self):
        indices = []
        for entry in self.idx_list:
            if isinstance(entry, slice):
                indices.append(Subtensor.str_from_slice(entry))
            else:
                indices.append(str(entry))
        if self.inplace:
            msg = 'Inplace'
        else:
            msg = ''
        if not self.set_instead_of_inc:
            msg += 'Inc'
        else:
            msg += 'Set'
        return "%s{%s;%s}" % (
            self.__class__.__name__,
            msg,
            ", ".join(indices))

    def make_node(self, x, y, *inputs):
        """
        Parameters
        ----------
        x
            The tensor to increment.
        y
            The value to increment by.
        inputs: TODO WRITEME

        """
        x, y = map(theano.tensor.as_tensor_variable, [x, y])
        if y.ndim > x.ndim:
            raise ValueError(("Trying to increment a %d-dimensional "
                              "subtensor with a %d-dimensional value.") % (
                                  x.ndim, y.ndim))
        inputs = tuple(map(Subtensor.my_as_scalar, inputs))

        idx_list = list(self.idx_list)
        if len(idx_list) > x.type.ndim:
            exception = ValueError(
                Subtensor.e_invalid % (
                    len(idx_list),
                    x.type.ndim))
            exception.subtensor_invalid = True
            raise exception

        input_types = Subtensor.collapse(
            idx_list,
            lambda entry: isinstance(entry, gof.Type))
        if len(inputs) != len(input_types):
            raise IndexError(
                "Not enough inputs to fill in the Subtensor template.",
                inputs, idx_list)
        for input, expected_type in izip(inputs, input_types):
            if input.type != expected_type:
                raise TypeError(
                    "Wrong type for Subtensor template. Expected %s, got %s."
                    % (input.type, expected_type))

        return gof.Apply(self,
                         (x, y) + inputs,
                         [x.type()])

    def decl_view(self):
        return "PyArrayObject * zview = NULL;"

    def perform(self, node, inputs, out_):
        out, = out_
        x, y = inputs[:2]
        indices = list(reversed(inputs[2:]))

        def convert(entry):
            if isinstance(entry, gof.Type):
                rval = indices.pop()
                if sys.version_info < (2, 5):
                    # Before Python 2.5, PySlice_GetIndicesEx requires
                    # Python int to be passed.
                    rval_ = int(rval)
                    if rval_ != rval:
                        raise IndexError((
                            "Invalid value for indexing: %s. "
                            "That value may be too big.") % rval)
                    return rval_
                return rval
            elif isinstance(entry, slice):
                return slice(convert(entry.start),
                             convert(entry.stop),
                             convert(entry.step))
            else:
                return entry

        cdata = tuple(map(convert, self.idx_list))
        if len(cdata) == 1:
            cdata = cdata[0]
        if not self.inplace:
            x = x.copy()
        sub_x = x.__getitem__(cdata)
        if sub_x.shape:
            # we've sliced out an N-D tensor with N > 0
            if not self.set_instead_of_inc:
                sub_x += y
            else:
                # sub_x += -sub_x + y
                x.__setitem__(cdata, y)
        else:
            # scalar case
            if not self.set_instead_of_inc:
                x.__setitem__(cdata, sub_x + y)
            else:
                x.__setitem__(cdata, y)
        out[0] = x

    def c_code(self, node, name, inputs, outputs, sub):

        # This method delegates much of the work to helper
        # methods. This method implements the main logic
        # but subclasses may override the helper methods
        # to change the particulars, e.g. GpuIncSubtensor
        # turns the view/copy operations on numpy arrays
        # into the same operations on cuda arrays.

        self.do_type_checking(node)

        if self.inplace:  # convert bool to int
            inplace = 1
        else:
            inplace = 0
        x = inputs[0]
        y = inputs[1]
        z, = outputs
        if self.set_instead_of_inc:  # convert bool to int
            op_is_set = 1
        else:
            op_is_set = 0
        fail = sub['fail']
        view_ndim = (node.inputs[0].ndim -
                     numpy.sum([not isinstance(idx, slice)
                                for idx in self.idx_list]))

        copy_of_x = self.copy_of_x(x)

        copy_input_if_necessary = """
        if (%(inplace)s)
        {
            if (%(x)s != %(z)s)
            {
                Py_XDECREF(%(z)s);
                Py_INCREF(%(x)s);
                %(z)s = %(x)s;
            }
        }
        else
        {
            Py_XDECREF(%(z)s);
            %(z)s = %(copy_of_x)s;
        }
        """ % locals()

        # get info needed to make zview: a view of %(z)s
        helper_args = self.get_helper_c_code_args()

        get_zview = Subtensor.helper_c_code(
            node=node,
            name=name,
            inputs=outputs[:1] + inputs[2:],
            outputs=outputs,
            sub=sub,
            idx_list=self.idx_list,
            view_ndim=view_ndim,
            ** helper_args
        )

        # Make a view on the output, as we will write into it.
        alloc_zview = self.make_view_array(z, view_ndim)

        build_view = """
        //TODO: give this Op a second output so that this view can be cached
        //TODO: alternatively, fix the memory leak on failure
        %(alloc_zview)s;
        if (!zview)
        {
            %(fail)s;
        }
        """ % locals()

        copy_into = self.copy_into("zview", y)

        add_to_zview = self.add_to_zview(name, y, fail)

        make_modification = """
        if (%(op_is_set)s)
        {
            if (%(copy_into)s) // does broadcasting
            {
                Py_DECREF(zview);
                %(fail)s;
            }
        }
        else
        {
            %(add_to_zview)s
        }
        """ % locals()
        return (self.decl_view() +
                copy_input_if_necessary +
                get_zview +
                build_view +
                make_modification +
                "Py_DECREF(zview);"
                )

    def do_type_checking(self, node):
        """
        Should raise NotImplementedError if c_code does not support
        the types involved in this node.

        """

        if not isinstance(node.inputs[0].type, theano.tensor.TensorType):
            raise NotImplementedError()

    def c_code_cache_version(self):
        hv = Subtensor.helper_c_code_cache_version()
        if hv:
            return (1, hv)
        else:
            return ()

    def copy_of_x(self, x):
        """
        Parameters
        ----------
        x
            A string giving the name of a C variable pointing to an array.

        Returns
        -------
        object
            C code expression to make a copy of x.

        Base class uses PyArrayObject *, subclasses may override for
        different types of arrays.

        """
        # Parameters of PyArrary_FromAny are:
        # array
        # dtype: we pass NULL to say any dtype is acceptable, so the existing
        #        dtype will be copied
        # min_depth: we pass 0 to have this parameter ignored
        # max_depth: we pass 0 to have this parameter ignored
        # requirements: here we pass NPY_ARRAY_ENSURECOPY to force a copy
        # context: this is almost always NULL, I'm not sure what it's used for
        return """(PyArrayObject*)PyArray_FromAny(py_%(x)s, NULL, 0, 0,
                NPY_ARRAY_ENSURECOPY, NULL)""" % locals()

    def make_view_array(self, x, view_ndim):
        """
        Parameters
        ----------
        x
            A string identifying an array to be viewed.
        view_ndim
            A string specifying the number of dimensions to have in the view.

        This doesn't need to actually set up the view with the right indexing;
        we'll do that manually later.

        """

        return """Py_INCREF(PyArray_DESCR(%(x)s));
        zview = (PyArrayObject*)PyArray_NewFromDescr(
                &PyArray_Type,
                PyArray_DESCR(%(x)s),
                %(view_ndim)s,
                xview_dims, //PyArray_DIMS(%(x)s),
                xview_strides, //PyArray_STRIDES(%(x)s),
                PyArray_BYTES(%(x)s) + xview_offset, //PyArray_DATA(%(x)s),
                PyArray_FLAGS(%(x)s),
                NULL);
        """ % locals()

    def get_helper_c_code_args(self):
        """
        Return a dictionary of arguments to pass to helper_c_code.

        """
        return Subtensor.default_helper_c_code_args()

    def copy_into(self, view, source):
        """
        Parameters
        ----------
        view : string
            C code expression for an array.
        source : string
            C code expression for an array.

        Returns
        -------
        object
            C code expression to copy source into view, and 0 on success.

        """
        return """PyArray_CopyInto(%(view)s, %(source)s)""" % locals()

    def add_to_zview(self, name, x, fail):
        """
        Return C code to add x to zview. Should DECREF zview if the
        add fails.

        """

        return """
            PyArrayObject * add_rval = (PyArrayObject*)PyNumber_InPlaceAdd(
                    (PyObject*)zview, py_%(x)s);
            if (add_rval)
            {
                assert (PyArray_Check((PyObject*)add_rval));
                assert (PyArray_DATA(add_rval) == PyArray_DATA(zview));
                Py_DECREF(add_rval);
            }
            else
            {
                Py_DECREF(zview);
                %(fail)s;
            }""" % locals()

    def infer_shape(self, node, shapes):
        return [shapes[0]]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None or eval_points[1] is None:
            return [None]
        # Again we ignore eval points for indices because incsubtensor is
        # not differentiable wrt to those
        return self(eval_points[0], eval_points[1], *inputs[2:],
                    **dict(return_list=True))

    def connection_pattern(self, node):

        rval = [[True], [True]]

        for ipt in node.inputs[2:]:
            rval.append([False])

        return rval

    def grad(self, inputs, grads):
        g_output, = grads
        x, y = inputs[:2]
        idx_list = inputs[2:]

        if x.dtype in theano.tensor.discrete_dtypes:
            # The output dtype is the same as x
            gx = x.zeros_like(dtype=theano.config.floatX)
            if y.dtype in theano.tensor.discrete_dtypes:
                gy = y.zeros_like(dtype=theano.config.floatX)
            else:
                gy = y.zeros_like()
        elif x.dtype in theano.tensor.complex_dtypes:
            raise NotImplementedError("No support for complex grad yet")
        else:
            if self.set_instead_of_inc:
                gx = set_subtensor(
                    Subtensor(idx_list=self.idx_list)(g_output, *idx_list),
                    theano.tensor.zeros_like(y))
            else:
                gx = g_output
            gy = Subtensor(idx_list=self.idx_list)(g_output, *idx_list)
            gy = _sum_grad_over_bcasted_dims(y, gy)

        return [gx, gy] + [DisconnectedType()()] * len(idx_list)


def _sum_grad_over_bcasted_dims(x, gx):
    """
    Sum of gx over dimensions to reproduce x.broadcastable.

    This is useful to sum gradients over certain dimensions when
    x has been broadcasted, and we need to sum the gradient contributions
    over all duplications.

    """
    if gx.broadcastable != x.broadcastable:
        x_dim_added = gx.ndim - x.ndim
        x_broad = (True,) * x_dim_added + x.broadcastable
        assert sum(gx.broadcastable) < sum(x_broad)
        axis_to_sum = []
        for i in xrange(gx.ndim):
            if gx.broadcastable[i] is False and x_broad[i] is True:
                axis_to_sum.append(i)
            elif (gx.broadcastable[i] is True and
                  x_broad[i] is False):
                # This means that Theano was able to infer that
                # gx.shape[i] is 1, so x.shape[i] is 1, but we
                # didn't know it. It is fine.
                pass
            else:
                assert gx.broadcastable[i] == x_broad[i]
        gx = gx.sum(axis=axis_to_sum, keepdims=True)
        if gx.ndim != x.ndim:
            assert gx.ndim > x.ndim
            for i in xrange(x_dim_added):
                assert gx.broadcastable[i]
            gx = gx.dimshuffle(*list(range(x_dim_added, gx.ndim)))
        assert gx.broadcastable == x.broadcastable
    return gx


#########################
# Advanced indexing
#########################
#
# Should reproduce numpy's behaviour, see url:
# docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing


class AdvancedSubtensor1(Op):
    """
    Implement x[ilist] where ilist is a vector of integers.

    """
    # sparse_grad doesn't go in here since it only affects the output
    # of the grad() method.
    __props__ = ()
    _f16_ok = True

    def __init__(self, sparse_grad=False):
        self.sparse_grad = sparse_grad

    def make_node(self, x, ilist):
        x_ = theano.tensor.as_tensor_variable(x)
        ilist_ = theano.tensor.as_tensor_variable(ilist)
        if ilist_.type.dtype not in theano.tensor.integer_dtypes:
            raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')
        bcast = (ilist_.broadcastable[0],) + x_.broadcastable[1:]
        return Apply(self, [x_, ilist_], [TensorType(dtype=x.dtype,
                                                     broadcastable=bcast)()])

    def perform(self, node, inp, out_):
        x, i = inp
        out, = out_
        # Copy always implied by numpy advanced indexing semantic.
        if out[0] is not None and out[0].shape == (len(i),) + x.shape[1:]:
            o = out[0]
        else:
            o = None

        # If i.dtype is more precise than numpy.intp (int32 on 32-bit machines,
        # int64 on 64-bit machines), numpy may raise the following error:
        # TypeError: array cannot be safely cast to required type.
        # We need to check if values in i can fit in numpy.intp, because
        # if they don't, that should be an error (no array can have that
        # many elements on a 32-bit arch).
        if i.dtype != numpy.intp:
            i_ = theano._asarray(i, dtype=numpy.intp)
            if not numpy.can_cast(i.dtype, numpy.intp):
                # Check if there was actually an incorrect conversion
                if numpy.any(i != i_):
                    raise IndexError(
                        'index contains values that are bigger '
                        'than the maximum array size on this system.', i)
            i = i_

        out[0] = x.take(i, axis=0, out=o)

    def connection_pattern(self, node):
        rval = [[True]]

        for ipt in node.inputs[1:]:
            rval.append([False])

        return rval

    def grad(self, inputs, grads):
        global sparse_module_ref
        x, ilist = inputs
        gz, = grads
        assert len(inputs) == 2
        if self.sparse_grad:
            if x.type.ndim != 2:
                raise TypeError(
                    "AdvancedSubtensor1: you can't take the sparse grad"
                    " from a tensor with ndim != 2. ndim is " +
                    str(x.type.ndim))
            if sparse_module_ref is None:
                import theano.sparse as sparse_module_ref

            rval1 = [sparse_module_ref.construct_sparse_from_list(x, gz,
                                                                  ilist)]
        else:
            rval1 = [advanced_inc_subtensor1(x.zeros_like(), gz, ilist)]
        return rval1 + [DisconnectedType()()] * (len(inputs) - 1)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self.make_node(eval_points[0], *inputs[1:]).outputs

    def infer_shape(self, node, ishapes):
        x, ilist = ishapes
        return [ilist + x[1:]]

    def c_support_code(self):
        # In some versions of numpy, NPY_MIN_INTP is defined as MIN_LONG,
        # which is not defined. It should be NPY_MIN_LONG instead in that case.
        return dedent("""\
                #ifndef MIN_LONG
                #define MIN_LONG NPY_MIN_LONG
                #endif""")

    def c_code(self, node, name, input_names, output_names, sub):
        if self.__class__ is not AdvancedSubtensor1:
            raise MethodNotDefined(
                "c_code defined for AdvancedSubtensor1,"
                " not for child class", type(self))
        a_name, i_name = input_names[0], input_names[1]
        output_name = output_names[0]
        fail = sub['fail']
        return """
            PyArrayObject *indices;
            int i_type = PyArray_TYPE(%(i_name)s);
            if (i_type != NPY_INTP) {
                // Cast %(i_name)s to NPY_INTP (expected by PyArray_TakeFrom),
                // if all values fit.
                if (!PyArray_CanCastSafely(i_type, NPY_INTP) &&
                    PyArray_SIZE(%(i_name)s) > 0) {
                    npy_int64 min_val, max_val;
                    PyObject* py_min_val = PyArray_Min(%(i_name)s, NPY_MAXDIMS,
                                                       NULL);
                    if (py_min_val == NULL) {
                        %(fail)s;
                    }
                    min_val = PyLong_AsLongLong(py_min_val);
                    Py_DECREF(py_min_val);
                    if (min_val == -1 && PyErr_Occurred()) {
                        %(fail)s;
                    }
                    PyObject* py_max_val = PyArray_Max(%(i_name)s, NPY_MAXDIMS,
                                                       NULL);
                    if (py_max_val == NULL) {
                        %(fail)s;
                    }
                    max_val = PyLong_AsLongLong(py_max_val);
                    Py_DECREF(py_max_val);
                    if (max_val == -1 && PyErr_Occurred()) {
                        %(fail)s;
                    }
                    if (min_val < NPY_MIN_INTP || max_val > NPY_MAX_INTP) {
                        PyErr_SetString(PyExc_IndexError,
                                     "Index contains values "
                                     "that are bigger than the maximum array "
                                     "size on this system.");
                        %(fail)s;
                    }
                }
                indices = (PyArrayObject*) PyArray_Cast(%(i_name)s, NPY_INTP);
                if (indices == NULL) {
                    %(fail)s;
                }
            }
            else {
                 indices = %(i_name)s;
                 Py_INCREF(indices);
            }
            if (%(output_name)s != NULL) {
                npy_intp nd, i, *shape;
                nd = PyArray_NDIM(%(a_name)s) + PyArray_NDIM(indices) - 1;
                if (PyArray_NDIM(%(output_name)s) != nd) {
                    Py_CLEAR(%(output_name)s);
                }
                else {
                    shape = PyArray_DIMS(%(output_name)s);
                    for (i = 0; i < PyArray_NDIM(indices); i++) {
                        if (shape[i] != PyArray_DIMS(indices)[i]) {
                            Py_CLEAR(%(output_name)s);
                            break;
                        }
                    }
                    if (%(output_name)s != NULL) {
                        for (; i < nd; i++) {
                            if (shape[i] != PyArray_DIMS(%(a_name)s)[
                                                i-PyArray_NDIM(indices)+1]) {
                                Py_CLEAR(%(output_name)s);
                                break;
                            }
                        }
                    }
                }
            }
            %(output_name)s = (PyArrayObject*)PyArray_TakeFrom(
                        %(a_name)s, (PyObject*)indices, 0, %(output_name)s, NPY_RAISE);
            Py_DECREF(indices);
            if (%(output_name)s == NULL) %(fail)s;
        """ % locals()

    def c_code_cache_version(self):
        return (0, 1, 2)

advanced_subtensor1 = AdvancedSubtensor1()


class AdvancedIncSubtensor1(Op):
    """
    Increments a subtensor using advanced slicing (list of index).

    """

    __props__ = ('inplace', 'set_instead_of_inc')

    def __init__(self, inplace=False, set_instead_of_inc=False):
        self.inplace = inplace
        self.set_instead_of_inc = set_instead_of_inc
        if inplace:
            self.destroy_map = {0: [0]}

    def clone_inplace(self):
        return self.__class__(
            inplace=True,
            set_instead_of_inc=self.set_instead_of_inc)

    def __str__(self):
        if self.inplace:
            msg = "inplace"
        else:
            msg = "no_inplace"
        if self.set_instead_of_inc:
            msg += ",set"
        else:
            msg += ",inc"

        return self.__class__.__name__ + "{%s}" % msg

    def make_node(self, x, y, ilist):
        x_ = theano.tensor.as_tensor_variable(x)
        y_ = theano.tensor.as_tensor_variable(y)
        ilist_ = theano.tensor.as_tensor_variable(ilist)

        if ilist_.type.dtype not in theano.tensor.integer_dtypes:
            raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')
        if y_.type.ndim > x_.type.ndim:
            if self.set_instead_of_inc:
                opname = 'set'
            else:
                opname = 'increment'
            raise TypeError(
                'cannot %s x subtensor with ndim=%s'
                ' by y with ndim=%s to x subtensor with ndim=%s ' % (
                    opname, x_.type.ndim, y_.type.ndim))

        return Apply(self, [x_, y_, ilist_], [x_.type()])

    def copy_of_x(self, x):
        """
        Parameters
        ----------
        x : string
            Gives the name of a C variable pointing to an array.

        Returns
        -------
        object
            C code expression to make a copy of x.

        Base class uses PyArrayObject *, subclasses may override for
        different types of arrays.

        """
        # Parameters of PyArrary_FromAny are:
        # array
        # dtype: we pass NULL to say any dtype is acceptable, so the existing
        #        dtype will be copied
        # min_depth: we pass 0 to have this parameter ignored
        # max_depth: we pass 0 to have this parameter ignored
        # requirements: here we pass NPY_ARRAY_ENSURECOPY to force a copy
        # context: this is almost always NULL, I'm not sure what it's used for
        return """(PyArrayObject*)PyArray_FromAny(py_%(x)s, NULL, 0, 0,
                NPY_ARRAY_ENSURECOPY, NULL)""" % locals()

    def c_support_code(self):
        from theano.gof.cutils import compile_cutils_code
        return compile_cutils_code()

    def c_code(self, node, name, input_names, output_names, sub):
        numpy_ver = [int(n) for n in numpy.__version__.split('.')[:2]]
        if bool(numpy_ver < [1, 8]):
            raise NotImplementedError
        x, y, idx = input_names
        out = output_names[0]
        fail = sub['fail']
        inc_or_set = 1 - self.set_instead_of_inc
        if self.inplace:  # convert bool to int
            inplace = 1
        else:
            inplace = 0
        copy_of_x = self.copy_of_x(x)

        return """
        PyObject* rval = NULL;
        if (%(inplace)s)
        {
            if (%(x)s != %(out)s)
            {
                Py_XDECREF(%(out)s);
                Py_INCREF(%(x)s);
                %(out)s = %(x)s;
            }
        }
        else
        {
            Py_XDECREF(%(out)s);
            %(out)s = %(copy_of_x)s;
        }
        PyObject *arglist = Py_BuildValue("OOOi",%(out)s, %(idx)s, %(y)s, %(inc_or_set)d);
        rval = inplace_increment(NULL, arglist);
        Py_XDECREF(arglist);
        if (rval == NULL) {
            %(fail)s;
        }
        Py_XDECREF(rval);
        """ % locals()

    def c_code_cache_version(self):
        return (3,)

    def perform(self, node, inp, out_):
        # TODO opt to make this inplace
        x, y, idx = inp
        out, = out_
        if not self.inplace:
            x = x.copy()
        # In Numpy, x[idx] += y doesn't work if the same index is present
        # many times: it does it only once. Is it a bug? In any case, for
        # this reason we implement our own 'inc' iteration.

        if self.set_instead_of_inc:
            x[idx] = y
        else:
            if config.cxx:
                increment = inplace_increment
            else:
                increment = self.inplace_increment1d_slow

            increment(x, idx, y)

        out[0] = x

    def inplace_increment1d_slow(self, x, idx, y):
        # If `y` has as many dimensions as `x`, then we want to iterate
        # jointly on `x` and `y`. Otherwise, it means `y` should be
        # broadcasted to fill all relevant rows of `x`.
        assert y.ndim <= x.ndim   # Should be guaranteed by `make_node`
        if y.ndim == x.ndim:
            if len(y) == 1:
                # Allow broadcasting of y[0]
                y_0 = y[0]
                for i in idx:
                    x[i] += y_0
            else:
                assert len(y) == len(idx)
                j = 0
                for i in idx:
                    x[i] += y[j]
                    j += 1
        else:
            for i in idx:
                x[i] += y

    def infer_shape(self, node, ishapes):
        x, y, ilist = ishapes
        return [x]

    def R_op(self, inputs, eval_points):
        if None in eval_points[:2]:
            return [None]
        return self.make_node(eval_points[0], eval_points[1],
                              *inputs[2:]).outputs

    def connection_pattern(self, node):

        rval = [[True], [True], [False]]
        return rval

    def grad(self, inputs, grads):
        g_output, = grads
        x, y, idx_list = inputs
        if x.dtype in theano.tensor.discrete_dtypes:
            # The output dtype is the same as x
            gx = x.zeros_like(dtype=theano.config.floatX)
            if y.dtype in theano.tensor.discrete_dtypes:
                gy = y.zeros_like(dtype=theano.config.floatX)
            else:
                gy = y.zeros_like()
        elif x.dtype in theano.tensor.complex_dtypes:
            raise NotImplementedError("No support for complex grad yet")
        else:
            if self.set_instead_of_inc:
                gx = advanced_set_subtensor1(
                    g_output,
                    y.zeros_like(),
                    idx_list)
            else:
                gx = g_output
            gy = advanced_subtensor1(g_output, idx_list)
            gy = _sum_grad_over_bcasted_dims(y, gy)

        return [gx, gy] + [DisconnectedType()()]

advanced_inc_subtensor1 = AdvancedIncSubtensor1()
advanced_set_subtensor1 = AdvancedIncSubtensor1(set_instead_of_inc=True)


def as_index_variable(idx):
    if idx is None:
        return NoneConst.clone()
    if isinstance(idx, slice):
        return make_slice(idx)
    if isinstance(idx, gof.Variable) and isinstance(idx.type, SliceType):
        return idx
    if isinstance(idx, gof.Variable) and isinstance(idx.type, NoneTypeT):
        return idx
    idx = theano.tensor.as_tensor_variable(idx)
    if idx.type.dtype not in theano.tensor.integer_dtypes:
        raise TypeError('index must be integers')
    return idx


def adv_index_broadcastable_pattern(a, idx):
    """
    This function is only used to determine the broadcast pattern for
    AdvancedSubtensor output variable.

    For this, we make a fake ndarray and a fake idx and call use ask numpy
    the output. From this, we find the output broadcast pattern.

    """

    def replace_slice(v):
        if isinstance(v, gof.Apply):
            if len(v.outputs) != 1:
                raise ValueError(
                    "It is ambiguous which output of a multi-output Op has"
                    " to be fetched.", v)
            else:
                v = v.outputs[0]

        if NoneConst.equals(v):
            return None
        if isinstance(v.type, SliceType):
            return slice(None, None)

        return numpy.zeros((2,) * v.ndim, int)

    newidx = tuple(map(replace_slice, idx))

    # 2 - True = 1; 2 - False = 2
    fakeshape = [2 - bc for bc in a.broadcastable]
    retshape = numpy.empty(fakeshape)[newidx].shape
    return tuple([dim == 1 for dim in retshape])


class AdvancedSubtensor(Op):
    """
    Return a subtensor copy, using advanced indexing.

    """

    # Should be used by __getitem__ and __getslice__, as follow:
    # AdvancedSubtensor()(self, *args),
    # if args contains and advanced indexing pattern
    __props__ = ()

    def make_node(self, x, *index):
        x = theano.tensor.as_tensor_variable(x)

        index = tuple(map(as_index_variable, index))
        bcast = adv_index_broadcastable_pattern(x, index)
        return gof.Apply(self,
                         (x,) + index,
                         [theano.tensor.tensor(dtype=x.type.dtype,
                                               broadcastable=bcast)])

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self.make_node(eval_points[0], *inputs[1:]).outputs

    def infer_shape(self, node, ishapes):
        # Really special case
        if len(ishapes) == 3:
            xshp, ind1shp, ind2shp = ishapes
            if (len(xshp) == 2 and
                    ind1shp is not None and len(ind1shp) == 1 and
                    ind2shp is not None and len(ind2shp) == 1):
                # if the graph is correct, we can assume ind1shp[0] and
                # ind2shp[0] will have the same value.
                # Try to return the one closest to the graph input.
                if node.inputs[2].owner is None:
                    return [ind2shp]
                else:
                    return [ind1shp]
        # Default case, we don't know
        raise theano.tensor.basic.ShapeError("case not implemented")

    def perform(self, node, inputs, out_):
        out, = out_
        # TODO: in general, we need to re-pack the inputs into a valid
        # index, just like subtensor
        out[0] = inputs[0].__getitem__(inputs[1:])

    def connection_pattern(self, node):
        rval = [[True]]

        for ipt in node.inputs[1:]:
            rval.append([False])

        return rval

    def grad(self, inputs, grads):
        gz, = grads
        x = inputs[0]
        rest = inputs[1:]
        return [advanced_inc_subtensor(theano.tensor.zeros_like(x), gz,
                                       *rest)] + \
            [DisconnectedType()()] * len(rest)
advanced_subtensor = AdvancedSubtensor()


class AdvancedIncSubtensor(Op):
    """
    Increments a subtensor using advanced indexing.
    """

    __props__ = ("inplace", "set_instead_of_inc")

    def __init__(self, inplace=False, set_instead_of_inc=False):
        self.inplace = inplace
        self.set_instead_of_inc = set_instead_of_inc
        # The assert is needed as in the pass the first argument was
        # something else that was not used.
        assert isinstance(inplace, bool)
        if self.inplace:
            raise NotImplementedError('In place computation is not'
                                      ' implemented')

    def __str__(self):
        return "%s{%s, %s}" % (self.__class__.__name__,
                               "inplace=" + str(self.inplace),
                               " set_instead_of_inc=" +
                               str(self. set_instead_of_inc))

    def make_node(self, x, y, *inputs):
        x = theano.tensor.as_tensor_variable(x)
        y = theano.tensor.as_tensor_variable(y)

        new_inputs = []
        for inp in inputs:
            if isinstance(inp, (list, tuple)):
                inp = theano.tensor.as_tensor_variable(inp)
            new_inputs.append(inp)
        return gof.Apply(self,
                         (x, y) + tuple(new_inputs),
                         [theano.tensor.tensor(
                             dtype=x.type.dtype,
                             broadcastable=x.type.broadcastable)])

    def perform(self, node, inputs, out_):
        # TODO: 1. opt to make this in place 2. generalize as described in
        # AdvancedSubtensor's perform TODO

        out, = out_
        if not self.inplace:
            out[0] = inputs[0].copy()
        else:
            out[0] = inputs[0]

        if self.set_instead_of_inc:
            out[0][inputs[2:]] = inputs[1]
        elif config.cxx:
            inplace_increment(out[0], tuple(inputs[2:]), inputs[1])
        else:
            raise NotImplementedError(
                'Could not import inplace_increment, so advanced '
                'indexing is disabled. '
                'Please make sure that you have a working C++ compiler '
                'and that config.cxx is correctly set.')

    def infer_shape(self, node, ishapes):
        return [ishapes[0]]

    def connection_pattern(self, node):

        rval = [[True], [True]]

        for ipt in node.inputs[2:]:
            rval.append([False])

        return rval

    def grad(self, inpt, output_gradients):
        x, y = inpt[:2]
        idxs = inpt[2:]
        outgrad, = output_gradients
        if x.dtype in theano.tensor.discrete_dtypes:
            # The output dtype is the same as x
            gx = x.zeros_like(dtype=theano.config.floatX)
            if y.dtype in theano.tensor.discrete_dtypes:
                gy = y.zeros_like(dtype=theano.config.floatX)
            else:
                gy = y.zeros_like()
        elif x.dtype in theano.tensor.complex_dtypes:
            raise NotImplementedError("No support for complex grad yet")
        else:
            if self.set_instead_of_inc:
                gx = advanced_set_subtensor(
                    outgrad,
                    y.zeros_like(),
                    *idxs)
            else:
                gx = outgrad
            gy = advanced_subtensor(outgrad, *idxs)
            # Make sure to sum gy over the dimensions of y that have been
            # added or broadcasted
            gy = _sum_grad_over_bcasted_dims(y, gy)
        return [gx, gy] + \
            [DisconnectedType()() for _ in idxs]

    def R_op(self, inputs, eval_points):
        if None in eval_points[:2]:
            return [None]
        return self.make_node(eval_points[0], eval_points[1],
                              *inputs[2:]).outputs
advanced_inc_subtensor = AdvancedIncSubtensor()
advanced_set_subtensor = AdvancedIncSubtensor(set_instead_of_inc=True)


def take(a, indices, axis=None, mode='raise'):
    a = theano.tensor.as_tensor_variable(a)
    indices = theano.tensor.as_tensor_variable(indices)
    # Reuse advanced_subtensor1 if indices is a vector
    if indices.ndim == 1:
        if mode == 'clip':
            indices = clip(indices, 0, a.shape[axis] - 1)
        elif mode == 'wrap':
            indices = indices % a.shape[axis]
        if axis is None:
            return advanced_subtensor1(a.flatten(), indices)
        elif axis == 0:
            return advanced_subtensor1(a, indices)
        else:
            if axis < 0:
                axis += a.ndim
            assert axis >= 0
            shuffle = list(range(a.ndim))
            shuffle[0] = axis
            shuffle[axis] = 0
            return advanced_subtensor1(
                a.dimshuffle(shuffle), indices).dimshuffle(shuffle)
    if axis is None:
        shape = indices.shape
        ndim = indices.ndim
    else:
        # If axis is 0, don't generate a useless concatenation.
        if axis == 0:
            shape = theano.tensor.concatenate(
                [indices.shape, a.shape[axis + 1:]])
        else:
            if axis < 0:
                axis += a.ndim
            shape = theano.tensor.concatenate(
                [a.shape[:axis], indices.shape, a.shape[axis + 1:]])
        ndim = a.ndim + indices.ndim - 1
    return take(a, indices.flatten(), axis, mode).reshape(shape, ndim)
