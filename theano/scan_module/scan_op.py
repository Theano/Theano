"""
This module provides the Scan Op.

See scan.py for details on scan.


Memory reuse in scan
--------------------

To reduce the number of memory allocations and copies associated with calling
the inner function and recovering the outputs at every iteration, Scan uses a
memory pre-allocation mechanism for some of its outputs. Instead of repeatedly
calling the inner function and copying the outputs to designated locations,
it tries to make the inner function write the outputs directly to the
designated locations.

This is achieved by initializing, at every iteration, the output storage
of the inner function with references to previously allocated memory. Other
than the code in the Python and Cython backends to do this and to ensure that
the pre-allocated memory has been used, the memory pre-allocation mechanism
relies on the following elements to work properly :
- In make_thunk(), when compiling the inner function, the borrow flag must
  be set to False for the inputs. This will prevent aliasing between the
  inputs and the outputs of the inner function which could lead to invalid
  results.
- In make_thunk(), again, the borrow flag must be set to True for the outputs.
  This will make Theano consider the output storages as persistent and make
  Theano provide them as pre-allocated storage to the ops that compute the
  outputs of the inner function instead of letting these ops allocate their
  own output storage.
- The ops that produce the outputs of the inner function must be prevented
  from working inplace because if they do, they're not using the pre-allocated
  storage. This is achieved by including the optimization
  'add_no_output_from_inplace' to the compilation mode used by scan. It
  prevents other optimizations from altering the graph such that outputs are
  produced by inplace operations.
- The ScanSaveMem optimization, whose goal is to limit the amount of memory
  used by scan, needs to allocate buffers large enough to be able, at every
  iteration, to simultaneously read the needed previous states and storing
  the new states. Before the memory reuse feature, the buffers could be
  smaller because, often, Scan only needed buffers large enough to read the
  needed previous states. This is because all the outputs of the inner
  function were computed before any of them was stored in the buffers. Now,
  the outputs are stored as they are computed which means that, if the buffer
  is too small, computing an output can overwrite an input that is still
  needed to compute another output.

"""
from __future__ import absolute_import, print_function, division

__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Frederic Bastien "
               "James Bergstra "
               "Pascal Lamblin ")
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import copy
import itertools
import logging
import time

import numpy
from six import iteritems, integer_types
from six.moves import xrange

import theano
from theano.compat import exc_message
from theano.compile import function, In, Out
from theano.compile.mode import AddFeatureOptimizer
from theano import compile, config, gradient, gof, tensor
from theano.gof import PureOp, Apply
from theano.gof.graph import io_connection_pattern
from theano.gof.toolbox import NoOutputFromInplace
from theano.compat import OrderedDict, izip
from theano.tensor import TensorType
from theano.tensor.opt import Shape_i
from theano.gradient import grad_undefined, DisconnectedType, NullType
from six import string_types
from theano.compile.profiling import ScanProfileStats

from theano.scan_module import scan_utils
from theano.scan_module.scan_utils import safe_new, forced_replace

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_module.scan_op')


class Scan(PureOp):
    """

    Parameters
    ----------
    inputs
        Inputs of the inner function of scan.
    outputs
        Outputs of the inner function of scan.
    info
        Dictionary containing different properties of the scan op (like number
        of different types of arguments, name, mode, if it should run on GPU or
        not, etc.).
    typeConstructor
        Function that constructs an equivalent to Theano TensorType.

    Notes
    -----
    ``typeConstructor`` had been added to refactor how
    Theano deals with the GPU. If it runs on the GPU, scan needs
    to construct certain outputs (those who reside in the GPU
    memory) as the GPU-specific type.  However we can not import
    gpu code in this file (as it is in sandbox, and not available
    on each machine) so the workaround is that the GPU
    optimization passes to the constructor of this class a
    function that is able to construct a GPU type. This way the
    class Scan does not need to be aware of the details for the
    GPU, it just constructs any tensor using this function (which
    by default constructs normal tensors).

    """

    def __init__(self,
                 inputs,
                 outputs,
                 info,
                 typeConstructor=None,
                ):
        if 'gpua' not in info:
            info['gpua'] = False
        # adding properties into self
        self.inputs = inputs
        self.outputs = outputs
        self.__dict__.update(info)
        # I keep a version of info in self, to use in __eq__ and __hash__,
        # since info contains all tunable parameters of the op, so for two
        # scan to be equal this tunable parameters should be the same
        self.info = info
        # build a list of output types for any Apply node using this op.
        self.output_types = []
        idx = 0
        jdx = 0
        tensorConstructor = lambda broadcastable, dtype: TensorType(
            broadcastable=broadcastable, dtype=dtype)
        if typeConstructor is None:
            typeConstructor = tensorConstructor

        while idx < self.n_mit_mot_outs:
            # Not that for mit_mot there are several output slices per
            # output sequence
            o = outputs[idx]
            self.output_types.append(
                typeConstructor(
                    broadcastable=(False,) + o.type.broadcastable,
                    dtype=o.type.dtype))

            idx += len(self.mit_mot_out_slices[jdx])
            jdx += 1

        # mit_sot / sit_sot / nit_sot
        end = idx + self.n_mit_sot + self.n_sit_sot + self.n_nit_sot

        for o in outputs[idx:end]:
            self.output_types.append(
                typeConstructor(
                    broadcastable=(False,) + o.type.broadcastable,
                    dtype=o.type.dtype))

        # shared outputs + possibly the ending condition
        for o in outputs[end:]:
            self.output_types.append(o.type)

        if self.as_while:
            self.output_types = self.output_types[:-1]

        mode_instance = compile.mode.get_mode(self.mode)
        # Clone mode_instance, altering "allow_gc" for the linker,
        # and adding a message if the mode is a ProfileMode.
        if self.name:
            message = self.name + " sub profile"
        else:
            message = "Scan sub profile"

        self.mode_instance = mode_instance.clone(
            link_kwargs=dict(allow_gc=self.allow_gc),
            message=message)

        if not hasattr(self, 'name') or self.name is None:
            self.name = 'scan_fn'
        # to have a fair __eq__ comparison later on, we update the info with
        # the actual mode used to compile the function and the name of the
        # function that we set in case none was given
        self.info['name'] = self.name

        # Pre-computing some values to speed up perform
        self.mintaps = [numpy.min(x) for x in self.tap_array]
        self.mintaps += [0 for x in xrange(self.n_nit_sot)]
        self.seqs_arg_offset = 1 + self.n_seqs
        self.shared_arg_offset = (self.seqs_arg_offset +
                                  self.n_mit_mot +
                                  self.n_mit_sot +
                                  self.n_sit_sot)
        self.nit_sot_arg_offset = (self.shared_arg_offset +
                                   self.n_shared_outs)
        self.n_outs = self.n_mit_mot + self.n_mit_sot + self.n_sit_sot
        self.n_tap_outs = self.n_mit_mot + self.n_mit_sot
        if self.info['gpu'] or self.info['gpua']:
            self._hash_inner_graph = self.info['gpu_hash']
        else:
            tmp_in, tmp_out = scan_utils.reconstruct_graph(self.inputs,
                                                           self.outputs)
            local_fgraph = gof.FunctionGraph(tmp_in, tmp_out, clone=False)
            self._cmodule_key = gof.CLinker().cmodule_key_variables(self.inputs,
                                                                    self.outputs,
                                                                    [])
            self._hash_inner_graph = hash(self._cmodule_key)

        # Compute mappings between outer inputs, outer outputs, inner
        # inputs and inner outputs to determine with variables are associated
        # with the same states.
        self.var_mappings = self.get_oinp_iinp_iout_oout_mappings()

    def validate_inner_graph(self):
        """
        Perform some elementary validations on the inner graph to ensure
        that it is coherent.

        """

        # For every recurrent output, iterate over the associated inner
        # inputs and output and ensure that they have the same dtype
        nb_recurr_outputs = self.n_mit_mot + self.n_mit_sot + self.n_sit_sot

        for outer_oidx in xrange(nb_recurr_outputs):

            inner_iidxs = self.var_mappings['inner_inp_from_outer_out'][outer_oidx]
            inner_oidxs = self.var_mappings['inner_out_from_outer_out'][outer_oidx]

            for (inner_iidx, inner_oidx) in itertools.product(inner_iidxs,
                                                              inner_oidxs):

                type_input = self.inputs[inner_iidx].type
                type_output = self.outputs[inner_oidx].type
                if (type_input != type_output):
                    raise TypeError("Inconsistency in the inner graph of "
                                    "scan '%s' : an input and an output are "
                                    "associated with the same recurrent state "
                                    "and should have the same type but have "
                                    "type '%s' and '%s' respectively." %
                                    (self.name, type_input, type_output))

        # If scan has the flag 'gpu' set to false (meaning that is shouldn't
        # use the CUDA gpu backend ), ensure that is has no input and no
        # output with type CudaNdarrayType
        from theano.sandbox.cuda import CudaNdarrayType
        if not self.info.get("gpu", False):
            for inp in self.inputs:
                if isinstance(inp.type, CudaNdarrayType):
                    raise TypeError("Inconsistency in the inner graph of "
                                    "scan '%s' : one of the inputs to the "
                                    "inner graph is of type CudaNdarray but "
                                    "the attributes of the scan op indicate "
                                    "that it shouldn't be the case")

            for out in self.outputs:
                if isinstance(out.type, CudaNdarrayType):
                    raise TypeError("Inconsistency in the inner graph of "
                                    "scan '%s' : one of the outputs to the "
                                    "inner graph is of type CudaNdarray but "
                                    "the attributes of the scan op indicate "
                                    "that it shouldn't be the case")

        # If scan has the flag 'gpua' set to false (meaning that is shouldn't
        # use the gpuarray gpu backend ), ensure that is has no input and no
        # output with type GpuArrayType
        from theano.sandbox.gpuarray import GpuArrayType
        if not self.info.get("gpua", False):
            for inp in self.inputs:
                if isinstance(inp.type, GpuArrayType):
                    raise TypeError("Inconsistency in the inner graph of "
                                    "scan '%s' : one of the inputs to the "
                                    "inner graph is of type GpuArrayType but "
                                    "the attributes of the scan op indicate "
                                    "that it shouldn't be the case")

            for out in self.outputs:
                if isinstance(out.type, GpuArrayType):
                    raise TypeError("Inconsistency in the inner graph of "
                                    "scan '%s' : one of the outputs to the "
                                    "inner graph is of type GpuArrayType but "
                                    "the attributes of the scan op indicate "
                                    "that it shouldn't be the case")

    def __setstate__(self, d):
        self.__dict__.update(d)
        if "allow_gc" not in self.__dict__:
            self.allow_gc = True
            self.info['allow_gc'] = True
        if not hasattr(self, 'gpua'):
            self.gpua = False
            self.info['gpua'] = False
        if not hasattr(self, 'var_mappings'):
            # Generate the mappings between inner and outer inputs and outputs
            # if they haven't already been generated.
            self.var_mappings = self.get_oinp_iinp_iout_oout_mappings()
        if hasattr(self, 'fn'):
            if not hasattr(self, 'thunk_mit_mot_out_slices'):
                # The thunk has been compiled before mit_mot preallocation
                # feature was implemented. Mark every mit_mot output tap as
                # not having been preallocated
                self.mitmots_preallocated = [False] * self.n_mit_mot_outs

            if not hasattr(self, 'outs_is_tensor'):
                # The thunk has been compiled before the analysis, at
                # compilation time, of the location of the inputs and outputs.
                # Perform this analysis here.
                self.inps_is_tensor = [isinstance(out, theano.tensor.TensorVariable)
                                       for out in self.fn.maker.fgraph.inputs]
                self.outs_is_tensor = [isinstance(out, theano.tensor.TensorVariable)
                                       for out in self.fn.maker.fgraph.outputs]

        # Ensure that the graph associated with the inner function is valid.
        self.validate_inner_graph()

    def make_node(self, *inputs):
        """
        Conventions:
            inner_X - the variable corresponding to X in the inner function
                      of scan (the lambda function executed at every time
                      step)
            outer_X - the variable corresponding to X in the outer graph,
                      i.e. the main graph (where the scan op lives)
            inner_X_out - the variable representing the new value of X after
                          executing one step of scan (i.e. outputs given by
                          the inner function)

        """
        assert numpy.all(isinstance(i, gof.Variable) for i in inputs)
        # Check that the number of inputs to the Scan node corresponds to
        # the number of inputs of the inner function of scan
        n_outer_ins = len(inputs) - len(self.outer_nitsot(inputs)) - 1
        n_inner_ins = (len(self.inner_seqs(self.inputs)) +
                       len(self.mitmot_taps()) +
                       len(self.mitsot_taps()) +
                       len(self.inner_sitsot(self.inputs)) +
                       len(self.inner_shared(self.inputs)) +
                       len(self.inner_non_seqs(self.inputs)))
        assert n_outer_ins == n_inner_ins, \
                ("The number of inputs given to the inner function of scan"
                 " does not match the number of inputs given to scan.")
        new_inputs = [inputs[0]]
        # assert dtype is consistent
        err_msg1 = ('When compiling the inner function of scan (the '
                    'function called by scan in each of its iterations) '
                    'the following error has been encountered: The '
                    '%s %s (argument number %d) has dtype '
                    '%s and %d dimension(s). The corresponding variable '
                    'in the inner function of scan %s '
                    'however has dtype %s and %d dimension(s). This '
                    'variable in the inner function of scan should '
                    'have the same dtype and one fewer dimension '
                    'compared to its corresponding variable in the initial '
                    'state (outputs_info in scan nomenclature). For example, '
                    'if the inner function of scan returns a vector '
                    'of size d and scan uses the values of '
                    'the previous time-step, then the initial state in scan '
                    'should be a matrix of shape (1, d). '
                    'The first dimension of this '
                    'matrix corresponds to the number of previous time-steps '
                    'that scan uses in each of its iterations. '
                    'In order to solve this issue if the two variable currently '
                    'have the same dimensionality, you can increase the '
                    'dimensionality of the varialbe in the initial state of scan '
                    'by using dimshuffle or shape_padleft. '
                   )
        err_msg2 = ('When compiling the inner function of scan the '
                    'following error has been encountered: The '
                    'initial state (`outputs_info` in scan nomenclature) '
                    'of variable %s (argument number %d) '
                    'has dtype %s, while the result of the inner function '
                    '(`fn`) has dtype %s. This can happen if the inner '
                    'function of scan results in an upcast or downcast.')
        err_msg3 = ('When compiling the inner function of scan (the '
                    'function called by scan in each of its iterations) '
                    'the following error has been encountered: The '
                    'initial state (`outputs_info` in scan nomenclature) '
                    'of variable %s (argument number %d) has %d dimension(s), '
                    'while the corresponding variable in the result of the inner '
                    'function of scan (`fn`) has %d dimension(s) (it should '
                    'be one less than the initial state). For example, '
                    'if the inner function of scan returns a vector '
                    'of size d and scan uses the values of '
                    'the previous time-step, then the initial state in scan '
                    'should be a matrix of shape (1, d). '
                    'The first dimension of this '
                    'matrix corresponds to the number of previous time-steps '
                    'that scan uses in each of its iterations. '
                    'In order to solve this issue if the two varialbe currently '
                    'have the same dimensionality, you can increase the '
                    'dimensionality of the variable in the initial state of scan '
                    'by using dimshuffle or shape_padleft. '
                   )

        def format(var, as_var):
            """
            This functions ensures that ``out`` has the same dtype as
            ``inp`` as well as calling filter_variable to make sure they are
            both TensorType or CudaNdarrayType. It internally deals with the
            corner case where inp.ndim + 1 = out.ndim

            """
            if not hasattr(var, 'dtype'):
                return var
            rval = var
            if rval.type.dtype != as_var.type.dtype:
                rval = rval.astype(as_var.type.dtype)
            if rval.ndim == as_var.ndim:
                rval = as_var.type.filter_variable(rval)
            else:
                tmp = as_var.type.clone(
                    broadcastable=(tuple(var.broadcastable[:1]) +
                                   tuple(as_var.broadcastable)))
                rval = tmp.filter_variable(rval)
            return rval

        # Check if input sequences and variables representing a slice of
        # them have the same dtype
        argoffset = 0
        for inner_seq, outer_seq in zip(self.inner_seqs(self.inputs),
                                        self.outer_seqs(inputs)):
            new_inputs.append(format(outer_seq, as_var=inner_seq))

        argoffset += len(self.outer_seqs(inputs))
        # Check that this 3 things have the same dtype for mit_mot:
        #   - initial state of the output
        #   - variable representing an input slice of the otuput
        #   - variable representing an output slice of the otuput
        ipos = 0
        opos = 0
        inner_mitmot = self.inner_mitmot(self.inputs)
        inner_mitmot_outs = self.inner_mitmot_outs(self.outputs)
        for idx, (itaps, otaps, _outer_mitmot) in enumerate(
                                     zip(self.mitmot_taps(),
                                         self.mitmot_out_taps(),
                                         self.outer_mitmot(inputs))):
            outer_mitmot = format(_outer_mitmot, as_var=inner_mitmot[ipos])
            new_inputs.append(outer_mitmot)
            for k in xrange(len(itaps)):
                if (inner_mitmot[ipos + k].type.dtype !=
                    outer_mitmot.type.dtype or
                    inner_mitmot[ipos + k].ndim != outer_mitmot.ndim - 1):
                    raise ValueError(err_msg1 % ('initial state (outputs_info'
                                           ' in scan nomenclature) ',
                                           str(outer_mitmot),
                                           argoffset + idx,
                                           outer_mitmot.type.dtype,
                                           outer_mitmot.type.ndim,
                                           str(inner_mitmot[ipos + k]),
                                           inner_mitmot[ipos +
                                                        k].type.dtype,
                                           inner_mitmot[ipos + k].type.ndim))
            ipos += len(itaps)
            for k in xrange(len(otaps)):
                if (inner_mitmot_outs[opos + k].type.dtype !=
                        outer_mitmot.type.dtype):
                    raise ValueError(err_msg2 %
                                     (str(outer_mitmot),
                                      argoffset + idx,
                                      outer_mitmot.type.dtype,
                                      inner_mitmot_outs[opos + k].type.dtype))
                if inner_mitmot_outs[opos + k].ndim != outer_mitmot.ndim - 1:
                    raise ValueError(err_msg3 %
                                     (str(outer_mitmot),
                                      argoffset + idx,
                                      outer_mitmot.ndim,
                                      inner_mitmot_outs[opos + k].ndim))
            opos += len(otaps)
        argoffset += len(self.outer_mitmot(inputs))
        # Same checks as above but for outputs of type mit_sot
        ipos = 0
        inner_mitsots = self.inner_mitsot(self.inputs)
        for idx, (itaps, _outer_mitsot, inner_mitsot_out) in enumerate(
            zip(self.mitsot_taps(),
                self.outer_mitsot(inputs),
                self.inner_mitsot_outs(self.outputs))):
            outer_mitsot = format(_outer_mitsot, as_var=inner_mitsots[ipos])
            new_inputs.append(outer_mitsot)

            for k in xrange(len(itaps)):
                if (inner_mitsots[ipos + k].type.dtype != \
                    outer_mitsot.type.dtype or
                    inner_mitsots[ipos + k].ndim != outer_mitsot.ndim - 1):
                    raise ValueError(err_msg1 % ('initial state (outputs_info'
                                               ' in scan nomenclature) ',
                                           str(outer_mitsot),
                                           argoffset + idx,
                                           outer_mitsot.type.dtype,
                                           outer_mitsot.type.ndim,
                                           str(inner_mitsots[ipos + k]),
                                           inner_mitsots[ipos + k].type.dtype,
                                           inner_mitsots[ipos + k].type.ndim))
            ipos += len(itaps)
            if inner_mitsot_out.type.dtype != outer_mitsot.type.dtype:
                raise ValueError(err_msg2 %
                                 (str(outer_mitsot),
                                  argoffset + idx,
                                  outer_mitsot.type.dtype,
                                  inner_mitsot_out.type.dtype))
            if inner_mitsot_out.ndim != outer_mitsot.ndim - 1:
                raise ValueError(err_msg3 %
                                 (str(outer_mitsot),
                                  argoffset + idx,
                                  outer_mitsot.ndim,
                                  inner_mitsot_out.ndim))

        argoffset += len(self.outer_mitsot(inputs))
        # Same checks as above but for outputs of type sit_sot
        for idx, (inner_sitsot, _outer_sitsot, inner_sitsot_out) in enumerate(
            zip(self.inner_sitsot(self.inputs),
                self.outer_sitsot(inputs),
                self.inner_sitsot_outs(self.outputs))):
            outer_sitsot = format(_outer_sitsot, as_var=inner_sitsot)
            new_inputs.append(outer_sitsot)
            if (inner_sitsot.ndim != outer_sitsot.ndim - 1):
                raise ValueError(err_msg1 % ('initial state (outputs_info'
                                           ' in scan nomenclature) ',
                                str(outer_sitsot),
                                argoffset + idx,
                                outer_sitsot.type.dtype,
                                outer_sitsot.type.ndim,
                                str(inner_sitsot),
                                inner_sitsot.type.dtype,
                                inner_sitsot.type.ndim))
            if inner_sitsot_out.type.dtype != outer_sitsot.type.dtype:
                raise ValueError(err_msg2 %
                                 (str(outer_sitsot),
                                  argoffset + idx,
                                  outer_sitsot.type.dtype,
                                  inner_sitsot_out.type.dtype))
            if inner_sitsot_out.ndim != outer_sitsot.ndim - 1:
                raise ValueError(err_msg3 %
                                 (str(outer_sitsot),
                                  argoffset + idx,
                                  outer_sitsot.type.ndim,
                                  inner_sitsot_out.type.ndim))

        argoffset += len(self.outer_sitsot(inputs))
        # Check that the shared variable and their update rule have the same
        # dtype. Maybe even same type ?!
        for idx, (inner_shared, inner_shared_out, _outer_shared) in enumerate(
            zip(self.inner_shared(self.inputs),
                self.inner_shared_outs(self.outputs),
                self.outer_shared(inputs))):
            outer_shared = format(_outer_shared, as_var=inner_shared)
            new_inputs.append(outer_shared)
            if (hasattr(outer_shared, 'dtype') and
                    outer_shared.dtype != inner_shared_out.dtype):
                raise ValueError(err_msg2 % (str(outer_shared),
                                             idx + argoffset,
                                             outer_shared.dtype,
                                             inner_shared_out.dtype))
            if (hasattr(outer_shared, 'dtype') and
                    outer_shared.ndim != inner_shared_out.ndim):
                raise ValueError(err_msg3 % (str(outer_shared),
                                             idx + argoffset,
                                             outer_shared.ndim,
                                             inner_shared_out.ndim))

            if (hasattr(outer_shared, 'dtype') and
                (outer_shared.dtype != inner_shared.dtype or
                 outer_shared.ndim != inner_shared.ndim)):
                raise ValueError(err_msg1 % ('initial state (outputs_info'
                                           ' in scan nomenclature) ',
                                           str(outer_shared),
                                           argoffset + idx,
                                           outer_shared.dtype,
                                           outer_shared.ndim,
                                           str(inner_shared),
                                           inner_shared.dtype,
                                           inner_shared.ndim))
        # We do not need to call `format` on outer_nisot arguments.
        # outer_nitsot stands for no input tap single output tap. This means
        # these are states that do not feed anything back in the recurrent
        # computation, and hence they do not have an initial state. The scan
        # node however receives an input for each such argument, the input
        # in this case is just a int saying how many steps of this output we
        # need to store. This input does not have the same dtype, nor is it the same
        # type of tensor as the output, it is always a scalar int.
        new_inputs += self.outer_nitsot(inputs)
        for inner_nonseq, _outer_nonseq in zip(
                            self.inner_non_seqs(self.inputs),
                            self.outer_non_seqs(inputs)):
            outer_nonseq = format(_outer_nonseq, as_var=inner_nonseq)
            new_inputs.append(outer_nonseq)
            if inner_nonseq.type != outer_nonseq.type:
                raise ValueError(('Argument %s given to scan node does not'
                                 ' match its correspondance %s') %
                                  (str(outer_nonseq), str(inner_nonseq)))

        for outer_nitsot in self.outer_nitsot(inputs):
            # For every nit_sot input we get as input a int/uint that
            # depicts the size in memory for that sequence. This feature is
            # used by truncated BPTT and by scan space optimization
            if (str(outer_nitsot.type.dtype)[:3] not in ('uin', 'int') or
                outer_nitsot.ndim != 0):
                raise ValueError('For output %s you need to provide a '
                                 'scalar int !', str(outer_nitsot))
        assert len(new_inputs) == len(inputs)

        # The vector_seqs and vector_outs are just a workaround
        # strange NumPy behavior: vector_ndarray[int] return a NumPy
        # scalar and not a NumPy ndarray of 0 dimensions.
        self.vector_seqs = [isinstance(seq, (tensor.TensorVariable,
                                             tensor.TensorConstant)) and
                            seq.ndim == 1 for seq in
                            new_inputs[1:1 + self.n_seqs]]
        self.vector_outs = [isinstance(arg, (tensor.TensorVariable,
                                             tensor.TensorConstant)) and
                            arg.ndim == 1 for arg in
                            new_inputs[1 + self.n_seqs: (1 + self.n_seqs +
                                                         self.n_outs)]]
        self.vector_outs += [False] * self.n_nit_sot

        apply_node = Apply(self,
                           new_inputs,
                           [t() for t in self.output_types])
        return apply_node

    def __eq__(self, other):
        # Check if we are dealing with same type of objects
        if not type(self) == type(other):
            return False
        if not 'destroy_map' in self.info:
            self.info['destroy_map'] = OrderedDict()
        if not 'destroy_map' in other.info:
            other.info['destroy_map'] = OrderedDict()
        keys_to_check = ['truncate_gradient', 'profile',
                         'n_seqs', 'tap_array',
                         'as_while', 'n_mit_sot', 'destroy_map',
                         'n_nit_sot', 'n_shared_outs',
                         'n_sit_sot', 'gpu', 'gpua', 'n_mit_mot_outs',
                         'n_mit_mot', 'mit_mot_out_slices']
        # This are some safety checks ( namely that the inner graph has the
        # same number of inputs and same number of outputs )
        if not len(self.inputs) == len(other.inputs):
            return False
        elif not len(self.outputs) == len(other.outputs):
            return False
        for key in keys_to_check:
            if self.info[key] != other.info[key]:
                return False
        # If everything went OK up to here, there is still one thing to
        # check. Namely, do the internal graph represent same
        # computations
        for self_in, other_in in izip(self.inputs, other.inputs):
            if self_in.type != other_in.type:
                return False

        return scan_utils.equal_computations(self.outputs,
                                             other.outputs,
                                             self.inputs,
                                             other.inputs)

    def __str__(self):
        if self.gpu:
            gpu_str = 'gpu'
        else:
            gpu_str = 'cpu'
        if self.as_while:
            name = 'do_while'
        else:
            name = 'for'
        aux_txt = '%s'
        if getattr(self, 'destroy_map', None) is None:
            self.destroy_map = OrderedDict()
        if len(self.destroy_map.keys()) > 0:
            # Check if all outputs are inplace
            if (sorted(self.destroy_map.keys()) == \
               sorted(range(self.n_mit_mot +
                            self.n_mit_sot +
                            self.n_sit_sot))):
                aux_txt += 'all_inplace,%s,%s}'
            else:
                aux_txt += '{inplace{'
                for k in self.destroy_map.keys():
                    aux_txt += str(k) + ','
                aux_txt += '},%s,%s}'
        else:
            aux_txt += '{%s,%s}'
        aux_txt = aux_txt % (name, gpu_str, str(self.name))
        return aux_txt

    def __hash__(self):
        return hash((type(self),
                     # and a hash representing the inner graph using the
                     # CLinker.cmodule_key_
                     self._hash_inner_graph,
                     scan_utils.hash_listsDictsTuples(self.info)))

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        """

        Parameters
        ----------
        node
            Something previously returned by self.make_node.
        storage_map
            dict variable -> one-element-list where a computed
            value for this variable may be found.
        compute_map
            dict variable -> one-element-list where a boolean
            value will be found. The boolean indicates whether the
            variable's storage_map container contains a valid value (True)
            or if it has not been computed yet (False).
        no_recycling
            List of variables for which it is forbidden to reuse memory
            allocated by a previous call.

        Notes
        -----
        If the thunk consults the storage_map on every call, it is safe
        for it to ignore the no_recycling argument, because elements of the
        no_recycling list will have a value of None in the storage map. If
        the thunk can potentially cache return values (like CLinker does),
        then it must not do so for variables in the no_recycling list.

        """

        # Before building the thunk, validate that the inner graph is
        # coherent
        self.validate_inner_graph()

        # Setting up all my variables in what I believe is a more Cython
        # friendly form

        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]
        node_input_compute = [compute_map[r] for r in node.inputs]
        node_output_compute = [compute_map[r] for r in node.outputs]
        #_logger.debug('Compiling node %i of graph' % node_idx)
        # If a shared variable is the result of a ViewOp it is a clear
        # indication that we need to copy that value after the perform of
        # scan is done
        slices = (self.n_mit_mot_outs +
                  self.n_mit_sot +
                  self.n_sit_sot +
                  self.n_nit_sot)

        if theano.config.scan.allow_output_prealloc:

            # Go through the mitmots. Whenever a mitmot has a tap both as an
            # input and an output, wrap the input such that the corresponding
            # output variable becomes an update to be performed on it, possibly
            # inplace at the end of the functions's execution.
            wrapped_inputs = [In(x, borrow=False)
                              for x in self.inputs[:self.n_seqs]]
            new_outputs = [x for x in self.outputs]
            preallocated_mitmot_outs = []
            new_mit_mot_out_slices = copy.deepcopy(self.mit_mot_out_slices)

            input_idx = self.n_seqs
            for mitmot_idx in range(self.n_mit_mot):
                for inp_tap in self.tap_array[mitmot_idx]:
                    if inp_tap in self.mit_mot_out_slices[mitmot_idx]:
                        inp = self.inputs[input_idx]

                        # Figure out the index of the corresponding output
                        output_idx = sum([len(m) for m in
                                          self.mit_mot_out_slices[:mitmot_idx]])
                        output_idx += self.mit_mot_out_slices[mitmot_idx].index(inp_tap)

                        # Make it so the input is automatically updated to the
                        # output value, possibly inplace, at the end of the
                        # function exectution. Also, since an update is
                        # defined, a default value must also be (this is
                        # verified by DebugMode). Use an array of size 0 but
                        # the right ndim and dtype.
                        default_val = numpy.zeros([0] * inp.ndim,
                                                  dtype=inp.dtype)
                        wrapped_inp = In(variable=inp, value=default_val,
                                         update=self.outputs[output_idx])
                        wrapped_inputs.append(wrapped_inp)
                        preallocated_mitmot_outs.append(output_idx)
                        new_mit_mot_out_slices[mitmot_idx].remove(inp_tap)
                    else:
                        # Wrap the corresponding input as usual. Leave the
                        # output as-is.
                        wrapped_inputs.append(In(self.inputs[input_idx],
                                                    borrow=False))
                    input_idx += 1

            # Wrap the inputs not associated to mitmots and wrap the remaining
            # outputs
            wrapped_inputs += [In(x, borrow=False) for x in
                               self.inputs[input_idx:]]
            wrapped_outputs = [Out(x, borrow=True) for x in
                               new_outputs[:slices]]
            wrapped_outputs += new_outputs[slices:]

            # Remove now useless outputs from the output list (start from the
            # end to avoid altering the indices of the other outputs to be
            # deleted.
            preallocated_mitmot_outs.sort()
            for p in preallocated_mitmot_outs[::-1]:
                del wrapped_outputs[p]

            # Store the list of mitmot output taps that have been altered
            # so they can be preallocated
            self.mitmots_preallocated = [i in preallocated_mitmot_outs
                                         for i in range(self.n_mit_mot_outs)]

            # Add an optimization to the compilation mode to attach a feature
            # to the function graph just before the inplace optimizations are
            # applied (inplace optimizations start at position 50 so the
            # optimization to attach the feature is registered at position 49.9
            # so that it runs before them). This feature will prevent mitsot,
            # sitsot and nitsot outputs from being computed inplace (to allow
            # their preallocation).
            mitsot_start = self.n_mit_mot_outs - len(preallocated_mitmot_outs)
            nitsot_end = (mitsot_start + self.n_mit_sot + self.n_sit_sot +
                          self.n_nit_sot)
            feature = NoOutputFromInplace(mitsot_start, nitsot_end)
            opt = AddFeatureOptimizer(feature)
            compilation_mode = self.mode_instance.register((opt, 49.9))

        else:
            # Output preallocation is not activated. Mark every mitmot output
            # tap as not being preallocated
            self.mitmots_preallocated = [False] * self.n_mit_mot_outs

            wrapped_inputs = [In(x, borrow=True) for x in
                              self.inputs]
            wrapped_outputs = [Out(x, borrow=False) for x in
                               self.outputs[:slices]]
            wrapped_outputs += self.outputs[slices:]

            compilation_mode = self.mode_instance

        profile = None
        if (theano.config.profile or
            (isinstance(self.profile, (string_types, bool, integer_types))
                                      and self.profile)):
            if isinstance(self.profile, string_types):
                profile = ScanProfileStats(name=self.profile)
            else:
                profile = ScanProfileStats(name=self.name)
        elif self.profile:
            profile = self.profile
        # make_thunk can be called many times on the same op
        # we do not want to recompile the inner fct every time.
        if not getattr(self, 'fn', None):
            self.fn = function(wrapped_inputs,
                               wrapped_outputs,
                               mode=compilation_mode,
                               name=self.name,
                               profile=profile,
                               on_unused_input='ignore')

        # Analyse the compile inner function to determine which inputs and
        # outputs are on the gpu and speed up some checks during the execution
        self.inps_is_tensor = [isinstance(out, theano.tensor.TensorVariable)
                               for out in self.fn.maker.fgraph.inputs]
        self.outs_is_tensor = [isinstance(out, theano.tensor.TensorVariable)
                               for out in self.fn.maker.fgraph.outputs]

        try:
            cython_mintaps = numpy.asarray(self.mintaps, dtype='int32')
            cython_tap_array_len = \
                numpy.asarray([len(x) for x in self.tap_array],
                              dtype='int32')
            if len(self.tap_array) == 0:
                d1 = 0
            else:
                d1 = numpy.max(cython_tap_array_len)
            d0 = len(self.tap_array)
            cython_tap_array = numpy.zeros((d0, d1), dtype='int32')
            for _d0 in xrange(d0):
                for _d1 in xrange(cython_tap_array_len[_d0]):
                    cython_tap_array[_d0, _d1] = self.tap_array[_d0][_d1]
            cython_mit_mot_out_nslices = \
                numpy.asarray([len(x) for x in self.mit_mot_out_slices],
                              dtype='int32')
            if len(self.mit_mot_out_slices) == 0:
                d1 = 0
            else:
                d1 = numpy.max(cython_mit_mot_out_nslices)
            d0 = len(self.mit_mot_out_slices)
            cython_mit_mot_out_slices = numpy.zeros((d0, d1),
                                                      dtype='int32')
            for _d0 in xrange(d0):
                for _d1 in xrange(cython_mit_mot_out_nslices[_d0]):
                    cython_mit_mot_out_slices[_d0, _d1] = \
                        self.mit_mot_out_slices[_d0][_d1]

            cython_vector_seqs = numpy.asarray(self.vector_seqs,
                                                    dtype='int32')
            cython_vector_outs = numpy.asarray(self.vector_outs,
                                                    dtype='int32')
            cython_mitmots_preallocated = numpy.asarray(self.mitmots_preallocated,
                                                        dtype='int32')

            cython_inps_is_tensor = numpy.asarray(self.inps_is_tensor,
                                                  dtype='int32')
            cython_outs_is_tensor = numpy.asarray(self.outs_is_tensor,
                                                  dtype='int32')

            if hasattr(self, 'destroy_map'):
                cython_destroy_map = [x in self.destroy_map
                                  for x in xrange(len(node.outputs))]
            else:
                cython_destroy_map = [0 for x in xrange(len(node.outputs))]
            cython_destroy_map = numpy.asarray(cython_destroy_map,
                                               dtype='int32')
            from . import scan_perform_ext
            p = lambda node, args, outs:\
                    scan_perform_ext.perform(
                        self.n_shared_outs,
                        self.n_mit_mot_outs,
                        self.n_seqs,
                        self.n_mit_mot,
                        self.n_mit_sot,
                        self.n_sit_sot,
                        self.n_nit_sot,
                        args[0],
                        self.as_while,
                        cython_mintaps,
                        cython_tap_array,
                        cython_tap_array_len,
                        cython_vector_seqs,
                        cython_vector_outs,
                        cython_mit_mot_out_slices,
                        cython_mit_mot_out_nslices,
                        cython_mitmots_preallocated,
                        cython_inps_is_tensor,
                        cython_outs_is_tensor,
                        self.fn.fn,
                        self.fn,
                        cython_destroy_map,
                        args,
                        outs,
                        self, node)
        except (ImportError, theano.gof.cmodule.MissingGXX):
            p = self.execute
        # default arguments are stored in the closure of `rval`

        # Big ugly hack since we can't get the real value of allow_gc
        # for the englobing function.
        allow_gc = config.allow_gc and not self.allow_gc

        def rval(p=p, i=node_input_storage, o=node_output_storage, n=node,
                 allow_gc=allow_gc):
            r = p(n, [x[0] for x in i], o)
            for o in node.outputs:
                compute_map[o][0] = True
            if allow_gc:
                self.fn.free()
            return r
        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.perform = p
        rval.lazy = False
        return rval

    def inner_seqs(self, list_inputs):
        # Given the list of inner inputs this function grabs those
        # corresponding to sequences
        return list_inputs[:self.n_seqs]

    def outer_seqs(self, list_inputs):
        if isinstance(list_inputs, Apply):
            list_inputs = list_inputs.inputs
        # Given the list of outter inputs this function grabs those
        # corresponding to sequences
        return list_inputs[1:1 + self.n_seqs]

    def inner_mitmot(self, list_inputs):
        n_taps = sum(len(x) for x in self.tap_array[:self.n_mit_mot])
        return list_inputs[self.n_seqs: self.n_seqs + n_taps]

    def outer_mitmot(self, list_inputs):
        if isinstance(list_inputs, Apply):
            list_inputs = list_inputs.inputs
        return list_inputs[1 + self.n_seqs:1 + self.n_seqs + self.n_mit_mot]

    def inner_mitmot_outs(self, list_outputs):
        n_taps = sum(len(x) for x in self.mit_mot_out_slices)
        return list_outputs[:n_taps]

    def outer_mitmot_outs(self, list_outputs):
        if isinstance(list_outputs, Apply):
            list_outputs = list_outputs.ouputs
        return list_outputs[:self.n_mit_mot]

    def mitmot_taps(self):
        return self.tap_array[:self.n_mit_mot]

    def mitmot_out_taps(self):
        return self.mit_mot_out_slices[:self.n_mit_mot]

    def inner_mitsot(self, list_inputs):
        n_mitmot_taps = sum(len(x) for x in self.tap_array[:self.n_mit_mot])
        ntaps_upto_sit_sot = sum(len(x) for x in
                                  self.tap_array[:(self.n_mit_mot +
                                                   self.n_mit_sot)])
        return list_inputs[self.n_seqs + n_mitmot_taps:
                           self.n_seqs + ntaps_upto_sit_sot]

    def outer_mitsot(self, list_inputs):
        if isinstance(list_inputs, Apply):
            list_inputs = list_inputs.inputs
        offset = 1 + self.n_seqs + self.n_mit_mot
        return list_inputs[offset:offset + self.n_mit_sot]

    def inner_mitsot_outs(self, list_outputs):
        n_taps = sum(len(x) for x in self.mit_mot_out_slices)
        return list_outputs[n_taps:n_taps + self.n_mit_sot]

    def outer_mitsot_outs(self, list_outputs):
        if isinstance(list_outputs, Apply):
            list_outputs = list_outputs.outputs
        return list_outputs[self.n_mit_mot:
                            self.n_mit_mot + self.n_mit_sot]

    def mitsot_taps(self):
        return self.tap_array[self.n_mit_mot:
                              self.n_mit_mot + self.n_mit_sot]

    def inner_sitsot(self, list_inputs):
        n_taps_upto_sit_sot = sum(len(x) for x in
                                  self.tap_array[:(self.n_mit_mot +
                                                   self.n_mit_sot)])
        offset = self.n_seqs + n_taps_upto_sit_sot
        return list_inputs[offset:offset + self.n_sit_sot]

    def outer_sitsot(self, list_inputs):
        if isinstance(list_inputs, Apply):
            list_inputs = list_inputs.inputs
        offset = 1 + self.n_seqs + self.n_mit_mot + self.n_mit_sot
        return list_inputs[offset:offset + self.n_sit_sot]

    def inner_sitsot_outs(self, list_outputs):
        n_taps = sum(len(x) for x in self.mit_mot_out_slices)
        offset = self.n_mit_sot + n_taps
        return list_outputs[offset:offset + self.n_sit_sot]

    def outer_sitsot_outs(self, list_outputs):
        if isinstance(list_outputs, Apply):
            list_outputs = list_outputs.outputs
        offset = self.n_mit_mot + self.n_mit_sot
        return list_outputs[offset:offset + self.n_sit_sot]

    def outer_nitsot(self, list_inputs):
        if isinstance(list_inputs, Apply):
            list_inputs = list_inputs.inputs
        offset = (1 + self.n_seqs + self.n_mit_mot + self.n_mit_sot +
                  self.n_sit_sot + self.n_shared_outs)
        return list_inputs[offset:offset + self.n_nit_sot]

    def inner_nitsot_outs(self, list_outputs):
        n_taps = sum(len(x) for x in self.mit_mot_out_slices)
        offset = self.n_mit_sot + n_taps + self.n_sit_sot
        return list_outputs[offset:offset + self.n_nit_sot]

    def outer_nitsot_outs(self, list_outputs):
        if isinstance(list_outputs, Apply):
            list_outputs = list_outputs.outputs
        offset = (self.n_mit_mot + self.n_mit_sot + self.n_sit_sot)
        return list_outputs[offset:offset + self.n_nit_sot]

    def inner_shared(self, list_inputs):
        n_taps_upto_sit_sot = sum(len(x) for x in
                                  self.tap_array[:(self.n_mit_mot +
                                                   self.n_mit_sot)])
        offset = self.n_seqs + n_taps_upto_sit_sot + self.n_sit_sot
        return list_inputs[offset:offset + self.n_shared_outs]

    def outer_shared(self, list_inputs):
        if isinstance(list_inputs, Apply):
            list_inputs = list_inputs.inputs
        offset = (1 + self.n_seqs + self.n_mit_mot + self.n_mit_sot +
                  self.n_sit_sot)
        return list_inputs[offset:offset + self.n_shared_outs]

    def inner_shared_outs(self, list_outputs):
        n_taps = sum(len(x) for x in self.mit_mot_out_slices)
        offset = self.n_mit_sot + n_taps + self.n_sit_sot + self.n_nit_sot
        return list_outputs[offset:offset + self.n_shared_outs]

    def outer_shared_outs(self, list_outputs):
        if isinstance(list_outputs, Apply):
            list_outputs = list_outputs.outputs
        offset = (self.n_mit_mot + self.n_mit_sot + self.n_sit_sot +
                    self.n_nit_sot)
        return list_outputs[offset:offset + self.n_shared_outs]

    def inner_non_seqs(self, list_inputs):
        n_taps_upto_sit_sot = sum(len(x) for x in
                                  self.tap_array[:(self.n_mit_mot +
                                                   self.n_mit_sot)])
        offset = (self.n_seqs + n_taps_upto_sit_sot + self.n_sit_sot +
                  self.n_shared_outs)
        return list_inputs[offset:]

    def outer_non_seqs(self, list_inputs):
        if isinstance(list_inputs, Apply):
            list_inputs = list_inputs.inputs
        offset = (1 + self.n_seqs + self.n_mit_mot + self.n_mit_sot +
                  self.n_sit_sot + self.n_nit_sot + self.n_shared_outs)
        return list_inputs[offset:]

    def execute(self, node, args, outs):
        """
        The args are packed like this:

            n_steps

            X sequence inputs x_1, x_2, ... x_<self.n_seqs>

            Y initial states (u_1, u_2, ... u_<self.n_outs>) for our
            outputs. Each must have appropriate length (T_1, T_2, ..., T_Y).

            W other inputs w_1, w_2, ... w_W

        There are at least 1 + self.n_seqs + self.n_outs inputs, and the
        ones above this number are passed to the scanned function as
        non-sequential inputs.

        The outputs are more straightforward:

            Y sequence outputs y_1, y_2, ... y_<self.n_outs>

        """
        # 1. Unzip the number of steps and sequences. If number of steps is
        # negative flip sequences around, and make n_steps positive
        t0_call = time.time()
        t_fn = 0
        n_steps = args[0]
        seqs = []
        if n_steps < 0:
            # History, in the past, this was used for backward
            # scan. Now we reverse the inputs outside of scan.
            raise IndexError(
                "Scan was asked to run for negative number of step %d" %
                n_steps)
        elif n_steps == 0:
            raise NotImplementedError(
                "We didn't implemented yet the case where scan do 0 iteration")
        else:
            for idx, seq in enumerate(args[1:self.seqs_arg_offset]):
                if seq.shape[0] < n_steps:
                    raise ValueError(('Sequence is shorter then the required '
                                     'number of steps : (n_steps, seq, '
                                      'seq.shape):'), n_steps,
                                      node.inputs[1 + idx],
                                      seq.shape)
                seqs.append(seq)

        # 2. Allocate memory for the outputs. Construct the list:
        #       store_steps  -- map containting the length of each output
        #       pos          -- map containing the current position of each
        #                       output

        store_steps = [arg.shape[0] for arg
                               in args[self.seqs_arg_offset:
                                       self.shared_arg_offset]]
        store_steps += [arg for arg in
                            args[self.nit_sot_arg_offset:
                                   self.nit_sot_arg_offset + self.n_nit_sot]
                       ]

        pos = [(-self.mintaps[idx]) % store_steps[idx] for idx
                         in xrange(self.n_outs + self.n_nit_sot)]
        if not getattr(self, 'destroy_map', None):
            self.destroy_map = OrderedDict()
        # 2.1 Create storage space for outputs
        for idx in xrange(self.n_outs):
            if idx in self.destroy_map:
                # ^ Case 1. Outputs should be computed inplace of their
                # initial state
                outs[idx][0] = args[self.seqs_arg_offset + idx]
            elif (outs[idx][0] is not None and
                  outs[idx][0].shape[1:] == args[self.seqs_arg_offset +
                                                 idx].shape[1:]
                  and outs[idx][0].shape[0] >= store_steps[idx]):
                # Put in the values of the initial state
                outs[idx][0] = outs[idx][0][:store_steps[idx]]
                if idx > self.n_mit_mot:
                    l = - self.mintaps[idx]
                    outs[idx][0][:l] = args[self.seqs_arg_offset + idx][:l]
                else:
                    outs[idx][0][:] = args[self.seqs_arg_offset + idx]
            else:
                outs[idx][0] = args[self.seqs_arg_offset + idx].copy()

        offset = self.nit_sot_arg_offset + self.n_nit_sot
        other_args = args[offset:]
        input_storage = self.fn.input_storage
        nb_mitmot_in = sum(map(len, self.tap_array[:self.n_mit_mot]))
        old_mitmot_input_storage = [None] * nb_mitmot_in
        old_mitmot_input_data = [None] * nb_mitmot_in
        output_storage = self.fn.output_storage
        old_output_storage = [None] * len(output_storage)
        old_output_data = [None] * len(output_storage)
        fn = self.fn.fn
        offset = (self.n_seqs + sum(map(len, self.tap_array[:self.n_outs])) +
                    self.n_shared_outs)
        for idx in xrange(len(other_args)):
            input_storage[idx + offset].storage[0] = other_args[idx]

        i = 0
        cond = True
        ############## THE MAIN LOOP #########################
        # for i in xrange(n_steps):
        while (i < n_steps) and cond:
            # sequences over which scan iterates
            # 3. collect input slices
            for idx in xrange(self.n_seqs):
                if self.vector_seqs[idx]:
                    input_storage[idx].storage[0] = \
                            seqs[idx][i:i + 1].reshape(())
                else:
                    input_storage[idx].storage[0] = seqs[idx][i]

            offset = self.n_seqs
            for idx in xrange(self.n_outs):
                if self.vector_outs[idx]:
                    for tap in self.tap_array[idx]:
                        _idx = (pos[idx] + tap) % store_steps[idx]
                        input_storage[offset].storage[0] =\
                                outs[idx][0][_idx:_idx + 1].reshape(())
                        offset += 1
                else:
                    for tap in self.tap_array[idx]:
                        _idx = (pos[idx] + tap) % store_steps[idx]
                        input_storage[offset].storage[0] = outs[idx][0][_idx]
                        offset += 1

            a_offset = self.shared_arg_offset
            o_offset = self.n_outs + self.n_nit_sot
            if i == 0:
                for j in xrange(self.n_shared_outs):
                    input_storage[offset].storage[0] = args[a_offset + j]
                    offset += 1
            else:
                for j in xrange(self.n_shared_outs):
                    input_storage[offset].storage[0] = outs[o_offset + j][0]
                    offset += 1

            # 4. collecting slices where the output should be stored

            # 4.1. Collect slices for mitmots
            offset = 0
            for idx in xrange(self.n_mit_mot_outs):
                if not self.mitmots_preallocated[idx]:
                    output_storage[offset].storage[0] = None
                    offset += 1

            # 4.2. Collect slices for mitsots, sitsots and nitsots
            if i != 0:
                for idx in xrange(self.n_outs + self.n_nit_sot -
                                  self.n_mit_mot):
                    if (store_steps[idx + self.n_mit_mot] == 1 or
                        self.vector_outs[idx + self.n_mit_mot]):
                        output_storage[idx + offset].storage[0] = None
                    else:
                        _pos0 = idx + self.n_mit_mot
                        output_storage[idx + offset].storage[0] =\
                            outs[_pos0][0][pos[_pos0]]
            else:
                for idx in xrange(self.n_outs + self.n_nit_sot -
                                  self.n_mit_mot):
                    output_storage[idx + offset].storage[0] = None

            # 4.3. Collect slices for shared outputs
            offset += self.n_outs + self.n_nit_sot - self.n_mit_mot
            for idx in xrange(self.n_shared_outs):
                output_storage[idx + offset].storage[0] = None

            # 4.4. If there is a condition add it to the mix
            if self.as_while:
                pdx = offset + self.n_shared_outs
                output_storage[pdx].storage[0] = None

            # 4.5. Keep a reference to the variables (ndarrays, CudaNdarrays,
            # etc) currently in the output_storage to be able to compare them
            # with the actual outputs of the inner function after its
            # execution. Also keep pointers to their data to be able to detect
            # cases where outputs reused the allocated object but alter the
            # memory region they refer to.
            for idx in xrange(len(output_storage)):

                var = output_storage[idx].storage[0]
                old_output_storage[idx] = var

                if var is None:
                    old_output_data[idx] = None
                elif self.outs_is_tensor[idx]:
                    old_output_data[idx] = var.data
                else:
                    old_output_data[idx] = var.gpudata

            # 4.6. Keep a reference to the variables (ndarrays, CudaNdarrays,
            # etc) associated with mitmot inputs currently in the
            # input_storage to be able to compare them with the content of the
            # input_storage after the execution of the function. Also keep
            # pointers to their data to be able to detect cases where outputs
            # reused the allocated object but alter the memory region they
            # refer to.
            for idx in xrange(nb_mitmot_in):
                var = input_storage[idx + self.n_seqs].storage[0]
                old_mitmot_input_storage[idx] = var

                if var is None:
                    old_mitmot_input_data[idx] = None
                elif self.inps_is_tensor[idx + self.n_seqs]:
                    old_mitmot_input_data[idx] = var.data
                else:
                    old_mitmot_input_data[idx] = var.gpudata

            # 5.1 compute outputs
            t0_fn = time.time()

            try:
                fn()
            except Exception:
                if hasattr(fn, 'position_of_error'):
                    # this is a new vm-provided function or c linker
                    # they need this because the exception manipulation
                    # done by raise_with_op is not implemented in C.
                    if hasattr(fn, 'thunks'):
                        # For the CVM
                        gof.link.raise_with_op(fn.nodes[fn.position_of_error],
                                               fn.thunks[fn.position_of_error])
                    else:
                        # For the c linker
                        # We don't have access from python to all the
                        # temps values So for now, we just don't print
                        # the extra shapes/strides info
                        gof.vm.raise_with_op(fn.nodes[fn.position_of_error])
                else:
                    # old-style linkers raise their own exceptions
                    raise

            dt_fn = time.time() - t0_fn
            if self.as_while:
                pdx = offset + self.n_shared_outs
                cond = output_storage[pdx].storage[0] == 0

            # 5.2. By calling fn() directly instead of calling the theano
            # function, it is possible that the updates have not been
            # performed. Perform the updates if needed.
            offset_out = len(output_storage) - 1
            if getattr(fn, 'need_update_inputs', True):
                # Update the inputs that have an update function
                for inp, storage in zip(self.fn.maker.expanded_inputs[::-1],
                                        self.fn.input_storage[::-1]):
                    if inp.update is not None:
                        storage.data = output_storage[offset_out].data
                        offset_out -= 1

            t_fn += dt_fn
            offset_out = 0

            # 5.3 Copy over the values for mit_mot outputs
            mitmot_inp_offset = 0
            mitmot_out_idx = 0
            for j in xrange(self.n_mit_mot):
                for k in self.mit_mot_out_slices[j]:
                    if self.mitmots_preallocated[mitmot_out_idx]:
                        # This output tap has been preallocated.
                        inp_idx = (mitmot_inp_offset +
                                   self.tap_array[j].index(k))

                        # Verify whether the input points to the same data as
                        # it did before the execution of the inner function.
                        old_var = old_mitmot_input_storage[inp_idx]
                        new_var = input_storage[self.n_seqs + inp_idx].storage[0]
                        if old_var is new_var:
                            old_data = old_mitmot_input_data[inp_idx]
                            if self.inps_is_tensor[self.n_seqs + inp_idx]:
                                same_data = (new_var.data == old_data)
                            else:
                                same_data = (new_var.gpudata == old_data)
                        else:
                            same_data = False

                        # If the corresponding input storage still points to
                        # the same data, it has been modified inplace and
                        # nothing needs to be done. Otherwise, recover the
                        # and store it in `outs` as usual
                        if not same_data:
                            outs[j][0][k + pos[j]] = \
                                input_storage[self.n_seqs + inp_idx].storage[0]

                    else:
                        # This output tap has not been preallocated, recover
                        # its value as usual
                        outs[j][0][k + pos[j]] = \
                                output_storage[offset_out].storage[0]
                        offset_out += 1

                    mitmot_out_idx += 1

                mitmot_inp_offset += len(self.tap_array[j])

            # 5.4 Copy over the values for mit_sot/sit_sot outputs
            begin = self.n_mit_mot
            end = self.n_outs
            offset_out -= self.n_mit_mot

            for j in xrange(begin, end):

                # Copy the output value to `outs`, if necessary
                if store_steps[j] == 1 or self.vector_outs[j]:
                    outs[j][0][pos[j]] = \
                            output_storage[offset_out + j].storage[0]
                else:
                    # Check whether the initialization of the output storage
                    # map for this output has been reused.
                    old_var = old_output_storage[offset_out + j]
                    new_var = output_storage[offset_out + j].storage[0]
                    if old_var is new_var:
                        old_data = old_output_data[offset_out + j]
                        if old_data is None:
                            output_reused = False
                        elif self.outs_is_tensor[offset_out + j]:
                            output_reused = (new_var.data == old_data)
                        else:
                            output_reused = (new_var.gpudata == old_data)
                    else:
                        output_reused = False

                    if not output_reused:
                        outs[j][0][pos[j]] = \
                            output_storage[offset_out + j].storage[0]

            # 5.5 Copy over the values for nit_sot outputs
            begin = end
            end += self.n_nit_sot
            for j in xrange(begin, end):

                if i == 0:
                    jout = j + offset_out
                    shape = (store_steps[j],) + \
                            output_storage[jout].storage[0].shape
                    if len(output_storage[jout].storage[0].shape) == 0:
                        self.vector_outs[j] = True
                    dtype = output_storage[jout].storage[0].dtype
                    if (outs[j][0] is None or
                            outs[j][0].shape[0] < store_steps[j] or
                            outs[j][0].shape[1:] != shape[1:] or
                            outs[j][0].dtype != dtype):
                        outs[j][0] = node.outputs[j].type.value_zeros(shape)
                    elif outs[j][0].shape[0] != store_steps[j]:
                        outs[j][0] = outs[j][0][:store_steps[j]]
                    outs[j][0][pos[j]] = output_storage[jout].storage[0]
                elif store_steps[j] == 1 or self.vector_outs[j]:
                    outs[j][0][pos[j]] = \
                        output_storage[j + offset_out].storage[0]
                else:
                    # Check whether the initialization of the output storage map
                    # for this output has been reused.
                    old_var = old_output_storage[offset_out + j]
                    old_data = old_output_data[offset_out + j]
                    new_var = output_storage[offset_out + j].storage[0]
                    if old_var is new_var:
                        if old_data is None:
                            output_reused = False
                        elif self.outs_is_tensor[offset_out + j]:
                            output_reused = (new_var.data == old_data)
                        else:
                            output_reused = (new_var.gpudata == old_data)
                    else:
                        output_reused = False

                    if not output_reused:
                        outs[j][0][pos[j]] = \
                            output_storage[j + offset_out].storage[0]

            # 5.6 Copy over the values for outputs corresponding to shared
            # variables
            begin = end
            end += self.n_shared_outs
            for j in xrange(begin, end):
                jout = j + offset_out
                outs[j][0] = output_storage[jout].storage[0]

            pos = [(idx + 1) % store for idx, store in
                               izip(pos, store_steps)]
            i = i + 1

        # 6. Check if you need to re-order output buffers
        begin = self.n_mit_mot
        end = self.n_outs + self.n_nit_sot
        for idx in xrange(begin, end):
            if (store_steps[idx] < i - self.mintaps[idx] and
                pos[idx] < store_steps[idx]):

                pdx = pos[idx]
                if pdx >= store_steps[idx] // 2:
                    # It seems inefficient to copy the bigger part of the
                    # array over, and back, but it is the only way that
                    # there is no overlap in the areas of out[idx][0] that
                    # are read and written.
                    # This way, there will be no information overwritten
                    # before it is read (as it used to happen).
                    shape = (pdx,) + outs[idx][0].shape[1:]
                    tmp = node.outputs[idx].type.value_zeros(shape)
                    tmp[:] = outs[idx][0][:pdx]
                    outs[idx][0][:store_steps[idx] - pdx] = outs[idx][0][pdx:]
                    outs[idx][0][store_steps[idx] - pdx:] = tmp
                    del tmp
                else:
                    shape = (store_steps[idx] - pdx,) + outs[idx][0].shape[1:]
                    tmp = node.outputs[idx].type.value_zeros(shape)
                    tmp[:] = outs[idx][0][pdx:]
                    outs[idx][0][store_steps[idx] - pdx:] = outs[idx][0][:pdx]
                    outs[idx][0][:store_steps[idx] - pdx] = tmp
                    del tmp
            # This would normally happen only when doing truncated
            # backpropagation through time. In such a scenarion Scan is
            # expected to return 0 for all entries for which the gradient is
            # not actually computed
            elif store_steps[idx] > i - self.mintaps[idx]:
                outs[idx][0][i - self.mintaps[idx]:] = 0
                # This is a fix for a bug introduced by while. If you say
                # you want to loop up to a condition, you expect the output
                # to have that length ( and not the maximal length possible)
                #
                # Without this the behaviour of a scan op is not consistent
                # if optimization gets applied compared to when optimization
                # do not get applied
                if i < n_steps:
                    # The reason I don't use out[idx][0][:i] is because for
                    # certain outputs (those with multiple taps),
                    # outs[idx][0] has more than n_steps entries, with the
                    # initial state  at the begining. When indexing in it I
                    # usually have to do something like
                    # outs[idx][0][i+offset]. To do something similar here,
                    # I would have first to compute the maximal tap for
                    # every output and then do outs[0][:i+maximal_tap],
                    # which implies I think more computations then this
                    # little trick that I used
                    outs[idx][0] = outs[idx][0][:-(n_steps - i)]

        # We never reuse the input or output storage of the
        # inner function so we clear it.
        for i_s in input_storage:
            i_s.storage[0] = None
        for o_s in output_storage:
            o_s.storage[0] = None

        t_call = time.time() - t0_call
        # NOTE: make this match what's in function_module.Function
        # and this little string helps us to find this spot:
        # "PROFILE_CODE"

        if hasattr(self.fn.maker, 'profile') and self.fn.maker.profile:
            profile = self.fn.maker.profile
            profile.callcount += 1
            profile.nbsteps += n_steps
            profile.call_time += t_call
            profile.vm_call_time += t_fn
            if hasattr(self.fn.fn, 'update_profile'):
                self.fn.fn.update_profile(profile)

        #/* Old ProfileMode
        # if hasattr(self.fn.maker.mode,'fct_call_time'):
        #    self.fn.maker.mode.fct_call_time[self.fn] += t_fn
        #    self.fn.maker.mode.fct_call[self.fn] += n_steps

        #self.fn.maker.mode.call_time += t_fn
        #self.fn.maker.mode.fn_time += t_fn
        # Old Profile Mode */
        self.t_call = t_call
        self.t_fn = t_fn

    # Infer Shape
    def infer_shape(self, node, input_shapes):
        # input_shapes correspond to the shapes of node.inputs
        for inp, inp_shp in izip(node.inputs, input_shapes):
            assert inp_shp is None or len(inp_shp) == inp.type.ndim

        # Here we build 2 variables;
        # - A list `inner_ins_shapes`, such that inner_ins_shapes[i] is the
        #   shape of self.inputs[i]
        # - A dictionary `out_equivalent` containing, for every inner input,
        #   an equivalent variable computed from the outer inputs.
        #   NOTE : For non-sequences, this equivalence is trivial. For
        #   sequences and recurrent states, there is no direct equivalence
        #   between outer and inner inputs. However, because every iteration
        #   of the Scan needs to give the same output shapes, we can give an
        #   equivalence between these inner inputs and the subelements of the
        #   corresponding outer inputs that the Scan would use as input for
        #   any given iteration. For simplicity, we use iteration 0.
        inner_ins_shapes = []
        out_equivalent = OrderedDict()

        # The two following blocks are commented as it cause in some
        # cases extra scans in the graph. See gh-XXX for the
        # investigation.

        # We skip the first outer input as it is the total or current number
        # of iterations.
        # sequences
        seqs_shape = [x[1:] for x in input_shapes[1:1 + self.n_seqs]]
        # We disable extra infer_shape for now. See gh-3765.
        extra_infer_shape = False

        if extra_infer_shape:
            inner_seqs = self.inputs[:self.n_seqs]
            outer_seqs = node.inputs[1:1 + self.n_seqs]
            for in_s, out_s in izip(inner_seqs, outer_seqs):
                out_equivalent[in_s] = out_s[0]

            # mit_mot, mit_sot, sit_sot
            outer_inp_idx = 1 + self.n_seqs
            inner_inp_idx = self.n_seqs
        else:
            outer_inp_idx = 0
        n_outs = self.n_mit_mot + self.n_mit_sot + self.n_sit_sot
        outs_shape = []
        for idx in xrange(n_outs):
            mintap = abs(min(self.tap_array[idx]))
            for k in self.tap_array[idx]:
                outs_shape += [input_shapes[idx + self.n_seqs + 1][1:]]
                if extra_infer_shape:
                    corresponding_tap = node.inputs[outer_inp_idx][mintap + k]
                    out_equivalent[self.inputs[inner_inp_idx]] = corresponding_tap
                    inner_inp_idx += 1
            outer_inp_idx += 1

        # shared_outs
        offset = 1 + self.n_seqs + n_outs
        for idx in xrange(self.n_shared_outs):
            outs_shape += [input_shapes[idx + offset]]

        # non_sequences
        offset += self.n_nit_sot + self.n_shared_outs
        inner_ins_shapes = seqs_shape + outs_shape + input_shapes[offset:]
        assert len(inner_ins_shapes) == len(self.inputs)

        # Non-sequences have a direct equivalent from self.inputs in
        # node.inputs
        inner_non_sequences = self.inputs[len(seqs_shape) + len(outs_shape):]
        for in_ns, out_ns in izip(inner_non_sequences, node.inputs[offset:]):
            out_equivalent[in_ns] = out_ns

        if self.as_while:
            self_outs = self.outputs[:-1]
        else:
            self_outs = self.outputs
        outs_shape = scan_utils.infer_shape(
                outs=self_outs,
                inputs=self.inputs,
                input_shapes=inner_ins_shapes)
        # Will be used to check if outs_shape can be expressed without using
        # variables in self.inputs.
        # The shapes of node.inputs are valid.
        validator = scan_utils.Validator(
                valid=input_shapes,
                invalid=self.inputs,
                valid_equivalent=out_equivalent)

        offset = 1 + self.n_seqs
        scan_outs = [x for x in input_shapes[offset:offset + n_outs]]
        offset += n_outs
        outs_shape_n = self.n_mit_mot_outs + self.n_mit_sot + self.n_sit_sot
        for x in xrange(self.n_nit_sot):
            out_shape_x = outs_shape[outs_shape_n + x]
            if out_shape_x is None:
                # This output is not a tensor, and has no shape
                scan_outs.append(None)
            else:
                # We need to make sure that we can compute the shapes from
                # node.inputs, and constants, without using the variables
                # in the inner function.
                r = node.outputs[n_outs + x]
                assert r.ndim == 1 + len(out_shape_x)
                shp = [node.inputs[offset + self.n_shared_outs + x]]
                for i, shp_i in izip(xrange(1, r.ndim), out_shape_x):
                    # Validate shp_i. v_shape_i is either None (if invalid),
                    # or a (variable, Boolean) tuple. The Boolean indicates
                    # whether variable is shp_i (if True), or an valid
                    # equivalent (if False). Here, we only need the variable.
                    v_shp_i = validator.check(shp_i)
                    if v_shp_i is None:
                        if hasattr(r, 'broadcastable') and r.broadcastable[i]:
                            shp.append(1)
                        else:
                            shp.append(Shape_i(i)(r))
                    else:
                        # It can (or at least, an equivalent variable can)
                        shp.append(v_shp_i[0])
                scan_outs.append(tuple(shp))

        scan_outs += [x for x in
                     input_shapes[offset:offset + self.n_shared_outs]]
        # if we are dealing with a repeat-until, then we do not know the
        # leading dimension so we replace it for every entry with Shape_i
        if self.as_while:
            scan_outs_init = scan_outs
            scan_outs = []
            for o, x in izip(node.outputs, scan_outs_init):
                if x is None:
                    scan_outs.append(None)
                else:
                    scan_outs.append((Shape_i(0)(o),) + x[1:])
        return scan_outs

    def connection_pattern(self, node):

        # We cache the result of this function because, with a previous
        # implementation that repeatedly called grad, there were cases
        # where calls to theano.grad() took as much as 4h for functions
        # containing many nested scans.
        if hasattr(node.tag, 'connection_pattern'):
            return node.tag.connection_pattern

        # Obtain the connection pattern of the inner function.
        inner_connect_pattern = io_connection_pattern(self.inputs, self.outputs)

        # Initially assume no outer input is connected to any outer output
        connection_pattern = [[False for output in node.outputs]
                              for x in node.inputs]

        # For every possible pair of outer input and outer output, iterate
        # over every possible pairing of their corresponding inner inputs
        # and inner outputs and, if one such pair of inner variables is
        # connected than the pair of outer variables is connected.
        for outer_oidx in xrange(len(node.outputs)):
            inner_oidxs = self.var_mappings['inner_out_from_outer_out'][outer_oidx]

            for outer_iidx in xrange(len(node.inputs)):
                inner_iidxs = self.var_mappings['inner_inp_from_outer_inp'][outer_iidx]

                for inner_oidx in inner_oidxs:
                    for inner_iidx in inner_iidxs:

                        if inner_connect_pattern[inner_iidx][inner_oidx]:
                            connection_pattern[outer_iidx][outer_oidx] = True
                            break

                    if connection_pattern[outer_iidx][outer_oidx]:
                        break

        # Applying Floyd-Warshall to find all paths connecting inputs to
        # outputs. Note that if `x` is an input to `y_t` and `y_tm1` is an
        # input to `z_t` then `x` is an input to `z_t`.

        n_outs = len(node.outputs)

        for steps in xrange(n_outs):
            for iidx in xrange(n_outs):
                for jidx in xrange(n_outs):

                    # Get the idx of the outer input corresponding to that
                    # outer output
                    j_inp_idx = self.var_mappings["outer_inp_from_outer_out"][jidx]

                    if j_inp_idx != -1:
                       if connection_pattern[j_inp_idx][iidx] == True:
                            for k in xrange(len(connection_pattern)):
                                if connection_pattern[k][jidx]:
                                    connection_pattern[k][iidx] = True

        node.tag.connection_pattern = connection_pattern
        return connection_pattern

    def get_oinp_iinp_iout_oout_mappings(self):
        """
        Compute and return dictionary mappings between the inputs and
        outputs of the inner function and the inputs and outputs of the Scan
        node in the outer graph.

        The return value is a dictionary in which the keys are the names of
        the individual mappings and the values are the mapping dictionaries
        themselves. In dictionaries representing mappings to outer variables,
        the values are individual integer indices. In dictionaries
        representing mappings to inner variables, the values are sequences of
        indices because multiple inner variables can be associated with the
        same state.

        """
        # Lists for outer variables contain individual indices, lists for
        # inner variables contain sequences of indices because many inner
        # variables can be associated with the same outer variable. The list
        # and indices are initialized already containing the data associated
        # with the timestep index, the first outer input.
        outer_input_indices = [0]
        inner_input_indices = [[]]
        inner_output_indices = [[]]
        outer_output_indices = [-1]

        outer_iidx = 1
        inner_iidx = 0
        inner_oidx = 0
        outer_oidx = 0

        # Handle sequences inputs
        for i in xrange(self.info['n_seqs']):
            outer_input_indices.append(outer_iidx)
            inner_input_indices.append([inner_iidx])
            inner_output_indices.append([])
            outer_output_indices.append(-1)

            outer_iidx += 1
            inner_iidx += 1
            inner_oidx += 0
            outer_oidx += 0

        # Handle mitmots, mitsots and sitsots variables
        for i in xrange(len(self.info['tap_array'])):
            nb_input_taps = len(self.info['tap_array'][i])

            if i < self.n_mit_mot:
                nb_output_taps = len(self.mit_mot_out_slices[i])
            else:
                nb_output_taps = 1

            outer_input_indices.append(outer_iidx)
            inner_input_indices.append(list(range(inner_iidx,
                                                  inner_iidx + nb_input_taps)))
            inner_output_indices.append(list(range(inner_oidx,
                                                   inner_oidx + nb_output_taps)))
            outer_output_indices.append(outer_oidx)

            outer_iidx += 1
            inner_iidx += nb_input_taps
            inner_oidx += nb_output_taps
            outer_oidx += 1

        # This is needed because, for outer inputs (and for outer inputs only)
        # nitsots come *after* shared variables.
        outer_iidx += self.info['n_shared_outs']

        # Handle nitsots variables
        for i in xrange(self.n_nit_sot):
            outer_input_indices.append(outer_iidx)
            inner_input_indices.append([])
            inner_output_indices.append([inner_oidx])
            outer_output_indices.append(outer_oidx)

            outer_iidx += 1
            inner_iidx += 0
            inner_oidx += 1
            outer_oidx += 1

        # This is needed because, for outer inputs (and for outer inputs only)
        # nitsots come *after* shared variables.
        outer_iidx -= (self.info['n_shared_outs'] + self.n_nit_sot)

        # Handle shared states
        for i in xrange(self.info['n_shared_outs']):
            outer_input_indices.append(outer_iidx)
            inner_input_indices.append([inner_iidx])
            inner_output_indices.append([inner_oidx])
            outer_output_indices.append(outer_oidx)

            outer_iidx += 1
            inner_iidx += 1
            inner_oidx += 1
            outer_oidx += 1

        # This is needed because, for outer inputs (and for outer inputs only)
        # nitsots come *after* shared variables.
        outer_iidx += self.n_nit_sot

        # Handle non-sequence inputs
        # Note : the number of non-sequence inputs is not stored in self.info
        # so it has to be inferred from the number of inner inputs that remain
        # to be handled
        for i in xrange(len(self.inputs) - inner_iidx):
            outer_input_indices.append(outer_iidx)
            inner_input_indices.append([inner_iidx])
            inner_output_indices.append([])
            outer_output_indices.append(-1)

            outer_iidx += 1
            inner_iidx += 1
            inner_oidx += 0
            outer_oidx += 0

        # With the global mapping inferred, the individual mappings
        # can be produced
        mappings = {"outer_inp_from_outer_out" : {},
                    "inner_inp_from_outer_out" : {},
                    "inner_out_from_outer_out" : {},
                    "inner_inp_from_outer_inp" : {},
                    "inner_out_from_outer_inp" : {},
                    "outer_out_from_outer_inp" : {},
                    "outer_inp_from_inner_inp" : {},
                    "inner_out_from_inner_inp" : {},
                    "outer_out_from_inner_inp" : {},
                    "outer_inp_from_inner_out" : {},
                    "inner_inp_from_inner_out" : {},
                    "outer_out_from_inner_out" : {}}

        for (oinp, iinp, iout, oout) in izip(outer_input_indices,
                                             inner_input_indices,
                                             inner_output_indices,
                                             outer_output_indices):

            if oout != -1:
                mappings["outer_inp_from_outer_out"][oout] = oinp
                mappings["inner_inp_from_outer_out"][oout] = iinp
                mappings["inner_out_from_outer_out"][oout] = iout

            if oinp != -1:
                mappings["inner_inp_from_outer_inp"][oinp] = iinp
                mappings["inner_out_from_outer_inp"][oinp] = iout
                mappings["outer_out_from_outer_inp"][oinp] = oout

            for idx in iinp:
                mappings["outer_inp_from_inner_inp"][idx] = oinp
                mappings["inner_out_from_inner_inp"][idx] = iout
                mappings["outer_out_from_inner_inp"][idx] = oout

            for idx in iout:
                mappings["outer_inp_from_inner_out"][idx] = oinp
                mappings["inner_inp_from_inner_out"][idx] = iinp
                mappings["outer_out_from_inner_out"][idx] = oout

        return mappings

    # GRAD FUNCTION
    def grad(self, inputs, dC_douts):
        outs = self(*inputs)
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        # `grad_step` equals the number of steps the original scan node has
        # done (if the original scan is a while loop than this number is the
        # length of the output sequence)
        # We do not know what kind of outputs the original scan has, so we
        # try first to see if it has a nit_sot output, then a sit_sot and
        # then a mit_sot
        if self.n_nit_sot > 0:
            grad_steps = self.outer_nitsot_outs(outs)[0].shape[0]
        elif self.n_sit_sot > 0:
            grad_steps = self.outer_sitsot_outs(outs)[0].shape[0] - 1
        elif self.n_mit_sot > 0:
            grad_steps = self.outer_mitsot_outs(outs)[0].shape[0] +\
                    self.mintaps[self.n_mit_mot]
        else:
            grad_steps = inputs[0]

        # Restrict the number of grad steps according to
        # self.truncate_gradient
        if self.truncate_gradient != -1:
            grad_steps = tensor.minimum(grad_steps, self.truncate_gradient)

        rval = scan_utils.reconstruct_graph(self.inputs,
                                            self.outputs)
        self_inputs = rval[0]
        self_outputs = rval[1]
        # differentiable inputs
        diff_inputs = (self.inner_seqs(self_inputs) +
                       self.inner_mitmot(self_inputs) +
                       self.inner_mitsot(self_inputs) +
                       self.inner_sitsot(self_inputs) +
                       self.inner_non_seqs(self_inputs))
        diff_outputs = (self.inner_mitmot_outs(self_outputs) +
                        self.inner_mitsot_outs(self_outputs) +
                        self.inner_sitsot_outs(self_outputs) +
                        self.inner_nitsot_outs(self_outputs))
        scan_node = outs[0].owner
        connection_pattern = self.connection_pattern(scan_node)

        def get_inp_idx(iidx):
            if iidx < self.n_seqs:
                return 1 + iidx
            oidx = 1 + self.n_seqs
            iidx = iidx - self.n_seqs
            for taps in self.mitmot_taps():
                if len(taps) > iidx:
                    return oidx
                else:
                    oidx += 1
                    iidx -= len(taps)
            for taps in self.mitsot_taps():
                if len(taps) > iidx:
                    return oidx
                else:
                    oidx += 1
                    iidx -= len(taps)

            if iidx < self.info['n_sit_sot']:
                return oidx + iidx
            else:
                return oidx + iidx + self.info['n_nit_sot']

        def get_out_idx(iidx):
            oidx = 0
            for taps in self.mitmot_out_taps():
                if len(taps) > iidx:
                    return oidx
                else:
                    oidx += 1
                    iidx -= len(taps)
            return oidx + iidx

        def compute_all_gradients(known_grads):
            y_s = known_grads.keys()
            g_y_s = known_grads.values()

            for g_y in g_y_s:
                if 'int' in str(g_y.dtype):
                    raise TypeError("Gradients may never be integers but g_y "
                                    "has type " + str(g_y.type))

            out_indices = [get_out_idx(self_outputs.index(y)) for y in y_s]

            connected_inputs = [i for i in range(len(scan_node.inputs)) if
                                any([connection_pattern[i][odx] for odx in out_indices])]

            wrt = [x for x in theano.gof.graph.inputs(y_s) if
                   (x in diff_inputs) and
                   get_inp_idx(self_inputs.index(x)) in connected_inputs]
            gmp = OrderedDict()

            # Required in case there is a pair of variables X and Y, with X
            # used to compute Y, for both of which there is an external
            # gradient signal. Without this, the total gradient signal on X
            # will be the external gradient  signalknown_grads[X]. With this,
            # it will be the sum of the external gradient signal and the
            # gradient obtained by propagating Y's external gradient signal
            # to X.
            known_grads = dict([(k.copy(), v) for (k, v) in known_grads.items()])

            grads = gradient.grad(
                        cost=None,
                        known_grads=known_grads,
                        wrt=wrt,
                        consider_constant=wrt,
                        disconnected_inputs='ignore',
                        return_disconnected='None',
                        null_gradients='return')

            for i in range(len(wrt)):
                gmp[wrt[i]] = grads[i]

            rval = [gmp.get(p, None) for p in diff_inputs]
            return rval

        dC_dinps_t = [None for inp in diff_inputs]
        disconnected_dC_dinps_t = [True for inp in diff_inputs]
        dC_dXts = []
        Xts = []
        for idx, Xt in enumerate(diff_outputs):

            # We are looking for x[t-1] for a given x[t]
            if idx >= self.n_mit_mot_outs:
                Xt_placeholder = safe_new(Xt)
                Xts.append(Xt_placeholder)

            # Different processing based on whether Xt is a nitsot output
            # or not. NOTE : This cannot be done by using
            # "if Xt not in self.inner_nitsot_outs(self_outputs)" because
            # the exact same variable can be used as multiple outputs.
            idx_nitsot_start = (self.info['n_mit_mot'] +
                                self.info['n_mit_sot'] +
                                self.info['n_sit_sot'])
            idx_nitsot_end = idx_nitsot_start + self.info['n_nit_sot']
            if idx < idx_nitsot_start or idx >= idx_nitsot_end:
                # What we do here is loop through dC_douts and collect all
                # those that are connected to the specific one and do an
                # upcast on all of their dtypes to get the dtype for this
                # specific output. Deciding if the gradient with this
                # specific previous step is defined or not is done somewhere
                # else.
                dtypes = []
                states = (self.inner_mitmot(self_inputs) +
                          self.inner_mitsot(self_inputs) +
                          self.inner_sitsot(self_inputs))

                for pos, inp in enumerate(states):
                    if inp in theano.gof.graph.inputs([Xt]):
                        # Get the index of the outer output that to which
                        # the state variable 'inp' corresponds.
                        outer_oidx = self.var_mappings['outer_out_from_inner_inp'][self.n_seqs +
                                                                                   pos]

                        if not isinstance(dC_douts[outer_oidx].type,
                                          DisconnectedType):
                            dtypes.append(dC_douts[outer_oidx].dtype)
                if dtypes:
                    new_dtype = theano.scalar.upcast(*dtypes)
                else:
                    new_dtype = theano.config.floatX
                dC_dXt = safe_new(Xt, dtype=new_dtype)
            else:
                if isinstance(dC_douts[idx].type, DisconnectedType):
                    continue
                dC_dXt = safe_new(dC_douts[idx][0])
            dC_dXts.append(dC_dXt)


        known_grads = {}
        dc_dxts_idx = 0
        for i in range(len(diff_outputs)):
            if i < idx_nitsot_start or i >= idx_nitsot_end:
                if diff_outputs[i] in known_grads:
                    known_grads[diff_outputs[i]] += dC_dXts[dc_dxts_idx]
                else:
                    known_grads[diff_outputs[i]] = dC_dXts[dc_dxts_idx]
                dc_dxts_idx += 1
            else:
                if isinstance(dC_douts[i].type, DisconnectedType):
                    continue
                else:
                    if diff_outputs[i] in known_grads:
                        known_grads[diff_outputs[i]] += dC_dXts[dc_dxts_idx]
                    else:
                        known_grads[diff_outputs[i]] = dC_dXts[dc_dxts_idx]
                    dc_dxts_idx += 1
        dC_dinps_t = compute_all_gradients(known_grads)


        # mask inputs that get no gradients
        for dx in xrange(len(dC_dinps_t)):
            if not dC_dinps_t[dx]:
                dC_dinps_t[dx] = tensor.zeros_like(diff_inputs[dx])
            else:
                disconnected_dC_dinps_t[dx] = False
                for Xt, Xt_placeholder in zip(
                        diff_outputs[self.n_mit_mot_outs:],
                        Xts):
                    tmp = forced_replace(
                        dC_dinps_t[dx],
                        Xt,
                        Xt_placeholder)
                    dC_dinps_t[dx] = tmp

        # construct dX_dtm1
        dC_dXtm1s = []
        for pos, x in enumerate(dC_dinps_t[self.n_seqs:]):

            # Get the index of the first inner input corresponding to the
            # pos-ieth inner input state
            idxs = self.var_mappings['inner_out_from_inner_inp'][self.n_seqs +
                                                                 pos]

            # Check if the pos-th input is associated with one of the
            # recurrent states
            x_is_state = pos < sum([len(t) for t in self.tap_array])

            if x_is_state and len(idxs) > 0:
                opos = idxs[0]
                dC_dXtm1s.append(safe_new(dC_dXts[opos]))
                if hasattr(x, 'dtype') and x.dtype != dC_dXts[opos].dtype:
                    dC_dinps_t[pos + self.n_seqs] = \
                            x.astype(dC_dXts[opos].dtype)
            else:
                dC_dXtm1s.append(safe_new(x))

        for dx, dC_dXtm1 in enumerate(dC_dXtm1s):
            if isinstance(dC_dinps_t[dx + self.n_seqs].type, NullType):
                # The accumulated gradient is undefined
                pass
            elif isinstance(dC_dXtm1.type, NullType):
                # The new gradient is undefined, this makes the accumulated
                # gradient undefined as weell
                dC_dinps_t[dx + self.n_seqs] = dC_dXtm1
            else:
                dC_dinps_t[dx + self.n_seqs] += dC_dXtm1
        # Construct scan op
        # Seqs
        outer_inp_seqs = [x[::-1] for x in inputs[1:1 + self.n_seqs]]
        for idx in xrange(self.n_mit_mot + self.n_mit_sot):
            mintap = numpy.min(self.tap_array[idx])
            maxtap = numpy.max(self.tap_array[idx])
            if idx < self.n_mit_mot:
                outmaxtap = numpy.max(self.mitmot_out_taps()[idx])
            else:
                outmaxtap = 0
            seq = outs[idx]
            for k in self.tap_array[idx]:
                if outmaxtap - k != 0:
                    nw_seq = seq[k - mintap: -(outmaxtap-k)][::-1]
                else:
                    nw_seq = seq[k - mintap:][::-1]
                outer_inp_seqs.append(nw_seq)
        outer_inp_seqs += [
            x[:-1][::-1] for x in self.outer_sitsot_outs(outs)]
        for x in self.outer_nitsot_outs(dC_douts):
            if not isinstance(x.type, DisconnectedType):
                outer_inp_seqs.append(x[::-1])

        if hasattr(inputs[0].tag, 'test_value'):
            # Here we tests that the new scan input sequence all have
            # the same shape[0]. This is a properties that the scan()
            # fct add and we want to keep it for all Scan op.  This is
            # used in T_Scan.test_grad_multiple_outs_taps to test
            # that.
            for taps, x in zip(self.mitsot_taps(),
                               self.outer_mitsot_outs(outs)):
                mintap = numpy.min(taps)
                if hasattr(x[::-1][:mintap], 'test_value'):
                    assert (x[::-1][:mintap].tag.test_value.shape[0] ==
                            inputs[0].tag.test_value)
            for x in self.outer_sitsot_outs(outs):
                if hasattr(x[::-1][:-1].tag, 'test_value'):
                    assert (x[::-1][:-1].tag.test_value.shape[0] ==
                            inputs[0].tag.test_value)
            for x in self.outer_nitsot_outs(outs):
                if hasattr(x[::-1].tag, 'test_value'):
                    assert (x[::-1].tag.test_value.shape[0] ==
                            inputs[0].tag.test_value)
        outer_inp_seqs += [x[::-1][:numpy.min(taps)]
                           for taps, x in zip(self.mitsot_taps(),
                                              self.outer_mitsot_outs(outs))]
        outer_inp_seqs += [x[::-1][:-1] for x in self.outer_sitsot_outs(outs)]
        outer_inp_seqs += [x[::-1] for x in self.outer_nitsot_outs(outs)]

        # Restrict the length of the outer sequences to the number of grad
        # steps
        outer_inp_seqs = [seq[:grad_steps] for seq in outer_inp_seqs]

        inner_inp_seqs = self.inner_seqs(self_inputs)
        inner_inp_seqs += self.inner_mitmot(self_inputs)
        inner_inp_seqs += self.inner_mitsot(self_inputs)
        inner_inp_seqs += self.inner_sitsot(self_inputs)
        inner_inp_seqs += self.inner_nitsot_outs(dC_dXts)
        inner_inp_seqs += Xts
        # mitmot
        outer_inp_mitmot = []
        outer_out_mitmot = []
        inner_inp_mitmot = []
        inner_out_mitmot = []
        mitmot_inp_taps = []
        mitmot_out_taps = []
        type_outs = []
        out_pos = 0
        ins_pos = self.n_seqs
        n_mitmot_outs = 0
        n_mitmot_inps = 0

        for idx in xrange(self.n_mit_mot):
            if isinstance(dC_douts[idx].type, DisconnectedType):
                out = outs[idx]
                outer_inp_mitmot.append(tensor.zeros_like(out))
            else:
                outer_inp_mitmot.append(dC_douts[idx][::-1])
            mitmot_inp_taps.append([])
            mitmot_out_taps.append([])
            undefined_msg = None
            through_shared = False
            disconnected = True

            for jdx in xrange(len(self.mit_mot_out_slices[idx])):
                inner_inp_mitmot.append(dC_dXts[out_pos])
                mitmot_inp_taps[idx].append(-self.mit_mot_out_slices[idx][jdx])
                n_mitmot_inps += 1
                out_pos += 1

            for jdx in xrange(len(self.tap_array[idx])):
                tap = -self.tap_array[idx][jdx]

                # Only create a new inner input if there is not already one
                # associated with this input tap
                if tap not in mitmot_inp_taps[idx]:
                    inner_inp_mitmot.append(dC_dXtm1s[ins_pos - self.n_seqs])

                if isinstance(dC_dinps_t[ins_pos].type, NullType):
                    # We cannot use Null in the inner graph, so we
                    # use a zero tensor of the appropriate shape instead.
                    inner_out_mitmot.append(
                        tensor.zeros(diff_inputs[ins_pos].shape,
                                     dtype=theano.config.floatX))
                    undefined_msg = dC_dinps_t[ins_pos].type.why_null
                else:
                    new_inner_out_mitmot = dC_dinps_t[ins_pos]

                    # If there is already an inner input associated with that
                    # input tap, make sure the computation of the new output
                    # uses it instead of the input it's currently using
                    if tap in mitmot_inp_taps[idx]:
                        to_replace = dC_dXtm1s[ins_pos - self.n_seqs]
                        replacement_idx = (len(mitmot_inp_taps[idx]) -
                                           mitmot_inp_taps[idx].index(tap))
                        replacement = inner_inp_mitmot[-replacement_idx]

                        self.tap_array[idx]
                        new_inner_out_mitmot = theano.clone(new_inner_out_mitmot,
                                                            replace=[(to_replace, replacement)])


                    inner_out_mitmot.append(new_inner_out_mitmot)

                if not disconnected_dC_dinps_t[ins_pos]:
                    disconnected = False

                for _sh in self.inner_shared(self_inputs):
                    if _sh in gof.graph.inputs([dC_dinps_t[ins_pos]]):
                        through_shared = True

                ins_pos += 1
                n_mitmot_outs += 1
                mitmot_out_taps[idx].append(-self.tap_array[idx][jdx])

                # Only add the tap as a new input tap if needed
                if tap not in mitmot_inp_taps[idx]:
                    n_mitmot_inps += 1
                    mitmot_inp_taps[idx].append(-self.tap_array[idx][jdx])

            if undefined_msg:
                type_outs.append(undefined_msg)
            elif through_shared:
                type_outs.append('through_shared')
            elif disconnected:
                type_outs.append('disconnected')
            else:
                type_outs.append('connected')

        offset = self.n_mit_mot
        for idx in xrange(self.n_mit_sot):
            if isinstance(dC_douts[idx + offset].type, DisconnectedType):
                outer_inp_mitmot.append(outs[idx + offset].zeros_like())
            else:
                outer_inp_mitmot.append(dC_douts[idx + offset][::-1])
            mitmot_inp_taps.append([])
            mitmot_out_taps.append([])
            idx_tap = idx + self.n_mit_mot
            inner_inp_mitmot.append(dC_dXts[out_pos])
            out_pos += 1
            n_mitmot_inps += 1
            undefined_msg = None
            through_shared = False
            disconnected = True
            mitmot_inp_taps[idx + offset].append(0)
            for jdx in xrange(len(self.tap_array[idx_tap])):
                inner_inp_mitmot.append(dC_dXtm1s[ins_pos - self.n_seqs])

                if isinstance(dC_dinps_t[ins_pos].type, NullType):
                    # We cannot use Null in the inner graph, so we
                    # use a zero tensor of the appropriate shape instead.
                    inner_out_mitmot.append(
                        tensor.zeros(diff_inputs[ins_pos].shape,
                                     dtype=theano.config.floatX))
                    undefined_msg = dC_dinps_t[ins_pos].type.why_null
                else:
                    inner_out_mitmot.append(dC_dinps_t[ins_pos])

                mitmot_inp_taps[idx + offset].append(
                    -self.tap_array[idx_tap][jdx])
                mitmot_out_taps[idx].append(
                    -self.tap_array[idx_tap][jdx])
                if not disconnected_dC_dinps_t[ins_pos]:
                    disconnected = False
                for _sh in self.inner_shared(self_inputs):
                    if _sh in gof.graph.inputs([dC_dinps_t[ins_pos]]):
                        through_shared = True

                n_mitmot_inps += 1
                ins_pos += 1
                n_mitmot_outs += 1

            if undefined_msg:
                type_outs.append(undefined_msg)
            elif through_shared:
                type_outs.append('through_shared')
            elif disconnected:
                type_outs.append('disconnected')
            else:
                type_outs.append('connected')

        offset += self.n_mit_sot
        for idx in xrange(self.n_sit_sot):
            mitmot_inp_taps.append([0, 1])
            mitmot_out_taps.append([1])
            through_shared = False
            if not isinstance(dC_douts[idx + offset].type, DisconnectedType):
                outer_inp_mitmot.append(dC_douts[idx + offset][::-1])
            else:
                if isinstance(dC_dinps_t[ins_pos].type, NullType):
                    # Cannot use dC_dinps_t[ins_pos].dtype, so we use
                    # floatX instead, as it is a dummy value that will not
                    # be used anyway.
                    outer_inp_mitmot.append(
                        tensor.zeros(outs[idx + offset].shape,
                                     dtype=theano.config.floatX))
                else:
                    outer_inp_mitmot.append(
                        tensor.zeros(outs[idx + offset].shape,
                                     dtype=dC_dinps_t[ins_pos].dtype))

            if isinstance(dC_dinps_t[ins_pos].type, NullType):
                # We cannot use Null in the inner graph, so we
                # use a zero tensor of the appropriate shape instead.
                inner_out_mitmot.append(
                    tensor.zeros(diff_inputs[ins_pos].shape,
                                 dtype=theano.config.floatX))
            else:
                inner_out_mitmot.append(dC_dinps_t[ins_pos])

            for _sh in self.inner_shared(self_inputs):
                if _sh in gof.graph.inputs([dC_dinps_t[ins_pos]]):
                    through_shared = True

            if isinstance(dC_dinps_t[ins_pos].type, NullType):
                type_outs.append(dC_dinps_t[ins_pos].type.why_null)
            elif through_shared:
                type_outs.append('through_shared')
            elif disconnected_dC_dinps_t[ins_pos]:
                type_outs.append('disconnected')
            else:
                type_outs.append('connected')

            inner_inp_mitmot += [dC_dXts[out_pos],
                                 dC_dXtm1s[ins_pos - self.n_seqs]]
            n_mitmot_outs += 1
            out_pos += 1
            ins_pos += 1
            n_mitmot_inps += 2

        n_nit_sot = self.n_seqs
        inner_out_nitsot = dC_dinps_t[:self.n_seqs]
        inner_out_sitsot = dC_dinps_t[ins_pos:]
        for _p, vl in enumerate(inner_out_sitsot):
            through_shared = False
            for _sh in self.inner_shared(self_inputs):
                if _sh in gof.graph.inputs([vl]):
                    through_shared = True
            if isinstance(vl.type, NullType):
                type_outs.append(vl.type.why_null)
                # Replace the inner output with a zero tensor of
                # the right shape
                inner_out_sitsot[_p] = tensor.zeros(
                    diff_inputs[ins_pos + _p].shape,
                    dtype=theano.config.floatX)
            elif through_shared:
                type_outs.append('through_shared')
            elif disconnected_dC_dinps_t[_p + ins_pos]:
                type_outs.append('disconnected')
            else:
                type_outs.append('connected')

        for _p, vl in enumerate(inner_out_nitsot):
            through_shared = False
            for _sh in self.inner_shared(self_inputs):
                if _sh in gof.graph.inputs([vl]):
                    through_shared = True
            if isinstance(vl.type, NullType):
                type_outs.append(vl.type.why_null)
                # Replace the inner output with a zero tensor of
                # the right shape
                inner_out_nitsot[_p] = tensor.zeros(
                    diff_inputs[_p].shape,
                    dtype=theano.config.floatX)

            if through_shared:
                type_outs.append('through_shared')
            elif disconnected_dC_dinps_t[_p]:
                type_outs.append('disconnected')
            else:
                type_outs.append('connected')

        inner_inp_sitsot = dC_dXtm1s[ins_pos - self.n_seqs:]
        outer_inp_sitsot = []
        for _idx, y in enumerate(inner_inp_sitsot):
            x = self.outer_non_seqs(inputs)[_idx]
            if isinstance(y.type, NullType):
                # Cannot use dC_dXtm1s.dtype, so we use floatX instead.
                outer_inp_sitsot.append(
                    tensor.zeros([grad_steps + 1] +
                                 [x.shape[i] for i in xrange(x.ndim)],
                                 dtype=theano.config.floatX))
                # replace y by a zero tensor of the right shape
                inner_inp_sitsot[_idx] = tensor.zeros(
                    diff_inputs[ins_pos + _idx].shape,
                    dtype=theano.config.floatX)

            else:
                outer_inp_sitsot.append(
                    tensor.zeros([grad_steps + 1] +
                                 [x.shape[i] for i in xrange(x.ndim)],
                                 dtype=y.dtype))

        n_sitsot_outs = len(outer_inp_sitsot)
        new_tap_array = mitmot_inp_taps + [[-1] for k in
                                           xrange(n_sitsot_outs)]

        info = OrderedDict()
        info['n_seqs'] = len(outer_inp_seqs)
        info['n_mit_sot'] = 0
        info['tap_array'] = new_tap_array
        info['gpu'] = False
        info['n_mit_mot'] = len(outer_inp_mitmot)
        info['n_mit_mot_outs'] = n_mitmot_outs
        info['mit_mot_out_slices'] = mitmot_out_taps
        info['truncate_gradient'] = self.truncate_gradient
        info['n_sit_sot'] = n_sitsot_outs
        info['n_shared_outs'] = 0
        info['n_nit_sot'] = n_nit_sot
        info['as_while'] = False
        info['profile'] = self.profile
        info['destroy_map'] = OrderedDict()
        if self.name:
            info['name'] = 'grad_of_' + self.name
        else:
            info['name'] = None
        info['mode'] = self.mode
        info['allow_gc'] = self.allow_gc

        outer_inputs = ([grad_steps] +
                        outer_inp_seqs +
                        outer_inp_mitmot +
                        outer_inp_sitsot +
                        [inputs[0] for x in xrange(n_nit_sot)] +
                        self.outer_shared(inputs) +
                        self.outer_non_seqs(inputs))

        inner_other_args = self_inputs[offset:]
        inner_gfn_ins = (inner_inp_seqs +
                         inner_inp_mitmot +
                         inner_inp_sitsot +
                         self.inner_shared(self_inputs) +
                         self.inner_non_seqs(self_inputs))
        inner_gfn_outs = (inner_out_mitmot +
                          inner_out_sitsot +
                          inner_out_nitsot)

        local_op = Scan(inner_gfn_ins, inner_gfn_outs, info)
        outputs = local_op(*outer_inputs)
        if type(outputs) not in (list, tuple):
            outputs = [outputs]
        # Re-order the gradients correctly
        gradients = [DisconnectedType()()]

        offset = (self.n_mit_mot +
                  self.n_mit_sot +
                  self.n_sit_sot +
                  n_sitsot_outs)
        for p, (x, t) in enumerate(
            zip(outputs[offset:offset + self.n_seqs],
                type_outs[offset:offset + self.n_seqs])):
            if t == 'connected':
                gradients.append(x[::-1])
            elif t == 'disconnected':
                gradients.append(DisconnectedType()())
            elif t == 'through_shared':
                gradients.append(
                    grad_undefined(self,
                                   p + 1,
                                   inputs[p + 1],
                                   'Depends on a shared variable'))
            else:
                # t contains the "why_null" string of a NullType
                gradients.append(NullType(t)())

        end = self.n_mit_mot + self.n_mit_sot + self.n_sit_sot
        for p, (x, t) in enumerate(
            zip(outputs[:end], type_outs[:end])):
            if t == 'connected':
                gradients.append(x[::-1])
            elif t == 'disconnected':
                gradients.append(DisconnectedType()())
            elif t == 'through_shared':
                gradients.append(
                    grad_undefined(self,
                                   p + 1 + self.n_seqs,
                                   inputs[p + 1 + self.n_seqs],
                                   'Depends on a shared variable'))
            else:
                # t contains the "why_null" string of a NullType
                gradients.append(NullType(t)())

        start = len(gradients)
        node = outs[0].owner
        for idx in xrange(self.n_shared_outs):
            disconnected = True
            connected_flags = self.connection_pattern(node)[idx + start]
            for dC_dout, connected in zip(dC_douts, connected_flags):
                if (not isinstance(dC_dout.type, DisconnectedType) and
                        connected):
                    disconnected = False
            if disconnected:
                gradients.append(DisconnectedType()())
            else:
                gradients.append(grad_undefined(
                    self, idx, inputs[idx],
                    'Shared Variable with update'))

        start = len(gradients)
        gradients += [DisconnectedType()()
                for x in xrange(self.n_nit_sot)]
        begin = end

        end = begin + n_sitsot_outs
        for p, (x, t) in enumerate(
            zip(outputs[begin:end], type_outs[begin:end])):
            if t == 'connected':
                gradients.append(x[-1])
            elif t == 'disconnected':
                gradients.append(DisconnectedType()())
            elif t == 'through_shared':
                gradients.append(
                    grad_undefined(self,
                                   p + begin + 1,
                                   inputs[p + begin + 1],
                                   'Depends on a shared variable'))
            else:
                # t contains the "why_null" string of a NullType
                gradients.append(NullType(t)())

        # Mask disconnected gradients
        # Ideally we would want to assert that the gradients we are
        # replacing do indeed evaluate to 0, though that is not practical
        # from a computational point of view
        # The gradients of scan are computed replacing Disconnected with 0,
        # because through the recurrence they can become nonzero
        for idx in xrange(len(gradients)):
            disconnected = True
            for kdx in xrange(len(node.outputs)):
                if connection_pattern[idx][kdx] and \
                   not isinstance(dC_douts[kdx].type, DisconnectedType):
                    disconnected = False
            if disconnected:
                gradients[idx] = DisconnectedType()()
        return gradients

    def R_op(self, inputs, eval_points):
        # Step 0. Don't work on the orignal tensor variables
        rval = scan_utils.reconstruct_graph(self.inputs,
                                            self.outputs, '_rop')
        self_inputs = rval[0]
        rop_of_inputs = rval[0][:self.n_seqs + self.n_outs] + \
                rval[0][self.n_seqs + self.n_outs + self.n_shared_outs:]
        self_outputs = rval[1]
        # Step 1. Compute the R_op of the inner function
        inner_eval_points = [scan_utils.safe_new(x, '_evalpoint')
                             for x in rop_of_inputs]
        if self.as_while:
            rop_self_outputs = self_outputs[:-1]
        else:
            rop_self_outputs = self_outputs
        if self.info['n_shared_outs'] > 0:
            rop_self_outputs = rop_self_outputs[:-self.info['n_shared_outs']]
        rop_outs = tensor.Rop(rop_self_outputs, rop_of_inputs,
             inner_eval_points)
        if type(rop_outs) not in (list, tuple):
            rop_outs = [rop_outs]
        # Step 2. Figure out what corresponds to what in the scan

        # When doing the R-op of scan, you end up having double of each type of
        # input, because for each sequence you need also its eval point, for
        # each mit_mot, mit_sot, sit_sot or other type of inputs the same.
        # Interestingly enough, all these types of eval points behave the same
        # way as the input to which they correspond
        # The only exception is the eval point for the number of sequences, and
        # evan point for the number of nit_sot which I think should just be
        # ignored (?)
        info = OrderedDict()
        info['n_seqs'] = self.n_seqs * 2
        info['n_mit_sot'] = self.n_mit_sot * 2
        info['n_sit_sot'] = self.n_sit_sot * 2
        info['n_mit_mot'] = self.n_mit_mot * 2
        info['n_nit_sot'] = self.n_nit_sot * 2
        info['n_shared_outs'] = self.n_shared_outs
        info['gpu'] = False
        info['as_while'] = self.as_while
        info['profile'] = self.profile
        info['truncate_gradient'] = self.truncate_gradient
        if self.name:
            info['name'] = 'rop_of_' + self.name
        else:
            info['name'] = None
        info['mode'] = self.mode
        info['allow_gc'] = self.allow_gc
        info['mit_mot_out_slices'] = self.mit_mot_out_slices * 2
        info['destroy_map'] = OrderedDict()
        new_tap_array = []
        b = 0
        e = self.n_mit_mot
        new_tap_array += self.tap_array[b:e] * 2
        b = e
        e += self.n_mit_sot
        new_tap_array += self.tap_array[b:e] * 2
        b = e
        e += self.n_sit_sot
        new_tap_array += self.tap_array[b:e] * 2
        info['tap_array'] = new_tap_array

        # Sequences ...
        b = 1
        ib = 0
        e = 1 + self.n_seqs
        ie = self.n_seqs
        clean_eval_points = []
        for inp, evp in zip(inputs[b:e], eval_points[b:e]):
            if evp is not None:
                clean_eval_points.append(evp)
            else:
                clean_eval_points.append(inp.zeros_like())

        scan_seqs = inputs[b:e] + clean_eval_points
        inner_seqs = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        # MIT_MOT sequences ...
        b = e
        e = e + self.n_mit_mot
        ib = ie
        ie = ie + int(numpy.sum([len(x) for x in
                                 self.tap_array[:self.n_mit_mot]]))
        clean_eval_points = []
        for inp, evp in zip(inputs[b:e], eval_points[b:e]):
            if evp is not None:
                clean_eval_points.append(evp)
            else:
                clean_eval_points.append(inp.zeros_like())

        scan_mit_mot = inputs[b:e] + clean_eval_points
        inner_mit_mot = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        # MIT_SOT sequences ...
        b = e
        e = e + self.n_mit_sot
        ib = ie
        ie = ie + int(numpy.sum([len(x) for x in
                         self.tap_array[self.n_mit_mot:\
                                        self.n_mit_mot + self.n_mit_sot]]))
        clean_eval_points = []
        for inp, evp in zip(inputs[b:e], eval_points[b:e]):
            if evp is not None:
                clean_eval_points.append(evp)
            else:
                clean_eval_points.append(inp.zeros_like())

        scan_mit_sot = inputs[b:e] + eval_points[b:e]
        inner_mit_sot = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        # SIT_SOT sequences ...
        b = e
        e = e + self.n_sit_sot
        ib = ie
        ie = ie + self.n_sit_sot
        clean_eval_points = []
        for inp, evp in zip(inputs[b:e], eval_points[b:e]):
            if evp is not None:
                clean_eval_points.append(evp)
            else:
                clean_eval_points.append(inp.zeros_like())

        scan_sit_sot = inputs[b:e] + clean_eval_points
        inner_sit_sot = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        # Shared outs ...
        b = e
        e = e + self.n_shared_outs
        ib = ie
        ie = ie + self.n_shared_outs
        scan_shared = inputs[b:e]
        inner_shared = self_inputs[ib:ie]

        # NIT_SOT sequences
        b = e
        e = e + self.n_nit_sot
        scan_nit_sot = inputs[b:e] * 2

        # All other arguments
        clean_eval_points = []
        for inp, evp in zip(inputs[e:], eval_points[e:]):
            if evp is not None:
                clean_eval_points.append(evp)
            else:
                clean_eval_points.append(inp.zeros_like())
        scan_other = inputs[e:] + clean_eval_points
        # inner_eval_points do not have entries for shared variables
        inner_other = self_inputs[ie:] + inner_eval_points[ib:]

        # Outputs
        n_mit_mot_outs = int(numpy.sum([len(x) for x in
                                        self.mit_mot_out_slices]))
        info['n_mit_mot_outs'] = n_mit_mot_outs * 2
        b = 0
        e = n_mit_mot_outs
        inner_out_mit_mot = self_outputs[b:e] + rop_outs[b:e]
        b = e
        e = e + self.n_mit_sot
        inner_out_mit_sot = self_outputs[b:e] + rop_outs[b:e]
        b = e
        e = e + self.n_sit_sot
        inner_out_sit_sot = self_outputs[b:e] + rop_outs[b:e]
        b = e
        e = e + self.n_nit_sot
        inner_out_nit_sot = self_outputs[b:e] + rop_outs[b:e]
        b = e
        e = e + self.n_shared_outs
        inner_out_shared = self_outputs[b:e]

        inner_ins = (inner_seqs +
                     inner_mit_mot +
                     inner_mit_sot +
                     inner_sit_sot +
                     inner_shared +
                     inner_other)
        inner_outs = (inner_out_mit_mot +
                      inner_out_mit_sot +
                      inner_out_sit_sot +
                      inner_out_nit_sot +
                      inner_out_shared)

        if self.as_while:
            inner_outs += [self_outputs[-1]]
        scan_inputs = ([inputs[0]] +
                       scan_seqs +
                       scan_mit_mot +
                       scan_mit_sot +
                       scan_sit_sot +
                       scan_shared +
                       scan_nit_sot +
                       scan_other)

        local_op = Scan(inner_ins, inner_outs, info)
        outputs = local_op(*scan_inputs)
        if type(outputs) not in (list, tuple):
            outputs = [outputs]
        # Select only the result of the R_op results
        final_outs = []
        b = self.n_mit_mot
        e = self.n_mit_mot * 2
        final_outs += outputs[b:e]
        b = e + self.n_mit_sot
        e = e + self.n_mit_sot * 2
        final_outs += outputs[b:e]
        b = e + self.n_sit_sot
        e = e + self.n_sit_sot * 2
        final_outs += outputs[b:e]
        b = e + self.n_nit_sot
        e = e + self.n_nit_sot * 2
        final_outs += outputs[b:e]
        final_outs += [None] * self.n_shared_outs

        return final_outs


# Since Scan is an op that contains a Theano compiled function, it is
# useful to let DebugMode know about it.
gof.ops_with_inner_function[Scan] = 'fn'


@theano.compile.profilemode.register_profiler_printer
def profile_printer(fct_name, compile_time, fct_call_time, fct_call,
                    apply_time, apply_cimpl, message, outputs_size,
                    other_time):
    # Scan overhead profile
    if any([isinstance(node.op, Scan) and v > 0 for (_, node), v in
            apply_time.items()]):
        print()
        print('Scan overhead:')
        print ('<Scan op time(s)> <sub scan fct time(s)> <sub scan op '
               'time(s)> <sub scan fct time(% scan op time)> <sub scan '
               'op time(% scan op time)> <node>')
        total_super_scan_time = 0
        total_scan_fct_time = 0
        total_scan_op_time = 0
        for (_, node), v in iteritems(apply_time):
            if isinstance(node.op, Scan):
                if v > 0:
                    scan_fct_time = node.op.mode_instance.fn_time
                    scan_op_time = node.op.mode_instance.local_time
                    total_super_scan_time += v
                    total_scan_fct_time += scan_fct_time
                    total_scan_op_time += scan_op_time
                    print('    %5.1fs  %5.1fs  %5.1fs  %5.1f%%  %5.1f%%' % (
                        v,
                        scan_fct_time,
                        scan_op_time,
                        scan_fct_time / v * 100,
                        scan_op_time / v * 100), node)
                else:
                    print((' The node took 0s, so we can not '
                           'compute the overhead'), node)
        print('    total %5.1fs  %5.1fs  %5.1fs  %5.1f%%  %5.1f%%' % (
            total_super_scan_time,
            total_scan_fct_time,
            total_scan_op_time,
            total_scan_fct_time / total_super_scan_time * 100,
            total_scan_op_time / total_super_scan_time * 100))
