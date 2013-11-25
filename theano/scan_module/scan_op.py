"""
This module provides the Scan Op

See scan.py for details on scan
"""

__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "Frederic Bastien "
               "James Bergstra "
               "Pascal Lamblin ")
__copyright__ = "(c) 2010, Universite de Montreal"
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import itertools
import logging
import time
from itertools import izip

import numpy

import theano
from theano.compile import function, Param, Out
from theano import compile
from theano import gradient
from theano.gof.python25 import any, OrderedDict
from theano.gof import PureOp, Apply
from theano import gof
from theano.tensor import TensorType
from theano import tensor
from theano.tensor.opt import Shape_i
from theano.gradient import grad_undefined
from theano.gradient import DisconnectedType
from theano.compile.profiling import ScanProfileStats

from theano.scan_module import scan_utils
from theano.scan_module.scan_utils import safe_new, forced_replace

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_module.scan_op')


class Scan(PureOp):
    def __init__(self,
                 inputs,
                 outputs,
                 info,
                 typeConstructor=None,
                ):
        """
        :param inputs: inputs of the inner function of scan
        :param outputs: outputs of the inner function of scan
        :param info: dictionary containing different properties of
            the scan op (like number of different types of
            arguments, name, mode, if it should run on GPU or
            not, etc.)
        :param typeConstructor: function that constructs a Theano TensorType
            able to represent a float32 ndarray.

        Note: ``typeConstructor`` had been added to refactor how Theano
        deals with the GPU. If it runs on the GPU, scan needs to construct
        certain outputs (those who reside in the GPU memory) as CudaNdarray.
        However we can not import cuda in this file (as it is in sandbox,
        and not available on each machine) so the workaround is that the GPU
        optimization (which is aware of cuda types) passes to the
        constructor of this class a function that is able to construct
        CudaNdarray. This way the class Scan does not need to be aware of
        CudaNdarray, it just constructs any float32 tensor using this
        function (which by default constructs normal tensors). Note that the
        second assumption in this code is that any float32 output or input
        will be moved on the GPU if the optimization gets applied (following
        Theano's philosophy of moving as much as possible on gpu).
        """
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
            # Scan assumes that only variables of dtype float32 might need a
            # special constructor (i.e. CudaNdarray constructor) when the
            # code is running on GPU, as it is the only type supported by
            # Theano yet. Therefore only for dtype float32 we use the passed
            # type constructor ``typeConstructor``. For anything else we
            # know that even if we run it on the GPU we still construct
            # normal Theano tensors.
            if o.type.dtype in ['float32']:
                self.output_types.append(
                    typeConstructor(
                        broadcastable=(False,) + o.type.broadcastable,
                        dtype=o.type.dtype))
            else:
                self.output_types.append(
                    tensorConstructor(
                        broadcastable=(False,) + o.type.broadcastable,
                        dtype=o.type.dtype))

            idx += len(self.mit_mot_out_slices[jdx])
            jdx += 1

        # mit_sot / sit_sot / nit_sot
        end = idx + self.n_mit_sot + self.n_sit_sot + self.n_nit_sot

        for o in outputs[idx:end]:
            # Scan assumes that only variables of dtype float32 might need a
            # special constructor (i.e. CudaNdarray constructor) when the
            # code is running on GPU, as it is the only type supported by
            # Theano yet. Therefore only for dtype float32 we use the passed
            # type constructor ``typeConstructor``. For anything else we
            # know that even if we run it on the GPU we still construct
            # normal Theano tensors.
            if o.type.dtype in ['float32']:
                self.output_types.append(
                    typeConstructor(
                        broadcastable=(False,) + o.type.broadcastable,
                        dtype=o.type.dtype))
            else:
                self.output_types.append(
                    tensorConstructor(
                        broadcastable=(False,) + o.type.broadcastable,
                        dtype=o.type.dtype))
        # shared outputs + possibly the ending condition
        for o in outputs[end:]:
            self.output_types.append(o.type)

        if self.as_while:
            self.output_types = self.output_types[:-1]

        mode_instance = compile.mode.get_mode(self.mode)
        # if the default mode is used, and that mode is ProfileMode
        # then we need to copy the mode otherwise the time for a given
        # op will be counted multiple times
        if (self.mode is None and
            isinstance(mode_instance, compile.profilemode.ProfileMode)):
            mode_instance = compile.profilemode.ProfileMode(
                optimizer=mode_instance.provided_optimizer,
                linker=mode_instance.provided_linker)
            compile.profilemode.prof_mode_instance_to_print.append(
                                                    mode_instance)
            self.mode_instance = mode_instance
            if self.name:
                self.mode_instance.message = self.name + " sub profile"
            else:
                self.mode_instance.message = "Scan sub profile"
        else:
            self.mode_instance = mode_instance

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
        if not self.info['gpu']:
            tmp_in, tmp_out = scan_utils.reconstruct_graph(self.inputs,
                                                           self.outputs)
            local_fgraph = gof.FunctionGraph(tmp_in, tmp_out, clone=False)
            self._cmodule_key = gof.CLinker().cmodule_key_(local_fgraph, [])
            self._hash_inner_graph = hash(self._cmodule_key)
        else:
            self._hash_inner_graph = self.info['gpu_hash']

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
        err_msg1 = ('When compiling the inner function of scan the '
                    'following error has been encountered: The '
                    '%s %s (argument number %d) has dtype '
                    '%s and %d dimension(s). The corresponding slice %s '
                    'however has dtype %s and %d dimension(s) (it should '
                    'have the same dtype and one fewer dimensions). This '
                    'should never happen, please '
                    'report to theano-dev mailing list'
                   )
        err_msg2 = ('When compiling the inner function of scan the '
                    'following error has been encountered: The '
                    'initial state (outputs_info in scan nomenclature) '
                    'of variable %s (argument number %d)'
                    ' has dtype %s and %d dimension(s), while the result '
                    'of the inner function for this output has dtype %s '
                    'and %d dimension(s). This could happen if the inner '
                    'graph of scan results in an upcast or downcast. '
                    'Please make sure that you use dtypes consistently')

        def format(var, as_var):
            """ This functions ensures that ``out`` has the same dtype as
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
                tmp = as_var.type.__class__(
                    broadcastable=tuple(var.broadcastable[:1])+\
                                  tuple(as_var.broadcastable),
                    dtype=as_var.dtype)
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
                if (inner_mitmot_outs[opos + k].type.dtype != \
                        outer_mitmot.type.dtype or
                    inner_mitmot_outs[opos + k].ndim != \
                         outer_mitmot.ndim - 1):
                    raise ValueError(err_msg2 %
                                      (str(outer_mitmot),
                                       argoffset + idx,
                                       outer_mitmot.type.dtype,
                                       outer_mitmot.ndim,
                                       inner_mitmot_outs[opos + k].type.dtype,
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
            if (inner_mitsot_out.type.dtype != outer_mitsot.type.dtype or
                inner_mitsot_out.ndim != outer_mitsot.ndim - 1):
                raise ValueError(err_msg2 %
                                 (str(outer_mitsot),
                                 argoffset + idx,
                                 outer_mitsot.type.dtype,
                                 outer_mitsot.type.ndim,
                                 inner_mitsot_out.type.dtype,
                                 inner_mitsot_out.type.ndim))

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
            if (inner_sitsot_out.type.dtype != outer_sitsot.type.dtype or
                inner_sitsot_out.ndim != outer_sitsot.ndim - 1):
                raise ValueError(err_msg2 %
                                (str(outer_sitsot),
                                argoffset + idx,
                                outer_sitsot.type.dtype,
                                outer_sitsot.type.ndim,
                                inner_sitsot_out.type.dtype,
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
                (outer_shared.dtype != inner_shared_out.dtype or
                 outer_shared.ndim != inner_shared_out.ndim)):
                raise ValueError(err_msg2 % (str(outer_shared),
                                             idx + argoffset,
                                             outer_shared.dtype,
                                             outer_shared.ndim,
                                             inner_shared_out.dtype,
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
        self.vector_seqs = [seq.ndim == 1 for seq in
                             new_inputs[1:1 + self.n_seqs]]
        self.vector_outs = [arg.ndim == 1 for arg in
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
                         'n_seqs', 'tap_array', 'name',
                         'as_while', 'n_mit_sot', 'destroy_map',
                         'n_nit_sot', 'n_shared_outs',
                         'n_sit_sot', 'gpu', 'n_mit_mot_outs',
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

        if not scan_utils.equal_computations(self.outputs,
                                             other.outputs,
                                             self.inputs,
                                             other.inputs):
            return False

        # If they do, then they need to match in other small details
        # like name, mode, etc.
        return True

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
        return (hash(type(self)) ^
                # and a hash representing the inner graph using the
                # CLinker.cmodule_key_
                self._hash_inner_graph ^
                scan_utils.hash_listsDictsTuples(self.info))

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        """
        :param node: something previously returned by self.make_node

        :param storage_map: dict variable -> one-element-list where a computed
                value for this variable may be found.

        :param compute_map: dict variable -> one-element-list where a boolean
                value will be found.  The boolean indicates whether the
                variable's storage_map container contains a valid value (True)
                or if it has not been computed yet (False).

        :param no_recycling: list of variables for which it is forbidden to
                reuse memory allocated by a previous call.

        :note: If the thunk consults the storage_map on every call, it is safe
            for it to ignore the no_recycling argument, because elements of the
            no_recycling list will have a value of None in the storage map.  If
            the thunk can potentially cache return values (like CLinker does),
            then it must not do so for variables in the no_recycling list.
        """
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
        wrapped_inputs = [Param(x, borrow=True) for x in self.inputs]
        wrapped_outputs = [Out(x, borrow=False) for x in
                           self.outputs[:slices]]
        wrapped_outputs += self.outputs[slices:]
        profile = None
        if (theano.config.profile or
            (isinstance(self.profile, (basestring, bool, int))
                                      and self.profile)):
            if isinstance(self.profile, basestring):
                profile = ScanProfileStats(name=self.profile)
            else:
                profile = ScanProfileStats(name=self.name)
        elif self.profile:
            profile = self.profile
        self.fn = function(wrapped_inputs,
                           wrapped_outputs,
                           mode=self.mode_instance,
                           name=self.name,
                           profile=profile,
                           on_unused_input='ignore')

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
            for _d0 in range(d0):
                for _d1 in range(cython_tap_array_len[_d0]):
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
            for _d0 in range(d0):
                for _d1 in range(cython_mit_mot_out_nslices[_d0]):
                    cython_mit_mot_out_slices[_d0, _d1] = \
                        self.mit_mot_out_slices[_d0][_d1]
            vector_seqs = [seq.ndim == 1 for seq in
                                 node.inputs[1:1 + self.n_seqs]]
            vector_outs = [arg.ndim == 1 for arg in
                                 node.inputs[1 + self.n_seqs:
                                             (1 + self.n_seqs + self.n_outs)]]
            vector_outs += [False] * self.n_nit_sot

            cython_vector_seqs = numpy.asarray(self.vector_seqs,
                                                    dtype='int32')
            cython_vector_outs = numpy.asarray(self.vector_outs,
                                                    dtype='int32')

            if hasattr(self, 'destroy_map'):
                cython_destroy_map = [x in self.destroy_map
                                  for x in xrange(len(node.outputs))]
            else:
                cython_destroy_map = [0 for x in xrange(len(node.outputs))]
            cython_destroy_map = numpy.asarray(cython_destroy_map,
                                               dtype='int32')
            import scan_perform_ext
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
                        self.fn.fn,
                        self.fn,
                        cython_destroy_map,
                        args,
                        outs,
                        self, node)
        except (ImportError, theano.gof.cmodule.MissingGXX):
            p = self.execute
        # default arguments are stored in the closure of `rval`

        def rval(p=p, i=node_input_storage, o=node_output_storage, n=node):
            r = p(n, [x[0] for x in i], o)
            for o in node.outputs:
                compute_map[o][0] = True
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
            n_steps = abs(n_steps)
            for idx, seq in enumerate(args[1:self.seqs_arg_offset]):
                if seq.shape[0] < n_steps:
                    raise ValueError(('Sequence is shorter then the required '
                                     'number of steps : (n_steps, seq, '
                                      'seq.shape):'), n_steps,
                                      node.inputs[1 + idx],
                                      seq.shape)
                seqs.append(seq[::-1])
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
        output_storage = self.fn.output_storage
        fn = self.fn.fn
        offset = (self.n_seqs + sum(map(len, self.tap_array[:self.n_outs])) +
                    self.n_shared_outs)
        for idx in xrange(len(other_args)):
            input_storage[idx + offset].storage[0] = other_args[idx]

        i = 0
        cond = True
        ############## THE MAIN LOOP #########################
        #for i in xrange(n_steps):
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
            for idx in xrange(self.n_mit_mot_outs):
                output_storage[idx].storage[0] = None

            offset = self.n_mit_mot_outs
            if i != 0 and self.n_nit_sot > 0:
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

            offset += self.n_outs + self.n_nit_sot - self.n_mit_mot
            for idx in xrange(self.n_shared_outs):
                output_storage[idx + offset].storage[0] = None
            # If condition add it to the mix
            if self.as_while:
                pdx = offset + self.n_shared_outs
                output_storage[pdx].storage[0] = None
            # 5. compute outputs
            t0_fn = time.time()
            try:
                fn()
            except Exception:
                if hasattr(fn, 'position_of_error'):
                    # this is a new vm-provided function or c linker
                    # they need this because the exception manipulation
                    # done by raise_with_op is not implemented in C.
                    if hasattr(self.fn, 'thunks'):
                        # For the CVM
                        gof.vm.raise_with_op(self.fn.nodes[self.fn.position_of_error],
                                             self.fn.thunks[self.fn.position_of_error])
                    else:
                        # For the c linker
                        # We don't have access from python to all the temps values
                        # So for now, we just don't print the extra shapes/strides info
                        gof.vm.raise_with_op(self.fn.nodes[self.fn.position_of_error])
                else:
                    # old-style linkers raise their own exceptions
                    raise
            dt_fn = time.time() - t0_fn
            if self.as_while:
                pdx = offset + self.n_shared_outs
                cond = output_storage[pdx].storage[0] == 0

            t_fn += dt_fn
            offset_out = 0
            # 5.1 Copy over the values for mit_mot outputs
            for j in xrange(self.n_mit_mot):
                for k in self.mit_mot_out_slices[j]:
                    outs[j][0][k + pos[j]] = \
                            output_storage[offset_out].storage[0]
                    offset_out += 1

            # 5.2 Copy over the values for mit_sot/sit_sot outputs
            begin = self.n_mit_mot
            end = self.n_outs
            offset_out -= self.n_mit_mot

            for j in xrange(begin, end):
                if (store_steps[j] == 1 or self.vector_outs[j] or
                    outs[j][0][pos[j]] is not
                      output_storage[offset_out + j].storage[0]):
                    outs[j][0][pos[j]] = \
                            output_storage[offset_out + j].storage[0]

            # 5.3 Copy over the values for nit_sot outputs
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
                elif (store_steps[j] == 1 or self.vector_outs[j] or
                      outs[j][0][pos[j]] is not
                      output_storage[j + offset_out].storage[0]):
                    outs[j][0][pos[j]] = \
                            output_storage[j + offset_out].storage[0]

            # 5.4 Copy over the values for outputs corresponding to shared
            # variables
            begin = end
            end += self.n_shared_outs
            for j in xrange(begin, end):
                jout = j + offset_out
                outs[j][0] = output_storage[jout].storage[0]

            pos = [(idx + 1) % store for idx, store in
                               itertools.izip(pos, store_steps)]
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
        #if hasattr(self.fn.maker.mode,'fct_call_time'):
        #    self.fn.maker.mode.fct_call_time[self.fn] += t_fn
        #    self.fn.maker.mode.fct_call[self.fn] += n_steps

        #self.fn.maker.mode.call_time += t_fn
        #self.fn.maker.mode.fn_time += t_fn
        # Old Profile Mode */
        self.t_call = t_call
        self.t_fn = t_fn

    ### Infer Shape
    def infer_shape(self, node, input_shapes):
        # input_shapes correspond to the shapes of node.inputs
        # Here, we build a list inner_ins_shape, such that inner_ins_shape[i]
        # is the shape of self.inputs[i]

        for inp, inp_shp in izip(node.inputs, input_shapes):
            assert inp_shp is None or len(inp_shp) == inp.type.ndim

        # sequences
        # We skip iputs_shapes[0] as it is the total or current number
        # of iterations.
        seqs_shape = [x[1:] for x in input_shapes[1:1 + self.n_seqs]]

        # mit_mot, mit_sot, sit_sot
        n_outs = self.n_mit_mot + self.n_mit_sot + self.n_sit_sot
        outs_shape = []
        for idx in xrange(n_outs):
            for k in self.tap_array[idx]:
                outs_shape += [input_shapes[idx + self.n_seqs + 1][1:]]

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
        out_equivalent = OrderedDict()
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
            scan_outs = [(Shape_i(0)(o),) + x[1:]
                         for o, x in izip(node.outputs, scan_outs)]
        return scan_outs

    def get_input_pos(self, output_index):
        """ For a given ``output_index``, an index in the inner outputs of
        scan, find a corresponding first index in the inner inputs of scan
        """
        ipos = self.n_seqs
        opos = output_index
        for otaps, itaps in zip(self.mitmot_out_taps(), self.mitmot_taps()):
            if len(otaps) > opos:
                return ipos
            else:
                opos = opos - len(otaps)
                ipos += len(itaps)
        for dx, taps in enumerate(self.mitsot_taps()):
            if opos == 0:
                return ipos
            else:
                opos = opos - 1
                ipos += len(taps)
        if opos < self.info['n_sit_sot']:
            return ipos + opos
        else:
            return -1

    def get_output_pos(self, input_index):
        """ For a given ``input_index``, an index in the inner inputs of
        scan, find a corresponding first index in the inner outputs of scan
        """
        ipos = input_index
        opos = 0
        for otaps, itaps in zip(self.mitmot_out_taps(), self.mitmot_taps()):
            if len(itaps) > ipos:
                return opos
            else:
                opos += len(otaps)
                ipos -= len(itaps)
        for dx, taps in enumerate(self.mitsot_taps()):
            if len(taps) > ipos:
                return opos
            else:
                opos += 1
                ipos -= len(taps)
        if ipos < self.info['n_sit_sot']:
            return ipos + opos
        else:
            return -1

    def get_output_slice_idx(self, output_index):
        """ For an ``output_index``, an index in the outter ouputs of scan,
        find a corresponding index in the inner outputs of scan.
        """
        ipos = 0
        opos = output_index
        for otaps in zip(self.mitmot_out_taps()):
            if len(otaps) > 0:
                return ipos
            else:
                opos = opos - 1
                ipos += len(otaps)
        return ipos + opos

    def connection_pattern(self, node):
        # The gradient wrt to n_steps is disconnected
        connection_pattern = [[False for output in node.outputs]]
        connection_pattern += [[False for output in node.outputs]
                              for x in node.inputs[1:]]

        def compute_gradient(y, g_y, diff_inputs):
            rval = []
            gmp = OrderedDict()
            consider_inps = [x for x in theano.gof.graph.inputs([y])
                             if x in diff_inputs]
            for x in consider_inps:
                try:
                    gmp[x] = gradient.grad(cost=None,
                                           known_grads={y: g_y}, wrt=x)
                except gradient.NullTypeGradError:
                    # It means the gradient is undefined (which implies
                    # is connected)
                    gmp[x] = x
                except gradient.DisconnectedInputError:
                    gmp[x] = None
            return [gmp.get(p, None) for p in diff_inputs]

        def _get_inner_outs(oidx):
            s = 0
            if self.n_mit_mot > 0:
                e = len(self.mitmot_out_taps()[0])
            else:
                e = 1
            for p in xrange(oidx):
                s = e
                if p < self.n_mit_mot:
                    e += len(self.mitmot_out_taps()[p])
                else:
                    e += 1
            return self.outputs[s:e]

        def _get_inner_inps(iidx):
            s = 0
            if self.n_seqs > 0:
                e = 1
            else:
                e = len(self.tap_array[0])
            p = iidx
            if node.inputs[iidx + 1] in self.outer_nitsot(node):
                return None
            if node.inputs[iidx + 1] in self.outer_non_seqs(node):
                loc_idx = self.outer_non_seqs(node).index(
                    node.inputs[iidx + 1])
                return [self.inner_non_seqs(self.inputs)[loc_idx]]

            for p in xrange(iidx):
                s = e
                if p < self.n_seqs:
                    e += 1
                elif p - self.n_seqs < len(self.tap_array):
                    e += len(self.tap_array[p - self.n_seqs])
                else:
                    e += 1

            return self.inputs[s:e]
        for oidx, out in enumerate(node.outputs):
            for iidx, inp in enumerate(node.inputs[1:]):
                ols = _get_inner_outs(oidx)
                ils = _get_inner_inps(iidx)

                if ils is None:
                    # The gradient should be disconnected
                    connection_pattern[iidx + 1][oidx] = False
                else:
                    for inner_out in ols:
                        # We check for the dtype because inner_out could be
                        # any Theano type like Generic or RandomState, for
                        # which we can not impose a dtype
                        if hasattr(inner_out, 'dtype'):
                            # Note that we do not care about the output of
                            # this compute gradient. We just care to see if
                            # it is None or not. (i.e. disconnected or not)
                            try:
                                old = theano.config.compute_test_value
                                theano.config.compute_test_value = 'off'
                                tmp = compute_gradient(
                                    inner_out,
                                    safe_new(inner_out, dtype='float64'),
                                    ils)
                            finally:
                                theano.config.compute_test_value = old
                        else:
                            # It should be undefined not disconnected
                            tmp = ils
                        if any([x is not None for x in tmp]):
                            connection_pattern[iidx + 1][oidx] = True
        # Applying Floyd-Warshall to find all paths connecting inputs to
        # outputs. Note that if `x` is an input to `y_t` and `y_tm1` is an
        # input to `z_t` then `x` is an input to `z_t`.

        n_outs = len(node.outputs)
        for steps in xrange(n_outs):
            for iidx in xrange(n_outs):
                for jidx in xrange(n_outs):
                    j_inp_idx = self.get_input_pos(jidx) + 1
                    if connection_pattern[j_inp_idx][iidx] == True:
                        for k in xrange(len(connection_pattern)):
                            if connection_pattern[k][jidx]:
                                connection_pattern[k][iidx] = True
        return connection_pattern

    ### GRAD FUNCTION
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

        rval = scan_utils.reconstruct_graph(self.inputs,
                                            self.outputs)
        self_inputs = rval[0]
        self_outputs = rval[1]
        #differentiable inputs
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

        def compute_gradient(y, g_y):
            if 'int' in str(g_y.dtype):
                raise TypeError("Gradients may never be integers but g_y "
                        "has type " + str(g_y.type))

            odx = get_out_idx(self_outputs.index(y))
            wrt = [x for x in theano.gof.graph.inputs([y])
                   if (x in diff_inputs) and
                   (connection_pattern[
                       get_inp_idx(self_inputs.index(x))][odx])]
            grads = gradient.grad(
                cost=None,
                known_grads={y: g_y},
                wrt=wrt,
                consider_constant=wrt,
                disconnected_inputs='ignore',
                return_disconnected='None')
            gmp = dict(zip(wrt, grads))
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
            if Xt not in self.inner_nitsot_outs(self_outputs):
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
                        oidx = self.get_output_pos(pos)
                        if not isinstance(dC_douts[oidx].type,
                                          DisconnectedType):
                            dtypes.append(dC_douts[oidx].dtype)
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
            _dC_dinps_t = compute_gradient(Xt, dC_dXt)
            for jdx in xrange(len(_dC_dinps_t)):
                if dC_dinps_t[jdx] is None:
                    dC_dinps_t[jdx] = _dC_dinps_t[jdx]
                elif _dC_dinps_t[jdx]:
                    dC_dinps_t[jdx] += _dC_dinps_t[jdx]
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
            opos = self.get_output_pos(pos)
            if opos >= 0:
                dC_dXtm1s.append(safe_new(dC_dXts[opos]))
                if x.dtype != dC_dXts[opos].dtype:
                    dC_dinps_t[pos + self.n_seqs] = \
                            x.astype(dC_dXts[opos].dtype)
            else:
                dC_dXtm1s.append(safe_new(x))
        for dx, dC_dXtm1 in enumerate(dC_dXtm1s):
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
                if outmaxtap -k != 0:
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
            undefined = False
            disconnected = True
            for jdx in xrange(len(self.mit_mot_out_slices[idx])):
                inner_inp_mitmot.append(dC_dXts[out_pos])
                mitmot_inp_taps[idx].append(-self.mit_mot_out_slices[idx][jdx])
                n_mitmot_inps += 1
                out_pos += 1

            for jdx in xrange(len(self.tap_array[idx])):
                inner_inp_mitmot.append(dC_dXtm1s[ins_pos - self.n_seqs])
                inner_out_mitmot.append(dC_dinps_t[ins_pos])
                if not disconnected_dC_dinps_t[ins_pos]:
                    disconnected = False

                for _sh in self.inner_shared(self_inputs):
                    if _sh in gof.graph.inputs([dC_dinps_t[ins_pos]]):
                        undefined = True

                n_mitmot_inps += 1
                ins_pos += 1
                n_mitmot_outs += 1
                mitmot_inp_taps[idx].append(-self.tap_array[idx][jdx])
                mitmot_out_taps[idx].append(-self.tap_array[idx][jdx])
            if undefined:
                type_outs.append('undefined')
            elif disconnected:
                type_outs.append('disconnected')
            else:
                type_outs.append('connected')

        offset = self.n_mit_mot
        for idx in xrange(self.n_mit_sot):
            mitmot_inp_taps.append([])
            mitmot_out_taps.append([])
            outer_inp_mitmot.append(dC_douts[idx + offset][::-1])
            idx_tap = idx + self.n_mit_mot
            inner_inp_mitmot.append(dC_dXts[out_pos])
            out_pos += 1
            n_mitmot_inps += 1
            undefined = False
            disconnected = True
            mitmot_inp_taps[idx + offset].append(0)
            for jdx in xrange(len(self.tap_array[idx_tap])):
                inner_inp_mitmot.append(dC_dXtm1s[ins_pos - self.n_seqs])
                inner_out_mitmot.append(dC_dinps_t[ins_pos])
                mitmot_inp_taps[idx + offset].append(
                    -self.tap_array[idx_tap][jdx])
                mitmot_out_taps[idx].append(
                    -self.tap_array[idx_tap][jdx])
                if not disconnected_dC_dinps_t[ins_pos]:
                    disconnected = False
                for _sh in self.inner_shared(self_inputs):
                    if _sh in gof.graph.inputs([dC_dinps_t[ins_pos]]):
                        undefined = True
                n_mitmot_inps += 1
                ins_pos += 1
                n_mitmot_outs += 1
            if undefined:
                type_outs.append('undefined')
            elif disconnected:
                type_outs.append('disconnected')
            else:
                type_outs.append('connected')

        offset += self.n_mit_sot
        for idx in xrange(self.n_sit_sot):
            mitmot_inp_taps.append([0, 1])
            mitmot_out_taps.append([1])
            undefined = False
            if not isinstance(dC_douts[idx + offset].type, DisconnectedType):
                outer_inp_mitmot.append(dC_douts[idx + offset][::-1])
            else:
                outer_inp_mitmot.append(
                    tensor.zeros(outs[idx + offset].shape,
                                 dtype=dC_dinps_t[ins_pos].dtype))
            inner_out_mitmot.append(dC_dinps_t[ins_pos])
            for _sh in self.inner_shared(self_inputs):
                if _sh in gof.graph.inputs([dC_dinps_t[ins_pos]]):
                    undefined = True
            if undefined:
                type_outs.append('undefined')
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

        if self.truncate_gradient != -1:
            grad_steps = tensor.minimum(grad_steps, self.truncate_gradient)

        n_nit_sot = self.n_seqs
        inner_out_nitsot = dC_dinps_t[:self.n_seqs]
        inner_out_sitsot = dC_dinps_t[ins_pos:]
        for _p, vl in enumerate(inner_out_sitsot):
            undefined = False
            for _sh in self.inner_shared(self_inputs):
                if _sh in gof.graph.inputs([vl]):
                    undefined = True
            if undefined:
                type_outs.append('undefined')
            elif disconnected_dC_dinps_t[_p + ins_pos]:
                type_outs.append('disconnected')
            else:
                type_outs.append('connected')

        for _p, vl in enumerate(inner_out_nitsot):
            undefined = False
            for _sh in self.inner_shared(self_inputs):
                if _sh in gof.graph.inputs([vl]):
                    undefined = True
            if undefined:
                type_outs.append('undefined')
            elif disconnected_dC_dinps_t[_p]:
                type_outs.append('disconnected')
            else:
                type_outs.append('connected')
        inner_inp_sitsot = dC_dXtm1s[ins_pos - self.n_seqs:]
        outer_inp_sitsot = [
            tensor.zeros([grad_steps + 1] +
                         [x.shape[i] for i in xrange(x.ndim)],
                         dtype=y.dtype)
            for y, x in zip(inner_inp_sitsot,
                            self.outer_non_seqs(inputs))]

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
            if t == 'disconnected':
                gradients.append(DisconnectedType()())
            elif t == 'undefined':
                gradients.append(
                    grad_undefined(self,
                                   p + 1,
                                   inputs[p + 1],
                                   'Depends on a shared variable'))
            else:
                gradients.append(x[::-1])
        end = self.n_mit_mot + self.n_mit_sot + self.n_sit_sot
        for p, (x, t) in enumerate(
            zip(outputs[:end], type_outs[:end])):
            if t == 'disconnected':
                gradients.append(DisconnectedType()())
            elif t == 'undefined':
                gradients.append(
                    grad_undefined(self,
                                   p + 1 + self.n_seqs,
                                   inputs[p + 1 + self.n_seqs],
                                   'Depends on a shared variable'))
            else:
                gradients.append(x[::-1])

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
            if t == 'disconnected':
                gradients.append(DisconnectedType()())
            elif t == 'undefined':
                gradients.append(
                    grad_undefined(self,
                                   p + begin + 1,
                                   inputs[p + begin + 1],
                                   'Depends on a shared variable'))
            else:
                gradients.append(x[-1])
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

        #SIT_SOT sequences ...
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
        print
        print 'Scan overhead:'
        print ('<Scan op time(s)> <sub scan fct time(s)> <sub scan op '
               'time(s)> <sub scan fct time(% scan op time)> <sub scan '
               'op time(% scan op time)> <node>')
        total_super_scan_time = 0
        total_scan_fct_time = 0
        total_scan_op_time = 0
        for (_, node), v in apply_time.items():
            if isinstance(node.op, Scan):
                if v > 0:
                    scan_fct_time = node.op.mode_instance.fn_time
                    scan_op_time = node.op.mode_instance.local_time
                    total_super_scan_time += v
                    total_scan_fct_time += scan_fct_time
                    total_scan_op_time += scan_op_time
                    print '    %5.1fs  %5.1fs  %5.1fs  %5.1f%%  %5.1f%%' % (
                        v,
                        scan_fct_time,
                        scan_op_time,
                        scan_fct_time / v * 100,
                        scan_op_time / v * 100), node
                else:
                    print (' The node took 0s, so we can not '
                           'compute the overhead'), node
        print '    total %5.1fs  %5.1fs  %5.1fs  %5.1f%%  %5.1f%%' % (
            total_super_scan_time,
            total_scan_fct_time,
            total_scan_op_time,
            total_scan_fct_time / total_super_scan_time * 100,
            total_scan_op_time / total_super_scan_time * 100)
