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
from theano.gof.python25 import any
from theano.gof import PureOp, Apply
from theano import gof
from theano.tensor import TensorType
from theano import tensor
from theano.tensor.opt import Shape_i
#from theano.sandbox import cuda
from theano.compile.profiling import ScanProfileStats

import scan_utils
from scan_utils import safe_new

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
        :param properties: dictionary containing different properties of
                        the scan op.
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
        if typeConstructor is None:
            typeConstructor = lambda broadcastable, dtype: TensorType(
                broadcastable=broadcastable, dtype=dtype)

        while idx < self.n_mit_mot_outs:
            # Not that for mit_mot there are several output slices per
            # output sequence
            o = outputs[idx]
            self.output_types.append(
                typeConstructor(
                    broadcastable=(False,) + o.type.broadcastable,
                    dtype=o.type.dtype)
                        )
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
        self.destroy_map = {}

        if hasattr(self, 'inplace') and self.inplace:
            for idx in xrange(self.n_mit_mot + self.n_mit_sot +
                              self.n_sit_sot):
                self.destroy_map[idx] = [idx + 1 + self.n_seqs]

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
            local_env = gof.Env(tmp_in, tmp_out)
            self._cmodule_key = gof.CLinker.cmodule_key_(local_env, [])
            self._hash_inner_graph = hash(self._cmodule_key)
        else:
            self._hash_inner_graph = self.info['gpu_hash']

    def make_node(self, *inputs):
        """
        Conventions:
            inner_? - the variable corresponding to ? in the inner function
                      of scan (the lambda function executed at every time
                      step)
            outer_? - the variable corresponding to ? in the outer graph,
                      i.e. the main graph (where the scan op lives)
            inner_?_out - the variable representing the new value of ? after
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

        # assert dtype is consistent
        err_msg1 = ('When compiling the inner function of scan the '
                    'following error has been encountered: The '
                    '%s %s (argument number %d) has dtype '
                    '%s and %d dimension(s). The corresponding slice %s '
                    'however has dtype %s and %d dimension(s). This '
                    'should never happen, please '
                    'report to theano-dev mailing list'
                   )
        err_msg2 = ('When compiling the inner function of scan the '
                    'following error has been encountered: The '
                    'initial state (outputs_info in scan nomenclature)'
                    'of variable %s (argument number %d)'
                    ' has dtype %s and %d dimension(s), while the result '
                    'of the inner function for this output has dtype %s '
                    'and %d dimension(s). This could happen if the inner '
                    'graph of scan results in an upcast or downcast. '
                    'Please make sure that you use dtypes consistently')
        # TODO make the assert exact
        # TODO assert the type(dtype, nbdim of self.inputs and
        #      inputs correspond)
        #assert len(inputs) >= len(self.inputs)
        #if self.info['as_while']:
        #    assert len(inputs) == len(self.inputs) + 2 + \
        #       self.info["n_nit_sot"]
        #else:
        #    assert len(inputs) == len(self.inputs) + 1 + \
        #       self.info["n_nit_sot"]
        # Flags that indicate which inputs are vectors

        self.vector_seqs = [seq.ndim == 1 for seq in
                             inputs[1:1 + self.n_seqs]]
        self.vector_outs = [arg.ndim == 1 for arg in
                             inputs[1 + self.n_seqs: (1 + self.n_seqs +
                                                    self.n_outs)]]
        self.vector_outs += [False] * self.n_nit_sot

        # Check if input sequences and variables representing a slice of
        # them have the same dtype
        argoffset = 0
        for idx, (inner_seq, outer_seq) in enumerate(
                                    zip(self.inner_seqs(self.inputs),
                                        self.outer_seqs(inputs))):
            if inner_seq.type.dtype != outer_seq[idx].type.dtype:
                raise ValueError(err_msg1 % ('sequence',
                                             str(outer_seq),
                                             idx,
                                             outer_seq.type.dtype,
                                             str(inner_seq),
                                             inner_seq.type.dtype))
        argoffset += len(self.outer_seqs(inputs))
        # Check that this 3 things have the same dtype for mit_mot:
        #   - initial state of the output
        #   - variable representing an input slice of the otuput
        #   - variable representing an output slice of the otuput
        ipos = 0
        opos = 0
        inner_mitmot = self.inner_mitmot(self.inputs)
        inner_mitmot_outs = self.inner_mitmot_outs(self.outputs)
        for idx, (itaps, otaps, outer_mitmot) in enumerate(
                                     zip(self.mitmot_taps(),
                                         self.mitmot_out_taps(),
                                         self.outer_mitmot(inputs))):
            for k in xrange(len(itaps)):
                if (inner_mitmot[ipos + k].type.dtype !=
                    outer_mitmot.type.dtype or
                    inner_mitmot[ipos + k].ndim != outer_mitmot.ndim - 1):
                    raise ValueError(err_msg1 % ('initial state (outputs_info'
                                           ' in scan nomenclature) ',
                                           str(outer_mitmot),
                                           argoffset + idx,
                                           outer_mitmot.type.dtype,
                                           str(inner_mitmot[ipos + k]),
                                           inner_mitmot[ipos + k].type.dtype))
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
                                       inner_mitmot_outs[opos + k].type.dtype))
            opos += len(otaps)
        argoffset += len(self.outer_mitmot(inputs))
        # Same checks as above but for outputs of type mit_sot
        ipos = 0
        inner_mitsots = self.inner_mitsot(self.inputs)
        for idx, (itaps, outer_mitsot, inner_mitsot_out) in enumerate(
            zip(self.mitsot_taps(),
                self.outer_mitsot(inputs),
                self.inner_mitsot_outs(self.outputs))):
            for k in xrange(len(itaps)):
                if (inner_mitsots[ipos + k].type.dtype != \
                        outer_mitsot.type.dtype or
                    inner_mitsots[ipos + k].ndim != outer_mitsot.ndim - 1):
                    raise ValueError(err_msg1 % ('initial state (outputs_info'
                                               ' in scan nomenclature) ',
                                           str(outer_mitsot),
                                           argoffset + idx,
                                           outer_mitsot.type.dtype,
                                           otuer_mitsot.type.ndim,
                                           str(inner_mitsot[ipos + k]),
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
        for idx, (inner_sitsot, outer_sitsot, inner_sitsot_out) in enumerate(
            zip(self.inner_sitsot(self.inputs),
                self.outer_sitsot(inputs),
                self.inner_sitsot_outs(self.outputs))):
            if (inner_sitsot.type.dtype != outer_sitsot.type.dtype or
                inner_sitsot.ndim != outer_sitsot.ndim - 1):
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
        for idx, (inner_shared, inner_shared_out, outer_shared) in enumerate(
            zip(self.inner_shared(self.inputs),
                self.inner_shared_outs(self.outputs),
                self.outer_shared(inputs))):
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
        for inner_nonseq, outer_nonseq in zip(
                            self.inner_non_seqs(self.inputs),
                            self.outer_non_seqs(inputs)):
            if (inner_nonseq.type.dtype != outer_nonseq.type.dtype or
                inner_nonseq.type.ndim != outer_nonseq.type.ndim):

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

        apply_node = Apply(self,
                           inputs,
                           [t() for t in self.output_types])
        return apply_node

    def __eq__(self, other):
        # Check if we are dealing with same type of objects
        if not type(self) == type(other):
            return False
        # This are some safety checks ( namely that the inner graph has the
        # same number of inputs and same number of outputs )
        elif not len(self.inputs) == len(other.inputs):
            return False
        elif not len(self.outputs) == len(other.outputs):
            return False
        elif self.info != other.info:
            return False
        else:
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

        if self.inplace:
            aux_txt = '%s{inplace,%s,%s}' % (name, gpu_str, str(self.name))
        else:
            aux_txt = '%s{%s,%s}' % (name, gpu_str, str(self.name))

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
        wrapped_outputs = [Out(x, borrow=True) for x in
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
                           profile=profile)

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
                        self.inplace,
                        args,
                        outs,
                        self)
        except ImportError:
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
        # In order to be able to allocate cuda ndarrays if needed
        from theano.sandbox import cuda
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
        # 2.1 Create storage space for outputs
        for idx in xrange(self.n_outs):
            if self.inplace:
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
                    # this is a new vm-provided function
                    # the C VM needs this because the exception manipulation
                    # done by raise_with_op is not implemented in C.
                    gof.vm.raise_with_op(fn.nodes[fn.position_of_error])
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
                        if self.gpu:
                            _cuda = cuda.cuda_ndarray.cuda_ndarray.CudaNdarray
                            outs[j][0] = _cuda.zeros(shape)
                        else:
                            outs[j][0] = numpy.zeros(shape, dtype)
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
            min_tap = self.mintaps[idx]
            if (store_steps[idx] < i - self.mintaps[idx] and
                pos[idx] < store_steps[idx]):

                pdx = pos[idx]
                if pdx < store_steps[idx] // 2:
                    shape = (pdx,) + outs[idx][0].shape[1:]
                    if cuda.cuda_available and isinstance(outs[idx][0],
                                                          cuda.CudaNdarray):
                        _cuda = cuda.cuda_ndarray.cuda_ndarray.CudaNdarray
                        tmp = _cuda.zeros(shape)
                    else:
                        tmp = numpy.empty(shape)
                    tmp[:] = outs[idx][0][:pdx]
                    outs[idx][0][:store_steps[idx] - pdx] = outs[idx][0][pdx:]
                    outs[idx][0][store_steps[idx] - pdx:] = tmp
                    del tmp
                else:
                    shape = (store_steps[idx] - pdx,) + outs[idx][0].shape[1:]
                    if cuda.cuda_available and isinstance(outs[idx][0],
                                                          cuda.CudaNdarray):
                        _cuda = cuda.cuda_ndarray.cuda_ndarray.CudaNdarray
                        tmp = _cuda.zeros(shape)
                    else:
                        tmp = numpy.empty(shape)
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
                    # outs[idx][0] has more then n_steps entries, with the
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
        out_equivalent = {}
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

    ### GRAD FUNCTION
    def grad(self, args, g_outs):
        # 1. forward pass - get the outputs after applying scan
        scan_outputs = self(*args)
        # 2. make sure they are given as a list
        if not(type(scan_outputs) in (list, tuple)):
            scan_outputs = [scan_outputs]
        # 3. un-group / unzip the inputs
        # Note ! We don't want to use the actual same variable as the ones
        # used by the original scan, rather create clones of them

        rval = scan_utils.reconstruct_graph(self.inputs,
                                            self.outputs, '_grad')
        self_inputs = rval[0]
        self_outputs = rval[1]

        seqs = self_inputs[:self.n_seqs]

        offset = self.n_seqs
        n_ins_mit_mot = numpy.sum([0] + [len(self.tap_array[x]) for x
                                   in xrange(self.n_mit_mot)])
        outs_mit_mot = self_inputs[offset:offset + n_ins_mit_mot]

        offset += n_ins_mit_mot
        n_ins_mit_sot = numpy.sum([0] + [len(self.tap_array[x]) for x
                                   in xrange(self.n_mit_mot,
                                             self.n_mit_mot + self.n_mit_sot)])
        outs_mit_sot = self_inputs[offset:offset + n_ins_mit_sot]
        offset += n_ins_mit_sot
        outs_sit_sot = self_inputs[offset:offset + self.n_sit_sot]
        offset += self.n_sit_sot
        old_scan_shared_ins = self_inputs[offset:offset + self.n_shared_outs]
        out_offset = (self.n_mit_mot_outs +
                      self.n_mit_sot +
                      self.n_nit_sot +
                      self.n_sit_sot)
        # shared variables as well as the condition
        old_scan_shared_outs = self_outputs[out_offset:]
        arg_offset = (1 +
                      self.n_seqs +
                      self.n_mit_mot +
                      self.n_mit_sot +
                      self.n_sit_sot)
        old_scan_init = args[arg_offset: arg_offset + self.n_shared_outs]
        offset += self.n_shared_outs
        other_args = self_inputs[offset:]

        # 4. Collect (possibly) differentiable inputs
        diff_inputs = (seqs +
                       outs_mit_mot +
                       outs_mit_sot +
                       outs_sit_sot +
                       other_args)
                       #args[-len(other_args):]    )

        # 5. construct the function that computes the gradient (we sum over
        # the gradients with respect to all outputs)
        def compute_gradient(y, g_y):
            gmp = gradient.grad_sources_inputs(
                        [(y, g_y)], diff_inputs, False)
            return [gmp.get(p, None) for p in diff_inputs]

        # 6. clean the outputs (i.e. remove update rules)
        end = (self.n_mit_mot_outs +
               self.n_mit_sot +
               self.n_sit_sot +
               self.n_nit_sot)
        clean_outputs = self_outputs[:end]
        g_outs_no_shared = g_outs[:end]

        # 7.1. empty lists to hold gradients
        # List of slices from outputs (used to compute the gradients)
        inner_g_outs = []
        g_out_slices = []
        # List of outputs of the gradient function
        inner_gfn_outs = []
        # slices of the input
        prev_inner_gfn_outs = []
        zeros_like_diff_ins = []
        pos = (self.n_seqs +
               n_ins_mit_mot +
               n_ins_mit_sot +
               self.n_sit_sot)
        offset = len(args) - len(other_args) - pos
        # 7.2. generate variables to represent previous steps of g_outs
        for idx, diff_in in enumerate(diff_inputs):
            prev_gfn_out = safe_new(diff_in)
            if hasattr(diff_in, 'name') and diff_in.name:
                prev_gfn_out.name = 'g_prev_' + diff_in.name
            else:
                prev_gfn_out.name = 'g_prev_' + str(idx)
            prev_inner_gfn_outs.append(prev_gfn_out)
            if idx < pos:
                zeros_like_diff_ins.append(tensor.zeros_like(diff_in))
            else:
                zeros_like_diff_ins.append(
                    tensor.zeros_like(args[idx + offset]))

        # 7.3. compute gradients of the inputs given one output
        for dx, out in enumerate(clean_outputs):
            inner_g_out = safe_new(out)
            ###
            #### I need to clip the gradient HERE !!

            if g_outs_no_shared[dx]:
                g_out_slices.append(g_outs_no_shared[dx][0])
            else:
                g_out_slices.append(None)
            if getattr(out, 'name', None) is not None:
                inner_g_out.name = 'g_' + out.name
            else:
                inner_g_out.name = 'g_' + str(dx)
            inner_g_outs.append(inner_g_out)
            _g_out = inner_g_out
            grad_outs = compute_gradient(out, _g_out)
            if not inner_gfn_outs:
                for idx, gfn_out in enumerate(grad_outs):
                    if idx >= self.n_seqs:
                        inner_gfn_outs.append(prev_inner_gfn_outs[idx])
                    else:
                        inner_gfn_outs.append(None)
            # 7.4 Sum the gradients
            # safety check, some of this inputs might still not be
            # differentiable, for those we don't add them to the mix
            # (assume their gradient is 0)
            for i, (x, y) in enumerate(zip(grad_outs, inner_gfn_outs)):
                if x and y:
                    inner_gfn_outs[i] = x + y
                elif y:
                    inner_gfn_outs[i] = y
                else:
                    inner_gfn_outs[i] = x

        ## 8. Mask the outputs that are not differentiable
        # backwards pass
        for i in xrange(len(inner_gfn_outs)):
            if inner_gfn_outs[i] == None:
                inner_gfn_outs[i] = tensor.zeros_like(diff_inputs[i])

        ## 9. Mask the g_outs that are Nones :
        for i, out in enumerate(scan_outputs):
            if g_outs[i] is None:
                try:
                    # this try is for catching non ndarray inputs (random
                    # states) it is more of a safety check ( all random
                    # states should be after n_outs_not_shared ...
                    g_outs[i] = tensor.zeros_like(scan_outputs[i])
                except Exception:
                    g_outs[i] = theano.tensor.constant(
                        numpy.array(0, theano.config.floatX))

        ## 10. Get your sequence in order for the scan:
        n_seqs = (self.n_seqs +
                  n_ins_mit_mot +
                  n_ins_mit_sot +
                  self.n_sit_sot +
                  self.n_nit_sot)
        offset = (self.n_mit_mot_outs +
                  self.n_mit_sot +
                  self.n_sit_sot)
        inner_seqs = (seqs +
                      outs_mit_mot +
                      outs_mit_sot +
                      outs_sit_sot +
                      inner_g_outs[offset:offset + self.n_nit_sot])

        scan_seqs = [x[::-1] for x in args[1:self.n_seqs + 1]]
        offset = 0
        for idx in xrange(self.n_mit_mot + self.n_mit_sot):
            mintap = numpy.min(self.tap_array[idx])
            maxtap = numpy.max(self.tap_array[idx])
            seq = scan_outputs[offset + idx]
            for k in self.tap_array[idx]:
                # We cut the sequence such that seq[i] to correspond to
                # seq[i-k]
                if maxtap < 0:
                    dim_offset = abs(maxtap)
                else:
                    dim_offset = 0
                if maxtap == mintap and maxtap != 0:
                    nw_seq = seq[:abs(maxtap)]
                elif maxtap - k != 0:
                    nw_seq = seq[dim_offset + k - mintap - 1:\
                                 -(maxtap - k + 1)][::-1]
                else:
                    nw_seq = seq[dim_offset + k - mintap - 1: -1][::-1]
                if getattr(seq, 'name', None) is not None:
                    nw_seq.name = seq.name + '[%d:]' % k
                scan_seqs.append(nw_seq)

        offset += self.n_mit_sot
        for idx in xrange(self.n_sit_sot):
            seq = scan_outputs[offset + idx][:-1]
            scan_seqs.append(seq[::-1])

        offset = (self.n_mit_mot_outs +
                  self.n_mit_sot +
                  self.n_sit_sot)
        scan_seqs += [x[::-1] for x in
                      g_outs[offset:offset + self.n_nit_sot]]

        scan_mit_mot = []
        inner_mit_mot = []
        scan_mit_mot_outs = []
        mit_mot_taps = []
        mit_mot_out_slices = []
        out_pos = 0
        ins_pos = n_seqs
        n_mit_mot_outs = 0
        n_mit_mot_ins = 0
        ins_pos = self.n_seqs
        for idx in xrange(self.n_mit_mot):
            scan_mit_mot.append(g_outs[idx][::-1])
            mit_mot_taps.append([])
            mit_mot_out_slices.append([])
            for jdx in xrange(len(self.mit_mot_out_slices[idx])):
                inner_mit_mot.append(inner_g_outs[out_pos])
                mit_mot_taps[idx].append(\
                    -self.mit_mot_out_slices[idx][jdx])
                n_mit_mot_ins += 1
                out_pos += 1

            for jdx in xrange(len(self.tap_array[idx])):
                inner_mit_mot.append(prev_inner_gfn_outs[ins_pos])
                scan_mit_mot_outs.append(\
                    inner_gfn_outs[ins_pos])
                n_mit_mot_ins += 1
                ins_pos += 1
                n_mit_mot_outs += 1
                mit_mot_taps[idx].append(-self.tap_array[idx][jdx])
                mit_mot_out_slices[idx].append(\
                    -self.tap_array[idx][jdx])

        offset = self.n_mit_mot
        for idx in xrange(self.n_mit_sot):
            mit_mot_taps.append([])
            mit_mot_out_slices.append([])
            scan_mit_mot.append(g_outs[idx + offset][::-1])
            idx_tap = idx + self.n_mit_mot
            for jdx in xrange(len(self.tap_array[idx_tap])):
                inner_mit_mot.append(prev_inner_gfn_outs[ins_pos])
                mit_mot_taps[idx + offset].append(\
                    -self.tap_array[idx_tap][jdx])
                mit_mot_out_slices[idx].append(\
                    -self.tap_array[idx_tap][jdx])
                scan_mit_mot_outs.append(inner_gfn_outs[ins_pos])
                n_mit_mot_ins += 1
                ins_pos += 1
                n_mit_mot_outs += 1
            inner_mit_mot.append(inner_g_outs[out_pos])
            out_pos += 1
            n_mit_mot_ins += 1
            mit_mot_taps[idx + offset].append(0)

        offset += self.n_mit_sot
        for idx in xrange(self.n_sit_sot):
            mit_mot_taps.append([0, 1])
            mit_mot_out_slices.append([1])
            scan_mit_mot.append(g_outs[idx + offset][::-1])
            scan_mit_mot_outs.append(inner_gfn_outs[ins_pos])
            inner_mit_mot += [inner_g_outs[out_pos],
                              prev_inner_gfn_outs[ins_pos]]
            n_mit_mot_outs += 1
            out_pos += 1
            ins_pos += 1
            n_mit_mot_ins += 2

        n_nit_sot = self.n_seqs
        scan_nit_sot_outs = inner_gfn_outs[:self.n_seqs]

        if self.truncate_gradient != -1:
            do_steps = tensor.minimum(args[0], self.truncate_gradient)
        else:
            do_steps = args[0]
        offset = (self.n_seqs +
                  n_ins_mit_sot +
                  n_ins_mit_mot +
                  self.n_sit_sot)
        # Instead of shared outs use sit_sot
        n_sitsot_outs = len(prev_inner_gfn_outs[offset:])
        scan_sitsot_ins = prev_inner_gfn_outs[offset:]
        scan_sitsot_init = []
        for x in zeros_like_diff_ins[offset:]:
            shapes = [x.shape[i] for i in xrange(x.ndim)]
            empty = tensor.zeros([do_steps + 1] + shapes,
                                 dtype=x.dtype)
            scan_sitsot_init.append(empty)
        scan_sitsot_outs = inner_gfn_outs[offset:]
        tap_array = mit_mot_taps + [[-1] for k in
                                           xrange(n_sitsot_outs)]
        info = {}
        info['n_seqs'] = n_seqs
        info['n_mit_sot'] = 0
        info['tap_array'] = tap_array
        info['gpu'] = False
        n_mit_mot = (self.n_mit_mot +
                     self.n_mit_sot +
                     self.n_sit_sot)
        info['n_mit_mot'] = n_mit_mot
        info['n_mit_mot_outs'] = n_mit_mot_outs
        info['mit_mot_out_slices'] = mit_mot_out_slices
        info['truncate_gradient'] = self.truncate_gradient
        info['n_sit_sot'] = n_sitsot_outs
        info['n_shared_outs'] = self.n_shared_outs
        info['n_nit_sot'] = n_nit_sot
        info['as_while'] = self.as_while
        info['profile'] = self.profile
        if self.name:
            info['name'] = 'grad_of_' + self.name
        else:
            info['name'] = None
        info['mode'] = self.mode
        info['inplace'] = False
        n_mit_sot = 0
        n_sit_sot = 0

        offset = (1 +
                  self.n_seqs +
                  self.n_mit_mot +
                  self.n_mit_sot +
                  self.n_sit_sot +
                  self.n_nit_sot +
                  self.n_shared_outs)

        scan_inputs = ([do_steps] +
                       scan_seqs +
                       scan_mit_mot +
                       scan_sitsot_init +
                       old_scan_init +
                       [args[0] for x in xrange(n_nit_sot)] +
                       args[offset:])

        offset = (self.n_seqs +
                  n_ins_mit_mot +
                  n_ins_mit_sot +
                  self.n_sit_sot +
                  self.n_shared_outs)

        inner_other_args = self_inputs[offset:]
        inner_gfn_ins = (inner_seqs +
                         inner_mit_mot +
                         scan_sitsot_ins +
                         old_scan_shared_ins +
                         inner_other_args)
        inner_gfn_outs = (scan_mit_mot_outs +
                          scan_sitsot_outs +
                          scan_nit_sot_outs +
                          old_scan_shared_outs)
        local_op = Scan(inner_gfn_ins, inner_gfn_outs, info)
        outputs = local_op(*scan_inputs)
        if type(outputs) not in (list, tuple):
            outputs = [outputs]
        # Re-order the gradients correctly
        gradients = [None]

        offset = (self.n_mit_mot +
                  self.n_mit_sot +
                  self.n_sit_sot +
                  n_sitsot_outs)
        gradients += [x[::-1] for x in outputs[offset:offset + self.n_seqs]]

        end = self.n_mit_mot + self.n_mit_sot + self.n_sit_sot
        gradients += [x[::-1] for x in outputs[:end]]
        gradients += [None for x in xrange(self.n_shared_outs)]
        gradients += [None for x in xrange(self.n_nit_sot)]
        begin = end

        end = begin + n_sitsot_outs
        gradients += [x[-1] for x in outputs[begin:end]]
        return gradients

    def R_op(self, inputs, eval_points):
        # Step 0. Don't work on the orignal tensor variables
        rval = scan_utils.reconstruct_graph(self.inputs,
                                            self.outputs, '_rop')
        self_inputs = rval[0]
        self_outputs = rval[1]
        # Step 1. Compute the R_op of the inner function
        inner_eval_points = [scan_utils.safe_new(x, '_evalpoint')
                             for x in self_inputs]
        if self.as_while:
            rop_self_outputs = self_outputs[:-1]
        else:
            rop_self_outputs = self_outputs
        rop_outs = tensor.Rop(rop_self_outputs, self_inputs, inner_eval_points)
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
        info = {}
        info['n_seqs'] = self.n_seqs * 2
        info['n_mit_sot'] = self.n_mit_sot * 2
        info['n_sit_sot'] = self.n_sit_sot * 2
        info['n_mit_mot'] = self.n_mit_mot * 2
        info['n_nit_sot'] = self.n_nit_sot * 2
        info['n_shared_outs'] = self.n_shared_outs * 2
        info['gpu'] = False
        info['as_while'] = self.as_while
        info['profile'] = self.profile
        info['truncate_gradient'] = self.truncate_gradient
        if self.name:
            info['name'] = 'rop_of_' + self.name
        else:
            info['name'] = None
        info['mode'] = self.mode
        info['inplace'] = False
        info['mit_mot_out_slices'] = self.mit_mot_out_slices * 2
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
        scan_seqs = inputs[b:e] + eval_points[b:e]
        inner_seqs = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        # MIT_MOT sequences ...
        b = e
        e = e + self.n_mit_mot
        ib = ie
        ie = ie + int(numpy.sum([len(x) for x in
                                 self.tap_array[:self.n_mit_mot]]))
        scan_mit_mot = inputs[b:e] + eval_points[b:e]
        inner_mit_mot = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        # MIT_SOT sequences ...
        b = e
        e = e + self.n_mit_sot
        ib = ie
        ie = ie + int(numpy.sum([len(x) for x in
                         self.tap_array[self.n_mit_mot:\
                                        self.n_mit_mot + self.n_mit_sot]]))
        scan_mit_sot = inputs[b:e] + eval_points[b:e]
        inner_mit_sot = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        #SIT_SOT sequences ...
        b = e
        e = e + self.n_sit_sot
        ib = ie
        ie = ie + self.n_sit_sot
        scan_sit_sot = inputs[b:e] + eval_points[b:e]
        inner_sit_sot = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        #Shared outs ...
        b = e
        e = e + self.n_shared_outs
        ib = ie
        ie = ie + self.n_shared_outs
        scan_shared = inputs[b:e] + eval_points[b:e]
        inner_shared = self_inputs[ib:ie] + inner_eval_points[ib:ie]

        # NIT_SOT sequences
        b = e
        e = e + self.n_nit_sot
        scan_nit_sot = inputs[b:e] * 2

        # All other arguments
        scan_other = inputs[e:] + eval_points[e:]
        inner_other = self_inputs[ie:] + inner_eval_points[ie:]

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
        inner_out_shared = self_outputs[b:e] + rop_outs[b:e]

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
        b = e + self.n_shared_outs
        e = e + self.n_shared_outs * 2
        final_outs += outputs[b:e]

        return final_outs


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
