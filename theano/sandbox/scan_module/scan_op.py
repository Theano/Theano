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

# Logging function for sending warning or info
_logger = logging.getLogger('theano.scan_module.scan_op')


class ScanOp(PureOp):
    def __init__(self,
                 inputs,
                 outputs,
                 lengths,
                 switches,
                 mintaps,
                 index,
                 options,
                 as_repeatUntil):
        self.inputs = inputs
        self.outputs = outputs

        self.index = index
        self.switches = switches
        self.lengths = lengths
        self.mintaps = mintaps
        self.as_repeatUntil = as_repeatUntil
        self.options = options
        self.name = options['name']
        self.mode = options['mode']
        self.inplace = options['inplace']
        self.gpu = options['gpu']
        self.profile = options['profile']
        self.hash_inner_graph = options['hash_inner_graph']
        # --Construct the destroy map--
        if self.inplace:
            for idx in xrange(len(outputs)):
                self.destroy_map[idx] = [idx + 1]
        # --Decide on the default mode--
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
        # --Adding default name--
        if not hasattr(self, 'name') or self.name is None:
            self.name = 'scan_fn'

    def make_node(self, *inputs):
        # Checking if arguments are of the right type is done in the scan
        # function
        out_types = [out.type() for out in self.outputs]
        return Apply(self, inputs, out_types)

    def __eq__(self, other):
        # Check if we are dealing with same type of objects
        if not type(self) == type(other):
            return False
        if self.options != other.options:
            return False
        if self.mintals != other.mintaps:
            return False
        # Check if the number of different types of arguments is the same
        diff_args = ['inputs', 'outputs', 'lengths', 'mintaps', 'switches']
        for arg in diff_args:
            if len(getattr(self, arg)) != len(getattr(other, arg)):
                return False
        for x, y in izip(self.inputs, other.inputs):
            if x.type != y.type:
                return False
        for x, y in izip(self.lengths, other.lengths):
            if x.type != y.type:
                return False

        s_ins = [self.index] + self.inputs + self.lengths + self.switches
        o_ins = [other.index] + other.inputs + other.lengths + other.switches
        givens = dict(izip(s_ins, o_ins))
        # This part might be slow
        for x, y in izip(self.outputs, other.outputs):
            if not gof.graph.is_same_graph(x, y, givens=givens):
                return False
        return True

    def __str__(self):
        if self.gpu:
            gpu_str = 'gpu'
        else:
            gpu_str = 'cpu'
        if self.as_repeatUntil is not None:
            name = 'repeat/until'
        else:
            name = 'loop'
        if self.inplace:
            aux_txt = '%s{inplace,%s,%s}' % (name, gpu_str, str(self.name))
        else:
            aux_txt = '%s{%s,%s}' % (name, gpu_str, str(self.name))

        return aux_txt

    def __hash__(self):
        rval = hash(type(self)) ^ self.hash_inner_graph
        for val in self.options.values():
            if isinstance(val, (list, tuple)):
                for el in val:
                    rval = rval ^ el
            else:
                rval = rval ^ val
        return rval

    def infer_shape(self, node, input_shapes):
        for inp, inp_shp in izip(node.inputs, input_shapes):
            assert inp_shp is None or len(inp_shp) == inp.type.ndim
        n_outs = len(self.outputs)
        if self.as_repeatUntil is not None:
            return [(Shape_i(0)(o),) + x[1:] for o, x
                    in izip(node.outputs, input_shapes[1: n_outs + 1])]
        else:
            return input_shapes[1: n_outs + 1]

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        """
        :param node: the Apply node returned by the ``make_node`` function
                     of the scan op class

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
        # 1. Collect all memory buffers
        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]
        node_input_compute = [compute_map[r] for r in node.inputs]
        node_output_compute = [compute_map[r] for r in node.outputs]
        # 2. If the op is not inplace we need to copy over the initial values
        if not self.inplace:
            for membuf1, membuf2 in izip(
                    node_output_storage,
                    node_input_storage[1: 1 + len(node_output_storage)]):
                membuf1[0][:] = membuf2[0]

        # 3. Construct fake shared variables around every argument of scan
        givens = {}
        base_inputs = self.inputs[:len(self.outputs)]
        aux_inputs = self.inputs[len(self.outputs):]
        # 3.1 First the auxiliary arguments, those that are parameters or
        # input
        for mem_buf, var in izip(ndoe_input_storage[1 + len(base_inputs):],
                                 aux_inputs):
            givens[var] = theano.shared(mem_buf[0], name=var.name,
                                        borrow=True)
        # 3.2. Next the states (numeric or not) and the outputs
        updates = {}
        n_numeric_values = len(self.lengths)
        for pos, (mem_buf, var, expr) in enumerate(
             izip(node_output_storage, base_inputs, self.outputs)):
            givens[var] = theano.shared(mem_buf[0], name=var.name,
                                       borrow=True)
            updates[givens[var]] = expr
            if pos < n_numeric_values:
                self.lengths[pos].set_value(mem_buf[0].shape[0])
                givens[self.lengths[pos]] = \
                        tensor.constant(mem_buf[0].shape[0])

        # 3.3 Add the update for the index of scan
        updates[self.t] = self.t + numpy.int64(1)
        # 4.1 Construct the inner function of scan
        fn_outs = []
        if self.as_repeatUntil is not None:
            fn_outs = self.as_repeatUntil
        self.fn = theano.function([], fn_outs,
                                  givens=givens,
                                  updates=updates,
                                  mode=self.mode_instance,
                                  name=self.name,
                                  profile=self.profile)

        # Construct the perform
        if self.as_repeatUntil is not None:

            def p(node, args, outs):
                pos = 0
                cont = 1
                # reset all switches if any
                for sw in self.swithces:
                    sw.set_value(numpy.int8(0), borrow=True)
                while cont and pos < node_input_storage[0][0]:
                    cont = self.fn()
                    pos = pos + 1
                # We need to trim the outputs if they are longer
                for pos, membuf in enumerate(
                                node_output_storage[:n_numeric_values]):
                    if membuf[0].shape[0] > pos + self.mintaps[pos]:
                        membuf[0] = membuf[0][:pos + self.mintaps[pos]]
        else:

            def p(node, args, outs):
                for sw in self.switches:
                    sw.set_value(numpy.int8(0), borrow=True)
                self.fn.fn(n_calls=node_input_storage[0][0])

        def rval(p=p, i=node_input_storage, o=node_output_storage, n=node):
            r = perform(n, [x[0] for x in i], o)
            for o in node.outputs:
                compute_map[o][0] = True
            return r
        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.perform = p
        rval.lazy = False
        return rval

    def grad(self, args, g_outs):
        pass

    def R_op(self, inputs, eval_points):
        pass


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
                        v, scan_fct_time, scan_op_time,
                        scan_fct_time / v * 100, scan_op_time / v * 100), node
                else:
                    print (' The node took 0s, so we can not compute the '
                           'overhead'), node
        print '    total %5.1fs  %5.1fs  %5.1fs  %5.1f%%  %5.1f%%' % (
            total_super_scan_time, total_scan_fct_time, total_scan_op_time,
            total_scan_fct_time / total_super_scan_time * 100,
            total_scan_op_time / total_super_scan_time * 100)
