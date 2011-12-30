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


class ScanOp(PureOp):
    def __init__(self,
                 inputs,
                 outputs,
                 lengths,
                 switches,
                 options,
                 as_repeatUntil):
        self.inputs = inputs
        self.outputs = outputs

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
        # This are some safety checks ( namely that the inner graph has the
        # same number of inputs and same number of outputs )
        elif not len(self.inputs) == len(other.inputs):
            return False
        elif not len(self.outputs) == len(other.outputs):
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

        s_inputs = [self.t] + self.inputs + self.lengths + self.switches
        o_inputs = [other.t] + other.inputs + other.lengths + other.switches
        givens = dict(izip(s_inputs, o_inputs))
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
        if self.as_repeatUntil:
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
        pass

    def infer_shape(self, node, input_shapes):
        pass

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
