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
                 us,
                 xs,
                 ws,
                 zs,
                 xs_results,
                 ys_results,
                 lengths,
                 mintaps,
                 name,
                 mode,
                 inplace,
                 gpu,
                 as_repeatUntil,
                 profile):
        pass

    def make_node(self, *inputs):
        pass

    def __eq__(self, other):
        pass

    def __str__(self):
        pass

    def __hash__(self):
        pass

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
    if any([isinstance(node.op, Scan) and v>0 for (_,node),v in
            apply_time.items()]):
        print
        print 'Scan overhead:'
        print '<Scan op time(s)> <sub scan fct time(s)> <sub scan op time(s)> <sub scan fct time(% scan op time)> <sub scan op time(% scan op time)> <node>'
        total_super_scan_time = 0
        total_scan_fct_time = 0
        total_scan_op_time = 0
        for (_,node),v in apply_time.items():
            if isinstance(node.op, Scan):
                if v> 0:
                    scan_fct_time = node.op.mode_instance.fn_time
                    scan_op_time = node.op.mode_instance.local_time
                    total_super_scan_time += v
                    total_scan_fct_time += scan_fct_time
                    total_scan_op_time += scan_op_time
                    print '    %5.1fs  %5.1fs  %5.1fs  %5.1f%%  %5.1f%%'%(
                        v, scan_fct_time, scan_op_time, scan_fct_time/v*100,
                        scan_op_time/v*100), node
                else:
                    print ' The node took 0s, so we can not compute the overhead', node
        print '    total %5.1fs  %5.1fs  %5.1fs  %5.1f%%  %5.1f%%'%(
            total_super_scan_time, total_scan_fct_time, total_scan_op_time, total_scan_fct_time/total_super_scan_time*100, total_scan_op_time/total_super_scan_time*100)
