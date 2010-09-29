"""
This file implement specialization optimization that break the canonicalization form
"""

# TODO: intelligent merge for mul/add
# TODO: 0*x -> 0



import logging
_logger = logging.getLogger('theano.tensor.opt')

import operator
import itertools
import sys

import theano
from theano import gof
from elemwise import CAReduce
import basic as T

from theano.gof.python25 import any, all
from theano.gof.opt import Optimizer
from theano.gof import InconsistencyError, toolbox

from basic import get_constant_value
from theano.tensor.opt import register_uncanonicalize
from theano import scalar as scal

@register_uncanonicalize
@gof.local_optimizer([T._shape])
def local_max_and_argmax_specialize(node):
    if node.op == T._max_and_argmax:
        if len(node.outputs[1].clients)==0:
            import pdb;pdb.set_trace()
            try:
                axis=get_constant_value(node.inputs[1])
            except ValueError:
                return False

            return [CAReduce(scal.maximum,axis)(node.inputs[0]), T.as_tensor_variable(0)]

    return False

class MaxAndArgmaxOptimizer(Optimizer):
    """Graph optimizer for Fusion of elemwise operations"""

    def add_requirements(self, env):
        env.extend(toolbox.ReplaceValidate())

    def apply(self, env):
        did_something = True
        while did_something:
            nodelist = list(env.nodes)
            did_something = False
            for node in nodelist:
                if node.op == T._max_and_argmax:
                    if len(node.outputs[1].clients)==0:
                        try:
                            axis=get_constant_value(node.inputs[1])
                        except ValueError:
                            return False

                        new = CAReduce(scal.maximum,axis)(node.inputs[0])
                        try:
                            env.replace_all_validate(
                                ((node.outputs[0],new),),
                                reason = self.__class__.__name__)
                            did_something = True
                            break
                        except InconsistencyError, e:
                            pass

register_uncanonicalize(MaxAndArgmaxOptimizer(),name='MaxAndArgmaxOptimizer')

