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

class MaxAndArgmaxOptimizer(Optimizer):
    """Replace MaxAndArgmax by CAReduce when the argmax is not used

       This is faster as MaxAndArgmax don't have c code and execute it 
       in two pass.
    """

    def add_requirements(self, env):
        env.extend(toolbox.ReplaceValidate())

    def apply(self, env):
        did_something = True
        while did_something:
            nodelist = env.toposort()
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

@register_uncanonicalize
@gof.local_optimizer([T._shape])
def local_max_to_min(node):
    """
    change -(max(-x)) to min

    This is tested in tensor/tests/test_basic.py:test_min_max

    :note: we don't need an opt that will do the reverse as by default 
           the interface put only MaxAndArgmax into the graph.
    """
    if node.op == T.neg and node.inputs[0].owner:
        max = node.inputs[0]
        if max.owner and isinstance(max.owner.op, CAReduce) and max.owner.op.scalar_op==scal.maximum:
            neg = max.owner.inputs[0]
            if neg.owner and neg.owner.op == T.neg:
                return [CAReduce(scal.minimum,max.owner.op.axis)(neg.owner.inputs[0])]

    return False


