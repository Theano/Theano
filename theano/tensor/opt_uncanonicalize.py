"""
This file implement specialization optimization that break the canonization form of the graph.

Currently there is problem with the order of optimization and the definition of definition of
canonized graph.

Right now there is a canonization optimization phase that try to make all equivalent graph
identical. This is not always the case, but it do many of the basic stuff canonical. We
need to extend the definition of canonization to make this true more often.

The problem this file indent to fix in the future is that in the "Equilibrium" specialization
optimization phase, there is optimization that request that the graph is canonical, some other
request that this is not true, and some other that break the canonicalization for
some optimization. As we can't control the order of those optimization, there is case that some
optimization requesting a canonical graph won't be applied as optimization that break the
canonicalization form of the graph executed before.

To fix this, we need to split the specialization phase into a phase where optimization can't break the canonicalization form and one where this is allowed. This is also needed for the stabilized optimization phase, but as it happen before the specialization phase, this cause less problem.

Also, we should make the fgraph refuse optimization that break the canonization of the graph in the optimizations phases where the graph is supposed to be canonical.
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

from basic import get_scalar_constant_value, NotScalarConstantError
from theano.tensor.opt import register_uncanonicalize
from theano import scalar as scal

class MaxAndArgmaxOptimizer(Optimizer):
    """Replace MaxAndArgmax by CAReduce when the argmax is not used

       This is faster as MaxAndArgmax don't have c code and execute it
       in two pass.
    """

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def apply(self, fgraph):
        did_something = True
        while did_something:
            nodelist = fgraph.toposort()
            did_something = False
            for node in nodelist:
                if node.op == T._max_and_argmax:
                    if len(node.outputs[1].clients)==0:
                        try:
                            axis=get_scalar_constant_value(node.inputs[1])
                        except NotScalarConstantError:
                            return False

                        new = CAReduce(scal.maximum,axis)(node.inputs[0])
                        try:
                            fgraph.replace_all_validate(
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
