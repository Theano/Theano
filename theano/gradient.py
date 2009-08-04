"""Driver for general gradient calculations."""

__docformat__ = "restructuredtext en"

import sys
import gof #, gof.variable
import numpy #for numeric_grad

from gof.python25 import all
import gof.utils

import logging
_logger=logging.getLogger("theano.gradient")
_logger.setLevel(logging.WARN)

def error(*args):
    #sys.stderr.write('ERROR:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.error("ERROR: "+' '.join(str(a) for a in args))
def warning(*args):
    #sys.stderr.write('WARNING:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.warning("WARNING: "+' '.join(str(a) for a in args))
def info(*args):
    #sys.stderr.write('INFO:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.info("INFO: "+' '.join(str(a) for a in args))
def debug(*args):
    #sys.stderr.write('DEBUG:'+ ' '.join(str(a) for a in args)+'\n')
    _logger.debug("DEBUG: "+' '.join(str(a) for a in args))

_msg_retType = 'op.grad(...) returned a non-list'
_msg_badlen = 'op.grad(...) returned wrong number of gradients'

def grad_sources_inputs(sources, graph_inputs):
    """
    A gradient source is a pair (``r``, ``g_r``), in which ``r`` is a `Variable`, and ``g_r`` is a
    `Variable` that is a gradient wrt ``r``.

    This function traverses the graph backward from the ``r`` sources,
    calling ``op.grad(...)`` for all ops with some non-None gradient on an output.

    The ``op.grad(...)`` functions are called like this:

    .. code-block:: python
        op.grad(op.inputs[:], [total_gradient(v for v in op.outputs)])

    This call to ``op.grad`` should return a list or tuple: one symbolic gradient per input.
    If ``op`` has a single input, then ``op.grad``  should return a list or tuple of length 1.

    For each input wrt to which ``op`` is not differentiable, it should return ``None`` instead
    of a `Variable` instance.

    If a source ``r`` receives a gradient from another source ``r2``, then the effective
    gradient on ``r`` is the sum of both gradients.

    :type sources: list of pairs of Variable: (v, gradient-on-v)
    :param sources: gradients to back-propagate using chain rule
    :type graph_inputs: list of Variable
    :param graph_inputs: variables considered to be constant (do not backpropagate through
    them)

    :rtype: dictionary whose keys and values are of type `Variable`
    :return: mapping from each Variable encountered in the backward traversal to its gradient.
    """
    gmap = {}
    for (r, g_r) in sources:
        if g_r is not None:
            if r in gmap:
                gmap[r] = gmap[r] + g_r
            else:
                gmap[r] = g_r

    graph_outputs = gof.utils.uniq([r for r,g in sources])

    if graph_inputs is None:
        graph_inputs = gof.graph.inputs(graph_outputs)

    for node in gof.graph.io_toposort(graph_inputs, graph_outputs).__reversed__():
        g_outputs = [gmap.get(o,None) for o in node.outputs]

        #if all output gradients are None, continue
        if all(map(lambda x:x is None, g_outputs)): continue

        output_arg = g_outputs
        input_arg = node.inputs

        try:
            dinputs = [node.inputs[x[0]] for x in node.op.destroy_map.values()]
        except AttributeError:
            dinputs = []

        new_input_arg = []
        for input in input_arg:
            if input in dinputs and hasattr(input, 'copy'):
                new_input_arg.append(input.copy())
            else:
                new_input_arg.append(input)
        input_arg = new_input_arg
        
        #note that this function is not in a try-except block
        # the rationale:
        #  If the op implements grad, then any exception should be passed to the
        #  caller
        #  If the op doesn't implement grad, this entire function should fail.
        #  Other possibilities:
        #    * return a partial back-prop
        #
        op_grad = node.op.grad(input_arg, output_arg)
        if not isinstance(op_grad, (list,tuple)):
            raise ValueError(_msg_retType, node.op)
        g_inputs = op_grad
        assert isinstance(g_inputs, (list, tuple))
        if len(g_inputs) != len(node.inputs):
            raise ValueError(_msg_badlen, 
                    node.op, 
                    len(g_inputs),
                    len(node.inputs))
        for ii, (r, g_r) in enumerate(zip(node.inputs, g_inputs)):
            if g_r and len(sources) == 1 and sources[0][0].name and r.name:
                g_r.name = "(d%s/d%s)" % (sources[0][0].name, r.name)
            if g_r is not None: 
                if r in gmap:
                    gmap[r] = gmap[r] + g_r
                else:
                    gmap[r] = g_r
    return gmap


