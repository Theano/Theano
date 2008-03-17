import gof, gof.result
import numpy #for numeric_grad

from gof.python25 import all

_msg_retNone = 'op.grad(...) returned None, consider returning [None]'
_msg_badlen = 'op.grad(...) returned wrong number of gradients'

def _unpack_result(lst):
    if len(lst) > 1:
        return lst
    else:
        return lst[0]

def _pack_result(arg):
    if isinstance(arg, gof.result.ResultBase):
        return [arg]
    else:
        return arg

def grad_sources_inputs(sources, graph_inputs):
    """Return a dictionary mapping each result necessary for a source to its gradient

    sources - a list of gradient sources (explained below)
    graph_inputs - a list of results considered to be constant

    A gradient source is a pair (r, g_r), in which r is a result, and g_r is a
    result that is a gradient wrt r.

    This function traverses the graph backward from the 'r' sources,
    calling op.grad(...) when it is provided by an op, and at least one of the
    outputs of the op has an associated gradient.

    The op.grad(...) functions may be called in several ways (for the
    convenience of the op implementer) depending on the number of inputs and
    outputs.  

    If there is one input and one output:
        op.grad( op.inputs[0], grad(op.outputs[0]))

    If there are several inputs and one output:
        op.grad( op.inputs, grad(op.outputs[0]))

    If there is one input and several outputs:
        op.grad( op.inputs[0], [grad(o) for o in op.outputs[0]])

    If there are multiple inputs and outputs:
        op.grad( op.inputs, [grad(o) for o in op.outputs[0]])

    This function expects the op.grad(...) function to return the gradient
    expression [results] associated with the inputs of the op.  If the op has a
    single input, it should return a single result; if the op has multiple
    inputs, it should return a list of results corresponding to the gradients in
    the same order as the inputs.

    For each input wrt to which an op is not differentiable, it should return
    None instead of a result instance.

    """
    gmap = {}
    for (r, g_r) in sources:
        if g_r is not None:
            if r in gmap:
                gmap[r] = gmap[r] + g_r
            else:
                gmap[r] = g_r

    graph_outputs = gmap.keys()
    
    if graph_inputs is None:
        graph_inputs = gof.graph.inputs(graph_outputs)
        
    for op in gof.graph.io_toposort(graph_inputs, graph_outputs).__reversed__():
        g_outputs = [gmap.get(o,None) for o in op.outputs]

        #if all output gradients are None, continue
        if all(map(lambda x:x is None, g_outputs)): continue

        output_arg = _unpack_result(g_outputs)
        input_arg = _unpack_result(op.inputs)
        op_grad = op.grad(input_arg, output_arg)
        if op_grad is None:
            raise ValueError(_msg_retNone, op.__class__)
        g_inputs = _pack_result(op_grad)
        if len(g_inputs) != len(op.inputs):
            raise ValueError(_msg_badlen, 
                    op.__class__, 
                    len(g_inputs),
                    len(op.inputs))
        for r, g_r in zip(op.inputs, g_inputs):
            if g_r is not None: 
                if r in gmap:
                    gmap[r] = gmap[r] + g_r
                else:
                    gmap[r] = g_r
    return gmap

def grad(cost, param, g_cost=1.0):
    """Return symbolic expression of gradient of <cost> wrt <param>.
    If <param> is a list, then return a list containing the gradient of cost wrt
    each element of the list.
    """
    inputs = gof.graph.inputs([cost])
    gmap = grad_sources_inputs([(cost, g_cost)], inputs)
    if isinstance(param, list):
        return [gmap.get(p, None) for p in param]
    else:
        return gmap.get(param, None)


class numeric_grad:
    def __init__(self, f, pt, eps=1.0e-7):
        """Return the gradient of f at pt.
        
        This function computes the gradient by a one-sided finite differences of a
        fixed step size (eps).
        
        It is assumed that f(...) will return a scalar.
        It is assumed that all f's inputs are numpy.ndarray objects.
        """
        gf = [numpy.ndarray(x.shape) for x in pt]
        f_pt = f(*pt)

        for idx in xrange(len(gf)):
            if len(pt[idx].shape) == 0:
                orig = pt[idx]
                pt[idx] = numpy.asarray(pt[idx] + eps)
                f_eps = f(*pt)
                gf[idx] = numpy.asarray((f_eps - f_pt)/eps)
                pt[idx] = orig

            elif len(pt[idx].shape) == 1:
                for i in xrange(pt[idx].shape[0]):
                    orig = pt[idx][i]
                    pt[idx][i] = pt[idx][i] + eps
                    f_eps = f(*pt)
                    gf[idx][i] = numpy.asarray((f_eps - f_pt)/eps)
                    pt[idx][i] = orig
            elif len(pt[idx].shape) == 2:
                for i in xrange(pt[idx].shape[0]):
                    for j in xrange(pt[idx].shape[1]):
                        orig = pt[idx][i,j]
                        pt[idx][i,j] = pt[idx][i,j] + eps
                        f_eps = f(*pt)
                        gf[idx][i,j] = numpy.asarray((f_eps - f_pt)/eps)
                        pt[idx][i,j] = orig
            else:
                raise NotImplementedError()

        self.gf = gf

    @staticmethod
    def abs_rel_err(a,b,eps=1.0e-10):
        """Return a small number when a and b are close, relative to how big they are"""
        return abs( (a-b) / (a+b+eps))

    def max_err(self, g_pt):
        """Return the biggest relative error between g_pt and self.gf"""
        assert len(g_pt) == len(self.gf)
        errs = []
        for a, b in zip(g_pt, self.gf):
            errs.append(numpy.max(numeric_grad.abs_rel_err(a,b)))
        return max(errs)


