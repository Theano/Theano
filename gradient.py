import gof #, gof.result
import numpy #for numeric_grad

from gof.python25 import all

_msg_retType = 'op.grad(...) returned a non-list'
_msg_badlen = 'op.grad(...) returned wrong number of gradients'

def _unpack_result(lst):
    if len(lst) > 1:
        return lst
    else:
        return lst[0]

def _pack_result(arg):
    if isinstance(arg, gof.result.Result):
        return [arg]
    else:
        return arg

def grad_sources_inputs(sources, graph_inputs):
    """
    A gradient source is a pair (r, g_r), in which r is a result, and g_r is a
    result that is a gradient wrt r.

    This function traverses the graph backward from the 'r' sources,
    calling L{Op.grad}(...) when it is provided by an L{Op}, and at least one of the
    outputs of the L{Op} has an associated gradient.

    The L{Op.grad}(...) functions are called as such:
        op.grad( op.inputs[0], grad(op.outputs[0]))

    This function expects the L{Op.grad}(...) function to return the gradient
    expression [results] associated with the inputs of the L{Op}. The L{Op} should
    return a list of results corresponding to the gradients in the same order
    as the inputs. If it has a single output it should return a list or tuple
    of length 1.

    For each input wrt to which an L{Op} is not differentiable, it should return
    None instead of a result instance.

    @type sources: list
    @param sources: gradient sources (explained below)
    @type graph_inputs: list
    @param graph_inputs: results considered to be constant

    @rtype: dictionary
    @return: dictionary mapping each result necessary for a source to its gradient.
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
        g_inputs = op_grad #_pack_result(op_grad)
        assert isinstance(g_inputs, (list, tuple))
        if len(g_inputs) != len(node.inputs):
            raise ValueError(_msg_badlen, 
                    node.op, 
                    len(g_inputs),
                    len(node.inputs))
        for r, g_r in zip(node.inputs, g_inputs):
            if g_r is not None: 
                if r in gmap:
                    gmap[r] = gmap[r] + g_r
                else:
                    gmap[r] = g_r
    return gmap

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
        if isinstance(f, (list, tuple)):
            f_pt = [numpy.copy(x) for x in f_pt]
        else:
            f_pt = numpy.copy(f_pt)

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
        return abs(a-b) / (abs(a)+abs(b)+eps)

    def max_err(self, g_pt):
        """Return the biggest relative error between g_pt and self.gf"""
        assert len(g_pt) == len(self.gf)
        errs = []
        for a, b in zip(g_pt, self.gf):
            errs.append(numpy.max(numeric_grad.abs_rel_err(a,b)))
        return max(errs)


