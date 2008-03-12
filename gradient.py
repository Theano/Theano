import gof

def _unpack_result(lst):
    if len(lst) > 1:
        return lst
    else
        return lst[0]

def _pack_result(arg):
    if gof.result.is_result(arg): return [arg]
    return arg

def grad_sources_inputs(sources, inputs):
    """Return a dictionary mapping each result necessary for a source to its gradient

    sources - a list of gradient sources (explained below)
    inputs - a list of results considered to be constant

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
    for (r, g_r) in self.sources:
        if r in gmap:
            gmap[r] = gmap[r] + dr
        else:
            gmap[r] = dr

    outputs = gmap.keys()
    
    if inputs is None:
        inputs = gof.graph.inputs(outputs)
        
    for op in gof.graph.io_toposort(inputs, outputs).__reversed__():
        g_outputs = [gmap[o] for o in self.outputs]
        if all(map(lambda x:x is None, g_outputs)):
            continue
        output_arg = unpack_singleton(g_outputs)
        input_arg = unpack_singleton(op.inputs)
        op_grad = op.grad(input_arg, output_arg)
        if op_grad is None:
            raise Exception('If you really mean for grad(...) to return None,
            please return [None]', op.__class__)
        g_inputs = pack_singleton(op_grad)
        assert len(g_inputs) == len(op.inputs)

        for r, g_r in zip(self.inputs, g_inputs):
            if g_r is not None: 
                if r in gmap:
                    gmap[r] = gmap[r] + g_r
                else:
                    gmap[r] = g_r
    return gmap

def diff(cost, param):
    """Return symbolic expression of gradient of <cost> wrt <param>.

    If <param> is a list, then return a list containing the gradient of cost wrt
    each element of the list.
    """
    inputs = gof.graph.inputs([cost])
    gmap = grad_sources_inputs([(cost, 1.0)], inputs)
    if isinstance(param, lst):
        return [gmap[p] for p in param]
    else:
        return gmap[param]


