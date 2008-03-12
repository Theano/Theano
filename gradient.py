import gof

class OrderError(Exception):
    """Grad has been manipulated in the wrong order"""

class Grad(object):
    """A dictionary-like class, into which derivative expressions may be added.

    Attributes:
    map - dict: result -> grad(result)
    outputs - list: results from which to backpropagate gradient
    did_bprop - bool: has bprop been called?
    items_got - set: results for which we have returned the gradient


    Methods:

    add() - accumulate a gradient expression
    bprop() - recursively construct gradient expressions
    __call__() - retrieve the gradient wrt a given Op or result
    __getitem__() - retrieve the gradient wrt a given Op or result

    This class operates on graphs of nodes which implement the UpdateGradient interface.

    """

    def __init__(self, dct={}):
        self.map = {}
        self.outputs = []
        self.did_bprop = False
        self.items_got = set([])
        for key,val in dct.items():
            self.add_output(key,val)

    def __contains__(self, item):
        return item in self.map

    def __getitem__(self, r):
        """Return the gradient wrt result r
        
        r is also added to the set of things for which the gradient has been
        given.  Subsequent attempts to modify the gradient wrt r will fail
        with exception FixedGradientError.
        """
        self.items_got.add(r)
        try:
            return self.map[r]
        except KeyError:
            return None
    def __call__(self, r):
        """Return the gradient wrt result r"""
        return self.__getitem__(r)

    def add_output(self, r, dr):
        self.add(r, dr)
        self.outputs.append(r)
        
    def add(self, r, dr):
        """Add dr to the sum of gradients associated with r."""
        if r in self.items_got:
            raise OrderError('gradient has already been retrieved', r)
        if r in self.map:
            self.map[r] = self.map[r] + dr
        else:
            self.map[r] = dr

    def bprop(self):
        """Build a backpropagation graph.

        This function traverses the graph backward from self.outputs, calling
        update_gradient on the ops as it goes.  Ops without an update_gradient
        function are considered not differentiable.  The update_gradient
        function is defined in the UpdateGradient class.

        maybe_redo
        """
        if self.did_bprop:
            raise OrderError('bprop has already been done')
        try:
            outputs = self.outputs
            inputs = gof.graph.inputs(outputs)
            for op in gof.graph.io_toposort(inputs, outputs).__reversed__():
                op.update_gradient(self)
        finally:
            self.did_bprop = True

def grad(cost, param=None, cost_grad = 1.0):
    """Return symbolic expression of gradient of <cost> wrt <param>.

    If <param> is None, then return a Grad instance, from which the gradients of
    multiple objects can be retrieved using the __getitem__ or __call__ methods
    (as in function currying in languages such as scheme and OCaML).

    If <param> is not None, then return the gradient expression for 
    d cost / d param.

    """
    rval = Grad({cost:cost_grad})
    rval.bprop()
    if param is None:
        return rval
    else:
        return rval(param)


class UpdateGradient:
    """This class defines the interface that Grad.bprop expects of each
    differentiable Op"""

    def update_gradient(self, grad_d):
        """Override this function to call grad_d.add(r,grad_r) for each
        differentiable input result, r.

        You can assume that the gradient with respect to all output results
        has been accumulated in grad_d.  These expressions are available by
        calling grad_d[o] for o in self.outputs.  If grad_d[o] returns None,
        then this function should assume that grad_d[o] is an appropriate sort
        of zero.
        
        """
        raise AbstractFunctionError()

class SelfGrad (UpdateGradient):
    """This class implements update_gradient in terms of the popular self.grad

    This class defines update_gradient (necessary for Grad.bprop) to call a
    self.grad function like this:

        if len(self.outputs) > 1:
            self.grad(self.inputs, [grad_d[o] for o in self.outputs])
        else
            self.grad(self.inputs, grad_d[output[0]])

    self.grad() is an Abstract function, see its documentation for the
    expected behaviour.
    
    """

    def update_gradient(self, grad_d):
        #Call self.grad(inputs, output_gradients) and add the result to grad_d

        if len(self.outputs) > 1:
            inputgs = self.grad(self.inputs, [grad_d[o] for o in self.outputs])
        else:
            inputgs = self.grad(self.inputs, grad_d[self.outputs[0]])

        if len(self.inputs) == 1 and is_result(inputgs):
            inputgs = [inputgs]
        else:
            assert len(inputgs) == len(self.inputs)
        for input, inputgrad in zip(self.inputs, inputgs):
            grad_d.add(input, inputgrad)

    def grad(self, *args):
        """Return gradient expressions wrt input arguments

        If len(self.inputs)==1 : return the input gradient expression
        If len(self.inputs)>=2 : return a list of input gradient expressions 
        """
        raise AbstractFunctionError()


