import pdb

import theano
import theano.tensor as T
from theano.gof import Op, Apply
from theano.gradient import DisconnectedType

class PdbBreakpoint(Op):
    """
    This is an identity-like op with the side effect of enforcing a
    conditional breakpoint, inside a theano function, based on a symbolic
    scalar condition.

    @type name: String
    @param name: name of the conditional breakpoint. To be printed when the
                 breakpoint is activated.

    :note: WARNING. At least one of the outputs of the op must be used
                    otherwise the op will be removed from the Theano graph
                    due to its outputs being unused

    :note: WARNING. Employing the function inside a theano graph can prevent
                    Theano from applying certain optimizations to improve
                    performance, reduce memory consumption and/or reduce
                    numerical instability.

            Detailed explanation:
            As of 2014-12-01 the PdbBreakpoint op is not known by any
            optimization. Setting a PdbBreakpoint op in the middle of a
            pattern that is usually optimized out will block the optimization.

    Example:

    .. code-block:: python

        import theano
        import theano.tensor as T
        from theano.tests.breakpoint import PdbBreakpoint

        input = T.fvector()
        target = T.fvector()

        # Mean squared error between input and target
        mse = (input - target) ** 2

        # Conditional breakpoint to be activated if the total MSE is higher
        # than 100. The breakpoint will monitor the inputs, targets as well
        # as the individual error values
        breakpointOp = PdbBreakpoint("MSE too high")
        condition = T.gt(mse.sum(), 100)
        mse, monitored_input, monitored_target = breakpointOp(condition, mse,
                                                              input, target)

        # Compile the theano function
        fct = theano.function([input, target], mse)

        # Use the function
        print fct([10, 0], [10, 5]) # Will NOT activate the breakpoint
        print fct([0, 0], [10, 5]) # Will activate the breakpoint


    """

    __props__ = ("name",)

    def __init__(self, name):
        self.name = name

    def make_node(self, condition, *monitored_vars):

        # Validate that the condition is a scalar (else it is not obvious how
        # is should be evaluated)
        assert (condition.ndim == 0)

        # Build the op's view_map; every output i is a view of the input i+1.
        self.view_map = {}
        for i in range(len(monitored_vars)):
            self.view_map[i] = [i+1]

        # Build the Apply node
        inputs = [condition] + list(monitored_vars)
        outputs = [inp.type.make_variable() for inp in monitored_vars]
        return Apply(op=self, inputs=inputs, outputs=outputs)

    def perform(self, node, inputs, output_storage):
        condition = inputs[0]
        monitored = inputs[1:]

        if condition:
            print "-------------------------------------------------"
            print "Conditional breakpoint %s activated" % self.name
            print "The monitored variables are stored, in order,"
            print "in the list variable 'monitored'"
            print "-------------------------------------------------"
            pdb.set_trace()

        for i in range(len(output_storage)):
            output_storage[i][0] = monitored[i]

    def grad(self, inputs, output_gradients):
        return ([DisconnectedType()()] + output_gradients)

    def infer_shape(self, inputs, input_shapes):
        # Return the shape of every input but the condition (first input)
        return input_shapes[1:]

    def connection_pattern(self, node):

        nb_inp = len(node.inputs)
        nb_out = nb_inp - 1

        # First input is connected to no output and every other input n is
        # connected to input n-1
        connections = [[out_idx == inp_idx - 1 for out_idx in range(nb_out)]
                       for inp_idx in range(nb_inp)]
        return connections
