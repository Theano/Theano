from __future__ import absolute_import, print_function, division
import numpy as np
import imp

import theano
from theano.gof import Op, Apply
from theano.gradient import DisconnectedType


class PdbBreakpoint(Op):
    """
    This is an identity-like op with the side effect of enforcing a
    conditional breakpoint, inside a theano function, based on a symbolic
    scalar condition. It automatically detects available debuggers and uses
    the first available in the following order: `pudb`, `ipdb`, or `pdb`.

    :type name: String
    :param name: name of the conditional breakpoint. To be printed when the
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

        # Ensure that condition is a theano tensor
        if not isinstance(condition, theano.Variable):
            condition = theano.tensor.as_tensor_variable(condition)

        # Validate that the condition is a scalar (else it is not obvious how
        # is should be evaluated)
        assert (condition.ndim == 0)

        # Because the user might be tempted to instantiate PdbBreakpoint only
        # once and apply it many times on different number of inputs, we must
        # create a new instance of the op here, define the instance attributes
        # (view_map and var_types) in that instance and then apply it on the
        # inputs.
        new_op = PdbBreakpoint(name=self.name)
        new_op.view_map = {}
        new_op.inp_types = []
        for i in range(len(monitored_vars)):
            # Every output i is a view of the input i+1 because of the input
            # condition.
            new_op.view_map[i] = [i + 1]
            new_op.inp_types.append(monitored_vars[i].type)

        # Build the Apply node
        inputs = [condition] + list(monitored_vars)
        outputs = [inp.type() for inp in monitored_vars]
        return Apply(op=new_op, inputs=inputs, outputs=outputs)

    def perform(self, node, inputs, output_storage):
        condition = inputs[0]

        if condition:
            try:
                monitored = [np.asarray(inp) for inp in inputs[1:]]
            except:
                raise ValueError("Some of the inputs to the PdbBreakpoint op "
                                 "'%s' could not be casted to NumPy arrays" %
                                 self.name)

            print("\n")
            print("-------------------------------------------------")
            print("Conditional breakpoint '%s' activated\n" % self.name)
            print("The monitored variables are stored, in order,")
            print("in the list variable 'monitored' as NumPy arrays.\n")
            print("Their contents can be altered and, when execution")
            print("resumes, the updated values will be used.")
            print("-------------------------------------------------")

            if imp.find_module('pudb'):
                import pudb
                pudb.set_trace()
            elif imp.find_module('ipdb'):
                import ipdb
                ipdb.set_trace()
            else:
                import pdb
                pdb.set_trace()

            # Take the new values in monitored, cast them back to their
            # original type and store them in the output_storage
            for i in range(len(output_storage)):
                output_storage[i][0] = self.inp_types[i].filter(monitored[i])

        else:
            # Simply return views on the monitored variables
            for i in range(len(output_storage)):
                output_storage[i][0] = inputs[i + 1]

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
