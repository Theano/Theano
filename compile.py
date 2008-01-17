
import gof

import opt
from copy import copy

#prog(inputs, outputs)
#single(*outputs)
#multi(*output_lists)

import core


class prog(gof.Prog):
    def __init__(self, inputs, outputs, optimizer = opt.optimizer([])):
        """Compile a subgraph.

        N.B. This triggers computation of the subgraph leading to the outputs
        that is not fed by the inputs (the orphans).

        TODO: think about whether orphan computation should be in this function,
        or in self.__call__()
        """
        outputs = gof.mark_outputs_as_destroyed(outputs)
        gof.Prog.__init__(self,
                          inputs,
                          outputs,
                          optimizer,
                          gof.link.perform_linker,
                          [])
        self.compute_orphans()

    def __call__(self, check_uncomputed = True):
        """Recompute the graph.

        If the inputs are uncomputed (and check_uncomputed is True) then an
        Exception is raised.
        """
        if check_uncomputed:
            for input in self.env.inputs:
                if input.data is core.UNCOMPUTED:
                    raise Exception("You must provide a value for input %s!" % input)
        return gof.Prog.__call__(self)

    def compute_orphans(self):
        for orphan in self.env.orphans():
            if orphan.data is core.UNCOMPUTED:
                if orphan.owner:
                    orphan.owner.compute()
                else:
                    raise Exception("Orphan %s is uncomputed but needed to calculate the function." % orphan)
            

def to_func(inputs, outputs):
#    print gof.Env(inputs, outputs).io_toposort()
##    p = prog([copy(input) for input in inputs], gof.graph.clone(inputs, outputs))
    p = prog(inputs, outputs)
    def f(*args):
        for input, value in zip(inputs, args):
            p[input] = value
        outputs = p()
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
    return f

def single(*outputs):
    return prog(gof.graph.inputs(outputs), outputs)

