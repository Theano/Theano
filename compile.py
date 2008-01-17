import time

import gof


import core
import opt
from copy import copy

class profile_linker:
    def __init__(self, env):
        self.order = env.toposort()
        self.thunks = [op._perform for op in self.order]
        self.n_calls = 0
        self.times = [0.0 for op in self.order]
    def __call__(self):
        for i, thunk in enumerate(self.thunks):
            start_time = time.time()
            thunk()
            self.times[i] += time.time() - start_time
        self.n_calls += 1

    def dump(self):
        total_time = sum(self.times)
        print self.n_calls, 'calls took', total_time, 'seconds'

        print 'Proportion of CPU per op'
        for op, t in zip(self.order, self.times):
            s_op = str(op).split()[0][1:]
            print "  %-35s %4.5f"% (s_op, t/total_time)

        print 'Proportion of CPU per op class'
        dct = {}
        for op, t in zip(self.order, self.times):
            s_op = str(op).split()[0][1:]
            dct[s_op] = dct.get(s_op, 0.0) + t
        for t, s_op in reversed(sorted([(t,op) for op, t in dct.items()])):
            print "  %-35s %4.5f"% (s_op, t/total_time)




class prog(gof.Prog):
    def __init__(self, inputs, outputs, optimizer = opt.optimizer([]), linker =
            gof.link.perform_linker):
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
                          linker,
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

def single(*outputs, **kwargs):
    return prog(gof.graph.inputs(outputs), outputs, **kwargs)

