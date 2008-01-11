
import gof

import opt

#prog(inputs, outputs)
#single(*outputs)
#multi(*output_lists)

import core


def to_func(inputs, outputs):
#    print gof.Env(inputs, outputs).io_toposort()
    p = prog(inputs, outputs)
    print p.env
    def f(*args):
        for input, value in zip(inputs, args):
            p[input] = value
        outputs = p()
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
    return f


# def prog(inputs, outputs):
#     outputs = gof.ext.mark_outputs_as_destroyed(outputs)
#     program = gof.Prog(inputs, outputs, opt.optimizer, gof.link.ThunkLinker(), [gof.features.PrintListener])
#     for orphan in program.env.orphans():
#         if orphan.storage is core.UNCOMPUTED:
#             raise Exception("Your program depends on a few uncomputed values.")
#     return program

class prog(gof.Prog):

    def __init__(self, inputs, outputs):
#        core.build_mode()
        outputs = gof.ext.mark_outputs_as_destroyed(outputs)
        gof.Prog.__init__(self,
                          inputs,
                          outputs,
                          opt.optimizer,
                          gof.link.perform_linker,
                          [])
#        core.pop_mode()

    def __call__(self, check_uncomputed = True):
        if check_uncomputed:
            for input in self.env.inputs:
                if input.data is core.UNCOMPUTED:
                    raise Exception("You must provide a value for input %s!" % input)
            for orphan in self.env.orphans():
                if orphan.data is core.UNCOMPUTED:
                    raise Exception("Orphan %s is uncomputed but needed to calculate the function. " % input + \
                                        "Try calling prog.compute_orphans() or set it manually.")
        return gof.Prog.__call__(self)

    def compute_orphans(self):
        raise NotImplementedError


def single(*outputs):
    return prog(gof.graph.inputs(outputs), outputs)

