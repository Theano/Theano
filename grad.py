
import gof
import core

class _GradD(dict):
    """A dictionary-like class, into which derivative expressions may be added"""

    def add(self, r, dr):
        """Add dv to the sum of gradients associated with v"""

        if r is core.UNDEFINED:
            self[r] = core.UNDEFINED
        elif r in self:
            self[r] = self[r] + dr
        else:
            self[r] = dr


def expand_grad(i, o, cost_derivs):
    grad_d = _GradD(cost_derivs)
    core.build_mode()
    for op in gof.graph.io_toposort(i, o).__reversed__():
        op.update_gradient(grad_d)
#        inputgs = op.grad(*(op.inputs + [grad_d[output] for output in op.outputs]))
#        if not isinstance(inputgs, (list, tuple)):
#            inputgs = [inputgs] * len(op.inputs)
#        for input, inputg in zip(op.inputs, inputgs):
#            grad_d.add(input, inputg)
    core.pop_mode()
    return grad_d


def grad(cost, wrt, cost_grad = 1.0):
    assert core.current_mode() != 'eval'
#    cost, wrt = core.wrap(cost), core.wrap(wrt)
    cost_derivs = expand_grad([wrt], [cost], {cost: core.wrap(cost_grad)})
#     print wrt
#     for k, v in cost_derivs.items():
#         print k, v
    ret = cost_derivs.get(wrt, None)
    if ret is core.UNDEFINED:
        raise Exception("The gradient wrt %s is undefined." % wrt)
    return ret


