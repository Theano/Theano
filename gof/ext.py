
# from copy import copy
# from op import Op
# from lib import DummyOp
# from features import Listener, Constraint, Orderings
# from env import InconsistencyError
# from utils import ClsInit
# import graph


__all__ = ['Destroyer', 'Viewer']









class Return(DummyOp):
    """
    Dummy op which represents the action of returning its input
    value to an end user. It "destroys" its input to prevent any
    other Op to overwrite it.
    """
    def destroy_map(self): return {self.out:[self.inputs[0]]}


def mark_outputs_as_destroyed(outputs):
    return [Return(output).out for output in outputs]

