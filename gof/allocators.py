
from constructor import Allocator
from op import Op


class OpAllocator(Allocator):

    def __init__(self, opclass):
        if not issubclass(opclass, Op):
            raise TypeError("Expected an Op instance.")
        self.opclass = opclass


class FilteredOpAllocator(OpAllocator):

    def filter(self, op):
        pass
    
    def __call__(self, *inputs):
        op = self.opclass(*inputs)
        self.filter(op)
        if len(op.outputs) == 1:
            return op.outputs[0]
        else:
            return op.outputs


class BuildAllocator(FilteredOpAllocator):
    pass


class EvalAllocator(FilteredOpAllocator):
    def filter(self, op):
        op.perform()
        for output in op.outputs:
            output.role = None


class BuildEvalAllocator(FilteredOpAllocator):
    def filter(self, op):
        op.perform()


