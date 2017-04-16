import numpy

import theano
from theano import Variable, Constant
from theano import tensor
from theano.compile import SharedVariable

from theano.opencl.type import CLArrayType
from theano.opencl.basic_ops import host_from_cl, cl_from_host


class _operators(tensor.basic._tensor_py_operators):
    def _as_TensorVariable(self):
        return host_from_cl()(self)
    def _as_cl_variable(self):
        return self

    dtype = property(lambda s: s.type.dtype)
    broadcastable = property(lambda s: s.type.broadcastable)
    ndim = property(lambda s: s.type.ndim)

class CLArrayVariable(Variable, _operators):
    pass
CLArrayType.Variable = CLArrayVariable

class CLArrayConstantSignature(tensor.TensorConstantSignature):
    pass

class CLArrayConstant(Constant, _operators):
    def signature(self):
        return CLArrayConstantSignature((self.type, self.data.get()))

    def __str__(self):
        if self.name is not None:
            return self.name
        return "CLArrayConstant{%s}"%(self.data.get(),)
    
CLArrayType.Constant = CLArrayConstant

class CLArraySharedVariable(SharedVariable, _operators):
    def get_value(self, borrow=False, return_internal_type=False):
        if return_internal_type:
            if borrow:
                return self.container.value
            else:
                return copy.deepcopy(self.container.value)

        else:
            return self.container.value.get()

    def set_value(self, value, borrow=False):
        if isinstance(value, cl.array.Array):
            # Should I filter here?
            v = self.type.filter(value)
            if borrow:
                self.container.value = v
            else:
                self.container.value = copy.deepcopy(v)

        elif isinstance(value, numpy.ndarray):
            if value.shape != self.container.value.shape:
                raise ValueError("Shape mismatch in set_value()")
            self.container.value.set(value)
    
    def filter_update(self, other):
        if hasattr(other, '_as_cl_variable'):
            return other._as_cl_variable()

        if not isinstance(other.type, tensor.TensorType):
            raise TypeError("Incompatible type")
        
        if other.type.dtype != self.dtype:
            raise TypeError("Incompatible dtype")

        if (other.type.broadcastable != self.broadcastable):
            raise TypeError("Incompatible broadcastable")
        
        return cl_from_host(other)
