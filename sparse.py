
import numpy
from scipy import sparse

import gof

# Wrapper type

class SparseR(gof.ResultBase):
    """
    Attribute:
    format - a subclass of sparse.spmatrix indicating self.data.__class__

    Properties:
    T - read-only: return a transpose of self

    Methods:

    Notes:

    """
    def __init__(self, data=None, role=None, constant = False, 
            format = sparse.csr_matrix):
        core.ResultBase.__init__(self, role, data, constant)
        if isinstance(data, sparse.spmatrix):
            self.format = data.__class__
        else:
            self.format = format
        self._dtype = None
        self._shape = None

    def data_filter(self, value):
        if isinstance(value, sparse.spmatrix): return value
        return sparse.csr_matrix(value)

    def __add__(left, right): return add(left, right)
    def __radd__(right, left): return add(left, right)

    T = property(lambda self: transpose(self), doc = "Return aliased transpose")

    # self._dtype is used when self._data hasn't been set yet
    def __dtype_get(self):
        if self._data is None:
            return self._dtype
        else:
            return self._data.dtype
    def __dtype_set(self, dtype):
        if self._data is None:
            self._dtype = dtype
        else:
            raise StateError('cannot set dtype after data has been set')
    dtype = property(__dtype_get, __dtype_set)

    # self._shape is used when self._data hasn't been set yet
    def __shape_get(self):
        if self._data is None:
            return self._shape
        else:
            return self._data.shape
    def __shape_set(self, shape):
        if self._data is None:
            self._shape = shape
        else:
            raise StateError('cannot set shape after data has been set')
    shape = property(__shape_get, __shape_set)

# convenience base class
class op(gof.PythonOp, grad.update_gradient_via_grad):
    """unite PythonOp with update_gradient_via_grad"""

#
# Conversion
#

# convert a sparse matrix to an ndarray
class sparse2dense(op):
    def gen_outputs(self): return [core.Numpy2()]
    def impl(x): return numpy.asarray(x.todense())
    def grad(self, x, gz): 
        if x.format is sparse.coo_matrix: return dense2coo(gz)
        if x.format is sparse.csc_matrix: return dense2csc(gz)
        if x.format is sparse.csr_matrix: return dense2csr(gz)
        if x.format is sparse.dok_matrix: return dense2dok(gz)
        if x.format is sparse.lil_matrix: return dense2lil(gz)

# convert an ndarray to various sorts of sparse matrices.
class _dense2sparse(op):
    def gen_outputs(self): return [SparseR()]
    def grad(self, x, gz): return sparse2dense(gz)
class dense2coo(_dense2sparse):
    def impl(x): return sparse.coo_matrix(x)
class dense2csc(_dense2sparse):
    def impl(x): return sparse.csc_matrix(x)
class dense2csr(_dense2sparse):
    def impl(x): return sparse.csr_matrix(x)
class dense2dok(_dense2sparse):
    def impl(x): return sparse.dok_matrix(x)
class dense2lil(_dense2sparse):
    def impl(x): return sparse.lil_matrix(x)


# Linear Algebra

class add(op):
    def gen_outputs(self): return [SparseR()]
    def impl(csr,y): return csr + y

class transpose(op):
    def gen_outputs(self): return [SparseR()]
    def impl(x): return x.transpose() 
    def grad(self, x, gz): return transpose(gz)

class dot(op):
    """
    Attributes:
    grad_preserves_dense - an array of boolean flags (described below)


    grad_preserves_dense controls whether gradients with respect to inputs are
    converted to dense matrices when the corresponding inputs are not in a
    SparseR wrapper.  This can be a good idea when dot is in the middle of a
    larger graph, because the types of gx and gy will match those of x and y.
    This conversion might be annoying if the gradients are graph outputs though,
    hence this mask.
    """
    def __init__(self, *args, **kwargs):
        op.__init__(self, *args, **kwargs)
        self.grad_preserves_dense = [True, True]
    def gen_outputs(self): return [SparseR()]
    def impl(x,y):
        if hasattr(x, 'getnnz'):
            return x.dot(y)
        else:
            return y.transpose().dot(x.transpose()).transpose()

    def grad(self, x, y, gz):
        rval = [dot(gz, y.T), dot(x.T, gz)]
        for i in 0,1:
            if not isinstance(self.inputs[i], SparseR):
                #assume it is a dense matrix
                if self.grad_preserves_dense[i]:
                    rval[i] = sparse2dense(rval[i])
        return rval

