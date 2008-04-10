import copy #for __copy__
import numpy
from scipy import sparse

import gof.op, gof.result
import tensor


# Wrapper type

def assparse(sp, **kwargs):
    """
    Wrapper around SparseR constructor.
    @param sp:  A sparse matrix. assparse reads dtype and format properties
                out of this sparse matrix.
    @return:    SparseR version of sp.
    """
    if isinstance(sp, SparseR):
        return sp
    else:
        rval = SparseR(str(sp.dtype), sp.format, **kwargs)
        rval.data = sp
        return rval

class SparseR(gof.result.ResultBase):
    """
    Attribute:
    format - a string identifying the type of sparsity

    Properties:
    T - read-only: return a transpose of self

    Methods:

    Notes:

    """
    format_cls = {
            'csr' : sparse.csr_matrix,
            'csc' : sparse.csc_matrix
            }
    dtype_set = set(['int', 'int32', 'int64', 'float32', 'float64'])

    def __init__(self, dtype, format, **kwargs):
        """
        Fundamental way to do create a sparse node.
        @param dtype:   Type of numbers in the matrix.
        @param format:  The sparse storage strategy.
        @return         An empty SparseR instance.
        """

        gof.ResultBase.__init__(self, **kwargs)
        if dtype in SparseR.dtype_set:
            self._dtype = dtype
        assert isinstance(format, str)

        #print format, type(format), SparseR.format_cls.keys(), format in SparseR.format_cls
        if format in SparseR.format_cls:
            self._format = format
        else:
            raise NotImplementedError('unsupported format "%s" not in list' % format, SparseR.format_cls.keys())

    def filter(self, value):
        if isinstance(value, SparseR.format_cls[self.format])\
                and value.dtype == self.dtype:
                    return value
        #print 'pass-through failed', type(value)
        sp = SparseR.format_cls[self.format](value)
        if str(sp.dtype) != self.dtype:
            raise NotImplementedError()
        if sp.format != self.format:
            raise NotImplementedError()
        return sp

    def __copy__(self):
        if self.name is not None:
            rval = SparseR(self._dtype, self._format, name=self.name)
        else:
            rval = SparseR(self._dtype, self._format)
        rval.data = copy.copy(self.data)
        return rval


    dtype = property(lambda self: self._dtype)
    format = property(lambda self: self._format)
    T = property(lambda self: transpose(self), doc = "Return aliased transpose")


    def __add__(left, right): return add(left, right)
    def __radd__(right, left): return add(left, right)


#
# Conversion
#

# convert a sparse matrix to an ndarray
class DenseFromSparse(gof.op.Op):
    def __init__(self, x, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        self.inputs = [assparse(x)]
        self.outputs = [tensor.Tensor(x.dtype,[0,0])]
    def impl(self, x):
        return numpy.asarray(x.todense())
    def grad(self, x, gz): 
        return sparse_from_dense(gz, x.format)
dense_from_sparse = gof.op.constructor(DenseFromSparse)

class SparseFromDense(gof.op.Op):
    def __init__(self, x, format, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        if isinstance(format, gof.result.ResultBase):
            self.inputs = [tensor.astensor(x), format]
        else:
            self.inputs =  [tensor.astensor(x), gof.result.PythonResult()]
            self.inputs[1].data = format
        self.outputs = [SparseR(x.dtype, self.inputs[1].data)]
    def impl(self, x, fmt):
        # this would actually happen anyway when we try to assign to
        # self.outputs[0].data, but that seems hackish -JB
        return SparseR.format_cls[fmt](x)
    def grad(self, (x, fmt), gz):
        return dense_from_sparse(gz)
sparse_from_dense = gof.op.constructor(SparseFromDense)

# Linear Algebra

class Transpose(gof.op.Op):
    format_map = {
            'csr' : 'csc',
            'csc' : 'csr'}
    def __init__(self, x, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        x = assparse(x)
        self.inputs = [x]
        self.outputs = [SparseR(x.dtype, Transpose.format_map[x.format])]
    def impl(self, x):
        return x.transpose() 
    def grad(self, x, gz): 
        return transpose(gz)
transpose = gof.op.constructor(Transpose)

class AddSS(gof.op.Op): #add two sparse matrices
    def __init__(self, x, y, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        x, y = [assparse(x), assparse(y)]
        self.inputs = [x, y]
        if x.dtype != y.dtype:
            raise NotImplementedError()
        if x.format != y.format:
            raise NotImplementedError()
        self.outputs = [SparseR(x.dtype, x.format)]
    def impl(self, x,y): 
        return x + y
    def grad(self, (x, y), gz):
        return gz, gz
add_s_s = gof.op.constructor(AddSS)


if 0:
    class dot(gof.op.Op):
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
            gof.op.Op.__init__(self, **kwargs)
            self.grad_preserves_dense = [True, True]
        def gen_outputs(self): return [SparseR()]
        def impl(x,y):
            if hasattr(x, 'getnnz'):
                # if x is sparse, then do this.
                return x.dot(y)
            else:
                # if x is dense (and y is sparse), we do this
                return y.transpose().dot(x.transpose()).transpose()

        def grad(self, x, y, gz):
            rval = [dot(gz, y.T), dot(x.T, gz)]
            for i in 0,1:
                if not isinstance(self.inputs[i], SparseR):
                    #assume it is a dense matrix
                    if self.grad_preserves_dense[i]:
                        rval[i] = dense_from_sparse(rval[i])
            return rval


