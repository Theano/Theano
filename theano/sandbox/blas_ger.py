"""

Optimization to specialize gemm -> ger are not written

Scipy implementation is not written

We need to call scipy.linalg.blas.[cf]blas.[sdcz]ger here to don't loose speed again the old Outer op.
Here is the scipy signature: ger(alpha,x,y,incx=1,incy=1,a=0.0,overwrite_x=1,overwrite_y=1,overwrite_a=0)

http://www.scipy.org/doc/api_docs/SciPy.lib.blas.info.html



C implementation is not written.

Tests are not written.
"""
class GER(Op):
    """
    General rank-1 update
    A <- A + a x' y

    For matrix A, vectors x, y, and scalar a.
    """
    def __init__(self, inplace):
        self.inplace = bool(inplace)
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __hash__(self):
        return hash((type(self), self.inplace))

    def __eq__(self, other):
        return hash((type(self), self.inplace))

    def make_node(self, *inputs):
        inputs = map(as_tensor_variable, inputs)
        A, a, x, y = inputs

        nx = x.type.ndim
        ny = y.type.ndim

        if nx != 1: raise TypeError('non-vector arg0 to outer()', x)
        if ny != 1: raise TypeError('not-vector arg1 to outer()', y)

        if A.dtype != a.dtype:
            raise TypeError('dtype mismatch', (A.dtype, a.dtype))
        if A.dtype != x.dtype:
            raise TypeError('dtype mismatch', (A.dtype, x.dtype))
        if A.dtype != y.dtype:
            raise TypeError('dtype mismatch', (A.dtype, y.dtype))

        return Apply(self, inputs, [A.type()])

    def perform(self, node, inp, out):
        A, a, x, y = inp
        if not self.inplace:
            A = A.copy()
        A += a * numpy.outer(x, y)
        out[0][0] = A

    # grad not needed because this is put in during optimization

    def __str__(self):
        return "GER"
