from __future__ import absolute_import, division, print_function

import warnings

import pkg_resources
import numpy as np
from numpy.linalg.linalg import LinAlgError

import theano
from theano import Op, config, tensor
from theano.scalar import bool as bool_t
from theano.gof import COp, ParamsType
from theano.gpuarray import GpuArrayType

from .basic_ops import (CGpuKernelBase, as_gpuarray_variable, gpu_contiguous, gpuarray_helper_inc_dir,
                        infer_context_name)
from .type import gpu_context_type

try:
    import pygpu
    from pygpu.basic import triu, tril
    pygpu_available = True
except ImportError:
    pygpu_available = False

cusolver_available = False
try:
    import skcuda
    from skcuda import cusolver
    cusolver_available = True
except (ImportError, OSError, RuntimeError, pkg_resources.DistributionNotFound):
    pass

cublas_available = False
try:
    from skcuda import cublas
    cublas_available = True
except (ImportError, OSError, RuntimeError, pkg_resources.DistributionNotFound):
    pass

if cusolver_available:
    # Add cusolver call as it is missing in skcuda
    # SPOTRS
    cusolver._libcusolver.cusolverDnSpotrs.restype = int
    cusolver._libcusolver.cusolverDnSpotrs.argtypes = [cusolver.ctypes.c_void_p,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_void_p,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_void_p,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_void_p]

    def cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda,
                         B, ldb, devInfo):
        """
        Solve real single precision linear system for hermitian matrices.
        References
        ----------
        `cusolverDn<t>potrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrs>`_
        """

        status = cusolver._libcusolver.cusolverDnSpotrs(handle, uplo, n, nrhs,
                                                        int(A), lda, int(B),
                                                        ldb, int(devInfo))
        cusolver.cusolverCheckStatus(status)


def attach_cusolver_handle_to_context(ctx):
    handle = getattr(ctx, 'cusolver_handle', None)
    if handle is None:
        with ctx:
            ctx.cusolver_handle = cusolver.cusolverDnCreate()


def attach_cublas_handle_to_context(ctx):
    handle = getattr(ctx, 'cublas_handle', None)
    if handle is None:
        with ctx:
            ctx.cublas_handle = cublas.cublasCreate()


# it is a subset of all cases available in slinalg's MATRIX_STRUCTURE
MATRIX_STRUCTURES_SOLVE = (
    'general',
    'symmetric',
    'lower_triangular',
    'upper_triangular')


class GpuCusolverSolve(Op):
    """
    CUSOLVER GPU solver OP.

    Parameters
    ----------
    trans
        Whether to take the transpose of the input matrix or not.

    """

    __props__ = ('A_structure', 'trans', 'inplace')

    def __init__(self, A_structure='general', trans='N', inplace=False):
        self.trans = trans
        self.inplace = inplace
        self.A_structure = A_structure
        if self.inplace:
            self.destroy_map = {0: [0]}
        assert A_structure in MATRIX_STRUCTURES_SOLVE
        super(GpuCusolverSolve, self).__init__()

    def make_node(self, inp1, inp2):
        if not cusolver_available:
            raise RuntimeError('CUSOLVER is not available and '
                               'GpuCusolverSolve Op can not be constructed.')
        if skcuda.__version__ <= '0.5.1':
            warnings.warn('The GpuSolve op requires scikit-cuda > 0.5.1 to work with CUDA 8')
        context_name = infer_context_name(inp1, inp2)

        inp1 = as_gpuarray_variable(inp1, context_name)
        inp2 = as_gpuarray_variable(inp2, context_name)

        inp1 = gpu_contiguous(inp1)
        inp2 = gpu_contiguous(inp2)

        # this op can only operate on float32 matrices
        assert inp1.ndim == 2
        assert inp2.ndim == 2
        assert inp1.dtype == 'float32'
        assert inp2.dtype == 'float32'

        return theano.Apply(
            self, [inp1, inp2],
            [GpuArrayType('float32',
                          broadcastable=inp1.broadcastable,
                          context_name=context_name)()])

    def prepare_node(self, node, storage_map, compute_map, impl):
        ctx = node.inputs[0].type.context
        attach_cusolver_handle_to_context(ctx)

    def check_dev_info(self, dev_info):
        val = np.asarray(dev_info)[0]
        if val > 0:
            raise LinAlgError('A is singular')

    def perform(self, node, inputs, outputs):
        context = inputs[0][0].context

        # Size of the matrices to invert.
        z = outputs[0]

        # Matrix.
        A = inputs[0]

        # Solution vectors.
        b = inputs[1]

        assert(len(A.shape) == 2)
        assert(len(b.shape) == 2)

        if self.trans in ['T', 'C']:
            trans = 1
            l, n = A.shape
            k, m = b.shape
        elif self.trans == 'N':
            trans = 0
            n, l = A.shape
            k, m = b.shape
        else:
            raise ValueError('Invalid value for trans')
        if l != n:
            raise ValueError('A must be a square matrix')
        if n != k:
            raise ValueError('A and b must be aligned.')

        lda = max(1, n)
        ldb = max(1, k)

        # We copy A and b as cusolver operates inplace
        b = pygpu.array(b, copy=True, order='F')
        if not self.inplace:
            A = pygpu.array(A, copy=True)
        A_ptr = A.gpudata
        b_ptr = b.gpudata

        # cusolver expects a F ordered matrix, but A is not explicitly
        # converted between C and F order, instead we switch the
        # "transpose" flag.
        if A.flags['C_CONTIGUOUS']:
            trans = 1 - trans

        if self.A_structure == 'symmetric':
            with context:
                workspace_size = cusolver.cusolverDnSpotrf_bufferSize(
                    context.cusolver_handle, 0, n, A_ptr, lda)

            workspace = pygpu.zeros(workspace_size, dtype='float32',
                                    context=context)

            dev_info = pygpu.zeros((1,), dtype='int32', context=context)

            workspace_ptr = workspace.gpudata
            dev_info_ptr = dev_info.gpudata

            with context:
                cusolver.cusolverDnSpotrf(
                    context.cusolver_handle, 0, n, A_ptr, lda, workspace_ptr,
                    workspace_size, dev_info_ptr)
                self.check_dev_info(dev_info)

                cusolverDnSpotrs(
                    context.cusolver_handle, 0, n, m, A_ptr, lda,
                    b_ptr, ldb, dev_info_ptr)

        else:
            # general case for A
            with context:
                workspace_size = cusolver.cusolverDnSgetrf_bufferSize(
                    context.cusolver_handle, n, n, A_ptr, lda)

            workspace = pygpu.zeros(workspace_size, dtype='float32',
                                    context=context)

            pivots = pygpu.zeros(n, dtype='int32', context=context)

            dev_info = pygpu.zeros((1,), dtype='int32', context=context)

            workspace_ptr = workspace.gpudata
            pivots_ptr = pivots.gpudata
            dev_info_ptr = dev_info.gpudata

            with context:
                cusolver.cusolverDnSgetrf(
                    context.cusolver_handle, n, n, A_ptr, lda, workspace_ptr,
                    pivots_ptr, dev_info_ptr)
                self.check_dev_info(dev_info)

                cusolver.cusolverDnSgetrs(
                    context.cusolver_handle, trans, n, m, A_ptr, lda,
                    pivots_ptr, b_ptr, ldb, dev_info_ptr)

        z[0] = b


class GpuCublasTriangularSolve(Op):
    """
    CUBLAS GPU Triangular Solve Op.

    Parameters
    ----------
    lower
        Whether system is lower-triangular (True) or upper-triangular (False).
    trans
        Whether to take the transpose of the input matrix or not.
    """
    __props__ = ('trans', 'lower')

    def __init__(self, lower=True, trans='N'):
        self.trans = trans
        self.lower = lower
        super(GpuCublasTriangularSolve, self).__init__()

    def make_node(self, inp1, inp2):
        if not cublas_available:
            raise RuntimeError('CUBLAS is not available and '
                               'GpuCublasTriangularSolve Op can not be constructed.')
        context_name = infer_context_name(inp1, inp2)

        inp1 = as_gpuarray_variable(inp1, context_name)
        inp2 = as_gpuarray_variable(inp2, context_name)

        inp1 = gpu_contiguous(inp1)
        inp2 = gpu_contiguous(inp2)

        # this op can only operate on float32 matrices
        assert inp1.ndim == 2
        assert inp2.ndim in [1, 2]
        assert inp1.dtype == 'float32'
        assert inp2.dtype == 'float32'

        return theano.Apply(self, [inp1, inp2],
                            [GpuArrayType('float32',
                                          broadcastable=inp2.broadcastable,
                                          context_name=context_name)()])

    def prepare_node(self, node, storage_map, compute_map, impl):
        ctx = node.inputs[0].type.context
        attach_cublas_handle_to_context(ctx)

    def perform(self, node, inputs, outputs):
        ctx = node.inputs[0].type.context

        # Solution set
        x = outputs[0]

        # Matrix.
        A = inputs[0]

        # right hand side
        b = inputs[1]

        assert(len(A.shape) == 2)
        assert(len(b.shape) in [1, 2])

        # implicitly deal with the difference between C order
        # and fortran order by flipping the trans and lower flags
        lower = not self.lower
        trans = self.trans
        if trans in ['T', 'C']:
            trans = 'N'
            l, n = A.shape
        elif trans == 'N':
            trans = 'T'
            n, l = A.shape
        else:
            raise ValueError('Invalid value for trans')

        if b.ndim == 2:
            k, m = b.shape
        else:
            k, = b.shape
            m = 1

        if l != n:
            raise ValueError('A must be a square matrix')
        if n != k:
            raise ValueError('A and b must be aligned.')

        lda = max(1, n)
        ldb = max(1, k)

        # solution overwrites right hand side on exit
        b = pygpu.array(b, copy=True, order='F')

        A_ptr = A.gpudata
        b_ptr = b.gpudata

        # unit scalar used for multiplication
        alpha = 1.0
        # indicates matrix A is on left of B
        side = 'l'
        # set whether upper or lower part of matrix A stored
        uplo = 'l' if lower else 'u'
        # indicates elements on diagonal of matrix A may not be unity
        diag = 'n'

        with ctx:
            if b.ndim == 1:
                # matrix vector solve
                cublas.cublasStrsv(ctx.cublas_handle, uplo, trans, diag, n,
                                   A_ptr, lda, b_ptr, 1)
            else:
                cublas.cublasStrsm(ctx.cublas_handle, side, uplo, trans, diag,
                                   n, m, alpha, A_ptr, lda, b_ptr, ldb)

        x[0] = b


def gpu_solve(A, b, A_structure='general', trans='N'):
    if A_structure == 'lower':
        return GpuCublasTriangularSolve(True, trans)(A, b)
    elif A_structure == 'upper':
        return GpuCublasTriangularSolve(False, trans)(A, b)

    return GpuCusolverSolve(A_structure, trans)(A, b)


class GpuCholesky(Op):
    """
    CUSOLVER GPU Cholesky Op.

    Given a real positive definite matrix `A` returns either a lower
    triangular matrix `L` such that `A == dot(L, L.T)` if `lower == True`
    else returns an upper triangular matrix `U` such that `A == dot(U.T, U)`
    if `lower == False`.

    Parameters
    ----------
    lower
        Whether to return a lower rather than upper triangular decomposition.

    """

    __props__ = ('lower', 'inplace')

    def __init__(self, lower=True, inplace=False):
        self.lower = lower
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}
        super(GpuCholesky, self).__init__()

    def clone_inplace(self):
        return self.__class__(lower=self.lower, inplace=True)

    def make_node(self, inp):
        if not cusolver_available:
            raise RuntimeError('CUSOLVER is not available and '
                               'GpuCholesky Op can not be constructed.')
        if skcuda.__version__ <= '0.5.1':
            warnings.warn('The GpuCholesky op requires scikit-cuda > 0.5.1 to work with CUDA 8')
        if not pygpu_available:
            raise RuntimeError('Missing pygpu or triu/tril functions.'
                               'Install or update libgpuarray.')
        context_name = infer_context_name(inp)

        inp = as_gpuarray_variable(inp, context_name)

        inp = gpu_contiguous(inp)

        # this op can only operate on float32 matrices
        # because of current implementation of triu/tril.
        # TODO: support float64 for triu/tril in GpuArray and for GpuCholesky/GpuCusolverSolve in Theano.
        assert inp.ndim == 2
        assert inp.dtype == 'float32'

        return theano.Apply(self, [inp], [inp.type()])

    def prepare_node(self, node, storage_map, compute_map, impl):
        ctx = node.inputs[0].type.context
        attach_cusolver_handle_to_context(ctx)

    def perform(self, node, inputs, outputs):
        context = inputs[0][0].context

        # Input matrix.
        A = inputs[0]

        l, n = A.shape
        if l != n:
            raise ValueError('A must be a square matrix')

        lda = max(1, n)

        # cusolver operates on F ordered matrices, but A is expected
        # to be symmetric so it does not matter.
        # We copy A if needed
        if self.inplace:
            L = A
        else:
            L = pygpu.array(A, copy=True)

        # The output matrix will contain only the upper or lower
        # triangular factorization of A. If L is C ordered (it
        # probably is as it is the default in Theano) we just switch
        # the fill mode parameter of cusolver
        l_parameter = 0 if self.lower else 1
        if L.flags['C_CONTIGUOUS']:
            l_parameter = 1 - l_parameter

        L_ptr = L.gpudata

        with context:
            workspace_size = cusolver.cusolverDnSpotrf_bufferSize(
                context.cusolver_handle, l_parameter, n, L_ptr, lda)

            workspace = pygpu.zeros(workspace_size, dtype='float32',
                                    context=context)

            dev_info = pygpu.zeros((1,), dtype='int32', context=context)

            workspace_ptr = workspace.gpudata
            dev_info_ptr = dev_info.gpudata

            cusolver.cusolverDnSpotrf(
                context.cusolver_handle, l_parameter, n, L_ptr, lda, workspace_ptr,
                workspace_size, dev_info_ptr)

            val_dev_info = np.asarray(dev_info)[0]
            if val_dev_info > 0:
                raise LinAlgError('Cholesky decomposition failed (is A SPD?)')

        # cusolver leaves the elements in the matrix outside the considered
        # upper or lower triangle unchanged, so we need to put zeros outside
        # the triangle
        if self.lower:
            tril(L)
        else:
            triu(L)

        outputs[0][0] = L


def gpu_cholesky(A, lower=True):
    return GpuCholesky(lower)(A)


# TODO: add support for float64
class GpuMagmaBase(COp):
    """Base class for magma related operations. Add the necessary headers,
    libraries and optionally the location of headers and library.
    """
    def c_headers(self):
        return ['gpuarray/types.h', 'gpuarray/array.h', 'gpuarray/ext_cuda.h',
                'gpuarray_helper.h', 'magma.h']

    def c_header_dirs(self):
        dirs = [gpuarray_helper_inc_dir(), pygpu.get_include()]
        if config.magma.include_path:
            dirs.append(config.magma.include_path)
        return dirs

    def c_libraries(self):
        return ['magma']

    def c_lib_dirs(self):
        if config.magma.library_path:
            return [config.magma.library_path]
        return []

    def prepare_node(self, node, storage_map, compute_map, impl):
        from skcuda.magma import magma_init
        ctx = node.inputs[0].type.context
        if not getattr(ctx, 'is_magma_initialized', False):
            with ctx:
                magma_init()
                ctx.is_magma_initialized = True


class GpuMagmaSVD(GpuMagmaBase):
    """Computes the svd of a matrix :math:`A` using magma library.

    .. warning::

        Because of implementation constraints, this Op returns outputs
        in order ``S, U, VT``. Use :func:`theano.gpuarray.linalg.gpu_svd`
        to get them in expected order ``U, S, VT``.

    """
    __props__ = ('full_matrices', 'compute_uv')
    _cop_num_inputs = 1
    _cop_num_outputs = 3
    check_input = False
    params_type = ParamsType(full_matrices=bool_t, context=gpu_context_type)

    def __init__(self, full_matrices=True, compute_uv=True):
        self.full_matrices = full_matrices
        self.compute_uv = compute_uv
        COp.__init__(self, ['c_code/magma_svd.c'], 'APPLY_SPECIFIC(magma_svd)')

    def make_node(self, A):
        ctx_name = infer_context_name(A)
        A = as_gpuarray_variable(A, ctx_name)
        A = gpu_contiguous(A)
        if A.ndim != 2:
            raise LinAlgError("Matrix rank error")
        if A.dtype != 'float32':
            raise TypeError("only `float32` is supported for now")
        if self.compute_uv:
            return theano.Apply(self, [A],
                                # return S, U, VT
                                [GpuArrayType(A.dtype, broadcastable=[False],
                                              context_name=ctx_name)(),
                                 A.type(),
                                 A.type()])
        else:
            return theano.Apply(self, [A],
                                # return only S
                                [GpuArrayType(A.dtype, broadcastable=[False],
                                              context_name=ctx_name)()])

    def prepare_node(self, node, storage_map, compute_map, impl):
        super(GpuMagmaSVD, self).prepare_node(node, storage_map, compute_map, impl)
        # Check node to prevent eventual errors with old pickled nodes.
        if self.compute_uv:
            A, B, C = node.outputs
            # We expect order: S (vector), U (matrix), VT (matrix)
            assert A.type.ndim == 1 and B.type.ndim == C.type.ndim == 2, \
                "Due to implementation constraints, GpuMagmaSVD interface has changed and now returns (S, U, VT) " \
                "instead of (U, S, VT). Either update your code, or use gpu_svd() to get the expected (U, S, VT) order."

    def get_params(self, node):
        return self.params_type.get_params(self, context=node.inputs[0].type.context)

    def infer_shape(self, node, shapes):
        x_shape, = shapes
        M, N = x_shape
        K = tensor.minimum(M, N)
        s_shape = (K, )
        if self.compute_uv:
            u_shape = (M, M) if self.full_matrices else (M, K)
            vt_shape = (N, N) if self.full_matrices else (K, N)
            return [s_shape, u_shape, vt_shape]
        else:
            return [s_shape]


def gpu_svd(a, full_matrices=1, compute_uv=1):
    """
    This function performs the SVD on GPU.

    Parameters
    ----------
    full_matrices : bool, optional
        If True (default), u and v have the shapes (M, M) and (N, N),
        respectively.
        Otherwise, the shapes are (M, K) and (K, N), respectively,
        where K = min(M, N).
    compute_uv : bool, optional
        Whether or not to compute u and v in addition to s.
        True by default.

    Returns
    -------
    U, V,  D : matrices

    """
    out = GpuMagmaSVD(full_matrices, compute_uv)(a)
    if compute_uv:
        S, U, VT = out
        out = [U, S, VT]
    return out


class GpuMagmaMatrixInverse(GpuMagmaBase):
    """Computes the inverse of a matrix :math:`A` using magma library.
    """
    __props__ = ('inplace', )
    check_input = False
    params_type = ParamsType(inplace=bool_t, context=gpu_context_type)

    def __init__(self, inplace=False):
        COp.__init__(self, ['c_code/magma_inv.c'], 'APPLY_SPECIFIC(magma_inv)')
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def clone_inplace(self):
        return self.__class__(inplace=True)

    def make_node(self, A):
        ctx_name = infer_context_name(A)
        A = as_gpuarray_variable(A, ctx_name)
        A = gpu_contiguous(A)
        if A.ndim != 2:
            raise LinAlgError("Matrix rank error")
        if A.dtype != 'float32':
            raise TypeError("only `float32` is supported for now")
        return theano.Apply(self, [A], [A.type()])

    def get_params(self, node):
        return self.params_type.get_params(self, context=node.inputs[0].type.context)

    def infer_shape(self, node, shapes):
        return shapes


def gpu_matrix_inverse(a):
    """
    This function performs the matrix inverse on GPU.

    Returns
    -------
    a_inv: matrix

    """
    return GpuMagmaMatrixInverse()(a)


class GpuMagmaCholesky(GpuMagmaBase, CGpuKernelBase):
    """Computes the cholesky decomposition of a matrix :math:`A` using magma
    library.

    """
    __props__ = ('lower', 'inplace')
    check_input = False
    params_type = ParamsType(lower=bool_t, inplace=bool_t, context=gpu_context_type)

    def __init__(self, lower=True, inplace=False):
        self.lower = lower
        COp.__init__(self, ['c_code/magma_cholesky.c'], 'APPLY_SPECIFIC(magma_cholesky)')
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def clone_inplace(self):
        return self.__class__(lower=self.lower, inplace=True)

    def make_node(self, A):
        ctx_name = infer_context_name(A)
        A = as_gpuarray_variable(A, ctx_name)
        A = gpu_contiguous(A)
        if A.ndim != 2:
            raise LinAlgError("Matrix rank error")
        if A.dtype != 'float32':
            raise TypeError("only `float32` is supported for now")
        return theano.Apply(self, [A], [A.type()])

    def get_params(self, node):
        return self.params_type.get_params(self, context=node.inputs[0].type.context)

    def infer_shape(self, node, shapes):
        return [shapes[0]]


class GpuMagmaQR(GpuMagmaBase, CGpuKernelBase):
    """Computes the qr decomposition of a matrix :math:`A` using magma
    library.

    Parameters
    ----------
    complete : If `False`, returns only r.

    .. warning::

        Because of implementation constraints, this Op returns outputs
        in order ``R, Q``. Use :func:`theano.gpuarray.linalg.gpu_qr`
        to get them in expected order ``Q, R``.
    """
    __props__ = ('complete', )
    _cop_num_inputs = 1
    _cop_num_outputs = 2
    check_input = False
    params_type = ParamsType(complete=bool_t, context=gpu_context_type)

    def __init__(self, complete=True):
        self.complete = complete
        COp.__init__(self, ['c_code/magma_qr.c'], 'APPLY_SPECIFIC(magma_qr)')

    def make_node(self, A):
        ctx_name = infer_context_name(A)
        A = as_gpuarray_variable(A, ctx_name)
        A = gpu_contiguous(A)
        if A.ndim != 2:
            raise LinAlgError("Matrix rank error")
        if A.dtype != 'float32':
            raise TypeError("only `float32` is supported for now")
        if self.complete:
            return theano.Apply(self, [A],
                                # return R, Q
                                [A.type(), A.type()])
        else:
            return theano.Apply(self, [A],
                                # return R
                                [A.type()])

    def get_params(self, node):
        return self.params_type.get_params(self, context=node.inputs[0].type.context)


def gpu_qr(a, complete=True):
    """
    This function performs the QR on GPU.

    Parameters
    ----------
    complete : bool, optional
        If `False`, returns only r.

    Returns
    -------
    Q, R : matrices

    """
    out = GpuMagmaQR(complete)(a)
    if complete:
        R, Q = out
        out = [Q, R]
    return out


class GpuMagmaEigh(GpuMagmaBase):
    """Computes the eigen decomposition of a symmetric matrix :math:`A` using magma
    library.

    Parameters
    ----------
    UPLO : Specifies whether the calculation is done with the lower triangular
           part of matrix (`L`, default) or the upper triangular part (`U`).
    compute_v : If `True`, computes eigenvalues and eigenvectors (`True`,
                default). If `False`, computes only eigenvalues of matrix.
    """
    __props__ = ('lower', 'compute_v')
    _cop_num_inputs = 1
    _cop_num_outputs = 2
    check_input = False
    params_type = ParamsType(lower=bool_t, compute_v=bool_t,
                             context=gpu_context_type)

    def __init__(self, UPLO='L', compute_v=True):
        assert UPLO in ['L', 'U']
        self.lower = UPLO == 'L'
        self.compute_v = compute_v
        COp.__init__(self, ['c_code/magma_eigh.c'], 'APPLY_SPECIFIC(magma_eigh)')

    def make_node(self, A):
        ctx_name = infer_context_name(A)
        A = as_gpuarray_variable(A, ctx_name)
        A = gpu_contiguous(A)
        if A.ndim != 2:
            raise LinAlgError("Matrix rank error")
        if A.dtype != 'float32':
            raise TypeError("only `float32` is supported for now")
        if self.compute_v:
            return theano.Apply(self, [A],
                                # return D, V
                                [GpuArrayType(A.dtype, broadcastable=[False],
                                              context_name=ctx_name)(),
                                 A.type()])
        else:
            return theano.Apply(self, [A],
                                # return D
                                [GpuArrayType(A.dtype, broadcastable=[False],
                                              context_name=ctx_name)()])

    def get_params(self, node):
        return self.params_type.get_params(self, context=node.inputs[0].type.context)
