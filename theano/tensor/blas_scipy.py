"""
Implementations of BLAS Ops based on scipy's BLAS bindings.
"""
import numpy

from blas import Ger, ger, ger_destructive
from blas import blas_optdb, optdb,local_optimizer

from theano.tensor.opt import in2out

try:
    import scipy.linalg.blas
    have_fblas = True
    _blas_gemv_fns = {
            numpy.dtype('float32'):scipy.linalg.blas.fblas.sgemv,
            numpy.dtype('float64'):scipy.linalg.blas.fblas.dgemv,
            numpy.dtype('complex64'):scipy.linalg.blas.fblas.cgemv,
            numpy.dtype('complex128'):scipy.linalg.blas.fblas.zgemv,
            }
    _blas_ger_fns = {
            numpy.dtype('float32'):scipy.linalg.blas.fblas.sger,
            numpy.dtype('float64'):scipy.linalg.blas.fblas.dger,
            numpy.dtype('complex64'):scipy.linalg.blas.fblas.cgeru,
            numpy.dtype('complex128'):scipy.linalg.blas.fblas.zgeru,
            }
except ImportError, e:
    have_fblas = False


class ScipyGer(Ger):

    # keep everything else, but override the make_thunk
    def make_thunk(self, node, storage_map, compute_map, no_recycling):

        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]

        # get vars for containers
        cA, calpha, cx, cy = node_input_storage
        cZ, = node_output_storage
        local_ger = _blas_ger_fns[numpy.dtype(node.inputs[0].type.dtype)]

        def rval():
            # N.B. some versions of scipy (e.g. mine) don't actually work
            # in-place on a, even when I tell it to.
            A = cA[0]
            if A.flags['C_CONTIGUOUS']:
                A = local_ger(calpha[0], cy[0], cx[0], a=A.T,
                        overwrite_a=int(self.destructive)).T
            else:
                A = local_ger(calpha[0], cx[0], cy[0], a=A,
                        overwrite_a=int(self.destructive))
            cZ[0] = A

        #TODO: If this is currently an unofficial part of the thunk API,
        #      then maybe it should be documented and made official?
        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.lazy = False
        return rval

@local_optimizer([ger, ger_destructive])
def use_scipy_ger(node):
    if node.op == ger:
        return [ScipyGer(False)(*node.inputs)]

@local_optimizer([ScipyGer(False)])
def make_ger_destructive(node):
    if node.op == ScipyGer(False):
        return [ScipyGer(True)(*node.inputs)]

use_scipy_blas = in2out(use_scipy_ger)
make_scipy_blas_destructive = in2out(make_ger_destructive)

if have_fblas:
    # scipy_blas is scheduled in the blas_optdb very late, because scipy sortof
    # sucks, but it is almost always present.
    # C implementations should be scheduled earlier than this, so that they take
    # precedence. Once the original Ger is replaced, then these optimizations
    # have no effect.
    blas_optdb.register('scipy_blas',
        use_scipy_blas,
        100, 'fast_run')

    # this matches the InplaceBlasOpt defined in blas.py
    optdb.register('make_scipy_blas_destructive',
            make_scipy_blas_destructive,
            70.0, 'fast_run', 'inplace')
