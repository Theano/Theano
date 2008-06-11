from __future__ import absolute_import
import numpy

from ..gof.link import WrapLinker
from ..gradient import numeric_grad

class Todo(Exception): """todo"""

#WrapLinker wrappers
def cmp_outputs(i, node, *thunks):
    """WrapLinker wrapper: raise an exception if outputs are different

    numpy.ndarrays of floating point types are compared approximately, rather
    than exactly.

    """
    class MisMatch(Exception): """Output mismatch"""

    #define a comparison function, which works for all the results in a graph
    #TODO: consider factoring this out (and maybe passing args explicitly
    # instead of by closure)
    def my_check_equal(x, y):
        x, y = x[0], y[0]
        if type(x) != type(y):
            raise MisMatch("Output type mismatch", (x, y))

        if hasattr(x, 'dtype'):
            # was: isinstance(x,numpy.ndarray), which doesn't
            # catch numpy.float64
            if x.dtype != y.dtype or x.shape != y.shape:
                raise MisMatch("ndarray type/shape.", (x,y))

            if str(x.dtype).startswith('float'):
                assert str(x.dtype) == 'float64' #otherwise we need to adjust
                #our constant below... but to what?
                abs_rel_err = numeric_grad.abs_rel_err(x, y)
                max_abs_rel_err = numpy.max(abs_rel_err)
                if max_abs_rel_err > 1.0e-7:
                    raise MisMatch('max_abs_rel_err exceeds tolerence', (max_abs_rel_err,
                        x, y))

            elif str(x.dtype).startswith('complex'):
                raise Todo()

            else:
                if not numpy.all(x==y):
                    raise MisMatch

        else:
            print 'wtf??', type(x), type(y), node.op
            if x != y:
                print 'wow!! wtf??'
                raise MisMatch("Output mismatch.", (x, y))

    #loop over all the thunks
    # ensure that the outputs from the first thunk match the outputs from
    # all subsequent thunks
    n_thunks = len(thunks)
    if n_thunks > 1:
        th0 = thunks[0]
        for th in thunks[1:]:
            for out0, outN in zip(th0.outputs, th.outputs):
                my_check_equal(out0[0], outN[0])

#TODO: better name for 'f'
def numpy_wrapper(f):
    def wrapper(i, node, *thunks):
        """WrapLinker wrapper: raise an exception if a NaN is found in outputs
        """
        for thunk in thunks:
            for output in thunk.outputs:
                if hasattr(output[0], 'dtype'):
                    if f(output[0]):
                        raise Exception('uh oh', (i, node, thunk, output[0]))
    return wrapper

numpy_any_isinf = numpy_wrapper(lambda a:numpy.any(numpy.isinf(a)))
numpy_any_isnan = numpy_wrapper(lambda a:numpy.any(numpy.isnan(a)))
numpy_notall_isfinite = numpy_wrapper(lambda a: not numpy.all(numpy.isfinite(a)))

def run_all(i, node, *thunks):
    """WrapLinker wrapper: run the thunks
    """
    for th in thunks:
        th()



def WrapLinkerMany(linkers, wrappers):
    """ Variant on WrapLinker that runs a series of wrapper functions instead of
    just one.
    """
    def wrapper(*args):
        for f in wrappers:
            f(*args)
    return WrapLinker(linkers, wrapper)


def DualLinker(linkers):
    return WrapLinkerMany(linkers, [run_all, cmp_outputs])

