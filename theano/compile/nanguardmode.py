import logging
import collections
import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.cuda as cuda
from theano.compile import Mode


logger = logging.getLogger("theano.compile.nanguardmode")


def flatten(l):
    """
    Turns a nested graph of lists/tuples/other objects into a list of objects.

    Parameters
    ----------
    l : List/tuple/other objects, might be nested.

    Returns
    -------
    A flattened list of objects
    """
    if isinstance(l, (list, tuple, collections.ValuesView)):
        rval = []
        for elem in l:
            if isinstance(elem, (list, tuple)):
                rval.extend(flatten(elem))
            else:
                rval.append(elem)
    else:
        return [l]
    return rval


def contains_nan(arr):
    """
    Test whether a numpy.ndarray contains any `np.nan` values.

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    contains_nan : bool
        `True` if the array contains any `np.nan` values, `False` otherwise.

    Notes
    -----
    Tests for the presence of `np.nan`'s using `np.isnan(np.min(ndarray))`.
    This approach is faster and more memory efficient than the obvious
    alternative, calling `np.any(np.isnan(ndarray))`, which requires the
    construction of a boolean array with the same shape as the input array.
    """
    if isinstance(arr, theano.gof.type.CDataType._cdata_type):
        return False
    elif isinstance(arr, np.random.mtrand.RandomState):
        return False
    return np.isnan(np.min(arr))


def contains_inf(arr):
    """
    Test whether a numpy.ndarray contains any `np.inf` values.

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    contains_inf : bool
        `True` if the array contains any `np.inf` values, `False` otherwise.

    Notes
    -----
    Tests for the presence of `np.inf`'s by determining whether the
    values returned by `np.nanmin(arr)` and `np.nanmax(arr)` are finite.
    This approach is more memory efficient than the obvious alternative,
    calling `np.any(np.isinf(ndarray))`, which requires the construction of a
    boolean array with the same shape as the input array.
    """
    if isinstance(arr, theano.gof.type.CDataType._cdata_type):
        return False
    elif isinstance(arr, np.random.mtrand.RandomState):
        return False
    return np.isinf(np.nanmax(arr)) or np.isinf(np.nanmin(arr))


class NanGuardMode(Mode):
    """
    A Theano compilation Mode that makes the compiled function automatically
    detect NaNs and Infs and detect an error if they occur.

    Parameters
    ----------
    nan_is_error : bool
        If True, raise an error anytime a NaN is encountered
    inf_is_error: bool
        If True, raise an error anytime an Inf is encountered.  Note that some
        pylearn2 modules currently use np.inf as a default value (e.g.
        mlp.max_pool) and these will cause an error if inf_is_error is True.
    big_is_error: bool
        If True, raise an error when a value greater than 1e10 is encountered.
    """
    def __init__(self, nan_is_error, inf_is_error, big_is_error=True):
        if cuda.cuda_available:
            self.guard_input = cuda.fvector('nan_guard')
            if nan_is_error or inf_is_error:
                self.gpumin = theano.function(
                    [self.guard_input], T.min(self.guard_input),
                    mode='FAST_RUN'
                )
            if inf_is_error:
                self.gpumax = theano.function(
                    [self.guard_input], T.max(self.guard_input),
                    mode='FAST_RUN'
                )
            if big_is_error:
                self.gpuabsmax = theano.function(
                    [self.guard_input], T.max(T.abs_(self.guard_input)),
                    mode='FAST_RUN'
                )

        def do_check_on(var, nd, f, is_input):
            """
            Checks `var` for NaNs / Infs. If detected, raises an exception
            and / or prints information about `nd`, `f`, and `is_input` to
            help the user determine the cause of the invalid values.

            Parameters
            ----------
            var : numpy.ndarray
                The value to be checked.
            nd : theano.gof.Apply
                The Apply node being executed
            f : callable
                The thunk for the apply node
            is_input : bool
                If True, `var` is an input to `nd`.
                If False, it is an output.
            """
            error = False
            if nan_is_error:
                err = False
                if cuda.cuda_available and isinstance(var, cuda.CudaNdarray):
                    err = np.isnan(self.gpumin(var.reshape(var.size)))
                else:
                    err = contains_nan(var)
                if err:
                    logger.error('NaN detected')
                    error = True
            if inf_is_error:
                err = False
                if cuda.cuda_available and isinstance(var, cuda.CudaNdarray):
                    err = (np.isinf(self.gpumin(var.reshape(var.size))) or
                           np.isinf(self.gpumax(var.reshape(var.size))))
                else:
                    err = contains_inf(var)
                if err:
                    logger.error('Inf detected')
                    error = True
            if big_is_error:
                err = False
                if cuda.cuda_available and isinstance(var, cuda.CudaNdarray):
                    err = (self.gpuabsmax(var.reshape(var.size)) > 1e10)
                elif isinstance(var, theano.gof.type.CDataType._cdata_type):
                    err = False
                elif isinstance(var, np.random.mtrand.RandomState):
                    err = False
                else:
                    err = (np.abs(var).max() > 1e10)
                if err:
                    logger.error('Big value detected')
                    error = True
            if error:
                if is_input:
                    logger.error('In an input')
                else:
                    logger.error('In an output')
                logger.error('Inputs: ')
                for ivar, ival in zip(nd.inputs, f.inputs):
                    logger.error('var')
                    logger.error(ivar)
                    logger.error(theano.printing.min_informative_str(ivar))
                    logger.error('val')
                    logger.error(ival)
                logger.error('Node:')
                logger.error(nd)
                assert False

        def nan_check(i, node, fn):
            """
            Runs `fn` while checking its inputs and outputs for NaNs / Infs

            Parameters
            ----------
            i : currently ignored (TODO: determine why it is here or remove)
            node : theano.gof.Apply
                The Apply node currently being executed
            fn : callable
                The thunk to execute for this Apply node
            """
            inputs = fn.inputs
            # TODO: figure out why individual inputs are themselves lists
            # sometimes
            for x in flatten(inputs):
                do_check_on(x, node, fn, True)
            fn()
            outputs = fn.outputs
            for j, x in enumerate(flatten(outputs)):
                do_check_on(x, node, fn, False)

        wrap_linker = theano.gof.WrapLinker([theano.gof.OpWiseCLinker()],
                                            nan_check)
        super(NanGuardMode, self).__init__(wrap_linker,
                                           optimizer=theano.config.optimizer)
