from __future__ import absolute_import, print_function, division
import numpy as np
import warnings
import theano
from theano import Op, Apply
import theano.tensor as T
from theano.scalar import as_scalar
import copy

class MultinomialFromUniform(Op):
    """
    Converts samples from a uniform distribution into samples from a multinomial distribution.
    
    Parameters
    ----------
    odtype : str
        Output data type, 'auto' for automatic determination.
    """

    __props__ = ("odtype",)

    def __init__(self, odtype):
        self.odtype = odtype

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.odtype)

    def __setstate__(self, dct):
        self.__dict__.update(dct)
        try:
            self.odtype
        except AttributeError:
            self.odtype = 'auto'

    def make_node(self, pvals, unis, n=1):
        pvals = T.as_tensor_variable(pvals)
        unis = T.as_tensor_variable(unis)
        if pvals.ndim != 2:
            raise NotImplementedError('pvals ndim should be 2', pvals.ndim)
        if unis.ndim != 1:
            raise NotImplementedError('unis ndim should be 1', unis.ndim)
        if self.odtype == 'auto':
            odtype = pvals.dtype
        else:
            odtype = self.odtype
        out = T.tensor(dtype=odtype, broadcastable=pvals.type.broadcastable)
        return Apply(self, [pvals, unis, as_scalar(n)], [out])

    def grad(self, ins, outgrads):
        pvals, unis, n = ins
        (gz,) = outgrads
        return [T.zeros_like(x, dtype=theano.config.floatX) if x.dtype in
                T.discrete_dtypes else T.zeros_like(x) for x in ins]

    def c_code_cache_version(self):
        return (8,)

    def c_code(self, node, name, ins, outs, sub):
        if len(ins) == 2:
            (pvals, unis) = ins
            n = 1
        else:
            (pvals, unis, n) = ins
        (z,) = outs

        return """
        if (PyArray_NDIM(%(pvals)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals ndim should be 2");
            %(fail)s;
        }
        if (PyArray_NDIM(%(unis)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "unis ndim should be 1");
            %(fail)s;
        }
        if (PyArray_DIMS(%(unis)s)[0] != (PyArray_DIMS(%(pvals)s)[0] * %(n)s))
        {
            PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0] * n");
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || ((PyArray_DIMS(%(z)s))[0] != (PyArray_DIMS(%(pvals)s))[0])
            || ((PyArray_DIMS(%(z)s))[1] != (PyArray_DIMS(%(pvals)s))[1])
        )
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*) PyArray_EMPTY(2,
                PyArray_DIMS(%(pvals)s),
                NPY_FLOAT64,
                0);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }

        { // NESTED SCOPE

        const int nb_multi = PyArray_DIMS(%(pvals)s)[0];
        const int nb_outcomes = PyArray_DIMS(%(pvals)s)[1];
        const int n_samples = %(n)s;

        for (int c = 0; c < n_samples; ++c){
            for (int n = 0; n < nb_multi; ++n)
            {
                int waiting = 1;
                double cummul = 0.;
                const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR1(%(unis)s, c*nb_multi + n);
                for (int m = 0; m < nb_outcomes; ++m)
                {
                    dtype_%(z)s* z_nm = (dtype_%(z)s*)PyArray_GETPTR2(%(z)s, n,m);
                    const dtype_%(pvals)s* pvals_nm = (dtype_%(pvals)s*)PyArray_GETPTR2(%(pvals)s, n,m);
                    cummul += *pvals_nm;
                    if (c == 0)
                    {
                        if (waiting && (cummul > *unis_n))
                        {
                            *z_nm = 1.;
                            waiting = 0;
                        }
                        else
                        {
                            *z_nm = 0.;
                        }
                    }
                    else {
                        if (cummul > *unis_n)
                        {
                            *z_nm = *z_nm + 1.;
                            break;
                        }
                    }
                }
            }
        }
        } // END NESTED SCOPE
        """ % locals()

    def perform(self, node, ins, outs):
        if len(ins) == 2:
            (pvals, unis) = ins
            n_samples = 1
        else:
            (pvals, unis, n_samples) = ins
        (z,) = outs

        if unis.shape[0] != pvals.shape[0] * n_samples:
            raise ValueError("unis.shape[0] != pvals.shape[0] * n_samples",
                             unis.shape[0], pvals.shape[0], n_samples)
        if z[0] is None or z[0].shape != pvals.shape:
            z[0] = np.zeros(pvals.shape, dtype=node.outputs[0].dtype)
        else:
            z[0].fill(0)

        nb_multi = pvals.shape[0]
        for c in range(n_samples):
            for n in range(nb_multi):
                unis_n = unis[c * nb_multi + n]
                cumsum = pvals[n].cumsum(dtype='float64')
                z[0][n, np.searchsorted(cumsum, unis_n)] += 1


class ChoiceFromUniform(MultinomialFromUniform):
    """
    Converts samples from a uniform distribution into samples (without replacement) 
    from a multinomial distribution.
    
    Parameters
    ----------
    odtype : str
        Output data type, 'auto' for automatic determination.
    replace : bool
        Whether to sample with replacement (default is False).
    """

    __props__ = ("odtype", "replace",)

    def __init__(self, odtype, replace=False, *args, **kwargs):
        self.replace = replace
        super(ChoiceFromUniform, self).__init__(odtype=odtype, *args, **kwargs)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "replace" not in state:
            self.replace = False

    def make_node(self, pvals, unis, n=1):
        pvals = T.as_tensor_variable(pvals)
        unis = T.as_tensor_variable(unis)
        if pvals.ndim != 2:
            raise NotImplementedError('pvals ndim should be 2', pvals.ndim)
        if unis.ndim != 1:
            raise NotImplementedError('unis ndim should be 1', unis.ndim)
        if self.odtype == 'auto':
            odtype = 'int64'
        else:
            odtype = self.odtype
        out = T.tensor(dtype=odtype, broadcastable=pvals.type.broadcastable)
        return Apply(self, [pvals, unis, as_scalar(n)], [out])

    def c_code_cache_version(self):
        return (1,)

    def c_code(self, node, name, ins, outs, sub):
        (pvals, unis, n) = ins
        (z,) = outs
        replace = int(self.replace)

        return """
        if (PyArray_NDIM(%(pvals)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals ndim should be 2");
            %(fail)s;
        }
        if (PyArray_NDIM(%(unis)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "unis ndim should be 1");
            %(fail)s;
        }
        if (PyArray_DIMS(%(unis)s)[0] != (PyArray_DIMS(%(pvals)s)[0] * %(n)s))
        {
            PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0] * n");
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || ((PyArray_DIMS(%(z)s))[0] != (PyArray_DIMS(%(pvals)s))[0])
            || ((PyArray_DIMS(%(z)s))[1] != (PyArray_DIMS(%(pvals)s))[1])
        )
        {
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*) PyArray_EMPTY(2,
                PyArray_DIMS(%(pvals)s),
                NPY_INT64,
                0);
            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }

        { // NESTED SCOPE

        const int nb_multi = PyArray_DIMS(%(pvals)s)[0];
        const int nb_outcomes = PyArray_DIMS(%(pvals)s)[1];
        const int n_samples = %(n)s;

        for (int c = 0; c < n_samples; ++c){
            for (int n = 0; n < nb_multi; ++n)
            {
                int waiting = 1;
                double cummul = 0.;
                const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR1(%(unis)s, c*nb_multi + n);
                for (int m = 0; m < nb_outcomes; ++m)
                {
                    dtype_%(z)s* z_nm = (dtype_%(z)s*)PyArray_GETPTR2(%(z)s, n,m);
                    const dtype_%(pvals)s* pvals_nm = (dtype_%(pvals)s*)PyArray_GETPTR2(%(pvals)s, n,m);
                    cummul += *pvals_nm;
                    if (c == 0)
                    {
                        if (waiting && (cummul > *unis_n))
                        {
                            *z_nm = 1.;
                            waiting = 0;
                        }
                        else
                        {
                            *z_nm = 0.;
                        }
                    }
                    else {
                        if (cummul > *unis_n)
                        {
                            *z_nm = *z_nm + 1.;
                            break;
                        }
                    }
                }
            }
        }
        } // END NESTED SCOPE
        """ % locals()

    def perform(self, node, ins, outs):
        if len(ins) == 2:
            (pvals, unis) = ins
            n_samples = 1
        else:
            (pvals, unis, n_samples) = ins
        (z,) = outs

        if unis.shape[0] != pvals.shape[0] * n_samples:
            raise ValueError("unis.shape[0] != pvals.shape[0] * n_samples",
                             unis.shape[0], pvals.shape[0], n_samples)
        if z[0] is None or z[0].shape != pvals.shape:
            z[0] = np.zeros(pvals.shape, dtype=node.outputs[0].dtype)
        else:
            z[0].fill(0)

        nb_multi = pvals.shape[0]
        for c in range(n_samples):
            for n in range(nb_multi):
                unis_n = unis[c * nb_multi + n]
                cumsum = pvals[n].cumsum(dtype='float64')
                z[0][n, np.searchsorted(cumsum, unis_n)] += 1
