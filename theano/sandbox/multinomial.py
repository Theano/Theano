from __future__ import absolute_import, print_function, division
import numpy as np
import warnings

import theano
from theano import Op, Apply
import theano.tensor as T
from theano.scalar import as_scalar
import copy


class MultinomialFromUniform(Op):
    # TODO : need description for parameter 'odtype'
    """
    Converts samples from a uniform into sample from a multinomial.

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
        # support old pickled graphs
        if len(ins) == 2:
            (pvals, unis) = ins
            n = 1
        else:
            (pvals, unis, n) = ins
        (z,) = outs
        if self.odtype == 'auto':
            t = "PyArray_TYPE(%(pvals)s)" % locals()
        else:
            t = theano.scalar.Scalar(self.odtype).dtype_specs()[1]
            if t.startswith('theano_complex'):
                t = t.replace('theano_complex', 'NPY_COMPLEX')
            else:
                t = t.upper()
        fail = sub['fail']
        return """
        if (PyArray_NDIM(%(pvals)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals ndim should be 2");
            %(fail)s;
        }
        if (PyArray_NDIM(%(unis)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "unis ndim should be 2");
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
                %(t)s,
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

        //
        // For each multinomial, loop over each possible outcome
        //
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
                            // if we re-used old z pointer, we have to clear it out.
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
        # support old pickled graphs
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
        # Original version that is not vectorized. I keep it here as
        # it is more readable.
        # For each multinomial, loop over each possible outcome
        # nb_outcomes = pvals.shape[1]
        # for c in range(n_samples):
        #    for n in range(nb_multi):
        #        waiting = True
        #        cummul = 0
        #        unis_n = unis[c * nb_multi + n]
        #        for m in range(nb_outcomes):
        #            cummul += pvals[n, m]
        #            if c == 0:
        #                if (waiting and (cummul > unis_n)):
        #                    z[0][n, m] = 1
        #                    waiting = False
        #                else:
        #                    # Only needed if we don't init the output to 0
        #                    z[0][n, m] = 0
        #            else:
        #                if (cummul > unis_n):
        #                    z[0][n, m] += 1
        #                    break

        # Vectorized version that is much faster as all the looping is
        # done in C even if this make extra work.
        for c in range(n_samples):
            for n in range(nb_multi):
                unis_n = unis[c * nb_multi + n]
                # The dtype='float64' is important. Otherwise we don't
                # have the same answer as the c code as in the c code
                # the cumul is in double precission.
                cumsum = pvals[n].cumsum(dtype='float64')
                z[0][n, np.searchsorted(cumsum, unis_n)] += 1


class ChoiceFromUniform(MultinomialFromUniform):
    """
    Converts samples from a uniform into sample (without replacement) from a
    multinomial.

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
        if self.odtype == 'auto':
            t = "NPY_INT64"
        else:
            t = theano.scalar.Scalar(self.odtype).dtype_specs()[1]
            if t.startswith('theano_complex'):
                t = t.replace('theano_complex', 'NPY_COMPLEX')
            else:
                t = t.upper()
        fail = sub['fail']
        return """
        // create a copy of pvals matrix
        PyArrayObject* pvals_copy = NULL;

        if (PyArray_NDIM(%(pvals)s) != 2)
        {
            PyErr_Format(PyExc_TypeError, "pvals ndim should be 2");
            %(fail)s;
        }
        if (PyArray_NDIM(%(unis)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "unis ndim should be 2");
            %(fail)s;
        }

        if ( %(n)s > (PyArray_DIMS(%(pvals)s)[1]) )
        {
            PyErr_Format(PyExc_ValueError, "Cannot sample without replacement n samples bigger than the size of the distribution.");
            %(fail)s;
        }

        if (PyArray_DIMS(%(unis)s)[0] != (PyArray_DIMS(%(pvals)s)[0] * %(n)s))
        {
            PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0] * n");
            %(fail)s;
        }

        pvals_copy = (PyArrayObject*) PyArray_EMPTY(2,
            PyArray_DIMS(%(pvals)s),
            PyArray_TYPE(%(pvals)s),
            0);

        if (!pvals_copy)
        {
            PyErr_SetString(PyExc_MemoryError, "failed to alloc pvals_copy");
            %(fail)s;
        }
        PyArray_CopyInto(pvals_copy, %(pvals)s);

        if ((NULL == %(z)s)
            || ((PyArray_DIMS(%(z)s))[0] != (PyArray_DIMS(%(pvals)s))[0])
            || ((PyArray_DIMS(%(z)s))[1] != %(n)s)
        )
        {
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = PyArray_DIMS(%(pvals)s)[0];
            dims[1] = %(n)s;
            %(z)s = (PyArrayObject*) PyArray_EMPTY(2,
                dims,
                %(t)s,
                -1);
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

        //
        // For each multinomial, loop over each possible outcome,
        // and set selected pval to 0 after being selected
        //
        for (int c = 0; c < n_samples; ++c){
            for (int n = 0; n < nb_multi; ++n)
            {
                double cummul = 0.;
                const dtype_%(unis)s* unis_n = (dtype_%(unis)s*)PyArray_GETPTR1(%(unis)s, c*nb_multi + n);
                dtype_%(z)s* z_nc = (dtype_%(z)s*)PyArray_GETPTR2(%(z)s, n, c);
                for (int m = 0; m < nb_outcomes; ++m)
                {
                    dtype_%(pvals)s* pvals_nm = (dtype_%(pvals)s*)PyArray_GETPTR2(pvals_copy, n, m);
                    cummul += *pvals_nm;
                    if (cummul > *unis_n)
                    {
                        *z_nc = m;
                        // No need to renormalize after the last samples.
                        if (c == (n_samples - 1))
                            break;
                        if (! %(replace)s )
                        {
                            // renormalize the nth row of pvals, reuse (cummul-*pvals_nm) to initialize the sum
                            dtype_%(pvals)s sum = cummul - *pvals_nm;
                            dtype_%(pvals)s* pvals_n = (dtype_%(pvals)s*)PyArray_GETPTR2(pvals_copy, n, m);
                            *pvals_nm = 0.;
                            for (int k = m; k < nb_outcomes; ++k)
                            {
                                sum = sum + *pvals_n;
                                pvals_n++;
                            }
                            pvals_n = (dtype_%(pvals)s*)PyArray_GETPTR2(pvals_copy, n, 0);
                            for (int k = 0; k < nb_outcomes; ++k)
                            {
                                *pvals_n = *pvals_n / sum;
                                pvals_n++;
                            }
                        }
                        break;
                    }
                }
            }
        }

        // delete pvals_copy
        {
            Py_XDECREF(pvals_copy);
        }
        } // END NESTED SCOPE
        """ % locals()

    def perform(self, node, ins, outs):
        (pvals, unis, n_samples) = ins
        # make a copy so we do not overwrite the input
        pvals = copy.copy(pvals)
        (z,) = outs

        if n_samples > pvals.shape[1]:
            raise ValueError("Cannot sample without replacement n samples "
                             "bigger than the size of the distribution.")

        if unis.shape[0] != pvals.shape[0] * n_samples:
            raise ValueError("unis.shape[0] != pvals.shape[0] * n_samples",
                             unis.shape[0], pvals.shape[0], n_samples)

        if self.odtype == 'auto':
            odtype = 'int64'
        else:
            odtype = self.odtype
        if (z[0] is None or
                not np.all(z[0].shape == [pvals.shape[0], n_samples])):
            z[0] = -1 * np.ones((pvals.shape[0], n_samples), dtype=odtype)

        nb_multi = pvals.shape[0]
        nb_outcomes = pvals.shape[1]

        # For each multinomial, loop over each possible outcome,
        # and set selected pval to 0 after being selected
        for c in range(n_samples):
            for n in range(nb_multi):
                cummul = 0
                unis_n = unis[c * nb_multi + n]
                for m in range(nb_outcomes):
                    cummul += pvals[n, m]
                    if (cummul > unis_n):
                        z[0][n, c] = m
                        # set to zero and re-normalize so that it's not
                        # selected again
                        if not self.replace:
                            pvals[n, m] = 0.
                            pvals[n] /= pvals[n].sum()
                        break


class MultinomialWOReplacementFromUniform(ChoiceFromUniform):
    def __init__(self, *args, **kwargs):
        warnings.warn("MultinomialWOReplacementFromUniform is deprecated, "
                      "use ChoiceFromUniform instead.",
                      DeprecationWarning,
                      stacklevel=2)
        super(MultinomialWOReplacementFromUniform, self).__init__(*args, **kwargs)
