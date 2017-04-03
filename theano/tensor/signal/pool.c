#section support_code

typedef struct _PoolDimUtils {
    int* z;
    int* r;
    int* ws;
    int* st;
    int* pd;
} PoolDimUtils;
int PoolDimUtils_init(PoolDimUtils* p, size_t nd) {
    p->z = (int*)malloc(nd * sizeof(int));
    p->r = (int*)malloc(nd * sizeof(int));
    p->ws = (int*)malloc(nd * sizeof(int));
    p->st = (int*)malloc(nd * sizeof(int));
    p->pd = (int*)malloc(nd * sizeof(int));
    return (p->z != NULL && p->r != NULL && p->ws != NULL && p->st != NULL && p->pd != NULL);
}
void PoolDimUtils_cleanup(PoolDimUtils* p) {
    free(p->z);
    free(p->r);
    free(p->ws);
    free(p->st);
    free(p->pd);
    p->z = p->r = p->ws = p->st = p->pd = NULL;
}

typedef struct _PoolRunUtils {
    int* r_st;
    int* r_end;
    int* r_idx;
    npy_intp* i_idx;
    npy_intp* o_idx;
} PoolRunUtils;
int PoolRunUtils_init(PoolRunUtils* p, size_t nd, size_t total_ndim) {
    p->r_st = (int*)malloc(nd * sizeof(int));
    p->r_end = (int*)malloc(nd * sizeof(int));
    p->r_idx = (int*)malloc(nd * sizeof(int));
    p->i_idx = (npy_intp*)malloc(total_ndim * sizeof(npy_intp));
    p->o_idx = (npy_intp*)malloc(total_ndim * sizeof(npy_intp));
    return (p->r_st != NULL && p->r_end != NULL && p->r_idx != NULL && p->i_idx != NULL && p->o_idx != NULL);
};
void PoolRunUtils_cleanup(PoolRunUtils* p) {
    free(p->r_st);
    free(p->r_end);
    free(p->r_idx);
    free(p->i_idx);
    free(p->o_idx);
    p->r_st = p->r_end = p->r_idx = NULL;
    p->i_idx = p->o_idx = NULL;
};

#section support_code_apply

void loop_over_pooling_region(PyArrayObject* _x, int dim_id, int total_ndim, int non_pool_ndim,
                              PoolRunUtils* pool_r, DTYPE_INPUT_0* collector, PARAMS_TYPE* p) {
    int* r_st = pool_r->r_st;
    int* r_end = pool_r->r_end;
    npy_intp* i_idx = pool_r->i_idx;
    // go through the pooled region in the unpadded input
    for (int m = r_st[dim_id]; m < r_end[dim_id]; ++m) {
        i_idx[non_pool_ndim + dim_id] = m;

        int next_dim = dim_id + 1;
        if (next_dim < p->ndim) {
            loop_over_pooling_region(_x, next_dim, total_ndim, non_pool_ndim, pool_r, collector, p);
        } else {
            // update maximum
            DTYPE_INPUT_0 a;
            if (total_ndim == 4)
                a = ((DTYPE_INPUT_0 *) (PyArray_GETPTR4(_x, i_idx[0], i_idx[1], i_idx[2], i_idx[3])))[0];
            else
                a = ((DTYPE_INPUT_0 *) (PyArray_GetPtr(_x, i_idx)))[0];

            if (p->mode == MODE_MAX) {
                *collector = (a > *collector) ? a : *collector;
            } else {
                *collector += a;
            }
        }

    } // for loop over region
}

void loop_over_pooling_dimension(PyArrayObject* _x, int dim_id, int total_ndim, int non_pool_ndim,
                                 PoolDimUtils* pool_d, PoolRunUtils* pool_r, DTYPE_INPUT_0* collector,
                                 PyArrayObject* out, PARAMS_TYPE* p) {
    int* r_st = pool_r->r_st;
    int* r_end = pool_r->r_end;
    int* r_idx = pool_r->r_idx;
    npy_intp* i_idx = pool_r->i_idx;
    npy_intp* o_idx = pool_r->o_idx;
    int* z = pool_d->z;
    int* r = pool_d->r;
    int* ws = pool_d->ws;
    int* st = pool_d->st;
    int* pd = pool_d->pd;
    for (r_idx[dim_id] = 0; r_idx[dim_id] < z[dim_id]; ++r_idx[dim_id]) {
        r_st[dim_id] = r_idx[dim_id] * st[dim_id];
        r_end[dim_id] = r_st[dim_id] + ws[dim_id];
        // skip the padding
        r_st[dim_id] = r_st[dim_id] < pd[dim_id] ? pd[dim_id] : r_st[dim_id];
        r_end[dim_id] = r_end[dim_id] > (r[dim_id] - pd[dim_id]) ? r[dim_id] - pd[dim_id] : r_end[dim_id];
        // from padded_img space to img space
        r_st[dim_id] -= pd[dim_id];
        r_end[dim_id] -= pd[dim_id];
        // handle the case where no padding, ignore border is True
        if (p->ignore_border) {
            r_end[dim_id] = r_end[dim_id] > r[dim_id] ? r[dim_id] : r_end[dim_id];
        }
        // use the index to find the correct position in the output
        o_idx[non_pool_ndim + dim_id] = r_idx[dim_id];

        int next_dim = dim_id + 1;
        if (next_dim < p->ndim) {
            /// New loop with i+1.
            loop_over_pooling_dimension(_x, next_dim, total_ndim, non_pool_ndim, pool_d, pool_r, collector, out, p);
        } else {
            /// Local code.
            // get a pointer to the correct position in the output
            DTYPE_OUTPUT_0 *z_ptr;
            if (total_ndim == 4)
                z_ptr = ((DTYPE_OUTPUT_0 *) (PyArray_GETPTR4(out, o_idx[0], o_idx[1], o_idx[2], o_idx[3])));
            else
                z_ptr = ((DTYPE_OUTPUT_0 *) (PyArray_GetPtr(out, o_idx)));
            if (p->mode == MODE_MAX) {
                for (int ii = 0; ii < p->ndim; ++ii) {
                    // set the first index of dimension ii.
                    i_idx[non_pool_ndim + ii] = r_st[ii];
                }
                // use the first element as the initial value of collector
                if (total_ndim == 4)
                    *collector = ((DTYPE_INPUT_0 *) (PyArray_GETPTR4(_x, i_idx[0], i_idx[1], i_idx[2], i_idx[3])))[0];
                else
                    *collector = ((DTYPE_INPUT_0 *) (PyArray_GetPtr(_x, i_idx)))[0];

                loop_over_pooling_region(_x, 0, total_ndim, non_pool_ndim, pool_r, collector, p);

                z_ptr[0] = *collector;
            } else {
                // Case if p->mode in {MODE_SUM, MODE_AVERAGE_EXC_PAD, MODE_AVERAGE_INC_PAD}.
                // initialize the sum at zero
                *collector = ((DTYPE_INPUT_0) (0));

                loop_over_pooling_region(_x, 0, total_ndim, non_pool_ndim, pool_r, collector, p);

                if (p->mode == MODE_SUM) {
                    z_ptr[0] = *collector;
                } else if (p->mode == MODE_AVERAGE_INC_PAD && p->ignore_border) {
                    DTYPE_OUTPUT_0 region_size = 1;
                    for (int ii = 0; ii < p->ndim; ++ii)
                        region_size *= ws[ii];
                    z_ptr[0] = *collector / region_size;
                } else {
                    DTYPE_OUTPUT_0 region_size = 1;
                    for (int ii = 0; ii < p->ndim; ++ii)
                        region_size *= (r_end[ii] - r_st[ii]);
                    z_ptr[0] = *collector / region_size;
                }
            }
        }
    }
}

int APPLY_SPECIFIC(pool)(PyArrayObject* _x, PyArrayObject* _ws, PyArrayObject* _stride, PyArrayObject* _pad,
                          PyArrayObject** out, PARAMS_TYPE* wrapper) {
    int typenum = PyArray_ObjectType((PyObject *) _x, 0);
    int total_ndim = PyArray_NDIM(_x);
    int nd = wrapper->ndim;
    int non_pool_ndim = total_ndim - nd;
    if (PyArray_NDIM(_x) != total_ndim) {
        PyErr_Format(PyExc_ValueError, "x must be a %d-D ndarray", total_ndim);
        return -1;
    }
    if (PyArray_DIM(_ws, 0) != nd) {
        PyErr_Format(PyExc_ValueError, "ws must be a vector of size %d", nd);
        return -1;
    }
    if (PyArray_DIM(_stride, 0) != nd) {
        PyErr_Format(PyExc_ValueError, "stride must be a vector of size %d", nd);
        return -1;
    }
    if (PyArray_DIM(_pad, 0) != nd) {
        PyErr_Format(PyExc_ValueError, "pad must be a vector of size %d", nd);
        return -1;
    }
    PoolDimUtils pool_d;
    if (!PoolDimUtils_init(&pool_d, nd)) {
        PyErr_NoMemory();
        PoolDimUtils_cleanup(&pool_d);
        return -1;
    }
    int* z = pool_d.z;		// shape of the output
    int* r = pool_d.r;		// shape of the padded_input
    int* ws = pool_d.ws;
    int* st = pool_d.st;
    int* pd = pool_d.pd;
    int nonzero_padding = 0;
    for (int i = 0; i < nd; ++i) {
        ws[i] = *((npy_intp *) PyArray_GETPTR1(_ws, i));
        st[i] = *((npy_intp *) PyArray_GETPTR1(_stride, i));
        pd[i] = *((npy_intp *) PyArray_GETPTR1(_pad, i));
        r[i] = PyArray_DIMS(_x)[non_pool_ndim + i] + 2 * pd[i];
        if (pd[i] > 0)
            nonzero_padding = 1;
    }
    if (!wrapper->ignore_border && nonzero_padding) {
        PyErr_SetString(PyExc_ValueError, "padding must be zero when ignore border is False");
        PoolDimUtils_cleanup(&pool_d);
        return -1;
    }
    if (wrapper->ignore_border) {
        for (int i = 0; i < nd; ++i) {
            // '/' in C is different from '/' in python
            if (r[i] - ws[i] < 0) {
                z[i] = 0;
            } else {
                z[i] = (r[i] - ws[i]) / st[i] + 1;
            }
        }
    } else for (int i = 0; i < nd; ++i) {
        // decide how many rows/cols the output has
        if (st[i] >= ws[i]) {
            z[i] = (r[i] - 1) / st[i] + 1;
        } else {
            z[i] = std::max(0, (r[i] - 1 - ws[i] + st[i]) / st[i]) + 1;
        }

        assert(z[i] > 0);
    }
    // memory allocation of z if necessary
    int mem_nec = 0;
    if ((!(*out)) || *PyArray_DIMS(*out) != total_ndim) {
        mem_nec = 1;
    }
    if (!mem_nec) {
        for (int i = 0; i < non_pool_ndim; ++i) {
            if (PyArray_DIMS(*out)[i] != PyArray_DIMS(_x)[i]) {
                mem_nec = 1;
                break;
            }
        }
    }
    if (!mem_nec) {
        for (int i = 0; i < nd; ++i) {
            if (PyArray_DIMS(*out)[non_pool_ndim + i] != z[i]) {
                mem_nec = 1;
                break;
            }
        }
    }
    if (mem_nec) {
        if (*out) Py_XDECREF(*out);
        npy_intp* dims = (npy_intp*)malloc(total_ndim * sizeof(npy_intp));
        if (dims == NULL) {
            PyErr_NoMemory();
            PoolDimUtils_cleanup(&pool_d);
            return -1;
        }
        for (int i = 0; i < non_pool_ndim; ++i) {
            dims[i] = PyArray_DIMS(_x)[i];
        }
        for (int i = 0; i < nd; ++i) {
            dims[non_pool_ndim + i] = z[i];
        }
        //TODO: zeros not necessary
        *out = (PyArrayObject *) PyArray_ZEROS(total_ndim, dims, typenum, 0);
        free(dims);
    }
    // initialize temp var for the value in a region
    DTYPE_INPUT_0 collector;
    int z_prod = 1;
    // do not run if any z[i] is zero
    for (int i = 0; i < nd; ++i) {
        z_prod *= z[i];
    }
    if (z_prod) {
        PoolRunUtils pool_r;
        if (!PoolRunUtils_init(&pool_r, nd, total_ndim)) {
            PyErr_NoMemory();
            PoolDimUtils_cleanup(&pool_d);
            PoolRunUtils_cleanup(&pool_r);
            return -1;
        }
        // will be used to hold start and end index of a region
        int* r_st = pool_r.r_st;
        int* r_end = pool_r.r_end;
        // index for iterating over the pooling regions
        int* r_idx = pool_r.r_idx;
        // placeholder for PyArray indexing (output)
        npy_intp* o_idx = pool_r.o_idx;
        // placeholder for PyArray indexing (input)
        npy_intp* i_idx = pool_r.i_idx;
        // loop over non-pooling dimensions
        int non_pooling_prod = 1;
        for (int i = 0; i < non_pool_ndim; ++i) {
            non_pooling_prod *= PyArray_DIMS(_x)[i];
        }
        /** first loop over non-pooling dimensions **/
        // NB: Pragma directive for OpenMP should be ignored by the compiler if OpenMP is not available.
        #pragma omp parallel for private(r_st, r_end, r_idx, i_idx, o_idx, maximum) schedule(static)
        for (int t = 0; t < non_pooling_prod; ++t) {
            // compute the non-pooling index in each dimension
            if (non_pool_ndim != 0) {
                o_idx[0] = t;
                i_idx[0] = t;
                for (int i = 1; i < non_pool_ndim; ++i) {
                    o_idx[i] = o_idx[i - 1] / PyArray_DIMS(_x)[i - 1];
                    o_idx[i - 1] = o_idx[i - 1] % PyArray_DIMS(_x)[i - 1];
                    i_idx[i] = o_idx[i];
                    i_idx[i - 1] = o_idx[i - 1];
                }
            }
            // then loop over each region in each pooling dimension
            loop_over_pooling_dimension(_x, 0, total_ndim, non_pool_ndim, &pool_d, &pool_r, &collector, *out, wrapper);
        } // for loop over non-pooling dimensions
        PoolRunUtils_cleanup(&pool_r);
    } // if z_prod
    PoolDimUtils_cleanup(&pool_d);
    return 0;
}
