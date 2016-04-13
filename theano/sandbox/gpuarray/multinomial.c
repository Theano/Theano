#section support_code_apply

static __global__ void k_multi_warp_APPLYSPECIFIC(multinomial)(
    const int nb_multi,
    const int nb_outcomes,
    float * global_pvals,
    const int pvals_row_stride,
    const int pvals_col_stride,
    float * global_unis,
    const int unis_stride,
    float * global_outs,
    const int outs_row_stride,
    const int outs_col_stride
)
{
    // each thread takes care of one multinomial draw
    int n = blockDim.x*blockIdx.x + threadIdx.x;
    if (n < nb_multi)
    {
        float cummul = 0.;
        bool done = false;
        const float unis_n = global_unis[n*unis_stride];
        for (int m = 0; m < nb_outcomes; ++m)
        {
            float current_out = 0.;
            if (!done)
            {
                cummul += global_pvals[m * pvals_col_stride + n * pvals_row_stride];
                if (unis_n < cummul)
                {
                    current_out = 1.;
                    done = true;
                }
            }
            //write out transposed for speed.
            global_outs[n * outs_col_stride + m * outs_row_stride] = current_out;
        }
    }
}

#section support_code_struct

int APPLY_SPECIFIC(multinomial)(PyGpuArrayObject *pvals,
                                PyGpuArrayObject *unis,
                                PyGpuArrayObject **out,
                                PyGpuContextObject *c) {
    if (PyGpuArray_NDIM(pvals) != 2)
    {
        PyErr_Format(PyExc_TypeError, "pvals wrong rank");
        FAIL;
    }
    if (PyGpuArray_NDIM(unis) != 1)
    {
        PyErr_Format(PyExc_TypeError, "unis wrong rank");
        FAIL;
    }
    if (PyGpuArray_HOST_DIMS(unis)[0] != PyGpuArray_HOST_DIMS(pvals)[0])
    {
        PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0]");
        FAIL;
    }

    //N.B. that the output is TRANSPOSED compared with pvals
    if ((NULL == *out)
        || (PyGpuArray_HOST_DIMS(*out[0] != PyGpuArray_HOST_DIMS(pvals)[1]
        || (PyGpuAarray_HOST_DIMS(*out[1] != PyGpuArray_HOST_DIMS(pvals)[0])
    {
        Py_XDECREF(*out);
        npy_intp dims[2];
        dims[0] = (PyGpuArray_HOST_DIMS(pvals)[1];
        dims[1] = (PyGpuArray_HOST_DIMS(pvals)[0]);
        *out = (PyGpuarray*)PyGpuArray_NewDims(2, dims);
        if (!*out)
        {
            PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
            FAIL;
        }
    }

    { // NESTED SCOPE
        int nb_multi = PyGpuArray_HOST_DIMS(pvals)[0];
        int nb_outcomes = PyGpuArray_HOST_DIMS(pvals)[1];
        //TODO : change this for a beautiful constant
        int max_nb_blocks = 2<<15 - 1;
        int nb_blocks = max_nb_blocks + 1;
        int nb_threads=16; // so it really starts at 32, because of the *2
        do
        {
            nb_threads*=2;
            if (nb_multi %% nb_threads == 0)
                nb_blocks = nb_multi/nb_threads;
            else
                nb_blocks = (int)((float)nb_multi/(float)nb_threads + 1.);
        } while (nb_blocks > max_nb_blocks);

        //printf("\\nN=%%i b=%%i t=%%i t*b=%%i", nb_multi, nb_blocks, nb_threads, nb_blocks*nb_threads);

        // TODO : next line is a bit hardcoded...
        if (nb_threads > 512)
        {
            PyErr_Format(PyExc_ValueError, "Mutinomial is not implemented for so many rows in the matrix (%%i)", nb_multi);
            FAIL;
        }
        dim3 n_blocks(nb_blocks,1,1);
        dim3 n_threads(nb_threads,1,1);
        int n_shared = 0;

        assert(nb_blocks*nb_threads >= nb_multi);

        k_multi_warp_APPLYSPECIFIC(multinomial)<<<n_blocks, n_threads, n_shared>>>(
            CudaNdarray_HOST_DIMS(%(z)s)[1],
            CudaNdarray_HOST_DIMS(%(z)s)[0],
            CudaNdarray_DEV_DATA(%(pvals)s),
            CudaNdarray_HOST_STRIDES(%(pvals)s)[0],
            CudaNdarray_HOST_STRIDES(%(pvals)s)[1],
            CudaNdarray_DEV_DATA(%(unis)s),
            CudaNdarray_HOST_STRIDES(%(unis)s)[0],
            CudaNdarray_DEV_DATA(%(z)s),
            CudaNdarray_HOST_STRIDES(%(z)s)[0],
            CudaNdarray_HOST_STRIDES(%(z)s)[1]
        );
        CNDA_THREAD_SYNC;
        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess != sts)
        {
            PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i; shared: %%i)\\n",
                "k_multi_warp_%(name)s",
                cudaGetErrorString(sts),
                n_blocks.x,
                n_blocks.y,
                n_threads.x,
                n_threads.y,
                n_threads.z,
                n_shared);
            FAIL;
        }

    } // END NESTED SCOPE
}