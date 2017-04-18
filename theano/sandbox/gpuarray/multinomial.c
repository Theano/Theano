#section support_code_apply

static __global__ void k_multi_warp_multinomial(
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
    size_t dims[2];
    if (PyGpuArray_NDIM(pvals) != 2)
    {
        PyErr_Format(PyExc_TypeError, "pvals wrong rank");
        return 1;
    }
    if (PyGpuArray_NDIM(unis) != 1)
    {
        PyErr_Format(PyExc_TypeError, "unis wrong rank");
        return 1;
    }
    if (PyGpuArray_DIMS(unis)[0] != PyGpuArray_DIMS(pvals)[0])
    {
        PyErr_Format(PyExc_ValueError, "unis.shape[0] != pvals.shape[0]");
        return 1;
    }

    dims[0] = PyGpuArray_DIMS(pvals)[1];
    dims[1] = PyGpuArray_DIMS(pvals)[0];
    if (theano_prep_output(out, 2, dims, unis->ga.typecode,
                           GA_C_ORDER, c) != 0)
      return 1;
    GpuArray_memset(&((*out)->ga), 0);

    { // NESTED SCOPE
        int nb_multi = PyGpuArray_DIMS(pvals)[0];
        int nb_outcomes = PyGpuArray_DIMS(pvals)[1];
        //TODO : change this for a beautiful constant
        int max_nb_blocks = 2<<15 - 1;
        int nb_blocks = max_nb_blocks + 1;
        int nb_threads=16; // so it really starts at 32, because of the *2
        do
        {
            nb_threads*=2;
            if (nb_multi % nb_threads == 0)
                nb_blocks = nb_multi/nb_threads;
            else
                nb_blocks = (int)((float)nb_multi/(float)nb_threads + 1.);
        } while (nb_blocks > max_nb_blocks);

        //printf("\\nN=%i b=%i t=%i t*b=%i", nb_multi, nb_blocks, nb_threads, nb_blocks*nb_threads);

        // TODO : next line is a bit hardcoded...
        if (nb_threads > 512)
        {
            PyErr_Format(PyExc_ValueError, "Multinomial is not implemented for so many rows in the matrix (%i)", nb_multi);
            return 1;
        }
        dim3 n_blocks(nb_blocks,1,1);
        dim3 n_threads(nb_threads,1,1);
        int n_shared = 0;

        assert(nb_blocks*nb_threads >= nb_multi);

        k_multi_warp_multinomial<<<n_blocks, n_threads, n_shared>>>(
            PyGpuArray_DIMS(*out)[1],
            PyGpuArray_DIMS(*out)[0],
            (float*)PyGpuArray_DEV_DATA(pvals),
            PyGpuArray_STRIDES(pvals)[0]/sizeof(float),
            PyGpuArray_STRIDES(pvals)[1]/sizeof(float),
            (float*)PyGpuArray_DEV_DATA(unis),
            PyGpuArray_STRIDES(unis)[0]/sizeof(float),
            (float*)PyGpuArray_DEV_DATA(*out),
            PyGpuArray_STRIDES(*out)[0]/sizeof(float),
            PyGpuArray_STRIDES(*out)[1]/sizeof(float)
        );

	//TODO
	//if(false)//SYNC)
	  //	  GpuArray_sync((*out)->ga);
 	//        SYNC;
        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess != sts)
        {
            PyErr_Format(PyExc_RuntimeError, "Cuda error: %s: %s. (grid: %i x %i; block: %i x %i x %i; shared: %i)\\n",
                "k_multi_warp_%(name)s",
                cudaGetErrorString(sts),
                n_blocks.x,
                n_blocks.y,
                n_threads.x,
                n_threads.y,
                n_threads.z,
                n_shared);
            return 1;
        }

    } // END NESTED SCOPE
	return 0;
}
