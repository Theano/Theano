#
# The following is based on acdc_wrapper.cu
# from https://github.com/mdenil/acdc-torch
#

import os

import theano
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable, gpu_contiguous

import cuda_ndarray

class FastACDC(GpuOp):

    def c_headers(self):
        return ['<stdio.h>', '<cufft.h>']

    def c_libraries(self):
        return ['cufft']

    def make_node(self, x, A, Ab, D, Db):
        x_ = gpu_contiguous(as_cuda_ndarray_variable(x))
        A_ = gpu_contiguous(as_cuda_ndarray_variable(A))
        Ab_ = gpu_contiguous(as_cuda_ndarray_variable(Ab))
        D_ = gpu_contiguous(as_cuda_ndarray_variable(D))
        Db_ = gpu_contiguous(as_cuda_ndarray_variable(Db))
        for t in [x_, A_, Ab_, D_, Db_]:
            assert t.dtype == 'float32'
        return theano.Apply(self, [x_, A_, Ab_, D_, Db_], [x_.type()])

    def grad(self, inputs, cost_grad):
        inp, A, Ab, D, Db = inputs
        gradOutput, = cost_grad
        gradOutput = gpu_contiguous(gradOutput)
        return FastACDCGrad()(inp, gradOutput, A, Ab, D, Db)

    def c_support_code_apply(self, node, nodename):
        # Based on a similar mechanism in sandbox/cuda/blas.py
        #
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        files = ['acdc.cu']
        codes = [open(os.path.join(os.path.split(__file__)[0], f)).read()
                for f in files]
        return reduce(str.__add__, codes)

    def c_code(self, node, name, inputs, outputs, sub):
        x, A, Ab, D, Db = inputs
        z, = outputs
        fail = sub['fail']
        return """
{
    // Validate that the output storage exists.
    if (NULL == %(z)s ||
        CudaNdarray_NDIM(%(z)s) != CudaNdarray_NDIM(%(x)s))
    {
        /* Reference received to invalid output variable.
        Decrease received reference's ref count and allocate new
        output variable */
        Py_XDECREF(%(z)s);
        %(z)s = (CudaNdarray*)CudaNdarray_NewDims(CudaNdarray_NDIM(%(x)s),
                                                  CudaNdarray_HOST_DIMS(%(x)s));
    }

    int batch_size;
    int input_size;
    int group_size;

    if (CudaNdarray_NDIM(%(x)s) == 1) {
        batch_size = 1;
        group_size = 1;
        input_size = CudaNdarray_HOST_DIMS(%(x)s)[0];
    }
    else if (CudaNdarray_NDIM(%(x)s) == 2) {
        batch_size = CudaNdarray_HOST_DIMS(%(x)s)[0];
        group_size = 1;
        input_size = CudaNdarray_HOST_DIMS(%(x)s)[1];
    }
    else if (CudaNdarray_NDIM(%(x)s) == 3) {
        batch_size = CudaNdarray_HOST_DIMS(%(x)s)[0];
        group_size = CudaNdarray_HOST_DIMS(%(x)s)[1];
        input_size = CudaNdarray_HOST_DIMS(%(x)s)[2];
    }
    else {
        PyErr_Format(PyExc_ValueError, "Input must have 1, 2, or 3 dims");
        %(fail)s;
    }

    int tmp_len = batch_size * input_size * group_size * 2;

    // Declare auxiliary arrays.
    CudaNdarray* tmp1 = (CudaNdarray*)CudaNdarray_NewDims(1, &tmp_len);
    CudaNdarray* tmp2 = (CudaNdarray*)CudaNdarray_NewDims(1, &tmp_len);

    // Verify.
    if (!%(x)s || !%(z)s || !%(A)s || !%(Ab)s || !%(D)s || !%(Db)s || !tmp1 || !tmp2) {
        PyErr_Format(PyExc_ValueError, "Could not allocate storage");
        %(fail)s;
    }

    // cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = 0; // Assign the default stream?

    // It should be up to the calling framework to provide plans.
    // This is a hack to make it work, though will be very slow if the
    // length/batchSize/groupSize change frequently.
    static cufftHandle planR2C;
    static cufftHandle planC2C;
    static int planLength = -1;
    static int planBatchSize = -1;

    if (planLength != input_size || planBatchSize != batch_size * group_size) {
       if (planLength != -1 && planBatchSize != -1) {
          cufftDestroy(planR2C);
          cufftDestroy(planC2C);
       }
       cufftPlan1d(&planR2C, input_size, CUFFT_R2C, batch_size * group_size);
       cufftPlan1d(&planC2C, input_size, CUFFT_C2C, batch_size * group_size);
       planLength = input_size;
       planBatchSize = batch_size * group_size;
    }

    acdc_fp<cufftReal, cufftComplex>(
                  stream,
                  input_size,
                  batch_size,
                  group_size,
                  planR2C, planC2C,
                  CudaNdarray_DEV_DATA(%(x)s),
                  CudaNdarray_DEV_DATA(%(A)s),
                  CudaNdarray_DEV_DATA(%(Ab)s),
                  CudaNdarray_DEV_DATA(%(D)s),
                  CudaNdarray_DEV_DATA(%(Db)s),
                  CudaNdarray_DEV_DATA(%(z)s),
                  CudaNdarray_DEV_DATA(tmp1),
                  CudaNdarray_DEV_DATA(tmp2));

    Py_XDECREF(tmp1);
    Py_XDECREF(tmp2);
}
""" % locals()

class FastACDCGrad(FastACDC):

    def make_node(self, inp, gradOutput, A, Ab, D, Db):
        inp_ = gpu_contiguous(as_cuda_ndarray_variable(inp))
        gradOutput_ = gpu_contiguous(as_cuda_ndarray_variable(gradOutput))
        A_ = gpu_contiguous(as_cuda_ndarray_variable(A))
        Ab_ = gpu_contiguous(as_cuda_ndarray_variable(Ab))
        D_ = gpu_contiguous(as_cuda_ndarray_variable(D))
        Db_ = gpu_contiguous(as_cuda_ndarray_variable(Db))
        for t in [inp_, gradOutput_, A_, Ab_, D_, Db_]:
            assert t.dtype == 'float32'
        return theano.Apply(self, [inp_, gradOutput_, A_, Ab_, D_, Db_],
                            [inp_.type(), A_.type(), Ab_.type(), D_.type(), Db_.type()])

    def c_code(self, node, name, inputs, outputs, sub):
        inp, gradOutput, A, Ab, D, Db = inputs
        gradInput, gradA, gradAb, gradD, gradDb = outputs
        fail = sub['fail']
        return """
{
    // Validate output storage.
    if (NULL == %(gradInput)s ||
        CudaNdarray_NDIM(%(gradInput)s) != CudaNdarray_NDIM(%(gradOutput)s))
    {
        Py_XDECREF(%(gradInput)s);
        %(gradInput)s = (CudaNdarray*)CudaNdarray_NewDims(
            CudaNdarray_NDIM(%(gradOutput)s), CudaNdarray_HOST_DIMS(%(gradOutput)s));
    }

    int batch_size;
    int input_size;
    int group_size;

    if (CudaNdarray_NDIM(%(gradOutput)s) == 1) {
        batch_size = 1;
        group_size = 1;
        input_size = CudaNdarray_HOST_DIMS(%(gradOutput)s)[0];
    }
    else if (CudaNdarray_NDIM(%(gradOutput)s) == 2) {
        batch_size = CudaNdarray_HOST_DIMS(%(gradOutput)s)[0];
        group_size = 1;
        input_size = CudaNdarray_HOST_DIMS(%(gradOutput)s)[1];
    }
    else if (CudaNdarray_NDIM(%(gradOutput)s) == 3) {
        batch_size = CudaNdarray_HOST_DIMS(%(gradOutput)s)[0];
        group_size = CudaNdarray_HOST_DIMS(%(gradOutput)s)[1];
        input_size = CudaNdarray_HOST_DIMS(%(gradOutput)s)[2];
    }
    else {
        PyErr_Format(PyExc_ValueError, "Input must have 1, 2, or 3 dims");
        %(fail)s;
    }

    if (NULL == %(gradA)s || CudaNdarray_NDIM(%(gradA)s) != 1)
    {
        Py_XDECREF(%(gradA)s);
        %(gradA)s = (CudaNdarray*)CudaNdarray_NewDims(1, &input_size);
    }
    if (NULL == %(gradAb)s || CudaNdarray_NDIM(%(gradAb)s) != 1)
    {
        Py_XDECREF(%(gradAb)s);
        %(gradAb)s = (CudaNdarray*)CudaNdarray_NewDims(1, &input_size);
    }
    if (NULL == %(gradD)s || CudaNdarray_NDIM(%(gradD)s) != 1)
    {
        Py_XDECREF(%(gradD)s);
        %(gradD)s = (CudaNdarray*)CudaNdarray_NewDims(1, &input_size);
    }
    if (NULL == %(gradDb)s || CudaNdarray_NDIM(%(gradDb)s) != 1)
    {
        Py_XDECREF(%(gradDb)s);
        %(gradDb)s = (CudaNdarray*)CudaNdarray_NewDims(1, &input_size);
    }

    // Declare tmp arrays.
    int tmp_len = batch_size * input_size * group_size * 2;
    int delta_mid_len = batch_size * input_size * group_size ;

    CudaNdarray* tmp1 = (CudaNdarray*)CudaNdarray_NewDims(1, &tmp_len);
    CudaNdarray* tmp2 = (CudaNdarray*)CudaNdarray_NewDims(1, &tmp_len);
    CudaNdarray* delta_mid = (CudaNdarray*)CudaNdarray_NewDims(1, &delta_mid_len);
    CudaNdarray* inputD = (CudaNdarray*)CudaNdarray_NewDims(
        CudaNdarray_NDIM(%(inp)s), CudaNdarray_HOST_DIMS(%(inp)s));

    // Verify.
    if (!%(gradA)s || !%(gradAb)s || !%(gradD)s || !%(gradDb)s ||
        !%(A)s || !%(Ab)s || !%(D)s || !%(Db)s ||
        !%(inp)s || !%(gradInput)s || !%(gradOutput)s ||
        !tmp1 || !tmp2 || !delta_mid || !inputD) {
        PyErr_Format(PyExc_ValueError, "Could not allocate storage");

        %(fail)s;
    }

    // Arrays grad{A,Ab,D,Db} have to be initialized to zeros. See FastACDC.lua.
    size_t total_size = input_size * sizeof(real);
    cudaMemset(CudaNdarray_DEV_DATA(%(gradA)s), 0, total_size);
    cudaMemset(CudaNdarray_DEV_DATA(%(gradAb)s), 0, total_size);
    cudaMemset(CudaNdarray_DEV_DATA(%(gradD)s), 0, total_size);
    cudaMemset(CudaNdarray_DEV_DATA(%(gradDb)s), 0, total_size);

    // cudaStream_t stream = THCState_getCurrentStream(state);
    cudaStream_t stream = 0; // Assign the default stream?

    // It should be up to the calling framework to provide plans.
    // This is a hack to make it work, though will be very slow if the
    // length/batchSize/groupSize change frequently.
    static cufftHandle planR2C;
    static cufftHandle planC2C;
    static int planLength = -1;
    static int planBatchSize = -1;

    if (planLength != input_size || planBatchSize != batch_size * group_size) {
       if (planLength != -1 && planBatchSize != -1) {
          cufftDestroy(planR2C);
          cufftDestroy(planC2C);
       }
       cufftPlan1d(&planR2C, input_size, CUFFT_R2C, batch_size * group_size);
       cufftPlan1d(&planC2C, input_size, CUFFT_C2C, batch_size * group_size);
       planLength = input_size;
       planBatchSize = batch_size * group_size;
    }

    acdc_bp<cufftReal, cufftComplex>(
                  stream,
                  input_size,
                  batch_size,
                  group_size,
                  planR2C, planC2C,
                  CudaNdarray_DEV_DATA(%(gradInput)s),
                  CudaNdarray_DEV_DATA(%(A)s),
                  CudaNdarray_DEV_DATA(%(Ab)s),
                  CudaNdarray_DEV_DATA(%(D)s),
                  CudaNdarray_DEV_DATA(%(Db)s),
                  CudaNdarray_DEV_DATA(%(gradOutput)s),
                  CudaNdarray_DEV_DATA(delta_mid),
                  CudaNdarray_DEV_DATA(tmp1),
                  CudaNdarray_DEV_DATA(tmp2));

    acdc_bp_acc<cufftReal, cufftComplex>(
                  stream,
                  input_size,
                  batch_size,
                  group_size,
                  planR2C, planC2C,
                  CudaNdarray_DEV_DATA(%(gradInput)s),
                  CudaNdarray_DEV_DATA(delta_mid),
                  CudaNdarray_DEV_DATA(%(A)s),
                  CudaNdarray_DEV_DATA(%(Ab)s),
                  CudaNdarray_DEV_DATA(%(D)s),
                  // CudaNdarray_DEV_DATA(%(Db)s),
                  CudaNdarray_DEV_DATA(%(inp)s),
                  CudaNdarray_DEV_DATA(inputD),
                  CudaNdarray_DEV_DATA(%(gradA)s),
                  CudaNdarray_DEV_DATA(%(gradAb)s),
                  CudaNdarray_DEV_DATA(%(gradD)s),
                  CudaNdarray_DEV_DATA(%(gradDb)s),
                  CudaNdarray_DEV_DATA(tmp1),
                  CudaNdarray_DEV_DATA(tmp2));

    Py_XDECREF(tmp1);
    Py_XDECREF(tmp2);
    Py_XDECREF(delta_mid);
    Py_XDECREF(inputD);
}
""" % locals()
