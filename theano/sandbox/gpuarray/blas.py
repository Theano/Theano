import numpy

from theano import Op, Apply, config

from theano.tensor.blas import Dot22, Gemv, Gemm, Ger
from theano.sandbox.gpuarray.basic_ops import (HideC, as_gpuarray_variable)
from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler

try:
    import pygpu
    from pygpu import blas
except ImportError, e:
    # To make sure theano is importable
    pass


class BlasOp(HideC):
    def c_headers(self):
        return ['<blas_api.h>']

    def c_header_dirs(self):
        return [pygpu.get_include()]

    def c_init_code(self):
        return ['import_pygpu__blas();']


class GpuGemv(BlasOp, Gemv):
    def make_node(self, y, alpha, A, x, beta):
        res = Gemv.make_node(self, y, alpha, A, x, beta)
        A = as_gpuarray_variable(A)
        x = as_gpuarray_variable(x)
        y = as_gpuarray_variable(y)
        assert A.dtype == x.dtype == y.dtype
        return Apply(self, [y, alpha, A, x, beta], [y.type()])

    def perform(self, node, inputs, out_storage):
        y, alpha, A, x, beta = inputs
        inplace = self.inplace
        if inplace and y.strides[0] < 0:
            inplace = False
        out_storage[0][0] = blas.gemv(alpha, A, x, beta, y,
                                      overwrite_y=inplace)

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], y=inp[0], alpha=inp[1], A=inp[2], x=inp[3],
                    beta=inp[4], fail=sub['fail'], name=name)
        if self.inplace:
            code = """
                   Py_XDECREF(%(out)s);
                   if (%(y)s->ga.strides[0] <= 0) {
                     %(out)s = pygpu_copy(%(y)s, GA_ANY_ORDER);
                     if (%(out)s == NULL) {
                       %(fail)s
                     }
                   } else {
                     %(out)s = %(y)s;
                     Py_INCREF(%(out)s);
                   }
                   """ % vars
        else:
            code = """
                   Py_XDECREF(%(out)s);
                   %(out)s = pygpu_copy(%(y)s, GA_ANY_ORDER);
                   if (%(out)s == NULL) {
                       %(fail)s
                   }
                   """ % vars
        code += """
        if (pygpu_blas_rgemv(cb_no_trans,
                             ((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
                             %(A)s, %(x)s,
                             ((dtype_%(beta)s *)PyArray_DATA(%(beta)s))[0],
                             %(out)s, 0) == -1) {
            %(fail)s
        }
        """ % vars
        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars
        return code

    def c_code_cache_version(self):
        return (2,)

gpugemv_no_inplace = GpuGemv(inplace=False)
gpugemv_inplace = GpuGemv(inplace=True)


class GpuGemm(BlasOp, Gemm):
    def make_node(self, C, alpha, A, B, beta):
        res = Gemm.make_node(self, C, alpha, A, B, beta)
        A = as_gpuarray_variable(A)
        B = as_gpuarray_variable(B)
        C = as_gpuarray_variable(C)
        assert A.dtype == B.dtype == C.dtype
        return Apply(self, [C, alpha, A, B, beta], [C.type()])

    def perform(self, node, inputs, outputs):
        C, alpha, A, B, beta = inputs
        inplace = self.inplace
        if inplace and not C.flags.forc:
            inplace = False
        outputs[0][0] = blas.gemm(alpha, A, B, beta, C,
                                  overwrite_c=inplace)

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], C=inp[0], alpha=inp[1], A=inp[2], B=inp[3],
                    beta=inp[4], fail=sub['fail'], name=name)
        if self.inplace:
            code = """
                   Py_XDECREF(%(out)s);
                   if (!GpuArray_ISONESEGMENT(&%(C)s->ga)) {
                     %(out)s = pygpu_copy(%(C)s, GA_ANY_ORDER);
                     if (%(out)s == NULL) {
                       %(fail)s
                     }
                   } else {
                     %(out)s = %(C)s;
                     Py_INCREF(%(out)s);
                   }
                   """ % vars
        else:
            code = """
                   Py_XDECREF(%(out)s);
                   %(out)s = pygpu_copy(%(C)s, GA_ANY_ORDER);
                   if (%(out)s == NULL) {
                       %(fail)s
                   }
                   """ % vars
        code += """
        if (pygpu_blas_rgemm(cb_no_trans, cb_no_trans,
                             ((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
                             %(A)s, %(B)s,
                             ((dtype_%(beta)s *)PyArray_DATA(%(beta)s))[0],
                             %(out)s, 0) == -1) {
            %(fail)s
        }
        """ % vars
        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars
        return code

    def c_code_cache_version(self):
        return (2,)


gpugemm_no_inplace = GpuGemm(inplace=False)
gpugemm_inplace = GpuGemm(inplace=True)


class GpuGer(BlasOp, Ger):
    def make_node(self, A, alpha, x, y):
        res = Ger.make_node(self, A, alpha, x, y)
        A = as_gpuarray_variable(A)
        x = as_gpuarray_variable(x)
        y = as_gpuarray_variable(y)
        assert A.dtype == x.dtype == y.dtype
        return Apply(self, [A, alpha, x, y], [A.type()])

    def perform(self, node, inp, out):
        A, alpha, x, y = inp
        inplace = self.destructive
        if inplace and not A.flags.forc:
            inplace = False
        outputs[0][0] = blas.ger(alpha, x, y, A,
                                 overwrite_a=inplace)

    def c_code(self, node, name, inp, out, sub):
        vars = dict(out=out[0], A=inp[0], alpha=inp[1], x=inp[2], y=inp[3],
                    fail=sub['fail'], name=name)
        if self.destructive:
            code = """
                   Py_XDECREF(%(out)s);
                   if (!GpuArray_ISONESEGMENT(&%(A)s->ga)) {
                     %(out)s = pygpu_copy(%(A)s, GA_ANY_ORDER);
                     if (%(out)s == NULL) {
                       %(fail)s
                     }
                   } else {
                     %(out)s = %(A)s;
                     Py_INCREF(%(out)s);
                   }
                   """ % vars
        else:
            code = """
                   Py_XDECREF(%(out)s);
                   %(out)s = pygpu_copy(%(A)s, GA_ANY_ORDER);
                   if (%(out)s == NULL) {
                       %(fail)s
                   }
                   """ % vars
        code += """
        if (pygpu_blas_rger(((dtype_%(alpha)s *)PyArray_DATA(%(alpha)s))[0],
                            %(x)s, %(y)s, %(out)s, 0) == -1) {
            %(fail)s
        }
        """ % vars
        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars
        return code

    def c_code_cache_version(self):
        return (1,)


gpuger_no_inplace = GpuGer(destructive=False)
gpuger_inplace = GpuGer(destructive=True)


class GpuDot22(BlasOp, Dot22):
    def make_node(self, x, y):
        res = Dot22.make_node(self, x, y)
        x = as_gpuarray_variable(x)
        y = as_gpuarray_variable(y)
        assert x.dtype == y.dtype
        return Apply(self, [x, y], [x.type()])

    def perform(self, node, inputs, outputs):
        x, y = inputs

        out = pygpu.empty((x.shape[0], y.shape[1]), dtype=x.dtype)
        outputs[0][0] = blas.gemm(1., x, y, 0., out,
                                  overwrite_c=True)

    def c_code(self, node, name, inputs, outputs, sub):
        dtype = node.inputs[0].dtype
        typecode = pygpu.gpuarray.dtype_to_typecode(dtype)
        vars = dict(A=inputs[0], B=inputs[1], dtype=dtype, out=outputs[0],
                    typecode=typecode,
                    fail=sub['fail'], name=name)
        code = """
        double one = 1.;
        double zero = 0.;

        size_t dims[] = {0, 0};
        dims[0] = PyGpuArray_DIMS(%(A)s)[0];
        dims[1] = PyGpuArray_DIMS(%(B)s)[1];

        %(out)s = pygpu_empty(2, dims,
                            %(typecode)s,
                            GA_C_ORDER,
                            pygpu_default_context(), Py_None);
        if (!%(out)s) {
            %(fail)s
        }

        if (pygpu_blas_rgemm(cb_no_trans, cb_no_trans,
                             one,
                             %(A)s, %(B)s,
                             zero,
                             %(out)s, 0) == -1) {
            %(fail)s
        }
        """ % vars
        if config.gpuarray.sync:
            code += """
            GpuArray_sync(&%(out)s->ga);
            """ % vars
        return code

    def c_code_cache_version(self):
        return (1,)

    def c_headers(self):
        ret = super(GpuDot22, self).c_headers()
        return ret + ['<numpy_compat.h>']

gpu_dot22 = GpuDot22()

from theano.compile import optdb
from theano.gof import local_optimizer, LocalOptGroup
from theano.tensor.opt import in2out


@local_optimizer([gpugemv_no_inplace], inplace=True)
def local_inplace_gpuagemv(node):
    if node.op == gpugemv_no_inplace:
        return [gpugemv_inplace(*node.inputs)]


@local_optimizer([gpugemm_no_inplace], inplace=True)
def local_inplace_gpuagemm(node):
    if node.op == gpugemm_no_inplace:
        return [gpugemm_inplace(*node.inputs)]

@local_optimizer([gpuger_no_inplace], inplace=True)
def local_inplace_gpuager(node):
    if node.op == gpuger_no_inplace:
        return [gpuger_inplace(*node.inputs)]

gpuablas_opt_inplace = in2out(LocalOptGroup(
        local_inplace_gpuagemv, local_inplace_gpuagemm, local_inplace_gpuager),
                              name='gpuablas_opt_inplace')
optdb.register('InplaceGpuaBlasOpt',
               gpuablas_opt_inplace,
               70.0, 'fast_run', 'inplace', 'gpuarray')


class GpuDownsampleFactorMax(Op):
    """
    Implement downsample with max on the gpu.
    """
    def __init__(self, ds, ignore_border=False):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border

    def __eq__(self, another_op):
        return (type(self) == type(another_op) and
                self.ds == other.ds and
                self.ignore_border == other.ignore_border)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.ds) ^ hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s}' % (self.__class__.__name__,
                              self.ds,
                              self.ignore_border)

    def make_node(self, x):
	x = as_gpuarray_variable(x)
        if not x.type.ndim == 4:
            raise TypeError()
        return Apply(self, [x], [x.type()])
    
    def c_code(self, node, nodename, inputs, outputs, sub):
        vars = dict(
                    ds0 = self.ds[0], 
                    ds1 = self.ds[1],
                    x = inputs[0],
                    z = outputs[0],
                    typecode = pygpu.gpuarray.dtype_to_typecode(node.inputs[0].dtype),
                    itemsize_x = numpy.dtype(inputs[0]).itemsize,
                    itemsize_z = numpy.dtype(outputs[0]).itemsize,
                    nodename = nodename,
                    fail = sub['fail'],
                    ignore_border = int(self.ignore_border)
        )
        return """
        int dims[4], xdim2, xdim3;
        if (%(x)s->nd != 4)
        {
            PyErr_SetString(PyExc_ValueError,
                            "GpuDownsampleFactorMax: rank error");
            %(fail)s;
        }
        xdim2 = PyGpuArray_DIMS(%(x)s)[2];
        xdim3 = PyGpuArray_DIMS(%(x)s)[3];
        dims[0] = PyGpuArray_DIMS(%(x)s)[0];
        dims[1] = PyGpuArray_DIMS(%(x)s)[1];
        dims[2] = xdim2 / %(ds0)s;
        dims[3] = xdim3 / %(ds1)s;
        if (! %(ignore_border)s)
        {
            dims[2] += (xdim2%%(%(ds0)s)?1:0);
            dims[3] += (xdim3%%(%(ds1)s)?1:0);
        }
        if(dims[3]>512){
            PyErr_Format(PyExc_ValueError,
                         "GpuDownsampleFactorMax: last dimention size of %%d"
                         " is bigger then 512. This case is not implemented.",
                         dims[3]);
            %(fail)s;
        }
        
        if ((NULL == %(z)s)
            || (PyGpuArray_DIMS(%(z)s)[0] != dims[0])
            || (PyGpuArray_DIMS(%(z)s)[1] != dims[1])
            || (PyGpuArray_DIMS(%(z)s)[2] != dims[2])
            || (PyGpuArray_DIMS(%(z)s)[3] != dims[3]))
        {
            Py_XDECREF(%(z)s);
            %(z)s = pygpu_empty(4, 
                              dims,
                              %(typecode)s,
                              GA_C_ORDER,
                              pygpu_default_context(),
                              Py_None);
            if (NULL == %(z)s)
            {
                Py_XDECREF(%(z)s);
                %(z)s = NULL;
                PyErr_SetString(PyExc_ValueError,
                                "GpuDownsampleFactorMax:"
                                "Was not able to allocate output!");
                %(fail)s;
            }
                  

        }
        {
            dim3 grid(std::min(dims[0] * dims[1], 65535),
                      dims[2]);
            //dim3 block(std::min(dims[3], 512));
            //TODO: implement this by supporting more outputs than threads
            dim3 block(dims[3]);
            if ((grid.x*grid.y) && dims[3])
            kMaxPool_%(nodename)s<%(ds0)s, %(ds1)s> <<<grid, block,
                                                       xdim3*sizeof(float)>>>(
                dims[0], dims[1], dims[2], dims[3], xdim2, xdim3,
                (const float *)(((char *)cuda_get_ptr(%(x)s -> ga.data)) + %(x)s -> ga.offset),
                PyGpuArray_STRIDES(%(x)s)[0] / %(itemsize_x)s,
                PyGpuArray_STRIDES(%(x)s)[1] / %(itemsize_x)s,
                PyGpuArray_STRIDES(%(x)s)[2] / %(itemsize_x)s,
                PyGpuArray_STRIDES(%(x)s)[3] / %(itemsize_x)s,
                (float *)(((char *)cuda_get_ptr(%(z)s -> ga.data)) + %(z)s -> ga.offset),
                PyGpuArray_STRIDES(%(z)s)[0] / %(itemsize_z)s,
                PyGpuArray_STRIDES(%(z)s)[1] / %(itemsize_z)s,
                PyGpuArray_STRIDES(%(z)s)[2] / %(itemsize_z)s,
                PyGpuArray_STRIDES(%(z)s)[3]) / %(itemsize_z)s;

            if config.gpuarray.sync:
                GpuArray_sync(&%(z)s->ga);
          
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err)
            {
                PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: %%s: %%s. (grid: %%i x %%i;"
                    " block: %%i x %%i x %%i)\\n",
                    "kMaxPool_%(nodename)s",
                    cudaGetErrorString(err),
                    grid.x,
                    grid.y,
                    block.x,
                    block.y,
                    block.z);
                %(fail)s;
            }
        }
        """ % vars

        

    def c_support_code_apply(self, node, nodename):
        ignore_border = int(self.ignore_border)
        return """
        template<int pf2, int pf3>
        __global__ void kMaxPool_%(nodename)s(
           int D0, int D1, int D2, int D3, int xD2, int xD3,
           const float * x, int xS0, int xS1, int xS2, int xS3,
           float *z, int zS0, int zS1, int zS2, int zS3)
        {
            float cur_max, cur_x;
            // Cast threadIdx.x into a signed int, to avoid problems with
            // indexing with negative offsets.
            int tx = threadIdx.x;
            for(int block_x_idx = blockIdx.x;
                block_x_idx < D0 * D1;
                block_x_idx += gridDim.x){

                int i0 = block_x_idx %% D0;
                int i1 = block_x_idx / D0;
                int i2 = blockIdx.y;

                extern __shared__ float xbuf[]; //size [xD3]

                for (int r2 = 0;
                     (r2 < pf2) && (%(ignore_border)s || (r2 + i2*pf2 < xD2));
                     ++r2)
                {
                    __syncthreads();
                    // load the current row of the image into shared memory
                    for (int j = tx; j < xD3; j += blockDim.x)
                    {
                        xbuf[j] = x[i0*xS0 + i1*xS1 + (i2*pf2+r2)*xS2 + j*xS3];
                    }
                    __syncthreads();

                    // initialize our max if this is the
                    // first row we're loading
                    cur_max = (r2 == 0) ? xbuf[tx*pf3] : cur_max;

                    // do a mini-reduction over the pf3 relevant elements
                    // in the current row

                    if (%(ignore_border)s)
                    {
                        for (int k = 0; k < pf3; ++k)
                        {
                            cur_x = xbuf[tx*pf3+k];
                            cur_max = (cur_x > cur_max) ? cur_x : cur_max;
                        }
                    }
                    else
                    {
                        for (int k = 0; k < pf3; ++k)
                        {
                            if (tx*pf3 + k < xD3)
                            {
                                cur_x = xbuf[tx*pf3+k];
                                cur_max = (cur_x > cur_max) ? cur_x : cur_max;
                            }
                        }
                    }
                }

                z[i0*zS0 + i1*zS1 + i2*zS2 + tx*zS3] = cur_max;
            }
        }
        """ % locals()

    def c_compiler(self):
      return NVCC_compiler

    def c_headers(self):
      return ['cuda.h', 'gpuarray/extension.h', 'numpy_compat.h']

    #def perform(self, node, input_storage, output_storage):
        #raise NotImplementedError('only C is implemented')
    def c_code_cache_version(self):
	return (6)


class GpuDownsampleFactorMaxGrad(Op):
    """
    Implement the grad of downsample with max on the gpu.
    """
    def __init__(self, ds, ignore_border):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.ds == other.ds and
                self.ignore_border == other.ignore_border)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.ds) ^ hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s}' % (self.__class__.__name__,
                              self.ds,
                              self.ignore_border)

    def make_node(self, x, z, gz):
        return Apply(self, [x, z, gz], [x.type()])
    
    def c_code(self, node, nodename, inp, out, sub):
        x, z, gz = inp
        gx, = out
	typecode = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        itemsize_x = numpy.dtype(node.inputs[0].dtype).itemsize
        itemsize_z = numpy.dtype(node.inputs[1].dtype).itemsize
        itemsize_gz = numpy.dtype(node.inputs[2].dtype).itemsize
        itemsize_gx = numpy.dtype(node.outputs[0].dtype).itemsize
        fail = sub['fail']
        ds0, ds1 = self.ds
        ignore_border = int(self.ignore_border)
        
        return """
        if (%(x)s->nd != 4
            || %(z)s->nd != 4
            || %(gz)s->nd != 4)
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if ((NULL == %(gx)s)
            || (PyGpuArray_DIMS(%(gx)s)[0] !=
                PyGpuArray_DIMS(%(x)s)[0])
            || (PyGpuArray_DIMS(%(gx)s)[1] !=
                PyGpuArray_DIMS(%(x)s)[1])
            || (PyGpuArray_DIMS(%(gx)s)[2] !=
                PyGpuArray_DIMS(%(x)s)[2])
            || (PyGpuArray_DIMS(%(gx)s)[3] !=
                PyGpuArray_DIMS(%(x)s)[3]))
        {
            Py_XDECREF(%(gx)s);
            %(gx)s = pygpu_empty(4,
                               dims,
                               %(typecode)s,
                               GA_C_ORDER,
                               pygpu_default_context(),
                               Py_None);
            if (NULL == %(gx)s)
            {
                Py_XDECREF(%(gx)s);
                %(gx)s = NULL;
                %(fail)s;
            }
        }
        {
            //TODO: supporting more output columns than threads
            // make sure we cover every x row when ignore border isset and
            // there's a border present to be ignored
            int needs_extra_z_col = %(ignore_border)s && (PyGpuArray_DIMS(%(x)s)[2] %% %(ds0)s);
            dim3 grid(std::min(PyGpuArray_DIMS(%(z)s)[0], 65535),
                      PyGpuArray_DIMS(%(z)s)[2] + (needs_extra_z_col ? 1 : 0));
            dim3 block(std::min(PyGpuArray_DIMS(%(x)s)[3], 512));

            kDownsampleMaxGrad_%(nodename)s<%(ds0)s, %(ds1)s> <<<grid, block>>>(
                PyGpuArray_DIMS(%(z)s)[0],
                PyGpuArray_DIMS(%(z)s)[1],
                PyGpuArray_DIMS(%(z)s)[2],
                PyGpuArray_DIMS(%(z)s)[3],
                PyGpuArray_DIMS(%(x)s)[2],
                PyGpuArray_DIMS(%(x)s)[3],
                (const float *)(((char *)cuda_get_ptr(%(x)s -> ga.data)) + %(x)s -> ga.offset),
                PyGpuArray_STRIDES(%(x)s)[0] / %(itemsize_x)s,
                PyGpuArray_STRIDES(%(x)s)[1] / %(itemsize_x)s,
                PyGpuArray_STRIDES(%(x)s)[2] / %(itemsize_x)s,
                PyGpuArray_STRIDES(%(x)s)[3] / %(itemsize_x)s,
                (const float *)(((char *)cuda_get_ptr(%(z)s -> ga.data)) + %(z)s -> ga.offset),
                PyGpuArray_STRIDES(%(z)s)[0] / %(itemsize_z)s,
                PyGpuArray_STRIDES(%(z)s)[1] / %(itemsize_z)s,
                PyGpuArray_STRIDES(%(z)s)[2] / %(itemsize_z)s,
                PyGpuArray_STRIDES(%(z)s)[3] / %(itemsize_z)s,
                (const float *)(((char *)cuda_get_ptr(%(gz)s -> ga.data)) + %(gz)s -> ga.offset),
                PyGpuArray_STRIDES(%(gz)s)[0] / %(itemsize_gz)s,
                PyGpuArray_STRIDES(%(gz)s)[1] / %(itemsize_gz)s,
                PyGpuArray_STRIDES(%(gz)s)[2] / %(itemsize_gz)s,
                PyGpuArray_STRIDES(%(gz)s)[3] / %(itemsize_gz)s,
                (float *)(((char *)cuda_get_ptr(%(gx)s -> ga.data)) + %(gx)s -> ga.offset),
                PyGpuArray_STRIDES(%(gx)s)[0] / %(itemsize_gx)s,
                PyGpuArray_STRIDES(%(gx)s)[1] / %(itemsize_gx)s,
                PyGpuArray_STRIDES(%(gx)s)[2] / %(itermsize_gx)s,
                PyGpuArray_STRIDES(%(gx)s)[3]) / %(itemsize_gx)s;

            if config.gpuarray.sync:
                GpuArray_sync(&%(gx)s->ga);
            
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err)
            {
                PyErr_Format(PyExc_RuntimeError,
    "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kDownsampleMaxGrad_%(nodename)s",
                    cudaGetErrorString(err),
                    grid.x,
                    grid.y,
                    block.x,
                    block.y,
                    block.z);
                %(fail)s;
            }
        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        # This code considers every position in the output z, andthen
        # computes the gradient for the input pixels that were
        # downsampled to that z-position. It does so by running along
        # every z row (sometimes plus one, to make sure every gx row
        # gets totally filled), and by running along every x col. This
        # code is not sensitive to the ignore_border flag along the
        # row dimension (since it runs for every position in the
        # output z), but it is sensitive along the col dimension.
        ignore_border = int(self.ignore_border)

        return """
        // ds0 is the downsampling factor in rows, ds1 in columns
        template<int ds0, int ds1>
        __global__ void kDownsampleMaxGrad_%(nodename)s(
           int D0, int D1, int D2, int D3, int xD2, int xD3,
           const float * x, int xS0, int xS1, int xS2, int xS3,
           const float * z, int zS0, int zS1, int zS2, int zS3,
           const float * gz, int gzS0, int gzS1, int gzS2, int gzS3,
           float *gx, int gxS0, int gxS1, int gxS2, int gxS3)
        {
            //  D0: number of image rows
            //  D1: number of image cols
            //  D2: number of z rows
            //  D3: number of z cols
            // xD2: number of x rows
            // xD3: number of x cols
            // various .S. variables are strides

            float cur_max, cur_x, my_z, my_gz;
            // Cast threadIdx.x into a signed int, to avoid problems with
            // indexing with negative offsets.
            int tx = threadIdx.x;
            int bdimx = blockDim.x;

            for(int i0 = blockIdx.x;
                i0 < D0;
                i0 += gridDim.x){

                int i1 = 0;                // image col
                // row wrt z and/or gz, ranges from 0 to D2 - 1 OR D2
                // (as needed to cover all x rows)
                int i2 = blockIdx.y;
                int x_col = tx;            // col wrt x, ranges from 0 to xD3 - 1
                int z_col = x_col/ds1;     // z_col corresponding to this x_col


                //TODO: raise occupancy.  Use threadIdx.y to run several
                //      iterations of this i1 loop in parallel

                for (i1 = 0; i1 < D1; ++i1) // loop over images (same for z and x)
                {
                    for(int col_iter = 0;
                        (tx + col_iter * bdimx < xD3) ; col_iter++){

                        //The if inside is to don't do the division if we
                        // need only 1 col_iter

                        if(tx + bdimx < xD3)
                        {
                            x_col = tx + col_iter * bdimx;
                            z_col = x_col/ds1;
                        }

                        if (%(ignore_border)s && ((x_col >= ds1 * D3) || (i2 >= D2)))
                        {
                            // This happens only if x_col, or i2*ds0, was ignored
                            // (via ignore_border)
                            // TODO: if ignore_border is False, this is impossible
                            //        and we don't even need to generate this code.

                            my_gz = 0.0f;

                            //any fp number suffices for my_z, so we don't even
                            //need to set it to anything in particular.

                        }
                        else
                        {
                            // this is effectively:
                            // my_gz = gz[image_row][image_col][z_row][z_col]
                            // my_z  = z[image_row][image_col][z_row][z_col]
                            my_gz = gz[i0 * gzS0 + i1 * gzS1 + i2 * gzS2 +
                                       z_col*gzS3];
                            my_z =   z[i0 *  zS0 + i1 *  zS1 + i2 *  zS2 +
                                       z_col* zS3];
                        }
                        for (int x_row = i2*ds0;
                              (x_row < i2*ds0+ds0) && (x_row < xD2); ++x_row)
                        {
                            // this is effectively:
                            // gx[image_row][image_col][x_row][x_col]
                            //   = (my_z == x[image_row][image_col][
                            //                x_row][x_col]) ? my_gz : 0.0f;
                            gx[i0*gxS0 + i1*gxS1 + x_row*gxS2 + x_col*gxS3]
                               = (my_z == x[i0*xS0 + i1*xS1 + x_row*xS2 +
                                            x_col*xS3]) ? my_gz : 0.0f;
                        }

                    }
                }
            }
        }
        """ % locals()

    def c_compiler(self):
      return NVCC_compiler

    def c_headers(self):
      return ['cuda.h', 'gpuarray/extension.h', 'numpy_compat.h']

    def c_code_cache_version(self):
        return (9,)

