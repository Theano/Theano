from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar
import StringIO

class GpuDot22(Op):
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, y):
        if x.type.ndim != 2:
            raise TypeError(x)
        if y.type.ndim != 2:
            raise TypeError(y)
        return Apply(self, [x,y], [x.type()])

    def c_code_cache_version(self):
        return ()

    def c_code(self, node, nodename, inputs, outputs, sub):
        x, y = inputs
        z, = outputs
        fail = sub['fail']
        return """
        if (cnda_%(x)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(x)==%%i must be 2", cnda_%(x)s->nd);
            %(fail)s;
        }
        if (cnda_%(y)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(y)==%%i must be 2", cnda_%(y)s->nd);
            %(fail)s;
        }
        if ((NULL == cnda_%(z)s)
            || (cnda_%(z)s->dim[0] != cnda_%(x)s->dim[0])
            || (cnda_%(z)s->dim[1] != cnda_%(y)s->dim[1]))
        {
            if (cnda_%(z)s) Py_DECREF(cnda_%(z)s);
            npy_intp dims[2];
            dims[0] = cnda_%(x)s->dim[0];
            dims[1] = cnda_%(y)s->dim[1];
            cnda_%(z)s = (CudaNdarray*)CudaNdarray_new_null();
            if ((NULL == cnda_%(z)s) || CudaNdarray_alloc_contiguous(cnda_%(z)s, 2, dims))
            {
                if (cnda_%(z)s)
                {
                    Py_DECREF(cnda_%(z)s);
                    cnda_%(z)s = NULL;
                }
                %(fail)s;
            }
        }
        if (CudaNdarray_gemm(1.0f, cnda_%(x)s, cnda_%(y)s, 0.0f, cnda_%(z)s))
        {
            if (cnda_%(z)s)
            {
                Py_DECREF(cnda_%(z)s);
                cnda_%(z)s = NULL;
            }
            %(fail)s;
        }
        """ % locals()
gpu_dot22 = GpuDot22()

class GpuGemm(Op):
    destroy_map = {0:[0]}
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, z, a, x, y, b):
        # the more complicated error checking performed by tensor.gemm is assumed to already
        # have been done
        return Apply(self, [z, a, x, y, b], [z.type()])

    def c_code_cache_version(self):
        return ()

    def c_code(self, node, name, inputs, outputs, sub):
        z_in, a, x, y, b = inputs
        z_out, = outputs
        fail = sub['fail']
        return """

        #define REAL float
        float %(name)s_a = (%(a)s->descr->type_num == PyArray_FLOAT) 
        ? (REAL)(((float*)%(a)s->data)[0])
        : (REAL)(((double*)%(a)s->data)[0]);

        float %(name)s_b = (%(b)s->descr->type_num == PyArray_FLOAT) ?
        (REAL)(((float*)%(b)s->data)[0])
        : (REAL)(((double*)%(b)s->data)[0]);
        #undef REAL

        if (CudaNdarray_gemm(%(name)s_a, cnda_%(x)s, cnda_%(y)s, %(name)s_b, cnda_%(z_in)s))
        {
            %(fail)s;
        }
        cnda_%(z_out)s = cnda_%(z_in)s;
        Py_INCREF(cnda_%(z_out)s);
        """ % locals()
gpu_gemm = GpuGemm()
