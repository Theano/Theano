import sys
import numpy

from theano import Op, Type, Apply, Variable, Constant
from theano import tensor
from theano.compile.sandbox.sharedvalue import shared, SharedVariable, shared_constructor

import cuda_ndarray # the module

class _tensor_operators(object):
    def _as_TensorVariable(self):
        return HostFromGpu()(self)
    def _as_CudaNdarrayVariable(self):
        return self

    dtype = property(lambda s:s.type.dtype)
    broadcastable = property(lambda s:s.type.broadcastable)
    ndim = property(lambda s:s.type.ndim)

class CudaNdarrayType(Type):

    def __init__(self, dtype, broadcastable, name=None):
        self.typenum = numpy.dtype(dtype).num
        self.dtype = str(dtype)
        self.broadcastable = tuple(broadcastable)
        self.name = name
        self.dtype_specs() # error checking is done there

    def filter(self, data, strict=False):
        typenum = numpy.dtype(self.dtype).num
        print >> sys.stderr, "bcastable", self.broadcastable
        return tensorview_module.filter(data, typenum, self.broadcastable, strict)

    def dtype_specs(self):
        """Return a tuple (python type, c type, numpy typenum) that corresponds to
        self.dtype.
        
        This function is used internally as part of C code generation.
        """
        #TODO: add more type correspondances for e.g. int32, int64, float32,
        #complex64, etc.
        try:
            return {'float32': (float, 'npy_float32', 'NPY_FLOAT32'),
                    'float64': (float, 'npy_float64', 'NPY_FLOAT64'),
                    'uint8': (int, 'npy_uint8', 'NPY_UINT8'),
                    'int8': (int, 'npy_int8', 'NPY_INT8'),
                    'uint16': (int, 'npy_uint16', 'NPY_UINT16'),
                    'int16': (int, 'npy_int16', 'NPY_INT16'),
                    'uint32': (int, 'npy_uint32', 'NPY_UINT32'),
                    'int32': (int, 'npy_int32', 'NPY_INT32'),
                    'uint64': (int, 'npy_uint64', 'NPY_UINT64'),
                    'int64': (int, 'npy_int64', 'NPY_INT64'),
                    'complex128': (complex, 'theano_complex128', 'NPY_COMPLEX128'),
                    'complex64': (complex, 'theano_complex64', 'NPY_COMPLEX64')}[self.dtype]
        except KeyError:
            raise TypeError("Unsupported dtype for %s: %s" % (self.__class__.__name__, self.dtype))

    def __eq__(self, other):
        """Compare True iff other is the same kind of CudaNdarrayType"""
        return type(self) == type(other) and other.typenum == self.typenum and other.broadcastable == self.broadcastable

    def values_eq_approx(self, a, b):
        if type(a) is numpy.ndarray and type(b) is numpy.ndarray:
            if a.shape != b.shape:
                return False
            if a.dtype != b.dtype:
                return False
            if 'int' in str(a.dtype):
                return numpy.all(a==b)
            elif a.shape == (): #for comparing scalars, use broadcasting.
                # Note: according to James B, there was a reason for the
                # following two lines, that may seem weird at first glance.
                # If someone can figure out what it is, please say it here!
                ones = numpy.ones(2)
                return numpy.allclose(ones * a, ones*b)
            #elif str(a.dtype).startswith('complex'):
            #    print >> sys.stderr, 'WARNING: skipping comparison of complex'
            #    return True
            else:
                cmp = numpy.allclose(a,b)
                if cmp:
                    # Numpy claims they are close, this is good enough for us.
                    return True
                # Numpy is unhappy, but it does not necessarily mean that a and
                # b are different. Indeed, Numpy does not like missing values
                # and will return False whenever some are found in a or b.
                # The proper way would be to use the MaskArray stuff available
                # in Numpy. However, it looks like it has been added to Numpy's
                # core recently, so it may not be available to everyone. Thus,
                # for now we use a home-made recipe, that should probably be
                # revisited in the future.
                a_missing = numpy.isnan(a)
                if not a_missing.any():
                    # There are no missing values in a, thus this is not the
                    # reason why numpy.allclose(a, b) returned False.
                    return False
                # The following line is what numpy.allclose bases its decision
                # upon, according to its documentation.
                rtol = 1.0000000000000001e-05
                atol = 1e-8
                cmp_elemwise = (numpy.absolute(a - b) <=
                        (atol + rtol * numpy.absolute(b)))
                # Find places where both a and b have missing values.
                both_missing = a_missing * numpy.isnan(b)
                # Combine all information.
                return (cmp_elemwise + both_missing).all()

        return False

    def __hash__(self):
        """Hash equal for same kinds of CudaNdarrayType"""
        return hash(type(self)) ^ hash(self.typenum) ^ hash(self.broadcastable)

    ndim = property(lambda self: len(self.broadcastable), doc = "number of dimensions")
    """Number of dimensions

    This read-only property is the preferred way to get the number of dimensions
    of a `CudaNdarrayType`.
    
    """

    def make_variable(self, name = None):
        """Return a `TensorVariable` of this type

        :Parameters:
         - `name`: str
           A pretty name to identify this `Variable` when printing and debugging

        """
        return CudaNdarrayVariable(self, name = name)

    def __str__(self):
        if self.name:
            return self.name
        else:
            b = self.broadcastable
            #bcast = str(self.broadcastable)
            bcast = {(): 'scalar',
                     (False,): 'vector',
                     (False, True): 'col',
                     (True, False): 'row',
                     (False, False): 'matrix'}.get(b, "%iD" % len(b) if not any(b) else str(b))
            return "CudaNdarrayType(%s, %s)" % (str(self.dtype), bcast)

    def __repr__(self):
        return str(self)
        #"CudaNdarrayType{%s, %s}" % (str(self.dtype), str(self.broadcastable))

    def c_declare(self, name, sub):
        ndim = self.ndim
        c_typename = self.dtype_specs()[1]
        return """ CudaNdarrayType::VoidTensor* vt_%(name)s;""" %locals()

    def c_init(self, name, sub):
        return "vt_%(name)s = NULL;" % locals()

    def c_extract(self, name, sub):
        return """
        vt_%(name)s = CudaNdarrayType::voidtensor_from_cobject(py_%(name)s);
        std::cerr << "extract  "<< py_%(name)s << " " << vt_%(name)s <<  "\\n";
        if (!vt_%(name)s)
        {
            PyErr_SetString(PyExc_TypeError, "Failed to extract VoidTensor");
            %(fail)s;
        }
        """ % dict(sub, name = name, type_num = self.dtype_specs()[2])

    def c_cleanup(self, name, sub):
        return """
        std::cerr << "cleanup " << py_%(name)s << "\\n";
        """ % locals()

    def c_sync(self, name, sub):
        """Override `CLinkerOp.c_sync` """
        return """
        std::cerr << "sync\\n";
        if (!vt_%(name)s) {  
            // failure: sync None to storage
            Py_XDECREF(py_%(name)s);
            py_%(name)s = Py_None;
            Py_XINCREF(py_%(name)s);
        }
        else if (PyCObject_AsVoidPtr(py_%(name)s) != (void*)vt_%(name)s) {
            // success, but a new gtt was allocated for us
            // we trust that the op code deleted the old gtt
            // we just pack the new gtt into a CObject
            Py_XDECREF(py_%(name)s);
            py_%(name)s = CudaNdarrayType::cobject_from_voidtensor(vt_%(name)s);
            std::cerr << "sync packing " << vt_%(name)s << " into new CObject "<< py_%(name)s << " "<< PyCObject_Check(py_%(name)s) << "\\n";
        }
        """ % locals()

    def c_headers(self):
        """Override `CLinkerOp.c_headers` """
        return []

    def c_libraries(self):
        return []

    def c_support_code(cls):
        rval = file('tensorview.cc').read()
        return rval

    def c_code_cache_version(self):
        return () #do not cache this stuff until it matures

class CudaNdarrayVariable(Variable, _tensor_operators):
    pass

class CudaNdarrayConstant(Constant, _tensor_operators):
    pass

class CudaNdarraySharedVariable(SharedVariable, _tensor_operators):

    def __getvalue(self):
        return tensorview_module.ndarray_from_voidtensor(self.container.value)
    def __setvalue(self, value):
        self.container.value = value #container does the filtering 
    value = property(__getvalue, __setvalue)

    def filter_update(self, other):
        if hasattr(other, '_as_CudaNdarrayVariable'):
            return other._as_CudaNdarrayVariable()

        if isinstance(other.type, tensor.TensorType) and (other.type.dtype == self.dtype) and (other.broadcastable == self.broadcastable):
            return GpuFromHost()(other)
        else:
            raise TypeError(other)

def gpu_tensor_shared_constructor(value, name, strict=False):
    """SharedVariable Constructor for TensorType"""
    if not isinstance(value, numpy.ndarray):
        raise TypeError

    bcast = [0 for b in value.shape]
    type = CudaNdarrayType(value.dtype, broadcastable=bcast)
    return CudaNdarraySharedVariable(type=type, value=value, name=name, strict=strict)

class HostFromGpu(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def make_node(self, x):
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError(x)
        return Apply(self, [x], [tensor.TensorType(dtype=x.dtype, broadcastable=x.broadcastable)()])
    def perform(self, node, (x,), (z,)):
        z[0] = tensorview_module.ndarray_from_voidtensor(x)
    def grad(self, inputs, (gz,)):
        return [GpuFromHost()(gz)]

class GpuFromHost(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [CudaNdarrayType(dtype=x.dtype, broadcastable=x.broadcastable)()])
    def perform(self, node, (x,), (z,)):
        z[0] = tensorview_module.filter(x, x.dtype.num, tuple([0]*x.ndim), 0)
    def grad(self, inputs, (gz,)):
        return [HostFromGpu()(gz)]

class GpuAdd(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def make_node(self, a, b):
        if not isinstance(a.type, CudaNdarrayType):
            raise TypeError(a)
        if not isinstance(b.type, CudaNdarrayType):
            raise TypeError(b)
        if a.type.broadcastable != b.type.broadcastable:
            raise NotImplementedError('different bcastable')
        if a.dtype != b.dtype:
            raise NotImplementedError('different dtype')
        return Apply(self, [a,b], [CudaNdarrayType(dtype=a.dtype, broadcastable=a.broadcastable)()])

    def perform(self, node, (a,b), (z,)):
        aval = tensorview_module.ndarray_from_voidtensor(a)
        bval = tensorview_module.ndarray_from_voidtensor(b)
        zval = aval + bval
        z[0] = tensorview_module.filter(zval,zval.dtype.num,(0,)*len(zval.shape), 0)

    def grad(self, inputs, (gz,)):
        return [gz for i in inputs]

    def c_support_code(self):
        return """
        template<typename T0, typename T1, typename T2>
        void gpu_tensor_add(const int nd, const int * dim, 
                T0 * __restrict__ z, const int * zstr, 
                const T1 * __restrict__ a, const int * astr,
                const T2 * __restrict__ b, const int * bstr)
        {
            if (0 == nd) //copy a scalar
            {
                z[0] = a[0] + b[0];
            }
            else
            {
                for (int i = 0; i< dim[0]; ++i)
                {
                    gpu_tensor_add(nd-1, dim+1, 
                        z + i * zstr[0], zstr+1,
                        a + i * astr[0], astr+1,
                        b + i * bstr[0], bstr+1);
                }
            }
        }
        """

    def c_code(self, node, nodename, (a,b), (z,), sub):
        asym, bsym = node.inputs
        zsym, = node.outputs
        nd_a = asym.ndim
        nd_b = bsym.ndim
        nd_z = zsym.ndim
        typename_a = asym.type.dtype_specs()[1]
        typename_b = bsym.type.dtype_specs()[1]
        typename_z = zsym.type.dtype_specs()[1]
        return """
        std::cerr << "GpuAdd start\\n";
        if (vt_%(z)s) delete vt_%(z)s;
        vt_%(z)s = new CudaNdarrayType::VoidTensor(vt_%(a)s->typenum, vt_%(a)s->elsize, %(nd_a)s, vt_%(a)s->dim);
        CudaNdarrayType::TensorView<%(nd_a)s, %(typename_a)s> view_%(a)s(vt_%(a)s);
        CudaNdarrayType::TensorView<%(nd_b)s, %(typename_b)s> view_%(b)s(vt_%(b)s);
        CudaNdarrayType::TensorView<%(nd_z)s, %(typename_z)s> view_%(z)s(vt_%(z)s);

        gpu_tensor_add(vt_%(a)s->nd, vt_%(a)s->dim,
            view_%(z)s.data, view_%(z)s.str,
            view_%(a)s.data, view_%(a)s.str,
            view_%(b)s.data, view_%(b)s.str);

        std::cerr << "GpuAdd done\\n";
        """ %locals()

    def c_code_cache_version(self):
        return ()

    #compiler = theano.gof.cmodule.nvcc_module_compile_str


@tensor.gof.local_optimizer([GpuFromHost(), None])
def local_gpu_host_gpu(node):
    if not tensor.opt.opt.check_chain(node, GpuFromHost(), HostFromGpu()):
        return False
    return [node.inputs[0].owner.inputs[0]]
tensor.opt.register_canonicalize(local_gpu_host_gpu, 'gpu_host_gpu')
@tensor.gof.local_optimizer([HostFromGpu(), None])
def local_host_gpu_host(node):
    if not tensor.opt.opt.check_chain(node, HostFromGpu(), GpuFromHost()):
        return False
    return [node.inputs[0].owner.inputs[0]]
tensor.opt.register_canonicalize(local_host_gpu_host, 'host_gpu_host')

@tensor.gof.local_optimizer([GpuFromHost(), None])
def local_gpu_add(node):
    if node.op == GpuFromHost():
        if node.inputs[0].owner and node.inputs[0].owner.op == tensor.add:
            add_inputs = node.inputs[0].owner.inputs
            if any(hasattr(i.owner, 'op') and isinstance(i.owner.op, HostFromGpu) for i in add_inputs):
                # move the add to a GpuAdd
                return [GpuAdd()(*(GpuFromHost()(i) for i in add_inputs))]
    return False

tensor.opt.register_canonicalize(local_gpu_add, 'gpu_add')






def unset_shared_for_numpy():
    raise NotImplementedError()

def set_shared_for_numpy():
    """
    Set the gpu_tensor_constructor as the handler for ndarray
    """
    shared_constructor(gpu_tensor_shared_constructor)


