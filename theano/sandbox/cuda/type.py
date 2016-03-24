"""
Provide CudaNdarrayType.

"""
from __future__ import absolute_import, print_function, division
import os
import six.moves.copyreg as copyreg
import warnings

import numpy

import theano
from theano import Type, Variable
from theano import tensor, config
from theano import scalar as scal
from six import StringIO

try:
    # We must do those import to be able to create the full doc when nvcc
    # is not available
    import cuda_ndarray.cuda_ndarray as cuda
    from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler
    import cuda_ndarray
    # Python 3 does not necessarily set __file__. May need manual setting.
    # The problem is known to occur on Windows 10 with Python 3.4 installed by Anaconda.
    try:
        cuda_ndarray.__file__
    except AttributeError:
        from theano.gof.cmodule import get_lib_extension
        # Only works with Python 3, but it's fine, because Python 2
        # guarantees to set __file__ when importing any module.
        cuda_ndarray.__file__ = os.path.join(cuda_ndarray.__path__._path[0],
                                             'cuda_ndarray.' + get_lib_extension())
except ImportError:
    # Used to know that `cuda` could not be properly imported.
    cuda = None


class CudaNdarrayType(Type):

    typenum = 11  # Until hardware improves, this class deals with floats.

    dtype = 'float32'

    Variable = None
    """
    This will be set to the Variable type corresponding to this class.

    That variable type is `CudaNdarrayVariable` defined in the
    ``var.py`` file beside this one.

    Notes
    -----
    The var file depends on the file basic_ops.py, which depends on this file.
    A cyclic dependency is avoided by not hardcoding
    ``Variable = CudaNdarrayVariable``.

    """

    Constant = None
    """
    This will be set to `CudaNdarrayConstant` defined in ``var.py``.

    Notes
    -----
    The var file depends on the file basic_ops.py, which depends on this file.
    A cyclic dependency is avoided by not hardcoding this class.

    """

    SharedVariable = None
    """
    This will be set to `CudaNdarraySharedVariable` defined in ``var.py``.

    Notes
    -----
    The var file depends on the file basic_ops.py, which depends on this file.
    A cyclic dependency is avoided by not hardcoding this class.

    """

    if cuda is not None:
        value_zeros = staticmethod(cuda.CudaNdarray.zeros)
    """
    Create an CudaNdarray full of 0 values.

    """

    def __init__(self, broadcastable, name=None, dtype=None):
        if dtype is not None and dtype != 'float32':
            raise TypeError('%s only supports dtype float32 for now. Tried '
                            'using dtype %s for variable %s' %
                            (self.__class__.__name__, dtype, name))
        self.broadcastable = tuple(bool(b) for b in broadcastable)
        self.name = name
        self.dtype_specs()  # error checking is done there

    def clone(self, dtype=None, broadcastable=None):
        if broadcastable is None:
            broadcastable = self.broadcastable
        return self.__class__(broadcastable, name=self.name, dtype=dtype)

    def filter(self, data, strict=False, allow_downcast=None):
        return self.filter_inplace(data, None, strict=strict,
                                   allow_downcast=allow_downcast)

    def filter_inplace(self, data, old_data, strict=False,
                       allow_downcast=None):
        if strict or allow_downcast or isinstance(data, cuda.CudaNdarray):
            return cuda.filter(data, self.broadcastable, strict, old_data)

        else:  # (not strict) and (not allow_downcast)
            # Check if data.dtype can be accurately cast to self.dtype
            if isinstance(data, numpy.ndarray):
                up_dtype = scal.upcast(self.dtype, data.dtype)
                if up_dtype == self.dtype:
                    return cuda.filter(data, self.broadcastable,
                                       strict, old_data)
                else:
                    raise TypeError(
                        '%s, with dtype %s, cannot store a value of '
                        'dtype %s without risking loss of precision.'
                        'If you do not mind, please cast your data to %s.'
                        % (self, self.dtype, data.dtype, self.dtype),
                        data)
            else:
                converted_data = theano._asarray(data, self.dtype)

                if (allow_downcast is None and
                        type(data) is float and
                        self.dtype == theano.config.floatX):
                    return cuda.filter(converted_data, self.broadcastable,
                            strict, old_data)
                elif numpy.all(data == converted_data):
                    return cuda.filter(converted_data, self.broadcastable,
                            strict, old_data)
                else:
                    raise TypeError(
                        '%s, with dtype %s, cannot store accurately value %s, '
                        'it would be represented as %s. If you do not mind, '
                        'you can cast your data to %s.'
                        % (self, self.dtype, data, converted_data, self.dtype),
                        data)

    def filter_variable(self, other, allow_convert=True):
        """
        Convert a Variable into a CudaNdarrayType, if compatible.

        This Variable should either already be a CudaNdarrayType, or be
        a TensorType. It has to have the right number of dimensions,
        broadcastable pattern, and dtype.

        """
        if hasattr(other, '_as_CudaNdarrayVariable'):
            other = other._as_CudaNdarrayVariable()

        if not isinstance(other, Variable):
            # The value is not a Variable: we cast it into
            # a Constant of the appropriate Type.
            other = self.Constant(type=self, data=other)

        if other.type == self:
            return other

        if not isinstance(other.type, (tensor.TensorType, CudaNdarrayType)):
            raise TypeError('Incompatible type', (self, other.type))
        if (other.type.dtype != self.dtype):
            raise TypeError('Incompatible dtype', (self.dtype,
                                                   other.type.dtype))
        if other.type.ndim != self.ndim:
            raise TypeError('Incompatible number of dimensions.'
                            ' Expected %d, got %d.' % (self.ndim, other.ndim))
        if other.type.broadcastable != self.broadcastable:
            if allow_convert:
                type2 = other.type.clone(broadcastable=self.broadcastable)
                other2 = type2.convert_variable(other)
            else:
                other2 = None
            if other2 is None:
                raise TypeError('Incompatible broadcastable dimensions.'
                                ' Expected %s, got %s.' %
                                (str(other.type.broadcastable),
                                 str(self.broadcastable)))
            other = other2

        return theano.sandbox.cuda.basic_ops.GpuFromHost()(other)

    @staticmethod
    def bound(a):
        high = a.gpudata
        low = a.gpudata
        # stride is in the number of element.
        # we must convert that to bytes in case we
        # will view the element as a different type.
        elem_size = numpy.zeros(0, dtype=a.dtype).dtype.itemsize

        for stri, shp in zip(a._strides, a.shape):
            if stri < 0:
                low += (stri * elem_size) * (shp - 1)
            else:
                high += (stri * elem_size) * (shp - 1)
        return low, high

    @staticmethod
    def may_share_memory(a, b):
        # when this is called with a an ndarray and b
        # a sparce matrix, numpy.may_share_memory fail.
        if a is b:
            return True
        if a.__class__ is b.__class__:
            a_l, a_h = CudaNdarrayType.bound(a)
            b_l, b_h = CudaNdarrayType.bound(b)
            if b_l >= a_h or a_l >= b_h:
                return False
            return True
        else:
            return False

    @staticmethod
    def values_eq(a, b):
        # TODO: make the comparaison without transfert.
        return tensor.TensorType.values_eq(numpy.asarray(a), numpy.asarray(b))

    @staticmethod
    def values_eq_approx(a, b, allow_remove_inf=False, allow_remove_nan=False,
                         rtol=None, atol=None):
        # TODO: make the comparaison without transfert.
        return tensor.TensorType.values_eq_approx(
            numpy.asarray(a),
            numpy.asarray(b),
            allow_remove_inf=allow_remove_inf,
            allow_remove_nan=allow_remove_nan,
            rtol=rtol, atol=atol
        )

    def dtype_specs(self):
        """
        Return a tuple (python type, c type, numpy typenum) that corresponds
        to self.dtype.

        This function is used internally as part of C code generation.

        """
        # TODO: add more type correspondances for e.g. int32, int64, float32,
        # complex64, etc.
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
                    'complex128': (complex, 'theano_complex128',
                                   'NPY_COMPLEX128'),
                    'complex64': (complex, 'theano_complex64',
                                  'NPY_COMPLEX64')}[self.dtype]
        except KeyError:
            raise TypeError("Unsupported dtype for %s: %s" % (
                    self.__class__.__name__, self.dtype))

    def __eq__(self, other):
        """
        Compare True iff other is the same kind of CudaNdarrayType.

        """
        return (type(self) == type(other) and
                other.broadcastable == self.broadcastable)

    def convert_variable(self, var):
        if (type(self) == type(var.type) and
            self.ndim == var.type.ndim and
            all(sb == ob or ob for sb, ob in zip(self.broadcastable,
                                                 var.type.broadcastable))):
            return theano.tensor.patternbroadcast(var, self.broadcastable)

    def __hash__(self):
        """
        Hash equal for same kinds of CudaNdarrayType.

        """
        return hash(type(self)) ^ hash(self.broadcastable)

    ndim = property(lambda self: len(self.broadcastable),
                    doc="number of dimensions")
    """
    Number of dimensions.

    This read-only property is the preferred way to get the number of
    dimensions of a `CudaNdarrayType`.

    """

    def make_variable(self, name=None):
        """
        Return a `TensorVariable` of this type.

        Parameters
        ----------
        name : str
            A pretty name to identify this `Variable` when printing and
            debugging.

        """
        return self.Variable(self, name=name)

    def __str__(self):
        if self.name:
            return self.name
        else:
            b = self.broadcastable
            #bcast = str(self.broadcastable)
            if not numpy.any(b):
                s = "%iD" % len(b)
            else:
                s = str(b)

            bcast = {(): 'scalar',
                     (False,): 'vector',
                     (False, True): 'col',
                     (True, False): 'row',
                     (False, False): 'matrix'}.get(b, s)
            return "CudaNdarrayType(%s, %s)" % (str(self.dtype), bcast)

    def __repr__(self):
        return str(self)
        #"CudaNdarrayType{%s, %s}" % (str(self.dtype), str(self.broadcastable))

    def c_declare(self, name, sub, check_input=True):
        return """ CudaNdarray * %(name)s;""" % locals()

    def c_init(self, name, sub):
        return "%(name)s = NULL;" % locals()

    def c_extract(self, name, sub, check_input=True,
                  check_broadcast=True):
        sio = StringIO()
        fail = sub['fail']
        nd = self.ndim
        print("""
        assert(py_%(name)s->ob_refcnt >= 2); // There should be at least one ref from the container object,
        // and one ref from the local scope.

        if (CudaNdarray_Check(py_%(name)s))
        {
            //fprintf(stderr, "c_extract CNDA object w refcnt %%p %%i\\n", py_%(name)s, (py_%(name)s->ob_refcnt));
            %(name)s = (CudaNdarray*)py_%(name)s;
            //std::cerr << "c_extract " << %(name)s << '\\n';
        """ % locals(), file=sio)
        if(check_input):
            print("""
                if (%(name)s->nd != %(nd)s)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has rank %%i, it was supposed to have rank %(nd)s",
                                 %(name)s->nd);
                    %(name)s = NULL;
                    %(fail)s;
                }
                //std::cerr << "c_extract " << %(name)s << " nd check passed\\n";
            """ % locals(), file=sio)
            for i, b in enumerate(self.broadcastable):
                if b and check_broadcast:
                    print("""
                if (CudaNdarray_HOST_DIMS(%(name)s)[%(i)s] != 1)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has dim %%i on broadcastable dimension %%i",
                                 CudaNdarray_HOST_DIMS(%(name)s)[%(i)s], %(i)s);
                    %(name)s = NULL;
                    %(fail)s;
                }
                //std::cerr << "c_extract " << %(name)s << "dim check %(i)s passed\\n";
                //std::cerr << "c_extract " << %(name)s << "checking bcast %(i)s <" << %(name)s->str<< ">\\n";
                //std::cerr << "c_extract " << %(name)s->str[%(i)s] << "\\n";
                if (CudaNdarray_HOST_STRIDES(%(name)s)[%(i)s])
                {
                    //std::cerr << "c_extract bad stride detected...\\n";
                    PyErr_Format(PyExc_RuntimeError,
                                 "c_extract: Some CudaNdarray has a nonzero stride %%i on a broadcastable dimension %%i",
                                 CudaNdarray_HOST_STRIDES(%(name)s)[%(i)s], %(i)s);
                    %(name)s = NULL;
                    %(fail)s;
                }
                //std::cerr << "c_extract " << %(name)s << "bcast check %(i)s passed\\n";
                    """ % locals(), file=sio)
            print("""
                assert(%(name)s);
                Py_INCREF(py_%(name)s);
            }
            else if (py_%(name)s == Py_None)
            {
                PyErr_SetString(PyExc_TypeError,
                                "expected a CudaNdarray, not None");
                %(name)s = NULL;
                %(fail)s;
            }
            else
            {
                //fprintf(stderr, "FAILING c_extract CNDA object w refcnt %%p %%i\\n", py_%(name)s, (py_%(name)s->ob_refcnt));
                PyErr_SetString(PyExc_TypeError, "Argument not a CudaNdarray");
                %(name)s = NULL;
                %(fail)s;
            }
            //std::cerr << "c_extract done " << %(name)s << '\\n';
            """ % locals(), file=sio)
        else:
            print("""
                assert(%(name)s);
                Py_INCREF(py_%(name)s);
            }
            """ % locals(), file=sio)
        # print sio.getvalue()
        return sio.getvalue()

    def c_extract_out(self, name, sub, check_input=True, check_broadcast=True):
        """ 
        To allow the hack to skip check_broadcast.

        """
        return """
        if (py_%(name)s == Py_None)
        {
            %(c_init_code)s
        }
        else
        {
            %(c_extract_code)s
        }
        """ % dict(
            name=name,
            c_init_code=self.c_init(name, sub),
            c_extract_code=self.c_extract(name, sub, check_input,
                                          check_broadcast))

    def c_cleanup(self, name, sub):
        return """
        //std::cerr << "cleanup " << py_%(name)s << " " << %(name)s << "\\n";
        //fprintf(stderr, "c_cleanup CNDA py_object w refcnt %%p %%i\\n", py_%(name)s, (py_%(name)s->ob_refcnt));
        if (%(name)s)
        {
            //fprintf(stderr, "c_cleanup CNDA cn_object w refcnt %%p %%i\\n", %(name)s, (%(name)s->ob_refcnt));
            Py_XDECREF(%(name)s);
        }
        //std::cerr << "cleanup done" << py_%(name)s << "\\n";
        """ % locals()

    def c_sync(self, name, sub):
        """
        Override `CLinkerOp.c_sync`.

        """
        return """
        //std::cerr << "sync\\n";
        if (NULL == %(name)s) {
            // failure: sync None to storage
            Py_XDECREF(py_%(name)s);
            py_%(name)s = Py_None;
            Py_INCREF(py_%(name)s);
        }
        else
        {
            if (py_%(name)s != (PyObject*)%(name)s)
            {
                Py_XDECREF(py_%(name)s);
                py_%(name)s = (PyObject*)%(name)s;
                Py_INCREF(py_%(name)s);
            }
            assert(py_%(name)s->ob_refcnt);
        }
        """ % locals()

    def c_headers(self):
        """
        Override `CLinkerOp.c_headers`.

        """
        return ['cuda_ndarray.cuh']

    def c_header_dirs(self):
        """
        Override `CLinkerOp.c_headers`.

        """
        ret = [os.path.dirname(cuda_ndarray.__file__)]
        cuda_root = config.cuda.root
        if cuda_root:
            ret.append(os.path.join(cuda_root, 'include'))
        return ret

    def c_lib_dirs(self):
        ret = [os.path.dirname(cuda_ndarray.__file__)]
        return ret

    def c_libraries(self):
        # returning cublas because the cuda_ndarray.cuh header
        # includes calls to SetVector and cublasGetError
        return ['cudart', config.cublas.lib, 'cuda_ndarray']

    def c_support_code(cls):
        return ""

    def c_code_cache_version(self):
        # return ()
        # no need to put nvcc.fastmath in the tuple as the
        # c_compile_args is put in the key.
        return (3,)  # cublas v2 changes

    def c_compiler(self):
        return NVCC_compiler

    def c_compile_args(self):
        return []

    def get_shape_info(self, obj):
        return obj.shape

    def get_size(self, shape_info):
        if shape_info:
            return numpy.prod(shape_info) * numpy.dtype(self.dtype).itemsize
        else:  # a scalar
            return numpy.dtype(self.dtype).itemsize

theano.compile.ops.expandable_types += (CudaNdarrayType,)

# Register C code for ViewOp on CudaNdarrayType
theano.compile.register_view_op_c_code(
        CudaNdarrayType,
        """
        Py_XDECREF(%(oname)s);
        %(oname)s = %(iname)s;
        Py_XINCREF(%(oname)s);
        """,
        version=1)

theano.compile.register_shape_i_c_code(
    CudaNdarrayType,
    """
    if(!%(oname)s)
        %(oname)s=(PyArrayObject*)PyArray_ZEROS(0, NULL, NPY_INT64, 0);
    ((npy_int64*)PyArray_DATA(%(oname)s))[0] =
                              CudaNdarray_HOST_DIMS(%(iname)s)[%(i)s];
    """,
    """
    if (%(i)s>=CudaNdarray_NDIM(%(iname)s)){
        PyErr_SetString(PyExc_TypeError,
            "Number of dimensions lower than expected");
        %(fail)s
    }
    """,
    version=(1,))

# Register CudaNdarrayType to the DeepCopyOp list of types with c code.
theano.compile.register_deep_copy_op_c_code(
        CudaNdarrayType,
        """
        int alloc = %(oname)s == NULL;
        for(int i=0; !alloc && i<CudaNdarray_NDIM(%(oname)s); i++) {
           if(CudaNdarray_HOST_DIMS(%(iname)s)[i] !=
              CudaNdarray_HOST_DIMS(%(oname)s)[i]) {
               alloc = true;
               break;
           }
        }
        if(alloc) {
            Py_XDECREF(%(oname)s);
            %(oname)s = (CudaNdarray*)CudaNdarray_Copy(%(iname)s);
            if (!%(oname)s)
            {
                PyErr_SetString(PyExc_ValueError,
                                "DeepCopyOp: the copy failed!");
                %(fail)s;
            }
        } else {
            if(CudaNdarray_CopyFromCudaNdarray(%(oname)s, %(iname)s)) {
                PyErr_SetString(PyExc_ValueError,
            "DeepCopyOp: the copy failed into already allocated space!");
                %(fail)s;
            }
        }
        """,
        version=3)


# THIS WORKS But CudaNdarray instances don't compare equal to one
# another, and what about __hash__ ?  So the unpickled version doesn't
# equal the pickled version, and the cmodule cache is not happy with
# the situation.
def CudaNdarray_unpickler(npa):

    if config.experimental.unpickle_gpu_on_cpu:
        # directly return numpy array
        warnings.warn("config.experimental.unpickle_gpu_on_cpu is set to True. Unpickling CudaNdarray as numpy.ndarray")
        return npa
    elif cuda:
        return cuda.CudaNdarray(npa)
    else:
        raise ImportError("Cuda not found. Cannot unpickle CudaNdarray")

copyreg.constructor(CudaNdarray_unpickler)


def CudaNdarray_pickler(cnda):
    return (CudaNdarray_unpickler, (numpy.asarray(cnda),))

# In case cuda is not imported.
if cuda is not None:
    copyreg.pickle(cuda.CudaNdarray, CudaNdarray_pickler,
                    CudaNdarray_unpickler)
