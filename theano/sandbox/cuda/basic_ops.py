from __future__ import absolute_import, print_function, division
import copy
import logging
import sys

import numpy
from six import iteritems
from six.moves import StringIO, xrange

import theano
from theano import gof, Type, Apply
from theano import tensor, scalar, config
from theano.gradient import grad_undefined
from theano.scalar import Scalar

scal = scalar  # somewhere scalar gets reassigned to be a function

try:
    # We must be able to import this file to create the full doc when nvcc
    # is not available
    from theano.sandbox.cuda import filter as type_support_filter
    from theano.sandbox.cuda import device_properties
    import cuda_ndarray
except ImportError:
    pass

from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.sandbox.cuda.elemwise import NaiveAlgo


_logger_name = 'theano.sandbox.cuda.basic_ops'
_logger = logging.getLogger(_logger_name)


def as_cuda_ndarray_variable(x):
    if getattr(x, 'owner', None):
        if isinstance(x.owner.op, HostFromGpu):
            return x.owner.inputs[0]
        elif (isinstance(x.owner.op, GpuFromHost) and
              x.owner.inputs[0].owner and
              isinstance(x.owner.inputs[0].owner.op, HostFromGpu)):
            return x.owner.inputs[0].owner.inputs[0]
    if hasattr(x, '_as_CudaNdarrayVariable'):
        return x._as_CudaNdarrayVariable()
    tensor_x = tensor.as_tensor_variable(x)
    return gpu_from_host(tensor_x)


def as_cuda_array(obj):
    if isinstance(obj, numpy.ndarray):
        return cuda_ndarray.cuda_ndarray.CudaNdarray(obj)
    elif isinstance(obj, cuda_ndarray.cuda_ndarray.CudaNdarray):
        return obj
    else:
        raise TypeError("Don't know how to cast to a CudaNdarray object")


class HostFromGpu(GpuOp):
    """
    Implement the transfer from gpu to the cpu.

    """

    check_input = False

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'HostFromGpu'

    def make_node(self, x):
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError("Expected a Theano variable with type "
                            "CudaNdarrayType. Got %s with type %s" % (x,
                                                                      x.type))
        return Apply(self, [x], [tensor.TensorType(dtype=x.dtype,
                                    broadcastable=x.broadcastable)()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = numpy.asarray(x)

    def grad(self, inputs, grads):
        gz, = grads
        return [gpu_from_host(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        return [self(ev)]

    def infer_shape(self, node, xshp):
        return xshp

    def c_code(self, node, name, inputs, outputs, sub):
        inp = inputs[0]
        out = outputs[0]
        fail = sub['fail']
        return """
        Py_XDECREF(%(out)s);
        %(out)s = (PyArrayObject *) CudaNdarray_CreateArrayObj(%(inp)s);
        if(!%(out)s){
            %(fail)s;
        }
        """ % locals()

    def c_code_cache_version(self):
        return (3,)
host_from_gpu = HostFromGpu()


class GpuFromHost(GpuOp):
    """
    Implement the transfer from cpu to the gpu.

    """

    check_input = False

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return 'GpuFromHost'

    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError("Expected a Theano variable with type "
                            "TensorType. Got %s with type %s" % (x,
                                                                 x.type))
        return Apply(self, [x], [CudaNdarrayType(broadcastable=x.broadcastable,
                                                 dtype=x.dtype)()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = type_support_filter(theano._asarray(x, dtype='float32'),
                                   tuple([0] * x.ndim), 0, z[0])

    def grad(self, inputs, grads):
        gz, = grads
        gz = as_cuda_ndarray_variable(gz)
        return [host_from_gpu(gz)]

    def R_op(self, inputs, eval_points):
        ev, = eval_points
        return [self(ev)]

    def infer_shape(self, node, xshp):
        return xshp

    def c_code(self, node, name, inputs, outputs, sub):
        inp = inputs[0]
        out = outputs[0]
        fail = sub['fail']
        return """
        int err = 0;
        Py_XDECREF(%(out)s);
        %(out)s = (CudaNdarray*) CudaNdarray_New();
        if(!%(out)s){
            %(fail)s;
        }
        err = CudaNdarray_CopyFromArray(%(out)s, %(inp)s);
        if(err){
            %(fail)s;
        }
        """ % locals()

    def c_code_cache_version(self):
        return (2,)

gpu_from_host = GpuFromHost()


class GpuElemwise(GpuOp):
    """
    Implement a generic elemwise on the gpu.

    """

    nin = property(lambda self: self.scalar_op.nin)
    nout = property(lambda self: self.scalar_op.nout)

    def __init__(self, scalar_op, inplace_pattern=None, sync=None):
        # TODO-- this looks like a bug-- either we should use the sync argument
        # or get rid of it, we shouldn't let the client think they can control
        # sync when they can't
        if inplace_pattern is None:
            inplace_pattern = {}
        sync = config.gpuelemwise.sync

        self.scalar_op = scalar_op

        self.inplace_pattern = inplace_pattern
        self.destroy_map = dict((o, [i]) for o, i in inplace_pattern.items())

        self.sync = sync

        self._rehash()

        self.src_generator = NaiveAlgo(self.scalar_op, sync=sync,
                                       inplace_pattern=self.inplace_pattern)

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        d.pop('__epydoc_asRoutine', None)
        d.pop('_hashval')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        # old objects defaulted to sync behaviour
        self.sync = d.get('sync', True)
        self._rehash()

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.scalar_op == other.scalar_op and
                self.inplace_pattern == other.inplace_pattern and
                self.sync == other.sync)

    def _rehash(self):
        items = list(self.inplace_pattern.items())
        items.sort()
        tuple_items = [k for k, v in items]
        for k, v in items:
            if isinstance(v, (tuple, list)):
                tuple_items += [tuple(v)]
            else:
                tuple_items += [v]
        tuple_items = tuple(tuple_items)
        h = (hash(type(self)) ^ hash(self.scalar_op) ^
             hash(tuple_items) ^ hash(self.sync))
        # don't change a code that has already been  computed for this object
        assert h == getattr(self, '_hashval', h)
        self._hashval = h

    def __hash__(self):
        return self._hashval

    def __str__(self):
        if self.inplace_pattern:
            items = list(self.inplace_pattern.items())
            items.sort()
            # We need to print the scalar_op, not only the its class name
            # to have the full definition of composite op.
            return "GpuElemwise{%s}%s" % (self.scalar_op, str(items))
        return "GpuElemwise{%s,no_inplace}" % (self.scalar_op)

    def __repr__(self):
        return self.__str__()

    def make_node(self, *inputs):
        _inputs = [as_cuda_ndarray_variable(i) for i in inputs]
        if self.nin > 0 and len(_inputs) != self.nin:
            raise TypeError('Wrong argument count', (self.nin, len(_inputs)))

        target_length = max([input.type.ndim for input in _inputs])

        args = []
        for input in _inputs:
            length = input.type.ndim
            difference = target_length - length
            if not difference:
                args.append(input)
            else:
                # TODO: use LComplete instead
                args.append(GpuDimShuffle(
                    input.type.broadcastable,
                    ['x'] * difference + list(range(length))
                    )(input))
        _inputs = args

        # output is broadcastable only along dimensions where all
        # inputs are broadcastable
        broadcastable = []
        for d in xrange(_inputs[0].type.ndim):
            bcast_d = True
            for i in _inputs:
                if not i.type.broadcastable[d]:
                    bcast_d = False
                    break
            broadcastable.append(bcast_d)
        assert len(broadcastable) == _inputs[0].type.ndim

        otype = CudaNdarrayType(broadcastable=broadcastable)
        assert self.nout > 0
        return Apply(self, _inputs, [otype() for o in xrange(self.nout)])

    def c_support_code(self, *args, **kwargs):
        return self.src_generator.c_support_code(*args, **kwargs)

    def c_support_code_apply(self, *args, **kwargs):
        return self.src_generator.c_support_code_apply(*args, **kwargs)

    def c_code(self, *args, **kwargs):
        return self.src_generator.c_code(*args, **kwargs)

    def c_compile_args(self):
        # TODO: compile ptx file without constraint and then use the number of
        # registers required to inform the maximum number of threads per block.
        return ["--maxrregcount=32"]

    def c_code_cache_version(self):
        return self.src_generator.cache_version


class GpuDimShuffle(GpuOp):
    """
    Implement DimShuffle on the gpu.

    """

    check_broadcast = False

    def __init__(self, input_broadcastable, new_order):
        input_broadcastable = tuple(input_broadcastable)
        self.input_broadcastable = input_broadcastable
        self.new_order = tuple(new_order)

        for i, b in enumerate(input_broadcastable):
            if i not in new_order:
                if not b:
                    # we cannot drop non-broadcastable dimensions
                    raise ValueError("You cannot drop a non-broadcastable"
                                     " dimension.",
                                     (input_broadcastable, new_order))

        # this is the list of the original dimensions that we keep
        self.shuffle = [x for x in new_order if x != 'x']

        # list of dimensions of the output that are broadcastable and were not
        # in the original input
        self.augment = [i for i, x in enumerate(new_order) if x == 'x']

        self.view_map = {0: [0]}

        self._rehash()

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_hashval']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._rehash()

    def make_node(self, input):
        ib = tuple(input.type.broadcastable)
        if not ib == self.input_broadcastable:
            if len(ib) != len(self.input_broadcastable):
                raise TypeError((
                    "The number of dimensions of the "
                    "input is incorrect for this op. Expected %s, got %s."
                    % (self.input_broadcastable, ib)))
            for expected, b in zip(self.input_broadcastable, ib):
                if expected is True and b is False:
                    raise TypeError((
                        "The broadcastable pattern of the "
                        "input is incorrect for this op. Expected %s, got %s."
                        % (self.input_broadcastable, ib)))
                # else, expected == b or expected is False and b is True
                # Both case are good.
        ob = []
        if not isinstance(input.type, CudaNdarrayType):
            input = as_cuda_ndarray_variable(input)
        for value in self.new_order:
            if value == 'x':
                ob.append(True)
            else:
                ob.append(ib[value])
        return Apply(self, [input], [CudaNdarrayType(broadcastable=ob)()])

    def __eq__(self, other):
        # it's probably not necessary to compare input_broadcastable
        return type(self) == type(other) \
            and self.new_order == other.new_order \
            and self.input_broadcastable == other.input_broadcastable

    def _rehash(self):
        self._hashval = (hash(type(self).__name__) ^
                         hash(type(self).__module__) ^
                         hash(self.new_order) ^
                         hash(self.input_broadcastable))

    def __hash__(self):
        return self._hashval

    def __str__(self):
        return "GpuDimShuffle{%s}" % ",".join(str(x) for x in self.new_order)

    def c_code(self, node, name, inp, out, sub):
        input, = inp
        res, = out
        basename = input + '__view_or_copy'

        nd_in = len(self.input_broadcastable)
        nd_out = len(self.new_order)
        sio = StringIO()
        fail = sub['fail']

        # check input
        print("""
        if (%(input)s->nd != %(nd_in)s)
        {
            PyErr_Format(PyExc_TypeError,
                         "required nd=%(nd_in)s, got nd=%%i", %(input)s->nd);
            %(fail)s;
        }
        """ % locals(), file=sio)

        # alloc an output
        print("""
        if (%(res)s && (%(res)s->nd == %(nd_out)s))
        {
            //re-use previously-allocated cnda
        }
        else
        {
            if (%(res)s)
            {
                if (CudaNdarray_set_nd(%(res)s, %(nd_out)s))
                {
                    Py_DECREF(%(res)s);
                    %(res)s = NULL;
                    %(fail)s;
                }
            }
            else
            {
                %(res)s = (CudaNdarray*) CudaNdarray_New(%(nd_out)s);
                if (NULL == %(res)s)
                {
                    %(fail)s;
                }
            }
        }
        """ % locals(), file=sio)

        print("""
        if (CudaNdarray_set_device_data(%(res)s,
                                        CudaNdarray_DEV_DATA(%(input)s),
                                        %(input)s))
        {
            // err message set
            Py_DECREF(%(res)s);
            %(res)s = NULL;
            %(fail)s;
        }
        """ % locals(), file=sio)

        # reassign the dimension and strides in the host pointers
        for i, o in enumerate(self.new_order):
            if o == 'x':
                # TODO: remove this assertion
                #      the correct thing to do is to insert a run-time check
                #      that the size in this dimension is 1
                assert node.outputs[0].type.broadcastable[i]
                print("""
        CudaNdarray_set_dim(%(res)s, %(i)s, 1);
        CudaNdarray_set_stride(%(res)s, %(i)s, 0);
                """ % locals(), file=sio)
            else:
                print("""
        CudaNdarray_set_dim(%(res)s, %(i)s,
                            CudaNdarray_HOST_DIMS(%(input)s)[%(o)s]);
        CudaNdarray_set_stride(%(res)s, %(i)s,
                               CudaNdarray_HOST_STRIDES(%(input)s)[%(o)s]);
                """ % locals(), file=sio)

        for i, o in enumerate(self.new_order):
            print("""
    //std::cerr << "GpuDimShuffle " << %(res)s << " str[%(i)s] = " << %(res)s->str[%(i)s] << "\\n";
            """ % locals(), file=sio)

        # copy the host dims and stride -> device
        if 0:
            print("""
            if (CudaNdarray_copy_structure_to_device(%(res)s))
            {
                //err msg set
                Py_DECREF(%(res)s);
                %(res)s = NULL;
                %(fail)s;
            }
            """ % locals(), file=sio)

        if 0:  # print full code to stdout
            print('--------------------------------------')
            print('C_CODE')
            print('')
            print(self)
            print("IN BROAD", self.input_broadcastable)
            print("NEW ORDER", self.new_order)
            print('------------')
            print('')
            print(sio.getvalue())
            print('--------------------------------------')
            if 0:
                sys.exit()

        return sio.getvalue()

    def c_code_cache_version(self):
        return (1, 0)

    def infer_shape(self, node, shapes):
        ishp, = shapes
        # transpose
        rval = [ishp[i] for i in self.shuffle]

        # augment
        for augm in self.augment:
            rval.insert(augm, 1)
        return [rval]


class GpuCAReduce(GpuOp):
    """
    GpuCAReduce is a Reduction along some dimensions by a scalar op.

    The dimensions along which to reduce is specified by the
    `reduce_mask` that you pass to the constructor.  The `reduce_mask`
    is a tuple of booleans (actually integers 0 or 1) that specify for
    each input dimension, whether to reduce it (1) or not (0).

    Parameters
    ----------
    pre_scalar_op 
        If present, must be a scalar op with only 1 input.
        We will execute it on the input value before reduction.
    
    Notes
    -----
    This Op is a work in progress.

    This op was recently upgraded from just GpuSum a general CAReduce. Not
    many code cases are supported for scalar_op being anything other than
    scal. Add instances yet.

    Important note: if you implement new cases for this op, be sure to
    benchmark them and make sure that they actually result in a speedup.
    GPUs are not especially well-suited to reduction operations so it is
    quite possible that the GPU might be slower for some cases.

    Examples
    --------
    When scalar_op is a theano.scalar.basic.Add instance:

    - reduce_mask == (1,) sums a vector to a scalar

    - reduce_mask == (1,0) computes the sum of each column in a matrix

    - reduce_mask == (0,1) computes the sum of each row in a matrix

    - reduce_mask == (1,1,1) computes the sum of all elements in a 3-tensor.

    ..note:: Any reduce_mask of all zeros is a sort of 'copy', and may
           be removed during graph optimization.

    """

    def __init__(self, reduce_mask, scalar_op, pre_scalar_op=None):
        self.reduce_mask = tuple(reduce_mask)
        self.scalar_op = scalar_op
        # used to make sure that calls to scalar op
        # have unique name arguments
        self._n_scalar_op_calls = 0
        self.pre_scalar_op = pre_scalar_op
        if pre_scalar_op:
            assert pre_scalar_op.nin == 1

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.reduce_mask == other.reduce_mask and
                self.scalar_op == other.scalar_op and
                self.pre_scalar_op == other.pre_scalar_op)

    def __hash__(self):
        return (hash(type(self)) ^
                hash(self.reduce_mask) ^
                hash(type(self.scalar_op)) ^
                hash(type(self.pre_scalar_op)))

    def __str__(self):
        pre = ""
        if self.pre_scalar_op:
            pre = "pre=%s,red=" % str(self.pre_scalar_op)
        return "GpuCAReduce{%s%s}{%s}" % (
                pre,
                str(self.scalar_op),
                ','.join(str(i) for i in self.reduce_mask)
                )

    def __setstate__(self, d):
        self.__dict__.update(d)
        # For unpickling of old ops.
        if not hasattr(self, "pre_scalar_op"):
            self.pre_scalar_op = None

    def make_node(self, x):
        x = as_cuda_ndarray_variable(x)
        if (x.type.ndim != len(self.reduce_mask)):
            raise TypeError("x must have rank %i" % len(self.reduce_mask))
        o_broadcast = [x.type.broadcastable[i] for i
                       in xrange(x.type.ndim) if not self.reduce_mask[i]]
        return Apply(self, [x], [CudaNdarrayType(o_broadcast)()])

    """
    This method must be commented, because there's no way
    to communicate that it's OK to call for + but not for
    max
    def perform(self, node, inp, out):
        x, = inp
        z, = out
        # reduce_max is declared but does nothing but
        # raise NotImplementedError.
        # We can't call it here anyway because it hasn't
        # been added to the python bindings yet
        z[0] = x.reduce_sum(self.reduce_mask)
    """

    def supports_c_code(self, inputs):
        """
        Returns True if the current op and reduce pattern has functioning C
        code.

        """

        # If we don't even have the right method, we certainly
        # don't support the C code
        # (This is the test that used to be implemented by
        # local_gpu_sum)
        pattern = (''.join(str(i) for i in self.reduce_mask))
        if not hasattr(self, 'c_code_reduce_%s' % pattern):
            return False

        # Now that this is a general reduction op, we might
        # have a method for a pattern, but that pattern
        # might not be implemented for the current scalar op.
        # To detect this more complicated situation, we
        # make fake arguments to c_code, try to run them,
        # and see if NotImplementedError gets raised.

        node = self.make_node(*inputs)

        name = 'fake_name'

        inp = ['fake_input_name_%d' % i for i in xrange(len(inputs))]
        out = ['fake_output_name_%d' % i for i in xrange(len(node.outputs))]

        sub = {'fail': 'fake failure code'}

        try:
            self.c_code(node, name, inp, out, sub)
            self.c_support_code_apply(node, name)
        except NotImplementedError:
            return False
        return True

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        nd_in = node.inputs[0].type.ndim
        nd_out = node.outputs[0].type.ndim

        assert nd_in - nd_out == sum(self.reduce_mask)

        sio = StringIO()
        fail = sub['fail']

        # check input
        print("""
        if (%(x)s->nd != %(nd_in)s)
        {
            PyErr_Format(PyExc_TypeError,
                         "required nd=%(nd_in)s, got nd=%%i", %(x)s->nd);
            %(fail)s;
        }
        """ % locals(), file=sio)

        # It might be nice to use a property of the op class to do this,
        # but tensor.elemwise.CAReduce has this exact same check so I guess
        # this is OK to do
        if self.scalar_op in [scal.minimum, scal.maximum]:
            conds = ["(CudaNdarray_HOST_DIMS(%s)[%d] == 0)" % (x, i)
                     for i in xrange(nd_in)
                     if self.reduce_mask[i]]
            assert len(conds) > 0
            cond = "(" + " || ".join(conds) + ")"
            print("""
            if %(cond)s
            {
                PyErr_Format(PyExc_ValueError," tried to reduce a 0-length axis.");
                %(fail)s;
            }
            """ % locals(), file=sio)

        #
        # alloc an output if we need one
        #

        # check the basics of out output
        print("""
        if (  !%(z)s
           || (%(z)s->nd != %(nd_out)s)
        """ % locals(), file=sio)

        # ensure that the output has the right non-reduced dimensions
        j = 0
        for i in xrange(nd_in):
            if not self.reduce_mask[i]:
                print(" || (CudaNdarray_HOST_DIMS(%(z)s)[%(j)s] != CudaNdarray_HOST_DIMS(%(x)s)[%(i)d]) " % locals(), file=sio)
                j += 1

        print("""
           )
        {
            """ % locals(), file=sio)
        if nd_out > 0:
            print("int new_dims[%(nd_out)s]; " % locals(), file=sio)
        else:
            print("int *new_dims=NULL; ", file=sio)

        j = 0
        for i in xrange(nd_in):
            if not self.reduce_mask[i]:
                print('new_dims[%(j)s] = CudaNdarray_HOST_DIMS(%(x)s)[%(i)s];' % locals(), file=sio)
                j += 1

        print("""
            Py_XDECREF(%(z)s);
            %(z)s = (CudaNdarray*) CudaNdarray_NewDims(%(nd_out)s, new_dims);
            if (NULL == %(z)s)
            {
                %(fail)s;
            }
        }
        """ % locals(), file=sio)

        # \begin bracket the reduction in a check that there is
        # actually work to do
        if getattr(self.scalar_op, 'identity', None) == 0:
            zero_shp = "cudaMemset(%(z)s->devdata, 0, CudaNdarray_SIZE(%(z)s) * sizeof(float))" % locals()
        # TODO: elif getattr(self.scalar_op, 'identity', None) == 1:
        else:
            zero_shp = """
            PyErr_Format(PyExc_NotImplementedError,
                         "GpuCAReduce not implemented when input shape is 0 for this scalar_op");
            %(fail)s;
            """ % locals()
        print("""
        if (CudaNdarray_SIZE(%(z)s) && ! CudaNdarray_SIZE(%(x)s)){
            %(zero_shp)s;
        }
        else if (CudaNdarray_SIZE(%(z)s))
        {
        """ % locals(), file=sio)

        #
        # Now perform the reduction
        #

        if all(i == 1 for i in self.reduce_mask):
            # check if the tensor is ccontiguous, if true, use the c_code_reduce_ccontig code.
            # TODO: check if we are ccontiguous when we un-dimshuffle
            # TODO: if only some dims are ccontiguous, call version with less dims.
            print('if(CudaNdarray_is_c_contiguous(%(x)s)){'%locals(), file=sio)
            self.c_code_reduce_ccontig(sio, node, name, x, z, fail)
            print("}else{", file=sio)
            getattr(self, 'c_code_reduce_%s'%(''.join(
                str(i) for i in self.reduce_mask)))(sio, node, name, x, z, fail)
            print("}", file=sio)
        else:
            getattr(self, 'c_code_reduce_%s'%(''.join(
                str(i) for i in self.reduce_mask)))(sio, node, name, x, z, fail)

        # \end bracket the reduction ...
        print("""
        }
        """ % locals(), file=sio)

        return sio.getvalue()

    def _makecall(self, node, name, x, z, fail, pattern=None):
        """
        Return a string for making a kernel call.

        The return value looks something like:

            .. code-block:: c

                if (verbose)
                    printf("running kernel_reduce_10_%(name)s\\n");
                int n_shared = sizeof(float) * n_threads.x * n_threads.y * n_threads.z;
                kernel_reduce_10_%(name)s<<<n_blocks, n_threads,
                                                n_shared>>>(
                        CudaNdarray_HOST_DIMS(%(x)s)[0],
                        CudaNdarray_HOST_DIMS(%(x)s)[1],
                        CudaNdarray_DEV_DATA(%(x)s),
                        CudaNdarray_HOST_STRIDES(%(x)s)[0],
                        CudaNdarray_HOST_STRIDES(%(x)s)[1],
                        CudaNdarray_DEV_DATA(%(z)s),
                        CudaNdarray_HOST_STRIDES(%(z)s)[0]
                        );
                CNDA_THREAD_SYNC;
                if (cudaSuccess != cudaGetLastError())
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: ... );
                    %(fail)s;
                }

        """
        sio = StringIO()
        if pattern is None:
            pattern = ''.join(str(c) for c in self.reduce_mask)
        ndim = len(self.reduce_mask)
        nd_out = ndim - sum(self.reduce_mask)
        shapes_format = "shape=(%s)" % ",".join(["%d"] * node.inputs[0].ndim)
        shapes_data = ",".join("CudaNdarray_HOST_DIMS(%s)[%d]" % (x, i)
                               for i in xrange(node.inputs[0].ndim))

        print("""
            if (verbose)
                printf("running kernel_reduce_%(pattern)s_%(name)s\\n");
            int n_shared = sizeof(float) * n_threads.x * n_threads.y * n_threads.z;
            if (verbose>1)
                printf("n_threads.x=%%d, n_threads.y=%%d, n_threads.z=%%d,"
                       " nb_threads=%%d, n_blocks.x=%%d, n_blocks.y=%%d,"
                       " nb_block=%%d, n_shared=%%d, %(shapes_format)s\\n",
                                  n_threads.x,n_threads.y,n_threads.z,
                                  n_threads.x*n_threads.y*n_threads.z,
                                  n_blocks.x,n_blocks.y,
                                  n_blocks.x*n_blocks.y, n_shared, %(shapes_data)s);
            kernel_reduce_%(pattern)s_%(name)s<<<n_blocks, n_threads, n_shared>>>(
            """ % locals(), file=sio)
        for i in xrange(ndim):
            print("""
                    CudaNdarray_HOST_DIMS(%(x)s)[%(i)s],
            """ % locals(), file=sio)
        print("""
                    CudaNdarray_DEV_DATA(%(x)s)
            """ % locals(), file=sio)
        for i in xrange(ndim):
            print("""
                    ,CudaNdarray_HOST_STRIDES(%(x)s)[%(i)s]
            """ % locals(), file=sio)
        print("""
                    ,CudaNdarray_DEV_DATA(%(z)s)
            """ % locals(), file=sio)
        for i in xrange(nd_out):
            print("""
                    ,CudaNdarray_HOST_STRIDES(%(z)s)[%(i)s]
            """ % locals(), file=sio)

        print("""
                    );
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: %%s: %%s."
                    " (grid: %%i x %%i; block: %%i x %%i x %%i)"
                    " %(shapes_format)s \\n",
                    "kernel_reduce_%(pattern)s_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z,
                    %(shapes_data)s);
                %(fail)s;
            }
        """ % locals(), file=sio)
        return sio.getvalue()

    def _k_decl(self, node, nodename, pattern=None,
                ndim=None, reduce_mask=None):
        """
        Return a string to declare a kernel function.

        The result will look something like this:

        .. code-block:: c

            static __global__ void kernel_reduce_110_%(nodename)s(
                    const int d0,
                    const int d1,
                    const int d2,
                    const float *A,
                    const int sA0,
                    const int sA1,
                    const int sA2,
                    float * Z,
                    const int sZ0)

            Since the nodename is unique, we don't need to put the name
            of the scalar_op in here.

        """
        if reduce_mask is None:
            reduce_mask = self.reduce_mask
        if ndim is None:
            ndim = len(reduce_mask)
        if pattern is None:
            pattern = ''.join(str(i) for i in reduce_mask)
        sio = StringIO()

        print("""
            static __global__ void kernel_reduce_%(pattern)s_%(nodename)s(
        """ % locals(), file=sio)
        for i in xrange(ndim):
            print("""
                    const int d%(i)s,
        """ % locals(), file=sio)
        print("""
                    const float *A,
        """ % locals(), file=sio)
        for i in xrange(ndim):
            print("""
                    const int sA%(i)s,
        """ % locals(), file=sio)
        print("""
                    float * Z
        """ % locals(), file=sio)
        for i in xrange(ndim - sum(reduce_mask)):
            print("""
                    , const int sZ%(i)s
        """ % locals(), file=sio)
        print(")", file=sio)
        return sio.getvalue()

    def _k_init(self, *args):
        return """
                const int threadCount = blockDim.x * blockDim.y * blockDim.z;
                const int threadNum = threadIdx.z * blockDim.x * blockDim.y
                + threadIdx.y * blockDim.x + threadIdx.x;
                extern __shared__ float buf[];
                float myresult = 0.0f;

                //This is caught in cuda/init.py when we init the gpu. I keep
                //it here to ease finding code that rely on this.
                if (warpSize != 32)
                {
                    Z[0] = -666;
                    return;
                }

        """

    def _assign_init(self, first_item):
        """
        This return the initial value for myresult.
        If the scalar op have an identity value, return it.

        Otherwise, check that the scalar op is maximum or minimum
        and return first_item. It should be the first element of the reduction.
        As the maximum and minimum of the same value don't change, this work.

        """
        if hasattr(self.scalar_op, 'identity'):
            return str(self.scalar_op.identity)
        else:
            assert isinstance(self.scalar_op, (scal.Maximum,
                                               scal.Minimum))
            if self.pre_scalar_op:
                #dtype = node.inputs[0].dtype
                dtype = 'float32'

                dummy_var = scal.Scalar(dtype=dtype)()

                dummy_node = self.pre_scalar_op.make_node(dummy_var)

                dummy_name = 'assign_init_pre_scalar_op' + str(self._n_scalar_op_calls)
                self._n_scalar_op_calls += 1
                t = self.pre_scalar_op.c_code(dummy_node, dummy_name,
                                              (first_item,), ("",), {})
                assert t.startswith(' = ')
                first_item = t[3:]
                if first_item[-1] == ';':
                    first_item = first_item[:-1]

            return first_item

    def _assign_reduce(self, node, name, left, right, sub, pre):
        """
        Parameters
        ----------
        node
            The node argument to this op's c_code.
        name
            The name argument to this op's c_code.
        left
            A C code string identifying an lvalue.
        right
            A C code string identifying an expression.
        sub
            The sub argument to this op's c_code.
        pre
            If True, we will add the pre_scalar_op.c_code.

        Returns
        -------
        str
            C code to reduce left and right, assigning the result to left.

        """
        x, = node.inputs

        dtype = x.dtype

        dummy_left = scal.Scalar(dtype=dtype)()
        dummy_right = scal.Scalar(dtype=dtype)()

        dummy_node = self.scalar_op.make_node(dummy_left, dummy_right)

        dummy_name = name + '_scalar_op' + str(self._n_scalar_op_calls)
        self._n_scalar_op_calls += 1
        if pre and self.pre_scalar_op:
            assert left == "myresult"
            dummy_node = self.pre_scalar_op.make_node(dummy_left)
            dummy_name = name + '_scalar_op' + str(self._n_scalar_op_calls)
            self._n_scalar_op_calls += 1
            t = self.pre_scalar_op.c_code(dummy_node, dummy_name,
                                          (right,), ("",), sub)
            assert t.startswith(' = ')
            right = t[3:]
            if right[-1] == ';':
                right = right[:-1]
        return self.scalar_op.c_code(dummy_node, dummy_name, (left, right),
                                     (left,), sub)

    def _k_reduce_buf(self, z_pos, node, name, sub):
        """
        WRITEME

        Parameters
        ----------
        node, name, sub: these should be passed through from the original
        call to c_code

        """

        # This code (the code in new_version) is currently ignored.
        # Code produced later in this function is returned instead.
        # The code here works with all nvidia driver
        # But only for powers or multiples of 2!
        new_version = """
        __syncthreads(); // some kernel do multiple reduction.
        buf[threadNum] = myresult;
        __syncthreads();


        if (threadNum >= ((threadCount >> 1) * 2))
        {
            int idx = threadNum - (threadCount >> 1) * 2;"""

        new_version += self._assign_reduce(node, name, 'buf[idx]',
                                           'buf[threadNum]', sub, False)

        new_version += """
        }
        __syncthreads();

        // Works for power of 2 only.
        int nTotalThreads = threadCount; // Total number of active threads
        while(nTotalThreads > 1)
        {
            int halfPoint = (nTotalThreads >> 1);        // divide by two
            // only the first half of the threads will be active.

            if (threadNum < halfPoint)
            {
              // Get the shared value stored by another thread
              float temp = buf[threadNum + halfPoint];
              """

        new_version += self._assign_reduce(node, name, 'buf[threadNum]',
                                           'temp', sub, False)

        new_version += """
            }
            __syncthreads();

            nTotalThreads = (nTotalThreads >> 1);        // divide by two.
        }
            __syncthreads();

        if (threadNum == 0)
        {
            %(z_pos)s = buf[0];
        }
            __syncthreads();"""

        new_version = new_version % locals()

        current_version = """
        __syncthreads(); // some kernel do multiple reduction.
        buf[threadNum] = myresult;
        __syncthreads();

        // rest of function is handled by one warp
        if (threadNum < warpSize)
        {
            //round up all the partial sums into the first `warpSize` elements
            for (int i = threadNum + warpSize; i < threadCount; i += warpSize)
            {
                """
        current_version += self._assign_reduce(node, name, 'myresult',
                                               'buf[i]', sub, False) + """
            }
            buf[threadNum] = myresult;
        /*Comment this optimization as it don't work on Fermi GPU.
        TODO: find why it don't work or put the GPU compute capability into the version
            // no sync because only one warp is running
            if(threadCount >32)
            {"""
        for num in [16, 8, 4, 2, 1]:
            current_version += self._assign_reduce(node, name,
                                                   'buf[threadNum]',
                                                   'buf[threadNum+%d]' % num,
                                                   sub, False)
        current_version += """
                if (threadNum == 0)
                {
                    %(z_pos)s = buf[0];
                }

            }
            else */
            if (threadNum < 16)
            {
                //reduce so that threadNum 0 has the reduction of everything
                """
        for num in [16, 8, 4, 2, 1]:
            this_if = "if (threadNum + %d < threadCount) " % num + \
                self._assign_reduce(node, name,
                                    'buf[threadNum]', 'buf[threadNum+%d]' % num,
                                    sub, False)
            current_version += this_if
        current_version += """
                if (threadNum == 0)
                {
                    %(z_pos)s = buf[0];
                }
            }
        }
        """

        current_version = current_version % locals()

        return current_version

    # Threads must be organized as: threadNum%nb_reduce correspond to the same sum
    # nb_reduce<=warpSize
    def _k_reduce_buf_multiple(self, z_pos, node, name, nb_reduce):
        reduce_fct = self._assign_reduce(node, name, 'myresult',
                                         'buf[i]', {}, True)
        return """
        __syncthreads(); // some kernel do multiple reduction.
        buf[threadNum] = myresult;
        __syncthreads();

        // rest of function is handled by one warp
        if (threadNum < %(nb_reduce)s)
        {
            //round up all the partial sums into the first `nb_reduce` elements
            for (int i = threadNum + %(nb_reduce)s; i < threadCount; i += %(nb_reduce)s)
            {
                %(reduce_fct)s;
            }
            %(z_pos)s = myresult;
        }
        """ % locals()

    def c_code_reduce_ccontig(self, sio, node, name, x, z, fail):
        """
        WRITEME

        IG: I believe, based on how this is called in c_code, that it
        is for the case where we are reducing on all axes and x is
        C contiguous.

        """
        if getattr(self.scalar_op, 'identity', None) == 0:
            zero_shp = "cudaMemset(%(z)s->devdata, 0, CudaNdarray_SIZE(%(z)s) * sizeof(float))" % locals()
        # TODO: elif getattr(self.scalar_op, 'identity', None) == 1:
        else:
            zero_shp = """
            PyErr_Format(PyExc_NotImplementedError,
                         "GpuCAReduce not implemented when input shape is 0 for this scalar_op");
            %(fail)s;
            """ % locals()

        print("""
        {
          if(CudaNdarray_SIZE(%(x)s)==0){
            %(zero_shp)s;
          }else{
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_SIZE(%(x)s),
                             (size_t) NUM_VECTOR_OP_THREADS_PER_BLOCK));
            dim3 n_blocks(1);
            if (verbose) printf("running kernel_reduce_ccontig_%(name)s"
                                " n_threads.x=%%d, size=%%d, ndim=%%d\\n",
                                n_threads.x,CudaNdarray_SIZE(%(x)s),%(x)s->nd);
            int n_shared = sizeof(float) * n_threads.x;
            kernel_reduce_ccontig_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                    CudaNdarray_SIZE(%(x)s),
                    CudaNdarray_DEV_DATA(%(x)s),
                    CudaNdarray_DEV_DATA(%(z)s));
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Cuda error: %%s: %%s."
                             " (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kernel_reduce_ccontig_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                %(fail)s;
            }
         }
        }
        """ % locals(), file=sio)

    def c_code_reduce_1(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            dim3 n_blocks(1);
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_11(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[1],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            while (n_threads.y * n_threads.x <= NUM_VECTOR_OP_THREADS_PER_BLOCK) ++n_threads.y;
            n_threads.y -= 1;
            if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[0])
                n_threads.y = CudaNdarray_HOST_DIMS(%(x)s)[0];

            dim3 n_blocks(1);
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_01X(self, sio, node, name, x, z, fail, N):
        """
        
        Parameters
        ----------
        N : int
            The number of 1 in the pattern
            N=1 -> 01, N=2 -> 011 N=3 ->0111
            Works for N=1,2,3.

        """

        assert N in [1, 2, 3]
        makecall = self._makecall(node, name, x, z, fail)
        N_pattern = ''.join(['1'] * N)
        param_dim = ",".join(["CudaNdarray_HOST_DIMS(%s)[%d]" % (x, i)
                              for i in xrange(N + 1)])
        strides_dim = ",".join(["CudaNdarray_HOST_STRIDES(%s)[%d]"
                                % (x, i) for i in xrange(N + 1)])

        threads_y = """
            //get as many y threads as we can fit
            while (n_threads.x * (n_threads.y+1) <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.y < CudaNdarray_HOST_DIMS(%(x)s)[%(N)s-1])
                    n_threads.y += 1;
                else
                    break;
            }""" % locals()

        threads_z = """
            //get as many z threads as we can fit
            while (n_threads.x * n_threads.y * (n_threads.z+1) <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.z < CudaNdarray_HOST_DIMS(%(x)s)[%(N)s-2])
                    n_threads.z += 1;
                else
                    break;
            }
            //Maximum for Fermi GPU on that dimensions.
            n_threads.z = std::min(n_threads.z, (unsigned)64);

        """ % locals()

        if len(self.reduce_mask) == 2:
            threads_y = ''
            threads_z = ''

        if len(self.reduce_mask) == 3:
            threads_z = ''

        print("""
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[%(N)s],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            %(threads_y)s
            %(threads_z)s
            dim3 n_blocks(std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                                   NUM_VECTOR_OP_BLOCKS));
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_01(self, sio, node, name, x, z, fail):
        self.c_code_reduce_01X(sio, node, name, x, z, fail, 1)

    def c_code_reduce_011(self, sio, node, name, x, z, fail):
        self.c_code_reduce_01X(sio, node, name, x, z, fail, 2)

    def c_code_reduce_0111(self, sio, node, name, x, z, fail):
        self.c_code_reduce_01X(sio, node, name, x, z, fail, 3)

    def c_code_reduce_10(self, sio, node, name, x, z, fail):
        print("""
    {
        int verbose = 0;
        if(CudaNdarray_HOST_STRIDES(%(x)s)[0] >
           CudaNdarray_HOST_STRIDES(%(x)s)[1]){
                // If there are a lot of summations to do, then we can use simple parallelization -
                // use each thread to do one sum.

                // we might as well launch blocks of 32 threads because that's the warp size.
                // we could schedule more threads if we were maxing out the gridsize below, but
                // the gridsize is way more than the physical hardware and I think 32 threads
                // on a huge grid is enough to fully use the hardware.
                dim3 n_threads(32,1,1);

                // We kindof reshape the input implicitly to something 4D:
                //  the shape A,B,C    ->   A, B, D, E
                //  where C <= D*E < C+32
                //  where E==32

                int A = 1;
                int B = CudaNdarray_HOST_DIMS(%(x)s)[0];
                int C = CudaNdarray_HOST_DIMS(%(x)s)[1];
                int D = C/32;
                if (32*D < C) D+= 1;
                assert ((C <= 32*D) && (32*D < C+32));

                // The gridsize would ideally be (A, D).  But we do the following logic to make
                // sure we don't ask for a grid that is too big.
                dim3 n_blocks(A,D);
                if (n_blocks.x > NUM_VECTOR_OP_BLOCKS) n_blocks.x = NUM_VECTOR_OP_BLOCKS;
                if (n_blocks.x*n_blocks.y > NUM_VECTOR_OP_BLOCKS) n_blocks.y = NUM_VECTOR_OP_BLOCKS/n_blocks.x;
                kernel_reduce_010_AD_%(name)s<<<n_blocks, n_threads>>>(
                A,B,C,D,
                        CudaNdarray_DEV_DATA(%(x)s),
                        1,
                        CudaNdarray_HOST_STRIDES(%(x)s)[0],
                        CudaNdarray_HOST_STRIDES(%(x)s)[1],
                        CudaNdarray_DEV_DATA(%(z)s),
                        1,
                        CudaNdarray_HOST_STRIDES(%(z)s)[0]
                        );

            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: %%s: %%s."
                    " (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kernel_reduce_10_AD%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                %(fail)s;
            }
        }else{
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            dim3 n_blocks(1,
                std::min(CudaNdarray_HOST_DIMS(%(x)s)[1],
                    NUM_VECTOR_OP_BLOCKS));
            if (verbose) {
              fprintf(stderr,
                "running kernel_reduce_10_%(name)s n_blocks=(%%i,%%i)\\n",
                n_blocks.x,
                n_blocks.y);
            }
            assert(CudaNdarray_HOST_DIMS(%(x)s)[1] == CudaNdarray_HOST_DIMS(%(z)s)[0]);
            int n_shared = sizeof(float) * n_threads.x;
            kernel_reduce_010_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                    1,
                    CudaNdarray_HOST_DIMS(%(x)s)[0],
                    CudaNdarray_HOST_DIMS(%(x)s)[1],
                    CudaNdarray_DEV_DATA(%(x)s),
                    1,
                    CudaNdarray_HOST_STRIDES(%(x)s)[0],
                    CudaNdarray_HOST_STRIDES(%(x)s)[1],
                    CudaNdarray_DEV_DATA(%(z)s),
                    1,
                    CudaNdarray_HOST_STRIDES(%(z)s)[0]
                    );
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: %%s: %%s."
                    " (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kernel_reduce_010_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z);
                %(fail)s;
            }
        }
    }
        """ % locals(), file=sio)

    def c_code_reduce_010(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        makecall_inner = self._makecall(node, name, x, z, fail,
                                        pattern="010_inner")
        pattern = ''.join(str(i) for i in self.reduce_mask)
        print("""
        {
            //int n_summations = CudaNdarray_HOST_DIMS(%(x)s)[0] * CudaNdarray_HOST_DIMS(%(x)s)[2];

            //if ((n_summations >= 15 * 32) && (CudaNdarray_HOST_DIMS(%(x)s)[2]>=16))
            if (1) // if the alternative is less buggy, consider not using this branch
            {
                // If there are a lot of summations to do, then we can use simple parallelization -
                // use each thread to do one sum.

                // we might as well launch blocks of 32 threads because that's the warp size.
                // we could schedule more threads if we were maxing out the gridsize below, but
                // the gridsize is way more than the physical hardware and I think 32 threads
                // on a huge grid is enough to fully use the hardware.
                dim3 n_threads(32,1,1);

                // We kindof reshape the input implicitly to something 4D:
                //  the shape A,B,C    ->   A, B, D, E
                //  where C <= D*E < C+32
                //  where E==32

                int A = CudaNdarray_HOST_DIMS(%(x)s)[0];
                int B = CudaNdarray_HOST_DIMS(%(x)s)[1];
                int C = CudaNdarray_HOST_DIMS(%(x)s)[2];
                int D = C/32;
                if (32*D < C) D+= 1;
                assert ((C <= 32*D) && (32*D < C+32));

                // The gridsize would ideally be (A, D).  But we do the following logic to make
                // sure we don't ask for a grid that is too big.
                dim3 n_blocks(A,D);
                if (n_blocks.x > NUM_VECTOR_OP_BLOCKS) n_blocks.x = NUM_VECTOR_OP_BLOCKS;
                if (n_blocks.x*n_blocks.y > NUM_VECTOR_OP_BLOCKS) n_blocks.y = NUM_VECTOR_OP_BLOCKS/n_blocks.x;
                int n_shared = 0;
                kernel_reduce_010_AD_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                A,B,C,D,
                        CudaNdarray_DEV_DATA(%(x)s),
                        CudaNdarray_HOST_STRIDES(%(x)s)[0],
                        CudaNdarray_HOST_STRIDES(%(x)s)[1],
                        CudaNdarray_HOST_STRIDES(%(x)s)[2],
                        CudaNdarray_DEV_DATA(%(z)s),
                        CudaNdarray_HOST_STRIDES(%(z)s)[0],
                        CudaNdarray_HOST_STRIDES(%(z)s)[1]
                        );
                CNDA_THREAD_SYNC;
                cudaError_t sts = cudaGetLastError();
                if (cudaSuccess != sts)
                {
                    PyErr_Format(PyExc_RuntimeError,
                        "Cuda error: %%s: %%s."
                        " (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                        "kernel_reduce_010_%(name)s",
                        cudaGetErrorString(sts),
                        n_blocks.x,
                        n_blocks.y,
                        n_threads.x,
                        n_threads.y,
                        n_threads.z);
                    %(fail)s;
                }
            }
            else
            {
                int verbose = 2;

                  dim3 n_threads(std::min(32,CudaNdarray_HOST_DIMS(%(x)s)[2]));
                  while(    (n_threads.x*(n_threads.y+1)<=NUM_VECTOR_OP_THREADS_PER_BLOCK)
                         && (n_threads.y<CudaNdarray_HOST_DIMS(%(x)s)[1])){
                      n_threads.y++;
                  }

                  dim3 n_blocks(std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                                (int)NUM_VECTOR_OP_BLOCKS));
                  n_blocks.y = std::min(
                      ceil_intdiv(CudaNdarray_HOST_DIMS(%(x)s)[2],(int)n_threads.x),
                      (int)(NUM_VECTOR_OP_BLOCKS / n_blocks.x)
                      );
                if(std::min(std::min(CudaNdarray_HOST_STRIDES(%(x)s)[0],
                                     CudaNdarray_HOST_STRIDES(%(x)s)[1]),
                            CudaNdarray_HOST_STRIDES(%(x)s)[2])
                   ==CudaNdarray_HOST_STRIDES(%(x)s)[2]
                  && n_blocks.y==ceil_intdiv(CudaNdarray_HOST_DIMS(%(x)s)[2],(int)n_threads.x)){
                  if(verbose>1)
                    printf("n_block.x.1=%%d, n_block.x.2=%%d, n_block.y.1=%%d, n_block.y.2=%%d,\\n",
                           CudaNdarray_HOST_DIMS(%(x)s)[0],NUM_VECTOR_OP_BLOCKS,
                           ceil_intdiv(CudaNdarray_HOST_DIMS(%(x)s)[2],(int)n_threads.x),
                                       (int)(NUM_VECTOR_OP_BLOCKS / n_blocks.x));
                  assert(n_threads.x<=32);
                  %(makecall_inner)s
                }else{
                  n_threads.x = std::min(CudaNdarray_HOST_DIMS(%(x)s)[1],
                                  (int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
                  n_blocks.x = std::min(CudaNdarray_HOST_DIMS(%(x)s)[0], (int)NUM_VECTOR_OP_BLOCKS);
                  n_blocks.y = std::min(
                      CudaNdarray_HOST_DIMS(%(x)s)[2],
                      (int)(NUM_VECTOR_OP_BLOCKS / n_blocks.x)
                      );
                  %(makecall)s
                }
                CNDA_THREAD_SYNC;
                cudaError_t sts = cudaGetLastError();
                if (cudaSuccess != sts)
                {
                    PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                        "kernel_reduce_%(pattern)s_%(name)s",
                        cudaGetErrorString(sts),
                        n_blocks.x,
                        n_blocks.y,
                        n_threads.x,
                        n_threads.y,
                        n_threads.z);
                    %(fail)s;
                }
            }
        }
        """ % locals(), file=sio)

    def c_code_reduce_0101(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[3],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            while (n_threads.x * n_threads.y <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[1]) break;
                n_threads.y += 1;
            }
            n_threads.y -= 1;
            dim3 n_blocks(CudaNdarray_HOST_DIMS(%(x)s)[0], CudaNdarray_HOST_DIMS(%(x)s)[2]);
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_100(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        # use threadIdx.x for i0
        # use blockIdx.x for i1
        # use blockIdx.y for i2
        print("""
        {
            int verbose = 0;
            if (CudaNdarray_HOST_STRIDES(%(x)s)[2] != 1){
                dim3 n_threads(
                        std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                                NUM_VECTOR_OP_THREADS_PER_BLOCK));
                dim3 n_blocks(std::min(CudaNdarray_HOST_DIMS(%(x)s)[1],
                              NUM_VECTOR_OP_BLOCKS));
                while (n_blocks.x * (n_blocks.y+1) <= NUM_VECTOR_OP_BLOCKS &&
                       n_blocks.y <= CudaNdarray_HOST_DIMS(%(x)s)[2])
                {
                    n_blocks.y += 1;
                }
                %(makecall)s
            }
            else
            {   // reuse 010_AD kernel, we transpose the 2 first dim
                // See the reduction for the real 010_AD kernel for
                // explanation. We do this to get coalesced read.
                dim3 n_threads(32,1,1);

                int A = CudaNdarray_HOST_DIMS(%(x)s)[1];
                int B = CudaNdarray_HOST_DIMS(%(x)s)[0];
                int C = CudaNdarray_HOST_DIMS(%(x)s)[2];
                int D = C/32;
                if (32*D < C) D+= 1;
                assert ((C <= 32*D) && (32*D < C+32));

                dim3 n_blocks(A,D);
                if (n_blocks.x > NUM_VECTOR_OP_BLOCKS)
                    n_blocks.x = NUM_VECTOR_OP_BLOCKS;
                if (n_blocks.x*n_blocks.y > NUM_VECTOR_OP_BLOCKS)
                    n_blocks.y = NUM_VECTOR_OP_BLOCKS/n_blocks.x;
                int n_shared = 0;
                kernel_reduce_010_AD_%(name)s<<<n_blocks, n_threads, n_shared>>>(
                A,B,C,D,
                        CudaNdarray_DEV_DATA(%(x)s),
                        CudaNdarray_HOST_STRIDES(%(x)s)[1],
                        CudaNdarray_HOST_STRIDES(%(x)s)[0],
                        CudaNdarray_HOST_STRIDES(%(x)s)[2],
                        CudaNdarray_DEV_DATA(%(z)s),
                        CudaNdarray_HOST_STRIDES(%(z)s)[0],
                        CudaNdarray_HOST_STRIDES(%(z)s)[1]
                        );
                CNDA_THREAD_SYNC;
                cudaError_t sts = cudaGetLastError();
                if (cudaSuccess != sts)
                {
                    PyErr_Format(PyExc_RuntimeError,
                        "Cuda error: %%s: %%s."
                        " (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                        "kernel_reduce_010_%(name)s",
                        cudaGetErrorString(sts),
                        n_blocks.x,
                        n_blocks.y,
                        n_threads.x,
                        n_threads.y,
                        n_threads.z);
                    %(fail)s;
                }
            }
        }
        """ % locals(), file=sio)

    def c_code_reduce_110(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[1],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            while (n_threads.x*n_threads.y <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[0])
                    break;
                n_threads.y += 1;
            }
            n_threads.y -= 1;

            dim3 n_blocks(CudaNdarray_HOST_DIMS(%(x)s)[2]);
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_001(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[2],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            dim3 n_blocks(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                        NUM_VECTOR_OP_BLOCKS));
            while (n_blocks.x * n_blocks.y <= NUM_VECTOR_OP_BLOCKS)
            {
                if (n_blocks.y > CudaNdarray_HOST_DIMS(%(x)s)[1])
                    break;
                n_blocks.y += 1;
            }
            n_blocks.y -= 1;
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_111(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[2],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));

            //get as many y threads as we can fit
            while (n_threads.x * n_threads.y <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[1])
                    break;
                n_threads.y += 1;
            }
            n_threads.y -= 1;

            //get as many z threads as we can fit
            while (n_threads.x * n_threads.y * n_threads.z <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.z > CudaNdarray_HOST_DIMS(%(x)s)[0])
                    break;
                n_threads.z += 1;
            }
            n_threads.z -= 1;
            //Maximum for Fermi GPU on that dimensions.
            n_threads.z = std::min(n_threads.z, (unsigned)64);

            dim3 n_blocks(1,1,1);
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_0011(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;

            dim3 n_blocks(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                        NUM_VECTOR_OP_BLOCKS));

            while (n_blocks.x * n_blocks.y <= NUM_VECTOR_OP_BLOCKS &&
                   n_blocks.y < CudaNdarray_HOST_DIMS(%(x)s)[1])
            {
                n_blocks.y += 1;
            }

            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[3],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));
            while (n_threads.x * n_threads.y <= NUM_VECTOR_OP_THREADS_PER_BLOCK
                   && n_threads.y < CudaNdarray_HOST_DIMS(%(x)s)[2]
                   && n_threads.x * n_threads.y * sizeof(float) <=(15*1024-200))
            {
                n_threads.y += 1;
            }

            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_1111(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[2],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));

            //get as many y threads as we can fit
            while (n_threads.x * n_threads.y <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[1])
                    break;
                n_threads.y += 1;
            }
            n_threads.y -= 1;

            //get as many z threads as we can fit
            while (n_threads.x * n_threads.y * n_threads.z <= NUM_VECTOR_OP_THREADS_PER_BLOCK)
            {
                if (n_threads.z > CudaNdarray_HOST_DIMS(%(x)s)[0])
                    break;
                n_threads.z += 1;
            }
            n_threads.z -= 1;

            //Maximum for Fermi GPU on that dimensions.
            n_threads.z = std::min(n_threads.z, (unsigned)64);

            dim3 n_blocks(1,1,1);
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_reduce_1011(self, sio, node, name, x, z, fail):
        makecall = self._makecall(node, name, x, z, fail)
        print("""
        {
            int verbose = 0;
            dim3 n_threads(
                    std::min(CudaNdarray_HOST_DIMS(%(x)s)[3],
                            NUM_VECTOR_OP_THREADS_PER_BLOCK));

            while (n_threads.x * (n_threads.y+1) <= NUM_VECTOR_OP_THREADS_PER_BLOCK) ++n_threads.y;
            if (n_threads.y > CudaNdarray_HOST_DIMS(%(x)s)[2])
                n_threads.y = CudaNdarray_HOST_DIMS(%(x)s)[2];

            while (n_threads.x * n_threads.y * (n_threads.z+1) <= NUM_VECTOR_OP_THREADS_PER_BLOCK) ++n_threads.z;
            if (n_threads.z > 64)
                n_threads.z = 64;
            if (n_threads.z > CudaNdarray_HOST_DIMS(%(x)s)[0])
                n_threads.z = CudaNdarray_HOST_DIMS(%(x)s)[0];

            dim3 n_blocks(CudaNdarray_HOST_DIMS(%(x)s)[1]);
            %(makecall)s
        }
        """ % locals(), file=sio)

    def c_code_cache_version_apply(self, node):
        version = [14]  # the version corresponding to the c code in this Op

        # now we insert versions for the ops on which we depend...
        scalar_node = Apply(self.scalar_op,
                [Scalar(dtype=input.type.dtype)() for input in node.inputs],
                [Scalar(dtype=output.type.dtype)() for output in node.outputs])
        version.extend(self.scalar_op.c_code_cache_version())
        for i in node.inputs + node.outputs:
            version.extend(Scalar(dtype=i.type.dtype).c_code_cache_version())
        if all(version):
            return tuple(version)
        else:
            return ()

    def c_support_code_apply(self, node, nodename):
        sio = StringIO()
        nd_in = len(self.reduce_mask)
        if all(i == 1 for i in self.reduce_mask):
            # this kernel is ok for up to a few thousand elements, but
            # it only runs on ONE multiprocessor
            reducebuf = self._k_reduce_buf('Z[0]', node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0]",
                                             {}, True)
            reduce_init = self._assign_init("A[0]")
            print("""
            static __global__ void kernel_reduce_ccontig_%(nodename)s(
                    const unsigned int d0,
                    const float *A,
                    float * Z)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ float buf[];
                float myresult = %(reduce_init)s;

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = threadIdx.x; i0 < d0; i0 += blockDim.x)
                {
                    %(reduce_fct)s
                }
                %(reducebuf)s
            }
            """ % locals(), file=sio)
        if self.reduce_mask == (1,):
            # this kernel is ok for up to a few thousand elements, but
            # it only runs on ONE multiprocessor
            reducebuf = self._k_reduce_buf('Z[0]', node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0]",
                                             {}, True)
            reduce_init = self._assign_init("A[0]")
            print("""
            static __global__ void kernel_reduce_1_%(nodename)s(
                    const unsigned int d0,
                    const float *A, const int sA0,
                    float * Z)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ float buf[];
                float myresult = %(reduce_init)s;

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = threadIdx.x; i0 < d0; i0 += blockDim.x)
                {
                    %(reduce_fct)s
                }
                %(reducebuf)s
            }
            """ % locals(), file=sio)
        if self.reduce_mask == (1, 1):
            # this kernel is ok for up to a few thousand elements, but
            # it only runs on ONE multiprocessor
            reducebuf = self._k_reduce_buf('Z[0]', node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1]",
                                             {}, True)
            reduce_init = self._assign_init("A[0]")

            print("""
            static __global__ void kernel_reduce_11_%(nodename)s(
                    const int d0,
                    const int d1,
                    const float *A, const int sA0, const int sA1,
                    float * Z)
            {
                const int threadCount = blockDim.x * blockDim.y;
                const int threadNum = threadIdx.y*blockDim.x + threadIdx.x;
                extern __shared__ float buf[];
                float myresult = %(reduce_init)s;

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = threadIdx.y; i0 < d0; i0 += blockDim.y)
                {
                    for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)
                    {
                        %(reduce_fct)s;
                    }
                }
                %(reducebuf)s
            }
            """ % locals(), file=sio)
        #01, 011, 0111
        if (0 == self.reduce_mask[0] and
            all(self.reduce_mask[1:]) and
            nd_in in[2, 3, 4]):
            # this kernel uses one block for each row.
            # threads per block for each element per row.

            N_pattern = ''.join(['1'] * (nd_in - 1))
            # TODO: is it faster to hardcode sA3, etc. in the later code, rather
            # than have the for_* variables declare them and the later code use
            # their names?
            if nd_in == 2:
                for_i1 = "for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)"
                first_i1 = 'threadIdx.x'
                sA1 = 'sA1'
                for_i2 = "int i2=0, sA2=0;"
                sA2 = '0'
                first_i2 = '0'
                for_i3 = "int i3=0, sA3=0;"
                sA3 = '0'
                first_i3 = '0'
            if nd_in == 3:
                for_i1 = "for (int i1 = threadIdx.y; i1 < d1; i1 += blockDim.y)"
                first_i1 = 'threadIdx.y'
                sA1 = 'sA1'
                for_i2 = "for (int i2 = threadIdx.x; i2 < d2; i2 += blockDim.x)"
                first_i2 = 'threadIdx.x'
                sA2 = 'sA2'
                for_i3 = "int i3=0, sA3=0;"
                first_i3 = 0
                sA3 = '0'
            if nd_in == 4:
                for_i1 = "for (int i1 = threadIdx.z; i1 < d1; i1 += blockDim.z)"
                first_i1 = 'threadIdx.z'
                sA1 = 'sA1'
                for_i2 = "for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y)"
                first_i2 = 'threadIdx.y'
                sA2 = 'sA2'
                for_i3 = "for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)"
                first_i3 = 'threadIdx.x'
                sA3 = 'sA3'

            reducebuf = self._k_reduce_buf('Z[i0 * sZ0]', node,
                                           nodename, sub={})
            param_dim = ",".join(["const int d%d" % i
                                  for i in xrange(nd_in)])
            param_strides = ",".join(["const int sA%d" % i
                                      for i in xrange(nd_in)])
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_init = self._assign_init("A[%(first_i3)s * %(sA3)s + %(first_i2)s * %(sA2)s + %(first_i1)s * %(sA1)s + i0 * sA0]" % locals())
            reduce_fct = self._assign_reduce(
                node, nodename, "myresult",
                "A[i3 * sA3 + i2 * sA2 + i1 * sA1 + i0 * sA0]",
                {}, True)

            print("""
                %(decl)s{
                    %(init)s
                    for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x){
                      myresult = %(reduce_init)s;
                      %(for_i1)s{
                        %(for_i2)s{
                          %(for_i3)s{
                            %(reduce_fct)s;
                          }
                        }
                      }
                      %(reducebuf)s
                    }
                }
                """ % locals(), file=sio)
        if self.reduce_mask == (0, 1, 0) or self.reduce_mask == (1, 0):
            # this kernel uses one block for each column,
            # threads per block for each element per column.

            # TODO: This kernel is pretty inefficient in terms of reading, because if A is
            #      c_contiguous (typical case) then each warp is accessing non-contigous
            #      memory (a segment of a column).
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i2*sZ1]',
                                           node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2]",
                                             {}, True)
            reduce_init = self._assign_init("A[i0 * sA0 + threadIdx.x * sA1 + i2 * sA2]")
            print("""
            static __global__ void kernel_reduce_010_%(nodename)s(
                    const int d0,
                    const int d1,
                    const int d2,
                    const float *A, const int sA0,
                    const int sA1, const int sA2,
                    float * Z, const int sZ0, const int sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ float buf[];

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }


                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i2 = blockIdx.y; i2 < d2; i2 += gridDim.y)
                    {
                        float myresult = %(reduce_init)s;
                        for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                        %(reducebuf)s
                    }
                }

            }
            """ % locals(), file=sio)
        if self.reduce_mask in [(0, 1, 0), (1, 0), (1, 0, 0)]:
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "X[a * sX0 + b * sX1 + c * sX2]",
                                             {}, True)
            reduce_init = self._assign_init("X[a * sX0 + 0 * sX1 + c * sX2]")
            print("""
            static __global__ void kernel_reduce_010_AD_%(nodename)s(
                    const int A,
                    const int B,
                    const int C,
                    const int D,
                    //const int E, // THIS is 32
                    const float *X, const int sX0,
                    const int sX1, const int sX2,
                    float * Z, const int sZ0, const int sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                float myresult = 0.0f;

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int a = blockIdx.x; a < A; a += gridDim.x)
                {
                    for (int i2_D = blockIdx.y; i2_D < D; i2_D += gridDim.y)
                    {
                        int c = i2_D * 32 + threadIdx.x;
                        if (c < C)
                        {
                            myresult = %(reduce_init)s;
                            for (int b = 0; b < B; ++b)
                            {
                                %(reduce_fct)s;
                            }
                            Z[a * sZ0 + c * sZ1] = myresult;
                        }
                    }
                }

            }
            """ % locals(), file=sio)
        if self.reduce_mask == (0, 1, 0):
            #
            # This kernel is optimized when the inner most dimensions
            # have the smallest stride.

            # this kernel uses one block for multiple column(up to 32TODO),
            # threads per block for each element per column.

# thread.x = dim 2 contiguous
# thread.y = dim 1
# block.x = dim 0
# block.y = dim 1 rest
            init = self._k_init(node, nodename)
            decl = self._k_decl(node, nodename, pattern="010_inner")
            reducebuf = self._k_reduce_buf_multiple('Z[i0 * sZ0 + i2*sZ1]',
                                                    node, nodename,
                                                    'blockDim.x')
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2]",
                                             {}, True)
            reduce_init = self._assign_init("A[i0 * sA0 + 0 * sA1 + i2 * sA2]")
            print("""
            %(decl)s
            {
             if(warpSize<blockDim.x){
               //TODO: set error code
               Z[0] = -666;
               return;
              }

              %(init)s
              for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
              {
                for (int i2 = blockIdx.y*blockDim.x+threadIdx.x; i2 < d2; i2 += gridDim.y*blockDim.x)
                 {
                  myresult = %(reduce_init)s;
                  for (int i1 = threadIdx.y; i1 < d1; i1 += blockDim.y)
                  {
                      %(reduce_fct)s;
                  }
                  %(reducebuf)s
                 }
              }
            }
            """ % locals(), file=sio)
        if self.reduce_mask == (1, 1, 0):
            # this kernel uses one block for each column,
            # threads per block for each element per column.

            # TODO: This kernel is pretty inefficient in terms of reading, because if A is
            #      c_contiguous (typical case) then each warp is accessing non-contigous
            #      memory (a segment of a column).
            reducebuf = self._k_reduce_buf('Z[blockIdx.x * sZ0]', node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + blockIdx.x * sA2]",
                                             {}, True)
            reduce_init = self._assign_init("A[blockIdx.x * sA2]")
            print("""
            static __global__ void kernel_reduce_110_%(nodename)s(
                    const int d0,
                    const int d1,
                    const int d2,
                    const float *A, const int sA0,
                    const int sA1, const int sA2,
                    float * Z, const int sZ0)
            {
                const int threadCount = blockDim.x * blockDim.y;
                const int threadNum = threadIdx.y * blockDim.x + threadIdx.x;
                extern __shared__ float buf[];
                float myresult = %(reduce_init)s;

                if (warpSize != 32)
                {
                    //TODO: set error code
                    Z[blockIdx.x * sZ0] = -666;
                    return;
                }

                for (int i0 = threadIdx.y; i0 < d0; i0 += blockDim.y)
                {
                    for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x)
                    {
                        %(reduce_fct)s;
                    }
                }

                %(reducebuf)s
            }
            """ % locals(), file=sio)
        if self.reduce_mask == (1, 0, 0):
            reducebuf = self._k_reduce_buf('Z[i1 * sZ0 + i2 * sZ1]',
                                           node, nodename, sub={})
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2]",
                                             {}, True)
            reduce_init = self._assign_init("A[i1 * sA1 + i2 * sA2]")
            print("""
            %(decl)s
            {
                %(init)s
                for (int i2 = blockIdx.y; i2 < d2; i2 += gridDim.y)
                {
                    for (int i1 = blockIdx.x; i1 < d1; i1 += gridDim.x)
                    {
                        myresult = %(reduce_init)s;
                        for (int i0 = threadIdx.x; i0 < d0; i0 += blockDim.x)
                        {
                            %(reduce_fct)s
                        }
                        %(reducebuf)s
                    }
                }
            }
            """ % locals(), file=sio)
        if self.reduce_mask == (1, 1, 1):
            reducebuf = self._k_reduce_buf('Z[0]', node,
                                           nodename, sub={})
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2]",
                                             {}, True)
            reduce_init = self._assign_init("A[0]")
            print("""
            %(decl)s
            {
                %(init)s
                myresult = %(reduce_init)s;
                for (int i0 = threadIdx.z; i0 < d0; i0 += blockDim.z)
                {
                    for (int i1 = threadIdx.y; i1 < d1; i1 += blockDim.y)
                    {
                        for (int i2 = threadIdx.x; i2 < d2; i2 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                    }
                }
                %(reducebuf)s
            }
            """ % locals(), file=sio)
        if self.reduce_mask == (0, 0, 1):
            # this kernel uses one block for each row,
            # threads per block for each element per row.
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i1 * sZ1]',
                                           node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2]",
                                             {}, True)
            reduce_init = self._assign_init("A[i0 * sA0 + i1 * sA1]")
            print("""
            static __global__ void kernel_reduce_001_%(nodename)s(
                    const int d0,
                    const int d1,
                    const int d2,
                    const float *A, const int sA0,
                    const int sA1, const int sA2,
                    float * Z, const int sZ0, const int sZ1)
            {
                const int threadCount = blockDim.x;
                const int threadNum = threadIdx.x;
                extern __shared__ float buf[];

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i1 = blockIdx.y; i1 < d1; i1 += gridDim.y)
                    {
                        float myresult = %(reduce_init)s;
                        for (int i2 = threadIdx.x; i2 < d2; i2 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                        %(reducebuf)s
                    }
                }
            }
            """ % locals(), file=sio)
        if self.reduce_mask == (0, 0, 1, 1):
             # this kernel uses one block for each row,
            # threads per block for each element per row.
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i1 * sZ1]',
                                           node, nodename, sub={})
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3]",
                                             {}, True)
            reduce_init = self._assign_init("A[i0 * sA0 + i1 * sA1]")
            print("""
            %(decl)s
            {
                %(init)s

                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i1 = blockIdx.y; i1 < d1; i1 += gridDim.y)
                    {
                        float myresult = %(reduce_init)s;
                    for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y)
                    {
                        for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                    }
                        %(reducebuf)s
                    }
                }
            }
            """ % locals(), file=sio)
        if self.reduce_mask == (0, 1, 0, 1):
            # this kernel uses one block for each row,
            # threads per block for each element per row.
            reducebuf = self._k_reduce_buf('Z[i0 * sZ0 + i2 * sZ1]',
                                           node, nodename, sub={})
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3]",
                                             {}, True)
            reduce_init = self._assign_init("A[i0 * sA0 + i2 * sA2]")
            print("""
            %(decl)s
            {
                %(init)s

                for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x)
                {
                    for (int i2 = blockIdx.y; i2 < d2; i2 += gridDim.y)
                    {
                        float myresult = %(reduce_init)s;
                    for (int i1 = threadIdx.y; i1 < d1; i1 += blockDim.y)
                    {
                        for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                    }
                        %(reducebuf)s
                    }
                }
            }
            """ % locals(), file=sio)
        if self.reduce_mask == (1, 1, 1, 1):
            reducebuf = self._k_reduce_buf('Z[0]', node, nodename,
                                           sub={})
            decl = self._k_decl(node, nodename)
            init = self._k_init(node, nodename)
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + i1 * sA1 + i2 * sA2 + i3 * sA3]",
                                             {}, True)
            reduce_init = self._assign_init("A[0]")
            print("""
            %(decl)s
            {
                %(init)s
                myresult = %(reduce_init)s;
              for (int i0 = 0; i0 < d0; i0++)
                for (int i1 = threadIdx.z; i1 < d1; i1 += blockDim.z)
                {
                    for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y)
                    {
                        for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                    }
                }
                %(reducebuf)s
            }
            """ % locals(), file=sio)
        if self.reduce_mask == (1, 0, 1, 1):
            reducebuf = self._k_reduce_buf('Z[blockIdx.x*sZ0]',
                                           node, nodename, sub={})
            reduce_fct = self._assign_reduce(node, nodename, "myresult",
                                             "A[i0 * sA0 + blockIdx.x * sA1 + i2 * sA2 + i3 * sA3]",
                                             {}, True)
            reduce_init = self._assign_init("A[blockIdx.x * sA1]")
            print("""
            static __global__ void kernel_reduce_1011_%(nodename)s(
                    const unsigned int d0,
                    const unsigned int d1,
                    const unsigned int d2,
                    const unsigned int d3,
                    const float *A, const int sA0, const int sA1,
                    const int sA2, const int sA3,
                    float * Z, const int sZ0)
            {
                const int threadCount = blockDim.x * blockDim.y * blockDim.z;
                const int threadNum = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
                extern __shared__ float buf[];
                float myresult = %(reduce_init)s;

                if (warpSize != 32)
                {
                    return;  //TODO: set error code
                }

                for (int i0 = threadIdx.z; i0 < d0; i0 += blockDim.z)
                {
                    for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y)
                    {
                        for (int i3 = threadIdx.x; i3 < d3; i3 += blockDim.x)
                        {
                            %(reduce_fct)s;
                        }
                    }
                }
                %(reducebuf)s
            }
            """ % locals(), file=sio)
        return sio.getvalue()


class GpuReshape(tensor.Reshape, GpuOp):
    """
    Implement Reshape on the gpu.

    """

    # __hash__, __eq__, __str__ come from tensor.Reshape
    def make_node(self, x, shp):
        x = as_cuda_ndarray_variable(x)
        shp = tensor.as_tensor_variable(shp)
        host_reshaped = host_from_gpu(x).reshape(shp, ndim=self.ndim)
        return Apply(self, [x, shp],
                     [CudaNdarrayType(host_reshaped.broadcastable)()])

    def perform(self, node, inp, out_):
        x, shp = inp
        out, = out_
        if (len(shp) != self.ndim):
            raise ValueError('shape argument to Reshape.perform'
                             ' has incorrect length %i'
                             ', should be %i' % (len(shp), self.ndim), shp)

        if shp.prod() != x.size:
            # We need to do check here to raise the same error as NumPy.
            # We should make pygpu do the same.
            ss = 1
            nb_m1 = 0
            m1_idx = -1
            for idx, i in enumerate(shp):
                if i == -1:
                    nb_m1 += 1
                    m1_idx = idx
                else:
                    ss *= i
            if nb_m1 > 1:
                raise ValueError("Only one -1 is accepted in the new shape")
            elif nb_m1 == 1:
                if (x.size % ss) != 0:
                    raise ValueError("When using -1 in new shape, the computed new shape must be an multiple of the original shape.")
                shp_new = numpy.copy(shp)
                shp_new[m1_idx] = x.size/ss
                shp = shp_new

            else:
                raise ValueError("total size of new array must be unchanged",
                                 x.shape, shp)

        out[0] = x.reshape(tuple(shp))

    def c_code_cache_version(self):
        return (2,)

    def c_code(self, node, name, inputs, outputs, sub):
        x, shape = inputs
        output, = outputs
        new_ndim = self.ndim
        sdtype = node.inputs[1].type.dtype_specs()[1]
        fail = sub['fail']
        return """
        PyObject *new_shape = PyTuple_New(%(new_ndim)s);
        size_t total = 1;
        int compute_axis = -1;

        assert (PyArray_NDIM(%(shape)s) == 1);
        if (PyArray_DIM(%(shape)s, 0) != %(new_ndim)s)
        {
            Py_XDECREF(new_shape);
            PyErr_Format(PyExc_ValueError,
                         "GpuReshape: given shape is of incorrect "
                         "length (%%d should be %%d).",
                         PyArray_DIM(%(shape)s, 0), %(new_ndim)s);
            %(fail)s;
        }

        for (size_t i = 0; i < %(new_ndim)s; ++i)
        {
            long dimension = ((%(sdtype)s*)(
                    PyArray_BYTES(%(shape)s) +
                    i * PyArray_STRIDES(%(shape)s)[0]))[0];
            if (dimension == -1)
            {
                if (compute_axis != -1)
                {
                    Py_XDECREF(new_shape);
                    PyErr_Format(PyExc_ValueError,
                                 "GpuReshape: only one -1 is accepted "
                                 "in the new shape, but got two at "
                                 "indices %%d and %%zu.",
                                 compute_axis, i);
                    %(fail)s;
                }
                compute_axis = i;
            }
            else
            {
                total *= dimension;
                PyObject *py_dimension = PyInt_FromLong(dimension);
                PyTuple_SetItem(new_shape, i, py_dimension);
            }
        }

        if (compute_axis != -1)
        {
            long dimension = CudaNdarray_SIZE(%(x)s) / total;
            total *= dimension;
            PyObject *py_dimension = PyInt_FromLong(dimension);
            PyTuple_SetItem(new_shape, compute_axis, py_dimension);
        }

        if (total != CudaNdarray_SIZE(%(x)s))
        {
            const int *shape_from_py = CudaNdarray_HOST_DIMS(%(x)s);

            char shape_from[128];
            size_t offset = 0;
            for (size_t i = 0; i < %(x)s->nd; ++i)
            {
                int ws = snprintf(shape_from + offset, 128 - offset,
                        " %%d,", shape_from_py[i]);
                offset += ws;
                if ( ws < 0 || offset >= 128 )
                    break;
            }

            shape_from[0]='(';
            if(offset < 128)
                shape_from[offset>0 ? offset-1 : 1] = ')';
            else
                for(size_t i=124; i<127; ++i)
                    shape_from[i] = '.';

            PyObject *shape_to_py = PyObject_Str(new_shape);
            const char *shape_to = PyString_AsString(shape_to_py);
            Py_XDECREF(new_shape);
            PyErr_Format(PyExc_ValueError,
                         "GpuReshape: cannot reshape input of shape "
                         "%%s to shape %%s.", shape_from, shape_to);
            %(fail)s;
        }

        Py_XDECREF(%(output)s);
        %(output)s = (CudaNdarray*) CudaNdarray_Reshape(%(x)s, new_shape);
        Py_XDECREF(new_shape);
        if (%(output)s == NULL)
        {
            %(fail)s;
        }
        """ % locals()


class GpuSubtensor(GpuOp, tensor.Subtensor):
    """
    Implement subtensor on the gpu.

    """

    check_broadcast = False

    # __hash__, __eq__, __str__ come from tensor.Subtensor
    def make_node(self, x, *inputs):
        assert isinstance(x.type, CudaNdarrayType)
        rval = tensor.Subtensor.make_node(self, x, *inputs)
        otype = CudaNdarrayType(rval.outputs[0].type.broadcastable)
        return Apply(self, [x] + rval.inputs[1:], [otype()])

    def perform(self, node, inputs, out_):
        out, = out_
        x = inputs[0]
        indices = list(reversed(inputs[1:]))

        def convert(entry):
            if isinstance(entry, Type):
                rval = indices.pop()
                if sys.version_info < (2, 5):
                    # Before Python 2.5, PySlice_GetIndicesEx requires
                    # Python int to be passed.
                    rval_ = int(rval)
                    if rval_ != rval:
                        raise IndexError((
                            "Invalid value for indexing: %s. "
                            "That value may be too big.") % rval)
                    return rval_
                return rval
            elif isinstance(entry, slice):
                return slice(convert(entry.start),
                             convert(entry.stop),
                             convert(entry.step))
            else:
                return entry

        cdata = tuple(map(convert, self.idx_list))
        if len(cdata) == 1:
            cdata = cdata[0]
        out[0] = x.__getitem__(cdata)

    def c_code(self, node, name, inputs, outputs, sub):
        x = inputs[0]
        z, = outputs
        view_ndim = node.outputs[0].ndim
        fail = sub['fail']

        decl = "CudaNdarray* xview = NULL;"

        get_xview = self.helper_c_code(node, name, inputs, outputs, sub,
                                       self.idx_list,
                                       view_ndim=view_ndim,
                                       c_prefix='CudaNdarray',
                                       strides_mul=4,
                                       )
        build_view = """
        //TODO: give this Op a second output so that this view can be cached
        //TODO: alternatively, fix the memory leak on failure
        xview = (CudaNdarray*) CudaNdarray_New(%(view_ndim)s);
        if (!xview)
        {
            %(fail)s;
        }

        if (CudaNdarray_set_device_data(
                xview,
                CudaNdarray_DEV_DATA(%(x)s) + xview_offset/4,
                (PyObject*) %(x)s))
        {
            PyErr_Format(PyExc_RuntimeError,
                         "GpuSubtensor is not able to set the"
                         " devdata field of the view");
            Py_XDECREF(xview);
            %(fail)s;
        }
        cnda_mark_dev_structure_dirty(xview);
        for(int idx=0;idx <%(view_ndim)s; idx++){
        //For broadcasted dimensions, set the strides to 0
        //We can't do that only for broadcasted dimensions as this can happen
        //for dimensions of size 0. That are rebroadcated later.
            if(xview_dims[idx]==1)
                CudaNdarray_set_stride(xview, idx, 0);
            else
                CudaNdarray_set_stride(xview, idx, xview_strides[idx]);
            CudaNdarray_set_dim(xview, idx, xview_dims[idx]);
        }
        """ % locals()

        finish_view = """
        Py_XDECREF(%(z)s);
        %(z)s = xview;
        """ % locals()

        return decl + get_xview + build_view + finish_view

    def c_code_cache_version(self):
        hv = self.helper_c_code_cache_version()
        # If `helper_c_code_cache_version` is not versioned we do not want to
        # have a versioned version of this op's C code.
        if len(hv) == 0:
            return ()
        return (3, hv)


class GpuAdvancedSubtensor1(tensor.AdvancedSubtensor1, GpuOp):
    """
    Implement AdvancedSubtensor1 on the gpu.

    """

    # If True or False, we assert that we use the take version or not
    # If None, we choose the best one applicable
    perform_using_take = None
    max_threads = 0

    def make_node(self, x, ilist):
        x_ = as_cuda_ndarray_variable(x)
        ilist_ = tensor.as_tensor_variable(ilist)
        if ilist_.type.dtype[:3] not in ('int', 'uin'):
            raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')

        # c code suppose it is int64
        if x.ndim in [1, 2, 3] and ilist_.dtype in [
            'int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']:
            ilist_ = tensor.cast(ilist_, 'int64')

        bcast = (ilist_.broadcastable[0],) + x_.broadcastable[1:]
        return Apply(self, [x_, ilist_],
                     [CudaNdarrayType(dtype=x.dtype,
                                      broadcastable=bcast)()])

    def perform(self, node, inp, out_):
        # This don't work as CudaNdarray_Subscript() don't support it.
        #super(GpuAdvancedSubtensor1, self).perform(node, inp, out_)
        x, idx = inp
        out, = out_
        x_orig = x
        # TODO: if more than 3 dims, reshape the inputs even if not all
        # dimensions are c contiguous
        if x.ndim > 3 and x.is_c_contiguous():
            x = x.reshape((x.shape[0], numpy.prod(x.shape[1:])))
        out_shape = (len(idx),) + x_orig.shape[1:]
        if x.ndim <= 3:
            # CudaNdarray.take only supports ndim <= 3
            if self.perform_using_take is not None:
                assert self.perform_using_take == True, (
                    "GpuAdvancedSubtensor1 used the fast version")
            if idx.dtype != numpy.int64:
                if idx.dtype in [numpy.int8, numpy.int16, numpy.int32,
                                 numpy.int64, numpy.uint8, numpy.uint16,
                                 numpy.uint32]:
                    idx = idx.astype(numpy.int64)
            if not idx.flags.c_contiguous:
                idx = numpy.ascontiguousarray(idx)

            idx = idx.view("float32")
            idx = cuda_ndarray.cuda_ndarray.CudaNdarray(idx)
            if self.max_threads == 0:
                num = theano.sandbox.cuda.use.device_number
                if device_properties(num)['regsPerBlock'] < (8192 * 2):
                    self.max_threads = 256
                else:
                    self.max_threads = 512

            o = x.take(idx,
                       0,  # axis
                       out_[0][0],  # return
                       "raise",
                       self.max_threads)
            if x is not x_orig:
                o = o.reshape(out_shape)
            out[0] = o
        else:
            if self.perform_using_take is not None:
                assert self.perform_using_take == False, (
                    "GpuAdvancedSubtensor1 didn't use the fast version")
            if out_[0][0] is None or out_[0][0].shape != out_shape:
                o = cuda_ndarray.cuda_ndarray.CudaNdarray.zeros(out_shape)
            else:
                o = out_[0][0]
            for (j, i) in enumerate(idx):
                o[j] = x[i]
            out[0] = o

    def c_code(self, node, name, inputs, outputs, sub):
        x, idx = inputs
        out, = outputs
        fail = sub['fail']
        if node.inputs[0].ndim not in [1, 2, 3]:
            raise NotImplementedError("This case does not have C code yet.")
        if node.inputs[1].dtype != 'int64':
            raise Exception("Index should have dtype int64. Check this node make_node().")
        return """
        //take(idx, 0, out, "raise", max_threads);
        PyObject * ret = NULL;
        PyObject * args = Py_BuildValue("OiOsi", %(idx)s, 0,
                                        %(out)s == NULL ? Py_None : (PyObject *)%(out)s,
                                        "raise", 512);
        if(args == NULL){
            //Error set by Py_BuildValue
            %(fail)s;
        }
        ret = CudaNdarray_TakeFrom(%(x)s, args);

        Py_DECREF(args);
        if (ret == NULL){
            %(fail)s;
        }
        // Even if we decref, we still try to reuse preallocated output
        Py_XDECREF(%(out)s);
        %(out)s = (CudaNdarray *) ret;
        """ % locals()

    def c_code_cache_version(self):
        return (2,)


class GpuAdvancedIncSubtensor1(tensor.AdvancedIncSubtensor1, GpuOp):
    """
    Implement AdvancedIncSubtensor1 on the gpu.

    """

    def make_node(self, x, y, ilist):
        x_ = as_cuda_ndarray_variable(x)
        y_ = as_cuda_ndarray_variable(y)
        ilist_ = tensor.as_tensor_variable(ilist)

        assert x_.type.dtype == y_.type.dtype
        assert x_.type.ndim >= y_.type.ndim

        if ilist_.type.dtype[:3] not in ('int', 'uin'):
            raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')
        if y_.type.ndim > x_.type.ndim:
            if self.set_instead_of_inc:
                opname = 'set'
            else:
                opname = 'increment'
            raise TypeError(
                'cannot %s x subtensor with ndim=%s'
                ' by y with ndim=%s to x subtensor with ndim=%s ' % (
                    opname, x_.type.ndim, y_.type.ndim))

        return Apply(self, [x_, y_, ilist_], [x_.type()])

    # CudaNdarray_Subscript() doesn't support Advanced slicing.
    # But we can't use the parent version that loops on each index
    # as we also need to loop when set_instead_of_inc is True and the
    # parent doesn't loop in that case.
    def perform(self, node, inp, out_):
        # TODO opt to make this inplace
        x, y, idx = inp
        out, = out_
        if not self.inplace:
            x = x.copy()
        assert y.ndim <= x.ndim   # Should be guaranteed by `make_node`
        if self.set_instead_of_inc:
            # CudaNdarray __setitem__ doesn't do broadcast nor support
            # list of index.
            if y.ndim == x.ndim:
                if len(y) == 1:
                    # Allow broadcasting of y[0]
                    y_0 = y[0]
                    for i in idx:
                        x[i] = y_0
                else:
                    assert len(y) == len(idx)
                    j = 0
                    for i in idx:
                        x[i] = y[j]
                        j += 1
            else:
                for i in idx:
                    x[i] = y
        else:
            # If `y` has as many dimensions as `x`, then we want to iterate
            # jointly on `x` and `y`. Otherwise, it means `y` should be
            # broadcasted to fill all relevant rows of `x`.
            if y.ndim == x.ndim:
                if len(y) == 1:
                    # Allow broadcasting of y[0]
                    y_0 = y[0]
                    for i in idx:
                        x[i] += y_0
                else:
                    assert len(y) == len(idx)
                    j = 0
                    for i in idx:
                        x[i] += y[j]
                        j += 1
            else:
                for i in idx:
                    x[i] += y
        out[0] = x

    def c_code_cache_version(self):
        return (7,)

    def c_code(self, node, name, inputs, outputs, sub):
        if (node.inputs[0].ndim != node.inputs[1].ndim):
            raise NotImplementedError("This case does not have C code yet.")

        x = inputs[0]
        y = inputs[1]
        ind = inputs[2]
        out = outputs[0]
        fail = sub['fail']
        inplace = int(self.inplace)
        set_instead_of_inc = int(self.set_instead_of_inc)

        return """
        PyObject *row_x, *row_y;
        PyObject *x_rowind_obj, *y_rowind_obj;
        dtype_%(ind)s *p_index;
        int num_indices, j;
        int ret;
        int broadcast_y;

        num_indices = PyArray_SIZE(%(ind)s);
        if ((num_indices - 1) > LONG_MAX) {
            PyErr_Format(PyExc_AssertionError,
                         "num_indices %%d exceeds LONG_MAX + 1", num_indices);
            %(fail)s;
        }

        Py_XDECREF(%(out)s);
        if (!%(inplace)s) {
            %(out)s = (CudaNdarray*)CudaNdarray_Copy(%(x)s);
        } else {
            %(out)s = %(x)s;
            Py_XINCREF(%(out)s);
        }
        broadcast_y = CudaNdarray_DIMS(%(y)s)[0] == 1;
        for (j = 0;j < num_indices; j++) {

             p_index = (dtype_%(ind)s *)PyArray_GETPTR1(%(ind)s, j);

             x_rowind_obj = PyInt_FromLong(*p_index);

             if (PyInt_AsLong(x_rowind_obj) != (*p_index)) {
                 PyErr_Format(PyExc_AssertionError,
                              "Error in converting row index to integer from long");
                 // Dec Ref what ever we have increfed or allocated so far
                 // We deallocate objects exactly in the reverse order they were allocated.
                 Py_XDECREF(x_rowind_obj);
                 %(fail)s;
             }

             row_x = CudaNdarray_Subscript((PyObject*)%(out)s, x_rowind_obj);
             if (row_x == NULL) {
                  Py_XDECREF(row_x);
                  Py_XDECREF(x_rowind_obj);
                  %(fail)s;
             }
             if (broadcast_y) {
                 y_rowind_obj = PyInt_FromLong(0);
             } else {
                 y_rowind_obj = PyInt_FromLong(j);
             }
             row_y = CudaNdarray_Subscript((PyObject*)%(y)s, y_rowind_obj);

             if (row_y == NULL) {
                  Py_XDECREF(row_y);
                  Py_XDECREF(row_x);
                  Py_XDECREF(y_rowind_obj);
                  Py_XDECREF(x_rowind_obj);
                  %(fail)s;
             }
             if (%(set_instead_of_inc)s) {
                 ret = CudaNdarray_CopyFromCudaNdarray((CudaNdarray *) row_x, (CudaNdarray *) row_y);
             } else {
                 ret = CudaNdarray_inplace_elemwise(row_x, row_y, IADD);
             }
             if (ret != 0) {
                 Py_XDECREF(row_y);
                 Py_XDECREF(row_x);
                 Py_XDECREF(y_rowind_obj);
                 Py_XDECREF(x_rowind_obj);
                 %(fail)s;
             }

             Py_XDECREF(row_y);
             Py_XDECREF(row_x);
             Py_XDECREF(y_rowind_obj);
             Py_XDECREF(x_rowind_obj);
        }


        if (!%(out)s) {
            %(fail)s
        }
        """ % locals()


class GpuAdvancedIncSubtensor1_dev20(GpuAdvancedIncSubtensor1):
    """
    Implement AdvancedIncSubtensor1 on the gpu, but use function
    only avail on compute capability 2.0 and more recent.

    """

    def make_node(self, x, y, ilist):
        """
        It defer from GpuAdvancedIncSubtensor1 in that it make sure
        the index are of type long.

        """
        x_ = as_cuda_ndarray_variable(x)
        y_ = as_cuda_ndarray_variable(y)
        ilist_ = tensor.as_tensor_variable(ilist)

        convert_map = {8: tensor.basic._convert_to_int8,
                       16: tensor.basic._convert_to_int16,
                       32: tensor.basic._convert_to_int32,
                       64: tensor.basic._convert_to_int64
        }
        intwidth = theano.configdefaults.python_int_bitwidth()
        ilist_ = convert_map[intwidth](ilist_)

        assert x_.type.dtype == y_.type.dtype
        assert x_.type.ndim >= y_.type.ndim

        if ilist_.type.dtype[:3] not in ('int', 'uin'):
            raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')
        if y_.type.ndim > x_.type.ndim:
            if self.set_instead_of_inc:
                opname = 'set'
            else:
                opname = 'increment'
            raise TypeError(
                'cannot %s x subtensor with ndim=%s'
                ' by y with ndim=%s to x subtensor with ndim=%s ' % (
                    opname, x_.type.ndim, y_.type.ndim))

        return Apply(self, [x_, y_, ilist_], [x_.type()])

    def c_code_cache_version(self):
        return (7,)

    def c_code(self, node, name, inputs, outputs, sub):
        active_device_no = theano.sandbox.cuda.active_device_number()
        compute_capability = device_properties(active_device_no)['major']
        if ((node.inputs[0].ndim != node.inputs[1].ndim) or
            (node.inputs[0].ndim != 2) or
            (compute_capability < 2)):
            raise NotImplementedError("This case does not have C code yet.")

        x = inputs[0]
        y = inputs[1]
        ind = inputs[2]
        out = outputs[0]
        fail = sub['fail']
        inplace = int(self.inplace)
        set_instead_of_inc = int(self.set_instead_of_inc)
        return """
        Py_XDECREF(%(out)s);
        if (!%(inplace)s) {
            %(out)s = (CudaNdarray*)CudaNdarray_Copy(%(x)s);
        } else {
            %(out)s = %(x)s;
            Py_XINCREF(%(out)s);
        }

        if (CudaNdarray_vector_add_or_replace_fast(%(out)s, %(y)s, %(ind)s, %(set_instead_of_inc)s) != 0){
            %(fail)s
        }

        if (!%(out)s) {
            %(fail)s
        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        return """

        __global__ void k_vector_add_or_replace_fast(int numRowsX,
                                          int numColsX,
                                          int stridesX0,
                                          int stridesX1,
                                          float *X,
                                          int numRowsY,
                                          int numColsY,
                                          int stridesY0,
                                          int stridesY1,
                                          float *Y ,
                                          long *d_indices_arr,
                                          int num,
                                          const int set_instead_of_inc,
                                          int* err)
        {
             for (int i = (blockIdx.x); i < num; i += gridDim.x)
             {
                  for(int j = (threadIdx.x); j < numColsX;j += blockDim.x)
                  {
                      int x_row = d_indices_arr[i];
                      if(x_row < 0)
                          x_row += numRowsX;
                      int y_row = i;
                      if(x_row < numRowsX && x_row >= 0){
                        if(set_instead_of_inc){
                            atomicExch(&X[(x_row * stridesX0) + (j * stridesX1)],
                                  Y[(y_row * stridesY0) + (j * stridesY1)]);
                        } else{
                            atomicAdd(&X[(x_row * stridesX0) + (j * stridesX1)],
                                  Y[(y_row * stridesY0) + (j * stridesY1)]);
                        }
                      } else {
                        *err = 1;
                      }
                  }
             }
             return;
        }

        int CudaNdarray_vector_add_or_replace_fast(CudaNdarray* py_self,
            CudaNdarray* py_other, PyArrayObject *indices_arr,
            const int set_instead_of_inc)
        {
            if(init_err_var()!= 0) return -1;

            const int *shapeX = CudaNdarray_HOST_DIMS(py_self);
            const int *shapeY = CudaNdarray_HOST_DIMS(py_other);
            const int *strX   = CudaNdarray_HOST_STRIDES(py_self);
            const int *strY   = CudaNdarray_HOST_STRIDES(py_other);
            unsigned int size = (unsigned int)PyArray_SIZE(indices_arr);
            if(size == 0){
                return 0;
            }
            unsigned int numcolsX = shapeX[1];
            unsigned int num_threads_per_block = std::min(
                numcolsX, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
            unsigned int num_blocks = std::min(
                size, (unsigned int)NUM_VECTOR_OP_BLOCKS);

            dim3 n_blocks(num_blocks);
            dim3 n_threads(num_threads_per_block);
            long *d_indices_arr = NULL;
            PyArrayObject *cpu_indices_arr = PyArray_GETCONTIGUOUS(
                indices_arr);
            d_indices_arr = (long*)device_malloc(
                PyArray_NBYTES(cpu_indices_arr));

            if(!d_indices_arr)
                return -1;

            cudaError_t err = cudaMemcpy(d_indices_arr,
                                         PyArray_DATA(cpu_indices_arr),
                                         PyArray_NBYTES(cpu_indices_arr),
                                         cudaMemcpyHostToDevice);
            if(err != cudaSuccess){
                PyErr_Format(
                    PyExc_RuntimeError,
                    "GpuAdvancedIncSubtensor1_dev20:"
                    " cudaMemcpy returned an error: %%s",
                    cudaGetErrorString(err));
                return -1;
            }

            k_vector_add_or_replace_fast<<<n_blocks, n_threads>>>(
                shapeX[0],
                shapeX[1],
                strX[0],
                strX[1],
                CudaNdarray_DEV_DATA(py_self),
                shapeY[0],
                shapeY[1],
                strY[0],
                strY[1],
                CudaNdarray_DEV_DATA(py_other),
                d_indices_arr,
                PyArray_SIZE(indices_arr),
                set_instead_of_inc,
                err_var
            );
            int index_err = check_err_var();

            device_free(d_indices_arr);
            Py_XDECREF(cpu_indices_arr);

            if(index_err != 0) return -1;

            err = cudaGetLastError();
            if(err != cudaSuccess){
                PyErr_Format(
                    PyExc_RuntimeError,
                    "GpuAdvancedIncSubtensor1_dev20: cuda error: %%s",
                    cudaGetErrorString(err));
                return -1;
            }
            return 0;
        }

        """ % locals()


class GpuIncSubtensor(tensor.IncSubtensor, GpuOp):
    """
    Implement IncSubtensor on the gpu.

    Notes
    -----
    The optimization to make this inplace is in tensor/opt.
    The same optimization handles IncSubtensor and GpuIncSubtensor.
    This Op has c_code too; it inherits tensor.IncSubtensor's c_code.
    The helper methods like do_type_checking, copy_of_x, etc. specialize
    the c_code for this Op.

    """

    def make_node(self, x, y, *inputs):
        x = as_cuda_ndarray_variable(x)
        y = as_cuda_ndarray_variable(y)
        rval = tensor.IncSubtensor.make_node(self, x, y, *inputs)
        return Apply(self, [x, y] + rval.inputs[2:], [x.type()])

    def do_type_checking(self, node):
        """ 
        Should raise NotImplementedError if c_code does not support
        the types involved in this node.

        """
        if not isinstance(node.inputs[0].type, CudaNdarrayType):
            raise NotImplementedError()

    def copy_of_x(self, x):
        """

        Parameters
        ----------
        x : str
            A string giving the name of a C variable pointing to an array.

        Returns
        -------
        str
            C code expression to make a copy of x.

        Notes
        -----
        Base class uses `PyArrayObject *`, subclasses may override for
        different types of arrays.

        """
        return """(CudaNdarray*) CudaNdarray_Copy(%(x)s)""" % locals()

    def decl_view(self):
        return "CudaNdarray* zview = NULL;"

    def make_view_array(self, x, view_ndim):
        """

        Parameters
        ----------        
        x : str
            A string identifying an array to be viewed.
        view_ndim : str
            A string specifying the number of dimensions to have in the view.
            This doesn't need to actually set up the view with the
            right indexing; we'll do that manually later.

        """
        ret = """zview = (CudaNdarray*) CudaNdarray_New(%(view_ndim)s);
        if (CudaNdarray_set_device_data(
                zview,
                CudaNdarray_DEV_DATA(%(x)s) + xview_offset/4,
                (PyObject*) %(x)s))
        {
            zview = NULL;
            PyErr_Format(PyExc_RuntimeError,
                         "GpuSubtensor is not able to set the"
                         " devdata field of the view");
        }else{
            cnda_mark_dev_structure_dirty(zview);
            for(int idx=0;idx <%(view_ndim)s; idx++){
                if(xview_dims[idx]==1)
                    CudaNdarray_set_stride(zview, idx, 0);
                else
                    CudaNdarray_set_stride(zview, idx, xview_strides[idx]);
                CudaNdarray_set_dim(zview, idx, xview_dims[idx]);
            }
        }
        """ % locals()
        return ret

    def get_helper_c_code_args(self):
        """
        Return a dictionary of arguments to use with helper_c_code.

        """
        return {'c_prefix': 'CudaNdarray',
                'strides_mul': 4
                }

    def copy_into(self, view, source):
        """

        Parameters
        ----------
        view : str
            C code expression for an array.
        source : str
            C code expression for an array

        Returns
        -------
        str
            A C code expression to copy source into view, and 0 on success.

        """
        # On the CPU it unbroadcast based on the run time shapes. We
        # need the same behavior on the GPU.
        return """CudaNdarray_CopyFromCudaNdarray(%(view)s, %(source)s, 1)""" % locals()

    def add_to_zview(self, name, x, fail):

        return """
        PyObject * add_result = CudaNdarray_inplace_add((PyObject *) zview,
                                                        (PyObject *) %(x)s);

        if (! add_result )
        {
            Py_DECREF(zview);
            %(fail)s;
        }
        else
        {
            Py_DECREF(add_result);
        }
        """ % locals()

    def c_code_cache_version(self):
        parent_version = super(GpuIncSubtensor, self).c_code_cache_version()
        if parent_version:
            return parent_version + (2,)
        return ()


class GpuFlatten(gof.HideC, tensor.Flatten, GpuOp):
    """
    Implement Flatten on the gpu.

    .. note:: The interface GpuFlatten is deprecated, you should use gpu_flatten.
    """
    def __init__(self):
        warnings.warn(
            "GpuFlatten class is deprecated, "
            "please use gpu_flatten method instead.",
            DeprecationWarning,
            stacklevel=4)

    def make_node(self, x):
        assert isinstance(x.type, CudaNdarrayType)
        rval = tensor.Flatten.make_node(self, x)
        host_out_broadcastable = rval.outputs[0].type.broadcastable
        out_type = CudaNdarrayType(broadcastable=host_out_broadcastable)
        return Apply(self, [x], [out_type()])



def gpu_flatten(x, outdim=1):
    """
    Implement flatten on the gpu.
    Reshapes the variable x by keeping
    the first outdim-1 dimension size(s) of x the same,
    and making the last dimension size of x equal to
    the multiplication of its remaining dimension size(s).

    Parameters
    ----------
        x : theano.tensor.var.TensorVariable
            the variable that should be reshaped.

        outdim : int
            the number of dimensions of the returned variable

    Returns
    -------
    theano.tensor.var.TensorVariable
        the flattend variable with dimensionality of outdim
    """
    x = as_cuda_ndarray_variable(x)
    if outdim > 1:
        dims = tuple(x.shape[:outdim-1])+(-1,)
    else:
        dims = (-1,)
    return  GpuReshape(outdim)(x, dims)


class GpuShape(tensor.Shape, GpuOp):
    """
    Implement Shape on the gpu.

    """

    def make_node(self, x):
        return Apply(self, [x], [tensor.lvector()])
gpu_shape = GpuShape()


class GpuJoin(tensor.Join, GpuOp):
    """
    Implement Join on the gpu.

    """

    def make_node(self, *axis_and_tensors):
        axis, tensors = axis_and_tensors[0], axis_and_tensors[1:]
        if not tensors:
            raise ValueError('Cannot join an empty list of tensors')
        as_tensor_variable_args = [as_cuda_ndarray_variable(x)
                                   for x in tensors]

        output_maker = \
                lambda bcast: CudaNdarrayType(broadcastable=bcast)()

        return tensor.Join._make_node_internal(self,
                        axis, tensors,
                        as_tensor_variable_args, output_maker)

    def perform(self, node, axis_and_tensors, out_):
        out, = out_
        axis, cndas = axis_and_tensors[0], axis_and_tensors[1:]
        # In case axis is numpy.int8 and has no __index__() method
        axis = int(axis)
        ndim = cndas[0].ndim
        if axis < -ndim:
            raise IndexError("Join axis %d out of bounds [0, %d)" %
                             (axis, ndim))
        if axis < 0:
            axis += ndim

        # compute size/shape
        width_sum = 0
        template_shape = cndas[0].shape
        for cnda in cndas:
            width_sum += cnda.shape[axis]
            # and while we're at it, check that shapes match
            tmp_shape = list(cnda.shape)
            # dimension in "axis" can be different, so make equal for ==
            tmp_shape[axis] = template_shape[axis]
            if tuple(tmp_shape) != template_shape:
                raise ValueError("Shape of input CudaNdarrays must"
                                 " agree except for the 'axis' dimension")

        if len(template_shape) != node.outputs[0].type.ndim:
            raise ValueError("Number of dimension of input tensors disagree"
                             " with dimensions passed at graph creation time.")

        # final shape must be the same as all input tensors
        # except for the "axis" dimension, so we can simply
        # copy the shape of the first one
        final_shape = list(cndas[0].shape)
        final_shape[axis] = width_sum

        # just to be explicit, check that dim=1 for broadcastable
        # dimensions
        for i, bcastable in enumerate(node.outputs[0].type.broadcastable):
            assert not bcastable or final_shape[i] == 1, (
                "Broadcastable dimension but dim != 1, this is invalid")

        rval = cuda_ndarray.cuda_ndarray.CudaNdarray.zeros(final_shape)

        curpos = 0

        # we use a [:] (copy all) slice for all dimensions
        # except for 'axis'

        def construct_slices(curlen):
            slices = [slice(None, None, None) for i in \
                            xrange(len(template_shape))]
            slices[axis] = slice(curpos, curpos + curlen, None)
            return tuple(slices)

        for i, cnda in enumerate(cndas):
            curlen = cnda.shape[axis]
            rval.__setitem__(construct_slices(curlen), cnda)
            curpos += curlen

        out[0] = rval

    def c_code(self, node, name, inputs, out_, sub):
        nd = node.inputs[1].ndim
        if not all(i.ndim == nd for i in node.inputs[2:]):
            # all inputs ndarray need to have the same number of dimensions
            raise NotImplementedError()
        axis = inputs[0]
        n_cndas = len(inputs[1:])
        input_1 = inputs[1]
        fail = sub['fail']
        out = out_[0]

        # getting the shapes of all the involved tensors (input[0]+out)
        str = """
        int axis = PyInt_AsLong((PyObject*)%(axis)s);
        const int nd = %(nd)s;
        int shape_out[nd];
        int width_sum = 0;
        int errorcode;
        int sum = 0;
        PyObject *slice_tuple = NULL;
        PyObject *section_slice = NULL;
        PyObject *full_slice = NULL;
        PyObject *out_sub = NULL;
        PyObject *start, *stop;
        start = NULL;
        stop = NULL;

        """ % locals()

        # Test negative axis
        str += """
        if( axis < -nd ){
            PyErr_Format(PyExc_IndexError,
                         "Join axis %%d out of bounds [0, %%d)", axis, nd);
            %(fail)s
        }

        if( axis < 0 ){
            axis = axis + nd;
        }
        """ % locals()

        # getting the shapes of all the involved tensors (input[1:])
        # + check: all input tensors have same shape as final out
        # except for "axis" dimension
        # shape_%(cdna)s[nd] is initialized before, to prevent following
        # error: jump to label __label_9 crosses initialization of
        # shape_%(cdna)s[nd]
        for i, cdna in enumerate(gof.utils.uniq(inputs[1:])):
            str += """
            int shape_%(cdna)s[nd];
            """ % locals()
        str += """
        if(-1 == axis && PyErr_Occurred()){
            %(fail)s;
        }
        full_slice = PySlice_New(NULL, NULL, NULL);
        if(full_slice == NULL){
            %(fail)s;
        }

        for(int i = 0; i<nd; i+=1)
        {
            shape_%(input_1)s[i] = CudaNdarray_HOST_DIMS(%(input_1)s)[i];
            shape_out[i] = shape_%(input_1)s[i];
        }
        """ % locals()
        for i, cdna in enumerate(gof.utils.uniq(inputs[2:])):
            str += """
            for(int i = 0; i<nd; i+=1)
            {
                shape_%(cdna)s[i] = CudaNdarray_HOST_DIMS(%(cdna)s)[i];
                if((i!=axis) && (shape_%(cdna)s[i]!=shape_out[i]))
                {
                    PyErr_Format(
                        PyExc_ValueError,
                        "GpuJoin: Wrong inputs for input %%d related"
                        " to inputs 0.!",
                        i);
                    %(fail)s;
                }
            }
            """ % locals()

        # computing the new shape for the out tensors
        for i, cdna in enumerate(inputs[1:]):
            str += "\t\twidth_sum += CudaNdarray_HOST_DIMS(%(cdna)s)[axis];\n" % locals()
        str += "\t\tshape_out[axis] = width_sum;\n"

        # preparing the output array + init of the necessary variables
        # for the data transfer
        str += """
        if (CudaNdarray_prep_output(&%(out)s, nd, shape_out))
        {
            %(fail)s;
        }
        """ % locals()
        # start copying the data into the new out tensors
        for i, cdna in enumerate(inputs[1:]):
            str += """
            sum += shape_%(cdna)s[axis];
            stop = PyInt_FromLong(sum);
            slice_tuple = PyTuple_New(nd);
            if(slice_tuple == NULL){
                %(fail)s;
            }
            section_slice = PySlice_New(start, stop, NULL);
            if(section_slice == NULL){
                %(fail)s;
            }
            for(int i=0; i<nd; i++)
            {
                if(i!=axis)
                {
                    Py_INCREF(full_slice);
                    PyTuple_SetItem(slice_tuple, i, full_slice);
                }
                else
                {
                    Py_INCREF(section_slice);
                    PyTuple_SetItem(slice_tuple, i, section_slice);
                }
            }
            out_sub = CudaNdarray_Subscript((PyObject*)%(out)s, slice_tuple);
            if(out_sub == NULL){
                Py_XDECREF(start);
                Py_XDECREF(stop);
                Py_XDECREF(slice_tuple);
                Py_XDECREF(out_sub);
                Py_XDECREF(%(out)s);
                %(fail)s;
            }
            Py_CLEAR(slice_tuple);
            Py_CLEAR(section_slice);

            errorcode = CudaNdarray_CopyFromCudaNdarray(
                (CudaNdarray*)out_sub, %(cdna)s);
            if(errorcode != 0)
            {
                Py_XDECREF(start);
                Py_XDECREF(stop);
                Py_XDECREF(out_sub);
                Py_XDECREF(%(out)s);
                %(fail)s;
            }
            Py_XDECREF(out_sub);
            Py_XDECREF(start);
            start = stop;
            stop = NULL;
            """ % locals()

        str += """
            Py_XDECREF(start);
            Py_XDECREF(stop);
        """
        return str

    def c_code_cache_version(self):
        return (6,)

gpu_join = GpuJoin()


class GpuSplit(tensor.Split, GpuOp):
    def make_node(self, x, axis, splits):
        x = as_cuda_ndarray_variable(x)
        node = tensor.Split.make_node(self, x, axis, splits)
        outs = [CudaNdarrayType(dtype=o.dtype,
                                broadcastable=o.type.broadcastable)()
                for o in node.outputs]
        return Apply(self, [x] + node.inputs[1:], outs)


class GpuAllocEmpty(GpuOp):
    """
    Implement Alloc on the gpu, but without initializing memory.

    """

    __props__ = ()

    @staticmethod
    def validate_shape(shape):
        sh = [tensor.as_tensor_variable(s) for s in shape]
        bcast = []
        for s in sh:
            if s.type.dtype[:3] not in ('int', 'uin'):
                raise TypeError('Shape arguments must be integers', s)
            # if s is constant 1, then we're broadcastable in that dim
            try:
                const_shp = tensor.get_scalar_constant_value(s)
            except tensor.NotScalarConstantError:
                const_shp = None
            bcast.append(1 == const_shp)
        otype = CudaNdarrayType(dtype='float32', broadcastable=bcast)
        output = otype()
        return sh, output

    def make_node(self, *shape):
        shape, output = self.validate_shape(shape)
        output.tag.values_eq_approx = tensor.type.values_eq_approx_always_true
        # The outut can contain nan/inf.  output.type is a new
        # instance, so we can do this only for that variable.
        output.type.filter_checks_isfinite = False
        output.tag.nan_guard_mode_check = False
        return Apply(self, shape, [output])

    def debug_perform(self, node, inputs, out_):
        self.perform(node, inputs, out_)
        # __setitem__ is limited on CudaNdarray
        tmp = numpy.empty(out_[0][0].shape, dtype='float32')
        tmp.fill(-123456789)
        out_[0][0][:] = tmp

    def perform(self, node, inputs, out_):
        out, = out_
        sh = tuple([int(i) for i in inputs])
        if out[0] is None or out[0].shape != sh:
            # XXX: We could implement and call CudaNdarray.empty(sh) instead.
            out[0] = cuda_ndarray.cuda_ndarray.CudaNdarray.zeros(sh)

    def c_code(self, node, name, inputs, out_, sub):
        out, = out_
        fail = sub['fail']
        shps = inputs
        nd = len(shps)
        str = "int dims[%(nd)s];\n" % locals()
        for idx, sh in enumerate(shps):
            str += "dims[%(idx)s] = PyInt_AsLong((PyObject*)%(sh)s);\n" % locals()

        str += "if(%(out)s==NULL\n" % locals()
        for idx, sh in enumerate(shps):
            str += "||CudaNdarray_HOST_DIMS(%(out)s)[%(idx)s]!=dims[%(idx)s]" % locals()
        str += """){
            Py_XDECREF(%(out)s);
            %(out)s = (CudaNdarray*)CudaNdarray_New();
            if (!%(out)s)
            {
                // exception already set
                %(fail)s;
            }
            if (CudaNdarray_alloc_contiguous(%(out)s, %(nd)s, dims))
            {
                // exception already set
                Py_XDECREF(%(out)s);
                %(out)s = NULL;
                %(fail)s;
            }
        }
        """ % locals()
        return str

    def infer_shape(self, node, input_shapes):
        return [node.inputs]

    def c_code_cache_version(self):
        return (1,)

    def do_constant_folding(self, node):
        return False

gpu_alloc_empty = GpuAllocEmpty()


class GpuAlloc(GpuAllocEmpty):
    """
    Implement Alloc on the gpu.

    The memset_0 param is an optimization. When True, we call
    cudaMemset that is faster.

    """

    __props__ = ('memset_0',)

    def __init__(self, memset_0=False):
        self.memset_0 = memset_0

    def __str__(self):
        # Hide the memset parameter when not used to prevent confusion.
        if self.memset_0:
            s = "%s{memset_0=%s}" % (self.__class__.__name__, self.memset_0)
        else:
            s = self.__class__.__name__
        return s

    def make_node(self, value, *shape):
        # if there is unneeded transfert generated by the next line
        # the optimizer will remove them.
        v = as_cuda_ndarray_variable(value)
        shape, output = self.validate_shape(shape)
        return Apply(self, [v] + shape, [output])

    # This is required because the superclass (GpuAllocEmpty) also has it.
    def debug_perform(self, node, inputs, out_):
        self.perform(node, inputs, out_)

    def perform(self, node, inputs, out_):
        # the super class (GpuAllocEmpty) allocates memory, we fill it
        value = inputs[0]
        shps = inputs[1:]
        super(GpuAlloc, self).perform(node, shps, out_)
        out, = out_
        out[0][...] = value  # broadcast value to fill us up

    def c_code(self, node, name, inputs, out_, sub):
        # the super class (GpuAllocEmpty) allocates memory, we fill it
        value = inputs[0]
        shps = inputs[1:]
        str = super(GpuAlloc, self).c_code(node, name, shps, out_, sub)
        out, = out_
        fail = sub['fail']
        memset_0 = int(self.memset_0)
        str += """
        if (%(memset_0)s && CudaNdarray_is_c_contiguous(%(out)s))
        {
            cudaError_t err = cudaMemset(%(out)s->devdata, 0,
                                         CudaNdarray_SIZE(%(out)s) * 4);
            if (cudaSuccess != err)
            {
                PyErr_Format(PyExc_MemoryError,
                             "GpuAlloc: Error memsetting %%ld"
                             " bytes of device memory. %%s",
                             (long)(CudaNdarray_SIZE(%(out)s) * 4),
                             cudaGetErrorString(err));
                Py_XDECREF(%(out)s);
                %(out)s = NULL;
                %(fail)s;
            }
        }
        else if (CudaNdarray_CopyFromCudaNdarray(%(out)s, %(value)s, true))
        {
            // exception already set
            Py_XDECREF(%(out)s);
            %(out)s = NULL;
            %(fail)s;
        }
        """ % locals()
        return str

    def infer_shape(self, node, input_shapes):
        return [node.inputs[1:]]

    def c_code_cache_version(self):
        parent_version = super(GpuAlloc, self).c_code_cache_version()
        return (parent_version, 10)

    def do_constant_folding(self, node):
        for client in node.outputs[0].clients:
            if client[0] == 'output':
                # If the output is a constant, it will have to be deepcopied
                # each time the function is called.  So we do not fold.
                return False
            elif (  # The following ops work inplace of their input id 0.
                  client[1] == 0 and
                  isinstance(client[0].op, (
                    # Ops that will work inplace on the Alloc. So if they
                    # get constant_folded, they would copy the
                    # constant and this is less efficients.

                    # Not doing the constant folding could also lower
                    # the peak memory usage, as we the "constant" won't
                    # always exists.
                      # theano.tensor.subtensor.AdvancedIncSubtensor,
                      GpuIncSubtensor,
                      GpuAdvancedIncSubtensor1,
                      theano.sandbox.cuda.blas.GpuGemm,
                      theano.sandbox.cuda.blas.GpuGemv,
                      theano.sandbox.cuda.blas.GpuGer,
                  ))):
                return False
            # If the clients is a transfer, we don't want to fold. We
            # let the moving opt finish before deciding what to do.
            elif isinstance(client[0].op, HostFromGpu):
                return False
        return True

gpu_alloc = GpuAlloc()


class CopyOnNegativeStrides(GpuOp):
    """
    Checks if the input has contains negative strides.
    
    If it does, returns a c contiguous copy.

    """

    view_map = {0: [0]}
    check_input = False
    __props__ = ()

    def grad(self, inputs, dout):

        x, = inputs
        dout, = dout
        dout = as_cuda_ndarray_variable(dout)

        return [dout]

    def make_node(self, input):
        input = as_cuda_ndarray_variable(input)
        return Apply(self, [input], [input.type()])

    def perform(self, node, inp, out):
        i = inp[0]
        if any(s < 0 for s in i.strides):
            i = i.copy()
        out[0][0] = i

    def infer_shape(self, node, xshp):
        return xshp

    def c_code(self, node, name, inp, out, sub):
        input, = inp
        z, = out
        fail = sub['fail']
        str = """
        {
            bool strides_all_positive = true;
            for (int i = 0; i < CudaNdarray_NDIM(%(input)s); i++){
                if (CudaNdarray_HOST_STRIDES(%(input)s)[i] < 0){
                    strides_all_positive = false;
                    break;
                }
            }
            if (strides_all_positive){
                Py_XDECREF(%(z)s);
                %(z)s = %(input)s;
                Py_INCREF(%(z)s);

            } else if ((NULL == %(z)s)""" % locals()
        for i in xrange(node.inputs[0].type.ndim):
            str += "\n|| (CudaNdarray_HOST_DIMS(%(input)s)[%(i)s] != CudaNdarray_HOST_DIMS(%(z)s)[%(i)s])" % locals()
        str += """
                || !CudaNdarray_is_c_contiguous(%(z)s))
            {
                Py_XDECREF(%(z)s);
                %(z)s = (CudaNdarray*)CudaNdarray_Copy(%(input)s);
                if (!%(z)s)
                {
                    %(fail)s;
                }
            }else if(CudaNdarray_CopyFromCudaNdarray(%(z)s,%(input)s)){
                %(fail)s;
            }
        }
        """ % locals()
        return str

    def c_code_cache_version(self):
        return (0,)

cp_on_negative_strides = CopyOnNegativeStrides()


class GpuContiguous(GpuOp):
    """
    Always return a c contiguous output. Copy the input only if it is
    not already c contiguous.

    """

    view_map = {0: [0]}
    check_input = False

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def grad(self, inputs, dout):

        x, = inputs
        dout, = dout
        dout = as_cuda_ndarray_variable(dout)

        return [dout]

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, input):
        input = as_cuda_ndarray_variable(input)
        return Apply(self, [input], [input.type()])

    def perform(self, node, inp, out):
        i = inp[0]
        if not i.is_c_contiguous():
            i = i.copy()
        assert i.is_c_contiguous()
        out[0][0] = i

    def infer_shape(self, node, xshp):
        return xshp

    def c_code(self, node, name, inp, out, sub):
        input, = inp
        z, = out
        fail = sub['fail']
        str = """
        {
            if (CudaNdarray_is_c_contiguous(%(input)s)){
                Py_XDECREF(%(z)s);
                %(z)s = %(input)s;
                Py_INCREF(%(z)s);

            } else if ((NULL == %(z)s)""" % locals()
        for i in xrange(node.inputs[0].type.ndim):
            str += "\n|| (CudaNdarray_HOST_DIMS(%(input)s)[%(i)s] != CudaNdarray_HOST_DIMS(%(z)s)[%(i)s])" % locals()
        str += """
                || !CudaNdarray_is_c_contiguous(%(z)s))
            {
                Py_XDECREF(%(z)s);
                %(z)s = (CudaNdarray*)CudaNdarray_Copy(%(input)s);
                if (!%(z)s)
                {
                    %(fail)s;
                }
            }else if(CudaNdarray_CopyFromCudaNdarray(%(z)s,%(input)s)){
                %(fail)s;
            }
        }
        """ % locals()
        return str

    def c_code_cache_version(self):
        return (2,)

gpu_contiguous = GpuContiguous()


# Those are predifined CudaNdarrayType as done in tensor.basic
# Useful mostly for test as the gpu op are inserted automatically...
def scalar(name=None, dtype=None):
    """
    Return a symbolic scalar variable.

    Parameters
    ----------
    dtype 
        Numeric type (None means to use theano.config.floatX).
    name : str
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=())
    return type(name)
fscalar = CudaNdarrayType(dtype='float32', broadcastable=())


def vector(name=None, dtype=None):
    """
    Return a symbolic vector variable.

    Parameters
    ----------
    dtype
        Numeric type (None means to use theano.config.floatX).
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=(False, ))
    return type(name)
fvector = CudaNdarrayType(dtype='float32', broadcastable=(False, ))


def matrix(name=None, dtype=None):
    """
    Return a symbolic matrix variable.

    Parameters
    ----------
    dtype
        Numeric type (None means to use theano.config.floatX).
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=(False, False))
    return type(name)
fmatrix = CudaNdarrayType(dtype='float32', broadcastable=(False, False))


def row(name=None, dtype=None):
    """
    Return a symbolic row variable (ndim=2, broadcastable=[True,False]).

    Parameters
    ----------
    dtype
        Numeric type (None means to use theano.config.floatX).
    name : str
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=(True, False))
    return type(name)
frow = CudaNdarrayType(dtype='float32', broadcastable=(True, False))


def col(name=None, dtype=None):
    """
    Return a symbolic column variable (ndim=2, broadcastable=[False,True]).

    Parameters
    ----------
    dtype
        Numeric type (None means to use theano.config.floatX).
    name : str
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=(False, True))
    return type(name)
fcol = CudaNdarrayType(dtype='float32', broadcastable=(False, True))


def tensor3(name=None, dtype=None):
    """
    Return a symbolic 3-D variable.

    Parameters
    ----------
    dtype
        Numeric type (None means to use theano.config.floatX).
    name : str
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype, broadcastable=(False, False, False))
    return type(name)
ftensor3 = CudaNdarrayType(dtype='float32', broadcastable=(False,) * 3)


def tensor4(name=None, dtype=None):
    """
    Return a symbolic 4-D variable.

    Parameters
    ----------
    dtype
        Numeric type (None means to use theano.config.floatX).
    name : str
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = CudaNdarrayType(dtype=dtype,
                           broadcastable=(False, False, False, False))
    return type(name)
ftensor4 = CudaNdarrayType(dtype='float32', broadcastable=(False,) * 4)


@theano.compile.profilemode.register_profiler_printer
def profile_printer(fct_name, compile_time, fct_call_time, fct_call,
                    apply_time, apply_cimpl, message, outputs_size,
                    other_time):
    if any([x[1].op.__class__.__name__.lower().startswith("gpu")
            for x in apply_time.keys()]):
        local_time = sum(apply_time.values())
        print()
        print('Some info useful for gpu:')

        cpu = 0
        gpu = 0
        trans = 0
        for (_, node), t in iteritems(apply_time):
            if isinstance(node.op.__class__.__name__,
                          (HostFromGpu, GpuFromHost)):
                trans += t
            elif node.op.__class__.__name__.lower().startswith("gpu"):
                gpu += t
            else:
                cpu += t
        print()
        print("    Spent %.3fs(%.3f%%) in cpu Op, %.3fs(%.3f%%) in gpu Op and %.3fs(%.3f%%) transfert Op" % (
            cpu, cpu / local_time * 100, gpu, gpu / local_time * 100,
            trans, trans / local_time * 100))

        print()
        print("    Theano function input that are float64")
        print("    <fct name> <input name> <input type> <str input>")
        for fct in fct_call:
            for i in fct.input_storage:
                if hasattr(i.type, 'dtype') and i.type.dtype == 'float64':
                    print('        ', fct.name, i.name, i.type, i)

        print()
        print("    List of apply that don't have float64 as input but have float64 in outputs")
        print("    (Useful to know if we forgot some cast when using floatX=float32 or gpu code)")
        print('    <Apply> <Apply position> <fct name> <inputs type> <outputs type>')
        for fct in fct_call:
            for idx, node in enumerate(fct.maker.fgraph.toposort()):
                if (any(hasattr(i, 'dtype') and i.dtype == 'float64'
                        for i in node.outputs) and
                    not any(hasattr(i, 'dtype') and i.dtype == 'float64'
                            for i in node.inputs)):

                    print('        ', str(node), idx, fct.name, end=' ')
                    print(str([getattr(i, 'dtype', None)
                               for i in node.inputs]), end=' ')
                    print(str([getattr(i, 'dtype', None)
                               for i in node.outputs]))


class GpuEye(GpuOp):

    def __init__(self, dtype=None):
        if dtype is None:
            dtype = config.floatX
        assert dtype == 'float32'
        self.dtype = dtype

    def make_node(self, n, m, k):
        n = tensor.as_tensor_variable(n)
        m = tensor.as_tensor_variable(m)
        k = tensor.as_tensor_variable(k)
        assert n.ndim == 0
        assert m.ndim == 0
        assert k.ndim == 0

        # k != 0 isn't implemented on the GPU yet.
        assert tensor.get_scalar_constant_value(k) == 0
        return Apply(self, [n, m], [matrix(dtype=self.dtype)])

    def infer_shape(self, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i]) for i in xrange(3)]

    def __eq__(self, other):
        return type(self) == type(other) and self.dtype == other.dtype

    def __hash__(self):
        return hash(self.dtype) ^ hash(type(self))

    def c_support_code(self):
        return """
//Only 1 block is used.
__global__ void kEye(float* a, int n, int m) {
    int nb_elem = min(n, m);
    for (unsigned int i = threadIdx.x; i < nb_elem; i += blockDim.x) {
        a[i*m + i] = 1;
    }
}"""

    def c_code(self, node, name, inp, out, sub):
        n, m = inp
        z, = out
        fail = sub['fail']
        s = """
        int dims[] = {0, 0};

        dims[0] = ((dtype_%(n)s*)PyArray_DATA(%(n)s))[0];
        dims[1] = ((dtype_%(m)s*)PyArray_DATA(%(m)s))[0];
        int total_size = dims[0] * dims[1] * sizeof(float);
        cudaError_t sts;
        void * orig_z = %(z)s;

        if (CudaNdarray_prep_output(&%(z)s, 2, dims))
        {
            %(fail)s;
        }

        sts = cudaMemset(CudaNdarray_DEV_DATA(%(z)s), 0, total_size);
        if (cudaSuccess != sts)
        {
            PyErr_Format(PyExc_MemoryError,
                         "GpuEye: Error in memset %%d bytes of device memory.",
                         total_size);
            if(orig_z == NULL)
                Py_XDECREF(%(z)s);
            %(fail)s;
        }

        kEye<<<1, 256>>>(CudaNdarray_DEV_DATA(%(z)s), dims[0], dims[1]);
        CNDA_THREAD_SYNC;

        sts = cudaGetLastError();
        if (cudaSuccess != sts)
        {
               PyErr_Format(PyExc_RuntimeError,
                    "Cuda error: kEye: %%s. n=%%d, m=%%d.",
                    cudaGetErrorString(sts),
                    dims[0], dims[1]);
            %(fail)s;
        }
        """ % locals()

        return s

    def c_code_cache_version(self):
        return (3,)
gpu_eye = GpuEye(dtype='float32')
