import StringIO, sys
import numpy

from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar

from .type import CudaNdarrayType
from .type_support import filter as type_support_filter

from .elemwise import NaiveAlgo

import logging, copy
_logger_name = 'theano_cuda_ndarray.basic_ops'
_logger = logging.getLogger(_logger_name)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler()) #TO REMOVE
def warning(*msg):
    _logger.warning(_logger_name+'WARNING: '+' '.join(str(m) for m in msg))
def info(*msg):
    _logger.info(_logger_name+'INFO: '+' '.join(str(m) for m in msg))
def debug(*msg):
    _logger.debug(_logger_name+'DEBUG: '+' '.join(str(m) for m in msg))

def as_cuda_ndarray_variable(x):
    if hasattr(x, '_as_CudaNdarrayVariable'):
        return x._as_CudaNdarrayVariable()
    tensor_x = tensor.as_tensor_variable(x)
    return GpuFromHost()(tensor_x)

class HostFromGpu(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return 'HostFromGpu'
    def make_node(self, x):
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError(x)
        return Apply(self, [x], [tensor.TensorType(dtype=x.dtype, broadcastable=x.broadcastable)()])
    def perform(self, node, (x,), (z,)):
        z[0] = numpy.asarray(x)
    def grad(self, inputs, (gz,)):
        return gz,
        #return [GpuFromHost()(gz)]
host_from_gpu = HostFromGpu()

class GpuFromHost(Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return 'GpuFromHost'
    def make_node(self, x):
        if not isinstance(x.type, tensor.TensorType):
            raise TypeError(x)
        return Apply(self, [x], [CudaNdarrayType(broadcastable=x.broadcastable)()])
    def perform(self, node, (x,), (z,)):
        z[0] = type_support_filter(numpy.asarray(x, dtype='float32'), tuple([0]*x.ndim), 0)
    def grad(self, inputs, (gz,)):
        return gz,
        #return [HostFromGpu()(gz)]
gpu_from_host = GpuFromHost()

class GpuElemwise(Op):
    nin = property(lambda self: self.scalar_op.nin)
    nout = property(lambda self: self.scalar_op.nout)

    def __init__(self, scalar_op, inplace_pattern):
        self.scalar_op = scalar_op
        self.inplace_pattern = inplace_pattern
        self.destroy_map = dict((o, [i]) for o, i in inplace_pattern.items())
        if scalar_op.nin > 0:
            self.ufunc = numpy.frompyfunc(scalar_op.impl, scalar_op.nin, scalar_op.nout)
        else:
            self.ufunc = None
        self._rehash()

        self.src_generator = NaiveAlgo(self.scalar_op)

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        d.pop('ufunc')
        d.pop('__epydoc_asRoutine', None)
        d.pop('_hashval')
        return d
    
    def __setstate__(self, d):
        self.__dict__.update(d)
        if self.scalar_op.nin > 0:
            self.ufunc = numpy.frompyfunc(self.scalar_op.impl, self.scalar_op.nin, self.scalar_op.nout)
        else:
            self.ufunc = None
        self._rehash()

    def __eq__(self, other):
        return type(self) == type(other) and (self.scalar_op == other.scalar_op)

    def _rehash(self):
        items = self.inplace_pattern.items()
        items.sort()
        tuple_items = tuple([k for k,v in items] + [(tuple(v) if isinstance(v, (tuple, list)) else v) for k,v in items])
        h = hash('Elemwise') ^ hash(self.scalar_op) ^ hash(tuple_items)
        assert h == getattr(self,'_hashval', h)
        self._hashval = h

    def __hash__(self):
        return self._hashval

    def __str__(self):
        if 0:
            # TODO:
            # Current implementation does not use inplace pattern
            # although since memory on card is precious... it should!
            if self.inplace_pattern:
                items = self.inplace_pattern.items()
                items.sort()
                return "GpuElemwise{%s}%s" % (self.scalar_op.__class__.__name__, str(items))
        return "GpuElemwise{%s}" % (self.scalar_op.__class__.__name__)

    def make_node(self, *inputs):
        _inputs = [as_cuda_ndarray_variable(i) for i in inputs]
        if self.nin > 0 and len(_inputs) != self.nin:
            raise TypeError('Wrong argument count', (self.nin, len(_inputs)))
        for i in _inputs[1:]:
            if i.type.ndim != inputs[0].type.ndim:
                raise TypeError('different ranks among inputs')

        # output is broadcastable only along dimensions where all inputs are broadcastable
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

    def c_code_cache_version(self):
        return self.src_generator.cache_version

class GpuDimShuffle(Op):
    def __init__(self, input_broadcastable, new_order):
        input_broadcastable = tuple(input_broadcastable)
        self.input_broadcastable = input_broadcastable
        new_order = tuple(new_order)
        self.new_order = new_order

        # list of dimensions of the input to drop
        self.drop = []
        i2j = {} # this maps i before dropping dimensions to j after dropping dimensions so self.shuffle can be set properly later on
        j = 0
        for i, b in enumerate(input_broadcastable):
            if i not in new_order:
                # we want to drop this dimension because it's not a value in new_order
                if b == 1: # 1 aka True
                    self.drop.append(i)
                else:
                    # we cannot drop non-broadcastable dimensions
                    raise ValueError("You cannot drop a non-broadcastable dimension.", (input_broadcastable, new_order))
            else:
                i2j[i] = j
                j += 1

        # transposition of non-broadcastable dimensions
        # This is how the dimensions will be permuted, without accounting for the extra
        # 'x' broadcastable dimensions to insert.
        self.shuffle = [i2j[x] for x in new_order if x != 'x']

        # list of dimensions of the output that are broadcastable and were not in the original input
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
            raise TypeError("The number of dimensions and/or broadcastable pattern of the input is incorrect for this op. Expected %s, got %s." % (self.input_broadcastable, ib))
        ob = []
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
        self._hashval = hash(type(self).__name__) ^ hash(type(self).__module__) \
                ^ hash(self.new_order) ^ hash(self.input_broadcastable)

    def __hash__(self):
        return self._hashval

    def __str__(self):
        return "GpuDimShuffle{%s}" % ",".join(str(x) for x in self.new_order)

    def c_code(self, node, name, (input,), (res,), sub):
        basename = input + '__view_or_copy'

        nd_in = len(self.input_broadcastable)
        nd_out = len(self.new_order)
        sio = StringIO.StringIO()
        fail = sub['fail']

        #check input
        print >> sio, """
        if (cnda_%(input)s->nd != %(nd_in)s)
        {
            PyErr_Format(PyExc_TypeError, "required nd=%(nd_in)s, got nd=%%i", cnda_%(input)s->nd);
            %(fail)s;
        }
        """ %locals()

        #alloc an output
        print >> sio, """
        if (cnda_%(res)s && (cnda_%(res)s->nd == %(nd_out)s))
        {
            //re-use previously-allocated cnda
        }
        else
        {
            if (cnda_%(res)s)
            {
                if (CudaNdarray_set_nd(cnda_%(res)s, %(nd_out)s))
                {
                    Py_DECREF(cnda_%(res)s);
                    cnda_%(res)s = NULL;
                    %(fail)s;
                }
            }
            else
            {
                cnda_%(res)s = (CudaNdarray*) CudaNdarray_New(%(nd_out)s);
                if (NULL == cnda_%(res)s)
                {
                    %(fail)s;
                }
            }
        }
        """ %locals()

        print >> sio, """
        if (CudaNdarray_set_device_data(cnda_%(res)s, CudaNdarray_DEV_DATA(cnda_%(input)s), cnda_%(input)s))
        {
            // err message set
            Py_DECREF(cnda_%(res)s);
            cnda_%(res)s = NULL;
            %(fail)s;
        }
        """ %locals()

        #reassign the dimension and strides in the host pointers
        for i, o in enumerate(self.new_order):
            if o == 'x':
                assert node.outputs[0].type.broadcastable[i]
                print >> sio, """
        CudaNdarray_set_dim(cnda_%(res)s, %(i)s, 1);
        CudaNdarray_set_stride(cnda_%(res)s, %(i)s, 0);
                """ %locals()
            else:
                assert not node.outputs[0].type.broadcastable[i]
                print >> sio, """
        CudaNdarray_set_dim(cnda_%(res)s, %(i)s, CudaNdarray_HOST_DIMS(cnda_%(input)s)[%(o)s]);
        CudaNdarray_set_stride(cnda_%(res)s, %(i)s, CudaNdarray_HOST_STRIDES(cnda_%(input)s)[%(o)s]);
                """ %locals()

        for i, o in enumerate(self.new_order):
                print >> sio, """
        //std::cerr << "GpuDimShuffle " << cnda_%(res)s << " str[%(i)s] = " << cnda_%(res)s->str[%(i)s] << "\\n";
                """ %locals()

        # copy the host dims and stride -> device
        if 0:
            print >> sio, """
            if (CudaNdarray_copy_structure_to_device(cnda_%(res)s))
            {
                //err msg set
                Py_DECREF(cnda_%(res)s);
                cnda_%(res)s = NULL;
                %(fail)s;
            }
            """ %locals()

        if 0: # print full code to stdout
            print '--------------------------------------'
            print 'C_CODE'
            print ''
            print self
            print "IN BROAD", self.input_broadcastable
            print "NEW ORDER", self.new_order
            print "SHUFFLE", self.shuffle
            print "AUGMENT", self.augment
            print '------------'
            print ''
            print sio.getvalue()
            print '--------------------------------------'
            if 0:
                import sys
                sys.exit()

        return sio.getvalue()
    
    def c_code_cache_version(self):
        return (1,0)

class GpuSum(Op):
    def __init__(self, reduce_mask):
        self.reduce_mask = tuple(reduce_mask)

    def __eq__(self, other):
        return type(self) == type(other) and self.reduce_mask == other.reduce_mask

    def __hash__(self):
        return hash(type(self)) ^ hash(self.reduce_mask)

    def __str__(self):
        return "GpuSum{%s}" % str(self.reduce_mask)

    def make_node(self, x):
        if (x.type.ndim != len(self.reduce_mask)):
            raise TypeError("x must have rank %i"%len(self.reduce_mask))
        o_broadcast = [x.type.broadcastable[i] for i in xrange(x.type.ndim) if not self.reduce_mask[i]]
        return Apply(self, [x], [CudaNdarrayType(o_broadcast)()])

    def perform(self, node, (x,), (z,)):
        z[0] = x.reduce_sum(self.reduce_mask)

class GpuReshape(tensor.Reshape):
    # __hash__, __eq__, __str__ come from tensor.Subtensor
    def make_node(self, x, shp):
        return Apply(self, [x, shp], [CudaNdarrayType([False]*self.ndim)()])
    def perform(self, node, (x, shp), (out,)):
        if (len(shp) != self.ndim):
            raise ValueError('shape argument to Reshape.perform has incorrect length %i'
                    ', should be %i' % (len(shp), self.ndim), shp)
        out[0] = x.reshape(tuple(shp))

class GpuSubtensor(tensor.Subtensor):
    # __hash__, __eq__, __str__ come from tensor.Subtensor
    def make_node(self, x, *inputs):
        rval = tensor.Subtensor.make_node(self, x, *inputs)
        rval.inputs[0] = x # clobber the 'astensor'
        rval.outputs[0].type = CudaNdarrayType(rval.outputs[0].type.broadcastable)
        return rval

    def perform(self, node, inputs, (out, )):
        x = inputs[0]
        indices = list(reversed(inputs[1:]))

        def convert(entry):
            if isinstance(entry, Type):
                return indices.pop()
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

    def old_perform(self, node, inputs, (out, )):
        indices = list(reversed(inputs[1:]))

        def convert(entry):
            if isinstance(entry, Type):
                return indices.pop()
            elif isinstance(entry, slice):
                return slice(convert(entry.start),
                             convert(entry.stop),
                             convert(entry.step))
            else:
                return entry

        x = inputs[0].view()
        out[0] = x
        #todo; when this works, put it into CudaNdarray.__getitem__
        #      (sequence protocol)
        x_shape = x.shape
        x_strides = x._strides
        offset = 0
        for i, thing in enumerate(map(convert, self.idx_list)):
            if isinstance(thing, int):
                #this requires reducing the rank of the 
                # view....
                raise NotImplementedError()

            if isinstance(thing, slice):
                #stride
                if thing.step is None:
                    stride = 1
                else:
                    stride = thing.step

                #start
                if thing.start is None:
                    if stride > 0:
                        start = 0
                    else:
                        start = x_shape[i]-1
                else:
                    if thing.start < 0:
                        start = x_shape[i] - thing.start
                    else:
                        start = thing.start

                #stop
                if thing.stop is None:
                    if stride > 0:
                        stop = x_shape[i]
                    else:
                        stop = -1
                else:
                    if thing.stop < 0:
                        stop = x_shape[i] - thing.stop
                    else:
                        stop = thing.stop

                newlen = (stop - start) // stride
                offset += x_strides[i] * start
                debug('GpuSubtensor slice', i, ': ', start, stop, stride)
                debug('GpuSubtensor shape', i, ': ', x_shape[i], newlen)
                x._set_shape_i(i, newlen)
                x._set_stride(i, x_strides[i] * stride)

            #print 'perform', id(x), x.shape, i, thing
        sizeof_float = 4
        x._dev_data += offset * sizeof_float
        #sys.stdout.flush()
        #sys.exit()

class GpuShape(tensor.Shape):
    def make_node(self, x):
        return Apply(self, [x], [tensor.lvector()])
gpu_shape = GpuShape()

