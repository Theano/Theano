"""
Implementation of MRG31k3p random number generator for Theano

Generator code in SSJ package (L'Ecuyer & Simard)
http://www.iro.umontreal.ca/~simardr/ssj/indexe.html

"""
import sys
import numpy

from theano import Op, Apply, shared, config
from theano.tensor import raw_random, TensorType, as_tensor_variable, get_vector_length, cast

def mulmod(a, b, c, m):
    r = numpy.int32(numpy.int64(a*b + c) % m)
    return r if r >= 0 else r+m

def matVecModM(A, s, m):
    # return (A * s) % m
    x = numpy.zeros_like(s)
    for i in xrange(len(x)):
        for j in xrange(len(s)):
            x[i] = mulmod(A[i][j], s[j], x[i], m)
    return x

def multMatVect(v, A, m1, B, m2):
   #multiply the first half of v by A with a modulo of m1
   #and the second half by B with a modulo of m2
   r = numpy.zeros_like(v)
   r[:3] = matVecModM(A, v[:3], m1)
   r[3:] = matVecModM(B, v[3:], m2)
   return r

#MRG31k3p
#generator constants :
M1 = numpy.int32(2147483647)    #2^31 - 1
M2 = numpy.int32(2147462579)    #2^31 - 21069
MASK12 = numpy.int32(511)       #2^9 - 1
MASK13 = numpy.int32(16777215)  #2^24 - 1
MASK2 = numpy.int32(65535)      #2^16 - 1
MULT2 = numpy.int32(21069)
NORM = 4.656612873077392578125e-10;

A1p0 = numpy.asarray([[0, 4194304, 129], [1, 0, 0], [0, 1, 0]])
A2p0 = numpy.asarray([[32768, 0, 32769], [1, 0, 0], [0, 1, 0]])

A1p72 = numpy.asarray([[1516919229, 758510237, 499121365],
       [1884998244, 1516919229, 335398200],
       [601897748, 1884998244, 358115744]]) 
A2p72 = numpy.asarray([[1228857673, 1496414766, 954677935],
   [1133297478, 1407477216, 1496414766],
   [2002613992, 1639496704, 1407477216]])

A1p134 = numpy.asarray(
  [[1702500920, 1849582496, 1656874625],
   [828554832, 1702500920, 1512419905],
   [1143731069, 828554832, 102237247]])
A2p134 = numpy.asarray(
  [[796789021, 1464208080, 607337906],
   [1241679051, 1431130166, 1464208080],
   [1401213391, 1178684362, 1431130166]])

def ff_2p134(rstate):
    return multMatVect(rstate, A1p134, M1, A2p134, M2)

def ff_2p72(rstate):
    return multMatVect(rstate, A1p72, M1, A2p72, M2)

def mrg_next_value(rstate, new_rstate):
    x11, x12, x13, x21, x22, x23 = rstate
    assert type(x11) == numpy.int32

    i0, i7, i9, i15, i16, i22, i24 = [numpy.int32(i) for i in (0,7, 9, 15, 16, 22, 24)]

    #first component
    y1 = ((x12 & MASK12) << i22) + (x12 >> i9) + ((x13 & MASK13) << i7) + (x13 >> i24);

    assert type(y1) == numpy.int32
    if (y1 < 0 or y1 >= M1):     #must also check overflow
        y1 -= M1;
    y1 += x13;
    if (y1 < 0 or y1 >= M1):
        y1 -= M1;

    x13 = x12;
    x12 = x11;
    x11 = y1;

    #second component
    y1 = ((x21 & MASK2) << i15) + (MULT2 * (x21 >> i16));
    assert type(y1) == numpy.int32
    if (y1 < 0 or y1 >= M2):
        y1 -= M2;
    y2 = ((x23 & MASK2) << i15) + (MULT2 * (x23 >> i16));
    assert type(y2) == numpy.int32
    if (y2 < 0 or y2 >= M2):
        y2 -= M2;
    y2 += x23;
    if (y2 < 0 or y2 >= M2):
        y2 -= M2;
    y2 += y1;
    if (y2 < 0 or y2 >= M2):
        y2 -= M2;

    x23 = x22;
    x22 = x21;
    x21 = y2;

    # Must never return either 0 or M1+1
    new_rstate[...] = [x11, x12, x13, x21, x22, x23]
    assert new_rstate.dtype == numpy.int32
    if (x11 <= x21):
        return (x11 - x21 + M1) * NORM
    else:
        return (x11 - x21) * NORM

class mrg_uniform(Op):
    def __init__(self, output_type, inplace=False):
        self.output_type = output_type
        self.inplace=inplace
        if inplace:
            self.destroy_map = {0:[0]}

    def __eq__(self, other):
        return type(self) == type(other) \
                and self.output_type == other.output_type \
                and self.inplace == other.inplace

    def __hash__(self):
        return hash(type(self)) ^ hash(self.output_type) ^ hash(self.inplace)

    @classmethod
    def new(cls, rstate, ndim, dtype, size):
        v_size = as_tensor_variable(size)
        if ndim is None:
            ndim = get_vector_length(v_size)
        op = cls(TensorType(dtype, (False,)*ndim))
        return op(rstate, cast(v_size, 'int32'))

    def make_node(self, rstate, size):
        return Apply(self, 
                [rstate, size], 
                [rstate.type(), self.output_type()])
    def perform(self, node, (rstate, size), (o_rstate, o_sample)):
        n_elements = 1
        if not self.inplace:
            rstate = rstate.copy()

        for s in size:
            n_elements *= s

        n_streams,_ = rstate.shape

        rval = numpy.zeros(n_elements, dtype=self.output_type.dtype)

        for i in xrange(n_elements):
            sample = mrg_next_value(rstate[i%n_streams], rstate[i%n_streams])
            rval[i] = sample

        o_rstate[0] = rstate.copy()
        o_sample[0] = rval.reshape(size)

    def c_code_cache_version(self):
        return ()

    def c_code(self, node, name, (rstate, size), (o_rstate, o_sample), sub):
        if self.inplace:
            o_rstate_requirement = 'NPY_C_CONTIGUOUS|NPY_ALIGNED'
        else:
            o_rstate_requirement = 'NPY_ENSURECOPY|NPY_C_CONTIGUOUS|NPY_ALIGNED' 
        ndim = self.output_type.ndim
        o_type_num = numpy.asarray(0, dtype=self.output_type.dtype).dtype.num
        fail = sub['fail']
        if self.output_type.dtype == 'float32':
            otype = 'float' 
            NORM = '4.6566126e-10f' #numpy.float32(1.0/(2**31+65))
            # this was determined by finding the biggest number such that
            # numpy.float32(number * M1) < 1.0
        else:
            otype = 'double' 
            NORM = '4.656612873077392578125e-10'
        return """
        //////// <code generated by mrg_uniform>

        npy_intp odims[%(ndim)s];
        int n_elements = 1;
        int n_streams = 0;
        int must_alloc_sample = ((NULL == %(o_sample)s) || (%(o_sample)s->nd != %(ndim)s));
        %(otype)s * sample_data;
        npy_int32 * state_data;

        const npy_int32 i0 = 0;
        const npy_int32 i7 = 7;
        const npy_int32 i9 = 9;
        const npy_int32 i15 = 15;
        const npy_int32 i16 = 16;
        const npy_int32 i22 = 22;
        const npy_int32 i24 = 24;

        const npy_int32 M1 = 2147483647;      //2^31 - 1
        const npy_int32 M2 = 2147462579;      //2^31 - 21069
        const npy_int32 MASK12 = 511;       //2^9 - 1
        const npy_int32 MASK13 = 16777215;  //2^24 - 1
        const npy_int32 MASK2 = 65535;      //2^16 - 1
        const npy_int32 MULT2 = 21069;

        if (%(size)s->nd != 1)
        {
            PyErr_SetString(PyExc_ValueError, "size must be vector");
            %(fail)s
        }
        if (%(size)s->dimensions[0] != %(ndim)s)
        {
            PyErr_Format(PyExc_ValueError, "size must have length %%i", %(ndim)s);
            %(fail)s
        }
        if (%(size)s->descr->type_num != PyArray_INT32)
        {
            PyErr_SetString(PyExc_ValueError, "size must be int32");
            %(fail)s
        }
        for (int i = 0; i < %(ndim)s; ++i)
        {
            odims[i] = ((npy_int32*)(%(size)s->data + %(size)s->strides[0] * i))[0];
            n_elements *= odims[i];
            must_alloc_sample = must_alloc_sample || (%(o_sample)s->dimensions[i] != odims[i]);
            //fprintf(stderr, "size %%i %%i\\n", i, (int)odims[i]);
            // TODO CHECK STRIDES OF o_sample?
        }
        if (must_alloc_sample)
        {
            Py_XDECREF(%(o_sample)s);
            %(o_sample)s = (PyArrayObject*)PyArray_SimpleNew(%(ndim)s, odims, %(o_type_num)s);
            if(!%(o_sample)s) {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc mrg_uniform output");
                %(fail)s
            }
        }
        Py_XDECREF(%(o_rstate)s);
        %(o_rstate)s = (PyArrayObject*)PyArray_FromAny(py_%(rstate)s, NULL, 0, 0, %(o_rstate_requirement)s,NULL);

        if (%(o_rstate)s->nd != 2)
        {
            PyErr_SetString(PyExc_ValueError, "rstate must be matrix");
            %(fail)s
        }
        if (%(o_rstate)s->dimensions[1] != 6)
        {
            PyErr_Format(PyExc_ValueError, "rstate must have 6 columns");
            %(fail)s
        }
        if (%(o_rstate)s->descr->type_num != PyArray_INT32)
        {
            PyErr_SetString(PyExc_ValueError, "rstate must be int32");
            %(fail)s
        }
        n_streams = %(o_rstate)s->dimensions[0];

        sample_data = (%(otype)s *) %(o_sample)s->data;
        state_data = (npy_int32 *) %(o_rstate)s->data;
        for (int i = 0; i < n_elements; ++i)
        {
            npy_int32 * state_data_i = state_data + (i%%n_streams)*6;
            npy_int32 y1, y2, x11, x12, x13, x21, x22, x23;

            x11 = state_data_i[0];
            x12 = state_data_i[1];
            x13 = state_data_i[2];
            x21 = state_data_i[3];
            x22 = state_data_i[4];
            x23 = state_data_i[5];

            y1 = ((x12 & MASK12) << i22) + (x12 >> i9) + ((x13 & MASK13) << i7) + (x13 >> i24);
            if ((y1 < 0 || y1 >= M1))     //must also check overflow
                y1 -= M1;
            y1 += x13;
            if ((y1 < 0 or y1 >= M1))
                y1 -= M1;
            x13 = x12;
            x12 = x11;
            x11 = y1;

            y1 = ((x21 & MASK2) << i15) + (MULT2 * (x21 >> i16));
            if (y1 < 0 || y1 >= M2)
                y1 -= M2;
            y2 = ((x23 & MASK2) << i15) + (MULT2 * (x23 >> i16));
            if (y2 < 0 || y2 >= M2)
                y2 -= M2;
            y2 += x23;
            if (y2 < 0 || y2 >= M2)
                y2 -= M2;
            y2 += y1;
            if (y2 < 0 or y2 >= M2)
                y2 -= M2;

            x23 = x22;
            x22 = x21;
            x21 = y2;

            if (x11 <= x21) {
                assert((x11 - x21 + M1) <= M1);
                sample_data[i] = (x11 - x21 + M1) * %(NORM)s;
            }
            else
            {
                assert(x11 - x21 <= M1);
                sample_data[i] = (x11 - x21) * %(NORM)s;
            }

            state_data_i[0]= x11;
            state_data_i[1]= x12;
            state_data_i[2]= x13;
            state_data_i[3]= x21;
            state_data_i[4]= x22;
            state_data_i[5]= x23;
        }
        //////// </ code generated by mrg_uniform>
        """ %locals()

class MRG_RandomStreams(object):
    """Module component with similar interface to numpy.random (numpy.random.RandomState)"""

    def __init__(self, seed=None):
        """
        :type seed: None or int

        :param seed: a default seed to initialize the RandomState instances after build.  See
        `RandomStreamsInstance.__init__` for more details.
        """
        super(MRG_RandomStreams, self).__init__()
        self.rstate = numpy.asarray([12345]*6, dtype='int32')

    def inc_rstate(self):
        """Update self.rstate to be skipped 2^134 steps forward to the next stream start"""
        self.rstate = ff_2p134(self.rstate)
        assert self.rstate.dtype == numpy.int32

    def get_substream_rstates(self, n_streams, inc_rstate=True):
        """Initialize a matrix in which each row is a MRG stream state,
        and they are spaced by 2**72 samples.
        """
        assert n_streams < 2**72
        assert n_streams > 0
        rval = numpy.zeros((n_streams,6), dtype='int32')
        rval[0] = self.rstate
        for i in xrange(1, n_streams):
            rval[i] = ff_2p72(rval[i-1])
        if inc_rstate:
            self.inc_rstate()
        return rval

    def n_streams(self, size):
        r = 1
        for s in size:
            r *= s
        return r

    def pretty_return(self, node_rstate, new_rstate, sample):
        sample.rstate = node_rstate
        sample.update = (node_rstate, new_rstate)
        node_rstate.default_update = new_rstate
        return sample


    def uniform(self, size=None, low=0.0, high=1.0, ndim=None, dtype=config.floatX):
        """
        Sample a tensor of given size whose element from a uniform
        distribution between low and high.

        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing
        information.
        """
        node_rstate = shared(self.get_substream_rstates(self.n_streams(size)))
        u = self.pretty_return(node_rstate, 
                *mrg_uniform.new(node_rstate, ndim, dtype, size))
        r = u * (high-low) + low
        
        if u.type.broadcastable != r.type.broadcastable:
            raise NotImplementedError( 'Increase the size to match the broadcasting pattern of `low` and `high` arguments')
        return  r

#
#
#
#
#
import time
import theano

def test_rng0():

    def basictest(f, steps, prefix=""):
        t0 = time.time()
        l = [f() for i in xrange(steps)]
        tt = time.time()

        mean, std, min, max = numpy.mean(l), numpy.std(l), numpy.min(l), numpy.max(l)

        print prefix, 'mean', mean
        print prefix, 'std', std
        print prefix, 'min', repr(min)
        print prefix, 'max', repr(max)
        print prefix, 'samples/sec', steps*sample_size[0]*sample_size[1] / (tt-t0)

        assert max < 1.0
        assert min >= 0.0
        assert abs(mean - 0.5) < .01, 'bad mean?'


    R = MRG_RandomStreams(234)

    sample_size = (200,20)

    u = R.uniform(size=sample_size)
    print "U dtype", u.dtype

    f = theano.function([], u)

    print 'random?', f()[0]
    print 'random?', f()[0]

    basictest(f, 1000, prefix='mrg  ')

    RR = theano.tensor.shared_randomstreams.RandomStreams(234)

    uu = RR.uniform(size=sample_size)
    ff = theano.function([], uu)

    basictest(ff, 1000, prefix='numpy')

