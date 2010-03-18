"""
Implementation of MRG31k3p random number generator for Theano

Generator code in SSJ package (L'Ecuyer & Simard)
http://www.iro.umontreal.ca/~simardr/ssj/indexe.html

"""
import sys
import numpy

from theano import Op, Apply, shared, config
from theano.tensor import raw_random, TensorType, as_tensor_variable, get_vector_length

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

    def __eq__(self, other):
        return type(self) == type(other) \
                and self.output_type == other.output_type \
                and self.inplace == other.inplace

    def __hash__(self):
        return hash(type(self)) ^ hash(self.output_type) ^ hash(self.inplace)

    @classmethod
    def new(cls, rstate, ndim, dtype, size, low, high):
        v_size = as_tensor_variable(size)
        if ndim is None:
            ndim = get_vector_length(v_size)
        op = cls(TensorType(dtype, (False,)*ndim))
        return op(rstate, v_size, as_tensor_variable(low), as_tensor_variable(high))

    def make_node(self, rstate, size, low, high):
        return Apply(self, 
                [rstate, size, low, high], 
                [rstate.type(), self.output_type()])
    def perform(self, node, (rstate, size, low, high), (o_rstate, o_sample)):
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
        return self.pretty_return(node_rstate, 
                *mrg_uniform.new(node_rstate, ndim, dtype, size, low, high))

#
#
#
#
#
import theano

def test_rng0():

    R = MRG_RandomStreams(234)

    u = R.uniform(size=(2,2), low=0, high=55)

    f = theano.function([], u)

    print 'random?', f()
    print 'random?', f()

    l = [f() for i in xrange(1000)]

    print 'mean', numpy.mean(l), numpy.std(l) / numpy.sqrt(1000)
