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

def mrg_next_value(rstate):
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
    new_rstate = numpy.asarray([x11, x12, x13, x21, x22, x23])
    assert new_rstate.dtype == numpy.int32
    if (x11 <= x21):
        return (x11 - x21 + M1) * NORM, new_rstate
    else:
        return (x11 - x21) * NORM, new_rstate



class mrg_uniform(Op):
    def __init__(self, output_type):
        self.output_type = output_type
    @classmethod
    def apply(cls, rstate, ndim, dtype, size, low, high):
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
        rstate = rstate.copy()

        for s in size:
            n_elements *= s

        rval = numpy.zeros(n_elements, dtype=self.output_type.dtype)

        for i in xrange(n_elements):
            sample, rstate = mrg_next_value(rstate)
            rval[i] = sample

        o_rstate[0] = rstate.copy()
        o_sample[0] = rval.reshape(size)

class MRG_RandomStreams(raw_random.RandomStreamsBase):
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
        """Skip self.rstate forward to the next stream point"""
        print >> sys.stderr, "TODO: skip forward the state"

    def gen(self, op, *args, **kwargs):
        """Create a new random stream in this container.

        :param op: one of the functions in numpy.raw_random

        :param args: interpreted by `op`

        :param kwargs: interpreted by `op`

        :returns: The symbolic random draw part of op()'s return value.  This function stores
        the updated RandomStateType Variable for use at `build` time.

        :rtype: TensorVariable
        """
        ndim = kwargs.pop('ndim', None)
        dtype = kwargs.pop('dtype', None)
        assert dtype is not None
        node_rstate = shared(self.rstate.copy())
        new_r, sample = globals()['mrg_'+op.__name__].apply(node_rstate, ndim, dtype, *args, **kwargs)
        sample.rstate = node_rstate
        sample.update = (node_rstate, new_r)
        node_rstate.default_update = new_r
        return sample


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

    print 'random sample?', f()
    print 'random sample?', f()
    print 'random sample?', f()
    print 'random sample?', f()
