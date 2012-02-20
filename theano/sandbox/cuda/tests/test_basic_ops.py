import sys, time, unittest

from theano.compile.pfunc import pfunc
from theano import tensor

import numpy
import theano
import theano.tensor as T

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

import theano.sandbox.cuda as tcn
import theano.sandbox.cuda as cuda
import theano.sandbox.cuda.basic_ops as B
from theano.tensor.basic import _allclose
from theano.tests import unittest_tools as utt

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')


def rand_cuda_ndarray(shape):
    return cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),
                                                    dtype='float32'))


#intentionally disabled
def tes_use():
    tcn.use()


def test_sum():
    """
    test sum pattern 1, 11, 10, 01, 001, 010, 100, 110, 011, 111,
    0011, 0101, 0111, 1011, 1111

    test sum pattern implemented with reshape:
    1000, 0100, 0010, 0001, 11111

    others implemented by reshape that are not tested
    0011,0101,0110,1001,1010,1100
    1110,1101,1011

    TODO: test with broadcast
    """
    for shape, pattern in [((100,3,1300),[1]),
                           ((0,),[0]),((5,),[0]),
                           ((0,0),[0,1]),((1,0),[0,1]),((5,4),[0,1]),((33,31),[0,1]),((5,4),[1]),((5,4),[0]),#need something bigger then 32 for some opt test.
                           ((5,4,3),[0]),((5,4,3),[1]),((5,4,3),[0,1]),((5,4,3),[2]),((5,4,3),[1,2]),((5,4,3),[0,1,2]),
                           ((0,0,0,0),[0,1,2,3]),
                           ((5,4,3,20),[2,3]), ((5,4,3,2),[0,1,2,3]), ((5,4,3,2),[0,2,3]),((5,4,3,2),[1,2,3]),
                           ((5,4,3,10,11),[1,2]),
                           ((5,4,3,20),[2,3]), ((5,4,3,2),[0,1,2,3]), ((5,4,3,2),[0,2,3]),((5,4,3,2),[1,2,3]),

                           #test shape bigger then 4096 on each dimension to make sure that we work correctly when we don't have enought thread/block in each dimensions
                           ((4100,3),[0]),((3,4101),[0]),#10
                           ((1024,33),[0]),((33,1024),[0]),#10
                           ((1025,33),[0]),((33,1025),[0]),#10

                           ((4100,3),[1]),((3,4101),[1]),#01
                           ((1024,33),[1]),((33,1024),[1]),#01
                           ((1025,33),[1]),((33,1025),[1]),#01

                           ((4100,3),[0,1]),((3,4101),[0,1]),#11
                           ((1024,33),[0,1]),((33,1024),[0,1]),#01
                           ((1025,33),[0,1]),((33,1025),[0,1]),#01

                           ((4100,4,3),[0]),((5,4100,3),[0]),((5,4,4100),[0]),#100
                           ((4100,4,3),[1]),((5,4100,3),[1]),((5,4,4100),[1]),#010
                           ((4100,4,3),[2]),((5,4100,3),[2]),((5,4,4100),[2]),#001
                           ((4100,4,3),[0,1]),((5,4100,3),[0,1]),((5,4,4100),[0,1]),#110
                           ((4100,4,3),[1,2]),((5,4100,3),[1,2]),((5,4,4100),[1,2]),#011
                           #((4100,4,3),[0,2]),((5,4100,3),[0,2]),((5,4,4100),[0,2]),#101 ##not implemented
                           ((4100,4,3),[0,1,2]),((5,4100,3),[0,1,2]),((5,4,4100),[0,1,2]),#111

                           ((4100,4,3,2),[2,3]),((4,4100,3,2),[2,3]),((4,3,4100,2),[2,3]),((4,3,2,4100),[2,3]),#0011
                           ((4100,4,3,2),[1,3]),((4,4100,3,2),[1,3]),((4,3,4100,2),[1,3]),((4,3,2,4100),[1,3]),#0101
                           ((4100,4,3,2),[0,2,3]),((4,4100,3,2),[0,2,3]),((4,3,4100,2),[0,2,3]),#((4,3,2,4100),[0,2,3]),#1011
                           ((4100,4,3,2),[1,2,3]),((4,4100,3,2),[1,2,3]),((4,3,4100,2),[1,2,3]),((4,3,2,4100),[1,2,3]),#0111
                           ((4100,2,3,4),[0,1,2,3]),((2,4100,3,4),[0,1,2,3]),((2,3,4100,4),[0,1,2,3]),((2,3,4,4100),[0,1,2,3]),#1111


                           #test pattern implemented by reshape
                           ((4100,4,3,2),[0]),((4,4100,3,2),[0]),((4,3,4100,2),[0]),((4,3,2,4100),[0]),#1000
                           ((4100,4,3,2),[1]),((4,4100,3,2),[1]),((4,3,4100,2),[1]),((4,3,2,4100),[1]),#0100
                           ((4100,4,3,2),[2]),((4,4100,3,2),[2]),((4,3,4100,2),[2]),((4,3,2,4100),[2]),#0010
                           ((4100,4,3,2),[3]),((4,4100,3,2),[3]),((4,3,4100,2),[3]),((4,3,2,4100),[3]),#0001
                           ((1100,2,3,4,5),[0,1,2,3,4]),((2,1100,3,4,5),[0,1,2,3,4]),((2,3,1100,4,5),[0,1,2,3,4]),((2,3,4,1100,5),[0,1,2,3,4]),((2,3,4,5,1100),[0,1,2,3,4]),#11111

                           ]:
        a = tensor.TensorType('float32', (False,) * len(shape))()
        b = T.Sum(pattern)(a)
        val = numpy.random.rand(numpy.prod(shape)).reshape(shape)
#        val = numpy.ones(shape)
#        val = numpy.arange(numpy.prod(shape)).reshape(shape)
        val = theano._asarray(val, dtype='float32')
        f = theano.function([a], b, mode=mode_with_gpu)
        f2 = theano.function([a], b, mode=mode_without_gpu)
        assert tcn.GpuSum in [x.op.__class__ for x in f.maker.env.toposort()]
        assert T.Sum in [x.op.__class__ for x in f2.maker.env.toposort()]
        if val.size == 0:
            assert f2(val) == f(val), ('shape', shape, 'pattern', pattern)
        else:
            try:
                #We raise the error threashold as we sum big matrix
                #and this cause small rounding difference with some seed
                #example in debug mode with unittests.rseed=9275
                orig_rtol = theano.tensor.basic.float32_rtol
                theano.tensor.basic.float32_rtol = 2e-5
                assert _allclose(f2(val), f(val)), ('shape', shape,
                                                    'pattern', pattern,
                                                    sum([shape[i] for i in pattern]))
            finally:
                theano.tensor.basic.float32_rtol = orig_rtol


        #test with dimshuffle
        #we shuffle the 2 outer dims.
    for shape, pattern in [#((5,),[0]),
                           ((5,4),[0,1]),((5,4),[0]),
                           ((5,4,3),[0]),((5,4,3),[0,1]),((5,4,3),[2]),((5,4,3),[0,1,2]),
                           ((5,4,3,2),[0,1,2,3]), ((5,4,3,2),[0,2,3])]:
        a = tensor.TensorType('float32', (False,) * len(shape))()
        dim_pattern = range(len(shape))
        dim_pattern[0] = 1
        dim_pattern[1] = 0
        a = a.dimshuffle(dim_pattern)
        b = T.Sum(pattern)(a)
        val = numpy.random.rand(numpy.prod(shape)).reshape(shape)
#        val = numpy.ones(shape)
#        val = numpy.arange(numpy.prod(shape)).reshape(shape)
        val = theano._asarray(val, dtype='float32')
        f = theano.function([a], b, mode=mode_with_gpu)
        f2 = theano.function([a], b, mode=mode_without_gpu)
        assert tcn.GpuSum in [x.op.__class__ for x in f.maker.env.toposort()]
        assert T.Sum in [x.op.__class__ for x in f2.maker.env.toposort()]
        assert _allclose(f2(val), f(val)), ('shape', shape,
                                            'pattern', pattern,
                                            sum([shape[i] for i in pattern]))


        #test with broadcast
    for shape, pattern in [((5,),[0]),
                           ((5,4),[0,1]),((5,4),[0]),
                           ((5,4,3),[0]),((5,4,3),[0,1]),((5,4,3),[2]),((5,4,3),[0,1,2]),
                           ((5,4,3,2),[0,1,2,3]), ((5,4,3,2),[0,2,3])]:
        shape = numpy.asarray(shape) * 2
        a = tensor.TensorType('float32', (False,) * len(shape))()
        a2 = tcn.CudaNdarrayType((False,) * len(shape))()
        b = T.Sum(pattern)(a)
        b2 = T.Sum(pattern)(a2)
        val = numpy.random.rand(numpy.prod(shape)).reshape(shape)
#        val = numpy.ones(shape)
#        val = numpy.arange(numpy.prod(shape)).reshape(shape)
        val = theano._asarray(val, dtype='float32')
        val2 = cuda.CudaNdarray(val)
        if len(shape) == 1:
            val = val[::2]
            val2 = val2[::2]
        elif len(shape) == 2:
            val = val[::2, ::2]
            val2 = val2[::2, ::2]
        elif len(shape) == 3:
            val = val[::2, ::2, ::2]
            val2 = val2[::2, ::2, ::2]
        elif len(shape) == 4:
            val = val[::2, ::2, ::2, ::2]
            val2 = val2[::2, ::2, ::2, ::2]
        f = theano.function([a], b, mode=mode_without_gpu)
        f2 = theano.function([a2], b2, mode=mode_with_gpu)
        assert tcn.GpuSum in [x.op.__class__ for x in f2.maker.env.toposort()]
        assert T.Sum in [x.op.__class__ for x in f.maker.env.toposort()]
        assert _allclose(f2(val2), f(val)), ('shape', shape,
                                             'pattern', pattern,
                                             sum([shape[i] for i in pattern]))


def test_flatten():
    x = cuda.fmatrix('x')
    f = theano.function([x], x.flatten())
    assert len(f([[0., 0.], [0., 0.]]).shape) == 1


def test_reshape():

    a = tcn.CudaNdarrayType((False,))()
    b = tcn.CudaNdarrayType((False, False))()
    c = T.reshape(a, [2, 3])

    #basic
    f = theano.function([a], c, mode=mode_with_gpu)
    fv = f(cuda_ndarray.CudaNdarray(theano._asarray([0, 1, 2, 3, 4, 5],
                                                    dtype='float32')))
    topo = f.maker.env.toposort()
    assert any([isinstance(node.op, B.GpuReshape) for node in topo])
    assert numpy.all(fv == numpy.asarray([[0, 1, 2], [3, 4, 5]]))

    #test that it works without inplace operations
    a_val = cuda_ndarray.CudaNdarray(theano._asarray([0, 1, 2, 3, 4, 5],
                                                     dtype='float32'))
    a_val_copy = cuda_ndarray.CudaNdarray(theano._asarray([0, 1, 2, 3, 4, 5],
                                                          dtype='float32'))
    b_val = cuda_ndarray.CudaNdarray(theano._asarray([[0, 1, 2], [3, 4, 5]],
                                                     dtype='float32'))

    f_sub = theano.function([a, b], c - b, mode=mode_with_gpu)
    topo = f_sub.maker.env.toposort()
    assert any([isinstance(node.op, B.GpuReshape) for node in topo])
    assert numpy.all(f_sub(a_val, b_val) == 0.0)
    assert numpy.all(numpy.asarray(a_val) == numpy.asarray(a_val_copy))

    #test that it works with inplace operations
    a_val = theano._asarray([0, 1, 2, 3, 4, 5], dtype='float32')
    a_val_copy = theano._asarray([0, 1, 2, 3, 4, 5], dtype='float32')
    b_val = theano._asarray([[0, 1, 2], [3, 4, 5]], dtype='float32')

    f_sub = theano.function([a, b], c - b, mode=mode_with_gpu)
    topo = f_sub.maker.env.toposort()
    assert any([isinstance(node.op, B.GpuReshape) for node in topo])
    assert numpy.all(f_sub(a_val, b_val) == 0.0)
    assert numpy.all(numpy.asarray(a_val) == numpy.asarray(a_val_copy))

    # verify gradient
    def just_vals(v):
        return T.Reshape(2)(v, theano._asarray([2, 3], dtype='int32'))
    utt.verify_grad(just_vals, [a_val])


def test_elemwise_empty():
    #test with 0 element
    a = tcn.shared_constructor(theano._asarray(numpy.random.rand(0, 0),
                                               dtype='float32'), 'a')

    b = tensor.fmatrix()

    f = pfunc([b], [], updates=[(a, a + b)], mode=mode_with_gpu)
    f2 = pfunc([b], [], updates=[(a, a + b)], mode=mode_without_gpu)

    a0 = a.get_value() * 1.0
    f(numpy.ones((0, 0), dtype='float32'))

    assert numpy.all(a0 + 1.0 == a.get_value())


def test_elemwise0():

    a = tcn.shared_constructor(theano._asarray(numpy.random.rand(4, 4),
                                               dtype='float32'), 'a')

    b = tensor.fmatrix()

    f = pfunc([b], [], updates=[(a, a + b)], mode=mode_with_gpu)

    #check that we work inplace.
    assert f.maker.env.toposort()[1].op.destroy_map.items() == [(0, [0])]

    a0 = a.get_value() * 1.0
    print 'BEFORE ADD', a.get_value()
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    f(numpy.ones((4, 4), dtype='float32'))
    print 'AFTER ADD', a.get_value()

    assert numpy.all(a0 + 1.0 == a.get_value())


def test_elemwise_bad_broadcast():
    x = cuda.fmatrix('x')
    y = cuda.fmatrix('y')

    f = theano.function([x, y], x * y, mode=mode_with_gpu)
    print f.maker.env.toposort()
    assert len(f.maker.env.toposort()) == 2
    assert isinstance(f.maker.env.toposort()[0].op, cuda.GpuElemwise)
    assert f.maker.env.toposort()[1].op == cuda.host_from_gpu

    try:
        f(rand_cuda_ndarray((10, 3)), rand_cuda_ndarray((10, 1)))
    except ValueError:
        pass
    else:
        raise Exception("Theano should have raised an error")


def test_elemwise1():
    """ Several kinds of elemwise expressions with no broadcasting,
    non power-of-two shape """

    shape = (3, 4)
    a = tcn.shared_constructor(theano._asarray(numpy.random.rand(*shape),
                                               dtype='float32') + 0.5, 'a')
    b = tensor.fmatrix()

    #let debugmode catch any mistakes
    print >> sys.stdout, "STARTING FUNCTION 1"
    f = pfunc([b], [], updates=[(a, b ** a)], mode=mode_with_gpu)
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    f(theano._asarray(numpy.random.rand(*shape), dtype='float32') + 0.3)

    print >> sys.stdout, "STARTING FUNCTION 2"
    #let debugmode catch any mistakes
    f = pfunc([b], [], updates=[(a, tensor.exp(b ** a))], mode=mode_with_gpu)
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
    f(theano._asarray(numpy.random.rand(*shape), dtype='float32') + 0.3)

    print >> sys.stdout, "STARTING FUNCTION 3"
    #let debugmode catch any mistakes
    f = pfunc([b], [], updates=[(a, a + b * tensor.exp(b ** a))],
              mode=mode_with_gpu)
    f(theano._asarray(numpy.random.rand(*shape), dtype='float32') + 0.3)


def test_elemwise2():
    """ Several kinds of elemwise expressions with dimension permutations """
    rng = numpy.random.RandomState(int(time.time()))
    print 'random?', rng.rand(3)
    shape = (3, 5)
    for pattern in [(0, 1), (1, 0)]:
        a = tcn.shared_constructor(theano._asarray(rng.rand(*shape),
                                                   dtype='float32'), name=None)
        b = tensor.Tensor(dtype='float32', broadcastable=[0] * len(shape))()
        f = pfunc([b], [], updates=[(a, (a + b).dimshuffle(pattern))],
                  mode=mode_with_gpu)
        has_elemwise = False
        for i, node in enumerate(f.maker.env.toposort()):
            print >> sys.stdout, i, node
            has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
        assert not has_elemwise
        #let debugmode catch errors
        print >> sys.stdout, 'pattern', pattern
        f(theano._asarray(rng.rand(*shape), dtype='float32') * .3)

    shape = (3, 4, 5, 6)
    a = tcn.shared_constructor(theano._asarray(rng.rand(*shape),
                                               dtype='float32'), 'a')
    b = tensor.Tensor(dtype='float32', broadcastable=[0] * len(shape))()
    f = pfunc([b], [], updates=[(a, (a + b).dimshuffle([2, 0, 3, 1]) *
        tensor.exp(b ** a).dimshuffle([2, 0, 3, 1]))], mode=mode_with_gpu)
    has_elemwise = False
    for i, node in enumerate(f.maker.env.toposort()):
        print i, node
        has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
    assert not has_elemwise
    #let debugmode catch errors
    f(theano._asarray(rng.rand(*shape), dtype='float32'))


def test_elemwise3():
    """ Several kinds of elemwise expressions with dimension
    permutations and broadcasting"""

    shape = (3, 4, 5, 6)
    a = tcn.shared_constructor(theano._asarray(numpy.random.rand(*shape),
                                               dtype='float32'), 'a')
    b = tensor.fvector()
    print b.type
    print tensor.constant(1).type
    print (1 + b).type
    print (1 + b ** a).type
    print tensor.exp((1 + b ** a)).type
    new_val = (a + b).dimshuffle([2, 0, 3, 1])
    new_val *= tensor.exp(1 + b ** a).dimshuffle([2, 0, 3, 1])
    f = pfunc([b], [], updates=[(a, new_val)], mode=mode_with_gpu)
    has_elemwise = False
    for i, node in enumerate(f.maker.env.toposort()):
        print >> sys.stdout, i, node
        has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
    assert not has_elemwise
    #let debugmode catch errors
    f(theano._asarray(numpy.random.rand(6), dtype='float32'))


def test_elemwise4():
    """ Test that two vectors can be broadcast to form an outer
    product (by performing rank-1 matrix update"""

    shape = (3, 4)
    a = tcn.shared_constructor(theano._asarray(numpy.random.rand(*shape),
                                               dtype='float32'), 'a')
    b = tensor.fvector()
    c = tensor.fvector()
    f = pfunc([b, c], [],
              updates=[(a, (a + b.dimshuffle('x', 0) * c.dimshuffle(0, 'x')))],
              mode=mode_with_gpu)
    has_elemwise = False
    for i, node in enumerate(f.maker.env.toposort()):
        print >> sys.stdout, i, node
        has_elemwise = has_elemwise or isinstance(node.op, tensor.Elemwise)
    assert not has_elemwise
    #let debugmode catch errors
    f(theano._asarray(numpy.random.rand(4), dtype='float32'),
      theano._asarray(numpy.random.rand(3), dtype='float32'))


def test_elemwise_comparaison_cast():
    """
    test if an elemwise comparaison followed by a cast to float32 are
    pushed to gpu.
    """

    a = tensor.fmatrix()
    b = tensor.fmatrix()
    av = theano._asarray(numpy.random.rand(4, 4), dtype='float32')
    bv = numpy.ones((4, 4), dtype='float32')

    for g, ans in [(tensor.lt, av < bv), (tensor.gt, av > bv),
                   (tensor.le, av <= bv), (tensor.ge, av >= bv)]:

        f = pfunc([a, b], tensor.cast(g(a, b), 'float32'), mode=mode_with_gpu)

        #theano.printing.debugprint(f)
        out = f(av, bv)
        assert numpy.all(out == ans)
        assert any([isinstance(node.op, cuda.GpuElemwise)
                    for node in f.maker.env.toposort()])


def test_elemwise_composite_float64():
    # test that we don't fuse composite elemwise with float64 somewhere inside
    # nvcc by default downcast them to float32. We would need to tell him not
    # to do so, but that possible only on some device.
    a = tensor.fmatrix()
    b = tensor.fmatrix()
    av = theano._asarray(numpy.random.rand(4, 4), dtype='float32')
    bv = numpy.ones((4, 4), dtype='float32')

    def get_all_basic_scalar(composite_op):
        l = []
        for i in composite_op.env.toposort():
            if isinstance(i, theano.scalar.Composite):
                l += get_all_basic_scalar(i)
            else:
                l.append(i)
        return l
    for mode in [mode_with_gpu, mode_with_gpu.excluding('gpu_after_fusion'),
                 mode_with_gpu.excluding('elemwise_fusion')]:
        f = pfunc([a, b],
                  tensor.cast(tensor.lt(tensor.cast(a, 'float64') ** 2,
                                               b),
                                     'float32'), mode=mode)

        #theano.printing.debugprint(f, print_type=True)
        out = f(av, bv)
        assert numpy.all(out == ((av ** 2) < bv))
        for node in f.maker.env.toposort():
            if isinstance(node.op, cuda.GpuElemwise):
                if isinstance(node.op.scalar_op, theano.scalar.Composite):
                    scals = get_all_basic_scalar(node.op.scalar_op)
                    for s in scals:
                        assert not any([i.type.dtype == 'float64'
                                        for i in s.inputs + s.outputs])


def test_elemwise_composite_support_code():
    """
    This was generating an error at compile time.
    Commit 3d1690fa346103594356ecaeceeb2c6757b45d2b fixed that.
    """
    X = tcn.shared_constructor(value=numpy.zeros((100, 10), dtype="float32"),
                               name='X')
    W = tcn.shared_constructor(value=numpy.zeros((10, 1), dtype="float32"),
                               name='W')
    U = T.dot(X, W)
    Y = tcn.shared_constructor(value=numpy.zeros((100, 1), dtype="float32"),
                               name='Y')
    P = T.exp(-(Y - U) ** 2)
    epsilon = numpy.asarray(0.001, dtype="float32")
    NLL = -T.mean(T.log(P + epsilon))  # SupportCodeError
    G = T.grad(NLL, wrt=[W])

    backup = theano.config.warn.identify_1pexp_bug
    theano.config.warn.identify_1pexp_bug = False
    try:
        f_grad = theano.function(inputs=[], outputs=G, mode=mode_with_gpu)
    finally:
        theano.config.warn.identify_1pexp_bug = backup
    f_grad()

    topo = f_grad.maker.env.toposort()
    assert sum([isinstance(node.op, T.Elemwise) for node in topo]) == 1
    assert sum([isinstance(node.op, tcn.GpuElemwise) for node in topo]) == 1


def speed_elemwise_collapse():
    """ used to time if the collapse of ccontiguous dims are useful """

    shape = (30, 40, 50, 600)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),
                                                 dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape), dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2[:, ::2, :, :]
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = a3 + b * tensor.exp(1 + b ** a3)
    f = pfunc([b], [c], mode=mode_with_gpu)

    v = theano._asarray(numpy.random.rand(*shape), dtype='float32')
    v = v[:, ::2, :, :]
    v = cuda_ndarray.CudaNdarray(v)
    for id, n in enumerate(f.maker.env.toposort()):
        print id, n
    t1 = time.time()
    for i in range(100):
        #let debugmode catch errors
        f(v)
    t2 = time.time()


def speed_elemwise_collapse2():
    """ used to test the speed up of the generalised collapse of
    ccontiguous dims"""

    shape = (30, 40, 50, 600)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),
                                                 dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape), dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2[:, :, :, ::2]
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = a3 + b * tensor.exp(1 + b ** a3)
    f = pfunc([b], [c], mode=mode_with_gpu)

    v = theano._asarray(numpy.random.rand(*shape), dtype='float32')
    v = v[:, :, :, ::2]
    v = cuda_ndarray.CudaNdarray(v)
    for id, n in enumerate(f.maker.env.toposort()):
        print id, n
    t1 = time.time()
    for i in range(100):
        #let debugmode catch errors
        f(v)
    t2 = time.time()


def test_elemwise_collapse():
    """ Test when all inputs have one(and the same) broadcastable dimension """

    shape = (4, 5, 60)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),
                                                 dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape), dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle(0, 'x', 1, 2)
    b = tcn.CudaNdarrayType((False, True, False, False))()
    c = a3 + b
    f = pfunc([b], [c], mode=mode_with_gpu)

    v = theano._asarray(numpy.random.rand(shape[0], 1, *shape[1:]),
                        dtype='float32')
    v = cuda_ndarray.CudaNdarray(v)
    if False:
        for id, n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out = f(v)[0]
    assert numpy.allclose(out, a.reshape(shape[0], 1, *shape[1:]) + v)
    print "Expected collapse of all dimensions"


def test_elemwise_collapse2():
    """ Test when only one inputs have one broadcastable dimension """

    shape = (4, 5, 9)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),
                                                 dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape), dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle(0, 'x', 1, 2)
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = a3 + b
    f = pfunc([b], [c], mode=mode_with_gpu)

    v = theano._asarray(numpy.random.rand(shape[0], 5, *shape[1:]),
                        dtype='float32')
    v = cuda_ndarray.CudaNdarray(v)
    if False:
        for id, n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out = f(v)[0]
    assert numpy.allclose(out, a.reshape(shape[0], 1, *shape[1:]) + v)
    print "Expected collapse to 3 dimensions"


def test_elemwise_collapse3():
    """ Test when only one inputs have two broadcastable dimension at each ends """

    shape = (4, 5)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),
                                                 dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape),
                        dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle('x', 0, 1, 'x')
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = (a3 + b)
    f = pfunc([b], [c], mode=mode_with_gpu)

    v = theano._asarray(numpy.random.rand(5, shape[0], shape[1], 4),
                        dtype='float32')
    v = cuda_ndarray.CudaNdarray(v)
    if False:
        for id, n  in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out = f(v)[0]
    assert numpy.allclose(out, a.reshape(1, shape[0], shape[1], 1) + v)
    print "Expected collapse to 3 dimensions"


def test_elemwise_collapse4():
    """ Test when only one inputs have two broadcastable dimension at
    each ends and we add a scalar"""

    shape = (4, 5)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),
                                                 dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape), dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle('x', 0, 1, 'x')
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = (a3 + b + 2)
    f = pfunc([b], [c], mode=mode_with_gpu)

    v = theano._asarray(numpy.random.rand(5, shape[0], shape[1], 4),
                        dtype='float32')
    v = cuda_ndarray.CudaNdarray(v)
    if False:
        for id, n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out = f(v)[0]
    assert numpy.allclose(out, a.reshape(1, shape[0], shape[1], 1) + v + 2)
    print "Expected collapse to 3 dimensions"


def test_elemwise_collapse5():
    """ Test when only one inputs have two broadcastable dimension at
    the beginning and we add a scalar"""

    shape = (4, 5)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),
                                                 dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape), dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle('x', 'x', 0, 1)
    b = tcn.CudaNdarrayType((False, False, False, False))()
    c = (a3 + b + 2)
    f = pfunc([b], [c], mode=mode_with_gpu)

    v = theano._asarray(numpy.random.rand(5, 4, shape[0], shape[1]),
                        dtype='float32')
    v = cuda_ndarray.CudaNdarray(v)
    if False:
        for id, n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out = f(v)[0]
    assert numpy.allclose(out, a.reshape(1, 1, shape[0], shape[1]) + v + 2)
    print "Expected collapse to 2 dimensions"


def test_elemwise_collapse6():
    """ Test when all inputs have two broadcastable dimension at the
    beginning"""

    shape = (4, 5)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),
                                                 dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape), dtype='float32')
    a2 = tcn.shared_constructor(a, 'a')
    a3 = a2.dimshuffle('x', 'x', 0, 1)
    b = tcn.CudaNdarrayType((True, True, False, False))()
    f = pfunc([b], [a3 + b], mode=mode_with_gpu)

    v = theano._asarray(numpy.random.rand(1, 1, shape[0], shape[1]),
                        dtype='float32')
    v = cuda_ndarray.CudaNdarray(v)
    if False:
        for id, n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out = f(v)[0]
    assert numpy.allclose(out, a.reshape(1, 1, shape[0], shape[1]) + v)
    print "Expected collapse to c contiguous"


def test_elemwise_collapse7(atol=1e-6):
    """ Test when one input have one broadcastable dimension and the
    other is a scalar"""

    shape = (5, 4, 1)
    a = cuda_ndarray.CudaNdarray(theano._asarray(numpy.random.rand(*shape),
                                                 dtype='float32'))
    a = theano._asarray(numpy.random.rand(*shape), dtype='float32')
    a2 = tcn.shared_constructor(a.copy(), 'a')
    a3 = a2.dimshuffle(0, 'x', 1, 2)
    f = pfunc([], [a3 + 2], mode=mode_with_gpu)

    if False:
        for id, n in enumerate(f.maker.env.toposort()):
            print id, n
    #let debugmode catch errors
    out = f()[0]
    ans = (a + 2).reshape(shape[0], 1, shape[1], shape[2])
    assert numpy.allclose(out, ans, atol=atol)
    print "Expected collapse to c contiguous"


def test_hostfromgpu_shape_i():
    """
    Test that the shape is lifted over hostfromgpu
    """
    pass

    m = mode_with_gpu.including('local_dot_to_dot22',
                                'local_dot22_to_dot22scalar','specialize')
    a = T.fmatrix('a')
    ca = theano.sandbox.cuda.var.CudaNdarrayType((False, False))()

    av = numpy.asarray(numpy.random.rand(5, 4), dtype='float32')
    cv = cuda.CudaNdarray(numpy.asarray(numpy.random.rand(5, 4),
                                      dtype='float32'))

    f = theano.function([a], cuda.basic_ops.gpu_from_host(a), mode=m)
    assert cuda.basic_ops.gpu_from_host in [x.op
                                            for x in f.maker.env.toposort()]
    f = theano.function([a], cuda.basic_ops.gpu_from_host(a).shape, mode=m)
    topo = f.maker.env.toposort()
    assert isinstance(topo[0].op, T.opt.Shape_i)
    assert isinstance(topo[1].op, T.opt.Shape_i)
    assert isinstance(topo[2].op, T.opt.MakeVector)
    assert tuple(f(av)) == (5, 4)



    f = theano.function([ca], cuda.basic_ops.host_from_gpu(ca), mode=m)
    assert cuda.basic_ops.host_from_gpu in [x.op
                                            for x in f.maker.env.toposort()]
    f = theano.function([ca], cuda.basic_ops.host_from_gpu(ca).shape, mode=m)
    topo = f.maker.env.toposort()
    assert isinstance(topo[0].op, T.opt.Shape_i)
    assert isinstance(topo[1].op, T.opt.Shape_i)
    assert isinstance(topo[2].op, T.opt.MakeVector)
    assert tuple(f(cv)) == (5, 4)

# -----------------------------------------------------------------------

import theano.sandbox.cuda as cuda_ndarray


def test_gpujoin_assert_cndas():
    # this will end up being an ndarray, as it's float64
    _a = numpy.asarray([[1, 2], [3, 4]], dtype='float64')
    a = theano.shared(_a)

    try:
        c = cuda.basic_ops.gpu_join(1, a)
        # can't "assert False" here, as we want the assertion
        # error from gpu_join
    except AssertionError:
        assert True
        return

    assert False


def test_gpujoin_no_rebroadcast():
    _a = numpy.asarray([[1, 2], [3, 4]], dtype='float32')
    a = tcn.shared_constructor(_a)
    f = theano.function([], T.join(1, a))
    l = f.maker.env.toposort()
    assert not any([isinstance(x.op, T.Rebroadcast) for x in l])


def test_gpualloc_input_on_gpu():
    a_val = numpy.asarray(numpy.random.rand(4, 5), dtype='float32')
    a = tcn.shared_constructor(a_val)

    b = T.fscalar()
    f = theano.function([b], T.ones_like(a) + b, mode=mode_without_gpu)
    f_gpu = theano.function([b], T.ones_like(a) + b, mode=mode_with_gpu)

    assert sum([node.op == T.alloc for node in f.maker.env.toposort()]) == 1
    assert sum([node.op == B.gpu_alloc
                for node in f_gpu.maker.env.toposort()]) == 1

    assert numpy.allclose(numpy.ones(a.get_value(borrow=True).shape) + 9,
                          f_gpu(9))
    assert numpy.allclose(f(5), f_gpu(5))


def test_gpujoin_gpualloc():
    a = T.fmatrix('a')
    a_val = numpy.asarray(numpy.random.rand(4, 5), dtype='float32')
    b = T.fmatrix('b')
    b_val = numpy.asarray(numpy.random.rand(3, 5), dtype='float32')

    f = theano.function([a, b], T.join(0, T.zeros_like(a),T.ones_like(b)) + 4,
                        mode=mode_without_gpu)
    f_gpu = theano.function([a, b], T.join(0, T.zeros_like(a), T.ones_like(b)),
                            mode=mode_with_gpu)
    f_gpu2 = theano.function([a, b], T.join(0, T.zeros_like(a),
                                           T.ones_like(b)) + 4,
                             mode=mode_with_gpu)

    assert sum([node.op == T.alloc for node in f.maker.env.toposort()]) == 2
    assert sum([node.op == T.join for node in f.maker.env.toposort()]) == 1
    assert sum([node.op == B.gpu_alloc
                for node in f_gpu.maker.env.toposort()]) == 2
    assert sum([node.op == B.gpu_join
                for node in f_gpu.maker.env.toposort()]) == 1
    assert sum([node.op == B.gpu_alloc
                for node in f_gpu2.maker.env.toposort()]) == 2
    assert sum([node.op == B.gpu_join
                for node in f_gpu2.maker.env.toposort()]) == 1
    assert numpy.allclose(f(a_val, b_val), f_gpu2(a_val, b_val))


def test_gpualloc_output_to_gpu():
    a_val = numpy.asarray(numpy.random.rand(4, 5), dtype='float32')
    a = tcn.shared_constructor(a_val)

    b = T.fscalar()
    f = theano.function([b], T.ones_like(a) + b, mode=mode_without_gpu)
    f_gpu = theano.function([b], B.gpu_from_host(T.ones_like(a)) + b,
                            mode=mode_with_gpu)

    print f.maker.env.toposort()
    print f_gpu.maker.env.toposort()
    print f(2)
    print f_gpu(2)

    assert sum([node.op == T.alloc for node in f.maker.env.toposort()]) == 1
    assert sum([node.op == B.gpu_alloc
                for node in f_gpu.maker.env.toposort()]) == 1

    assert numpy.allclose(numpy.ones(a.get_value(borrow=True).shape) + 9,
                          f_gpu(9))
    assert numpy.allclose(f(5), f_gpu(5))


import theano.tensor.tests.test_basic


class TestAlloc(theano.tensor.tests.test_basic.TestAlloc):
    dtype = "float32"
    mode = mode_with_gpu
    shared = staticmethod(cuda.shared_constructor)
    allocs = [B.GpuAlloc, B.GpuAlloc, tensor.Alloc]


class T_Join_and_Split(theano.tensor.tests.test_basic.T_Join_and_Split):
    def setUp(self):
        utt.seed_rng()
        self.mode = mode_with_gpu.excluding('constant_folding')
        self.join_op = cuda.GpuJoin
        # No gpu split.
        self.split_op = tensor.Split
        # No Make vector on the gpu, Join used instead
        self.make_vector_op = cuda.GpuJoin
        self.floatX = "float32"
        # In FAST_COMPILE mode, we force the FAST_RUN mode for optimization.
        self.hide_error = theano.config.mode not in ['DebugMode', 'DEBUG_MODE']
        self.shared = cuda.shared_constructor


# This is to don't duplicate test.
class T_subtensor(theano.tensor.tests.test_basic.T_subtensor):
    shared = staticmethod(cuda.shared_constructor)
    sub = cuda.GpuSubtensor
    inc_sub = cuda.GpuIncSubtensor
    adv_sub1 = cuda.GpuAdvancedSubtensor1
    adv_incsub1 = cuda.GpuAdvancedIncSubtensor1
    mode = mode_with_gpu
    dtype = 'float32'
    ignore_topo = (B.HostFromGpu, B.GpuFromHost)
    fast_compile = theano.config.mode == 'FAST_COMPILE'

    def __init__(self, name):
        return super(theano.tensor.tests.test_basic.T_subtensor,
                     self).__init__(name)


def test_advinc_subtensor1():
    """ Test the second case in the opt local_gpu_advanced_incsubtensor1 """
    shared = cuda.shared_constructor
    #shared = tensor.shared
    xval = numpy.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                      dtype='float32')
    yval = numpy.asarray([[10, 10, 10], [10, 10, 10]],
                      dtype='float32')
    x = shared(xval, name='x')
    y = T.fmatrices('y')
    expr = T.advanced_inc_subtensor1(x, y, [0, 2])
    f = theano.function([y], expr, mode=mode_with_gpu)
    assert sum([isinstance(node.op, cuda.GpuAdvancedIncSubtensor1)
                for node in f.maker.env.toposort()]) == 1
    assert numpy.allclose(f(yval), [[11., 12., 13.], [4., 5., 6.],
                                    [17., 18., 19.]])


def test_inc_subtensor():
    shared = cuda.shared_constructor
    #shared = tensor.shared
    x, y = T.fmatrices('x', 'y')
    xval = numpy.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                      dtype='float32')
    yval = numpy.asarray([[10, 10, 10], [10, 10, 10], [10, 10, 10]],
                      dtype='float32')
    expr = T.inc_subtensor(x[:, 1:3], y[:, 1:3])
    f = theano.function([x, y], expr, mode=mode_with_gpu)
    print f.maker.env.toposort()
    assert sum([isinstance(node.op, cuda.GpuSubtensor)
                for node in f.maker.env.toposort()]) == 1
    assert sum([isinstance(node.op, cuda.GpuIncSubtensor) and
                node.op.set_instead_of_inc==False
                for node in f.maker.env.toposort()]) == 1
    assert numpy.allclose(f(xval, yval), [[1., 12., 13.],
                                          [4., 15., 16.], [7., 18., 19.]])


def test_set_subtensor():
    shared = cuda.shared_constructor
    #shared = tensor.shared
    x, y = T.fmatrices('x', 'y')
    xval = numpy.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                      dtype='float32')
    yval = numpy.asarray([[10, 10, 10], [10, 10, 10], [10, 10, 10]],
                      dtype='float32')
    expr = T.set_subtensor(x[:, 1:3], y[:, 1:3])
    f = theano.function([x, y], expr, mode=mode_with_gpu)
    assert sum([isinstance(node.op, cuda.GpuSubtensor)
                for node in f.maker.env.toposort()]) == 1
    assert sum([isinstance(node.op, cuda.GpuIncSubtensor) and
                node.op.set_instead_of_inc == True
                for node in f.maker.env.toposort()]) == 1
    print f(xval, yval)


def test_many_arg_elemwise():
    """this test checks whether the + and * elemwise ops can handle extremely large numbers of
    arguments on gpu
    i.e., it is a test of the optimization theano/sandbox/cuda/opt.py:local_gpu_huge_add_or_mul """
    rng = numpy.random.RandomState([1, 2, 3])

    for num_args in [25]:
        for op_to_test in [theano.tensor.add, theano.tensor.mul]:
            for nb_dim in [2, 3, 4, 5]:
                shapes = [rng.randint(1, 5) for i in range(nb_dim)]
                args = [numpy.cast['float32'](rng.randn(*shapes))
                        for arg in xrange(0, num_args)]

                symb_args = [theano.tensor.TensorType('float32',
                                                      (False,)*nb_dim)()
                             for arg in xrange(0, num_args)]


                outputs = []
                for mode in [mode_with_gpu, mode_without_gpu]:
                    #test the optijmization local_gpu_elemwise_0
                    f = theano.function(
                        symb_args, op_to_test(*symb_args),
                        mode=mode.excluding("local_gpu_elemwise_1"))
                    outputs.append(f(*args))
                    #assert that the test was done on the gpu.
                    if mode is mode_with_gpu:
                        assert any([isinstance(node.op, cuda.GpuElemwise)
                                    for node in f.maker.env.nodes])

                    #test the optijmization local_gpu_elemwise_1
                    f = theano.function(
                        symb_args,
                        cuda.gpu_from_host(op_to_test(*symb_args)),
                        mode=mode.excluding("local_gpu_elemwise_0"))
                    out = f(*args)
                    #assert that the test was done on the gpu.
                    if mode is mode_with_gpu:
                        assert any([isinstance(node.op, cuda.GpuElemwise)
                                    for node in f.maker.env.nodes])
                    assert numpy.allclose(out, outputs[-1])

                results_gpu, results_cpu = outputs

                assert numpy.allclose(results_gpu, results_cpu)


def test_duplicate_arg_elemwise():
    A = theano.tensor.fmatrix()
    B = A + A

    f = theano.function([A], B, mode=mode_with_gpu)

    Aval = numpy.random.RandomState([1, 2, 3]).randn(5, 5).astype('float32')
    Bval = Aval + Aval

    assert numpy.allclose(Bval, f(Aval))


def test_shared_float32():
    '''Test use of cuda.shared_constructor through theano.shared'''
    # Register cuda.shared_constructor in theano.shared
    theano.shared.constructors.append(cuda.shared_constructor)

    a = theano.shared(numpy.ones((2, 3), dtype='float32'))
    assert isinstance(a.type, tcn.CudaNdarrayType)

    # Unregister
    del theano.shared.constructors[-1]


def test_shared_cudandarray():
    '''Test that we can create a CudaNdarraySharedVariable from a
    CudaNdarray'''
    a = cuda.shared_constructor(cuda.CudaNdarray.zeros((2, 3)))
    assert isinstance(a.type, tcn.CudaNdarrayType)


class test_tensordot_reshape(unittest.TestCase):
    '''Test alternative tensordot implementation.

    Test that the tensordot implementation using dimshuffle, reshape and dot
    gives the same results as the default (numpy) version.
    '''

    def setUp(self):
        self.rng = numpy.random.RandomState(utt.fetch_seed())

    def test1(self):
        # define some tensors
        tensor1 = self.rng.rand(20, 10, 5, 8).astype(theano.config.floatX)
        tensor2 = self.rng.rand(5, 8, 20).astype(theano.config.floatX)
        tensor3 = self.rng.rand(8, 20, 5).astype(theano.config.floatX)

        x = T.tensor4('x')
        y = T.tensor3('y')

        # case 1: number of axes to sum over
        default1 = theano.function([x, y], T.tensordot(x, y, 2))(
                tensor1, tensor2)
        reshape1 = theano.function([x, y], B.tensordot(x, y, 2))(
                tensor1, tensor2)
        assert numpy.allclose(default1, reshape1)

        # case 2: axis pairs
        default2 = theano.function(
                [x, y],
                T.tensordot(x, y, axes=[(0, 3), (1, 0)])
                )(tensor1, tensor3)
        reshape2 = theano.function(
                [x, y],
                B.tensordot(x, y, axes=[(0, 3), (1, 0)])
                )(tensor1, tensor3)
        assert numpy.allclose(default2, reshape2)

        default3 = theano.function(
                [x, y],
                T.tensordot(x, y, axes=[(0, 3, 2), (1, 0, 2)])
                )(tensor1, tensor3)
        reshape3 = theano.function(
                [x, y],
                B.tensordot(x, y, axes=[(0, 3, 2), (1, 0, 2)])
                )(tensor1, tensor3)
        assert numpy.allclose(default3, reshape3)


class test_size(unittest.TestCase):

    """
    Ensure the `size` attribute of CUDA tensors behaves as in numpy.
    """

    def test_matrix(self):
        x = cuda.fmatrix()
        y = numpy.zeros((5, 7), dtype='float32')
        assert y.size == theano.function([x], x.size)(y)

    def test_vector(self):
        x = cuda.fvector()
        y = numpy.zeros(7, dtype='float32')
        assert y.size == theano.function([x], x.size)(y)

    def test_scalar(self):
        x = cuda.fscalar()
        y = numpy.array(7, dtype='float32')
        assert y.size == theano.function([x], x.size)(y)

    def test_shared(self):
        # NB: we also test higher order tensors at the same time.
        y = cuda.CudaNdarray.zeros((1, 2, 3, 4))
        x = cuda.shared_constructor(y)
        assert y.size == theano.function([], x.size)()


import theano.tensor.tests.test_sharedvar
#This test the case when the shared constructor view an CudaNdarray as input
test_shared_options = theano.tensor.tests.test_sharedvar.makeSharedTester(
    shared_constructor_=tcn.shared_constructor,
    dtype_='float32',
    get_value_borrow_true_alias_=True,
    shared_borrow_true_alias_=True,#True when the original value is already a CudaNdarray!
    set_value_borrow_true_alias_=True,
    set_value_inplace_=True,
    set_cast_value_inplace_=False,
    shared_constructor_accept_ndarray_=True,
    internal_type_=cuda_ndarray.CudaNdarray,
    test_internal_type_=lambda a: isinstance(a, cuda_ndarray.CudaNdarray),
    theano_fct_=theano.tensor.exp,
    ref_fct_=numpy.exp,
    cast_value_=cuda.as_cuda_array,
    op_by_matrix_=True,
    name='test_shared_options')

#This test the case when the shared constructor view an ndarray as input
test_shared_options2 = theano.tensor.tests.test_sharedvar.makeSharedTester(
    shared_constructor_=tcn.shared_constructor,
    dtype_='float32',
    get_value_borrow_true_alias_=False,
    shared_borrow_true_alias_=False,
    set_value_borrow_true_alias_=False,
    set_value_inplace_=True,
    set_cast_value_inplace_=True,
    shared_constructor_accept_ndarray_=True,
    internal_type_=cuda_ndarray.CudaNdarray,
    test_internal_type_=lambda a: isinstance(a, cuda_ndarray.CudaNdarray),
    theano_fct_=theano.tensor.exp,
    ref_fct_=numpy.exp,
    cast_value_=numpy.asarray,
    op_by_matrix_=True,
    name='test_shared_options')


def speed_adv_sub1():
    data = numpy.random.rand(50000, 21).astype("float32")
    var = tcn.shared_constructor(data)
    vec = tensor.lvector()
    for batch_size in [100, 1000, 10000, 100000]:
        idx = numpy.random.randint(0, 50000, batch_size)
        mode_with_gpu = theano.compile.ProfileMode().including('gpu')
        f = theano.function([vec], var[vec], mode=mode_with_gpu)
        for i in range(100):
            f(idx)
        print "ProfileMode with batch size", batch_size
        mode_with_gpu.print_summary()

if __name__ == '__main__':
    test_many_arg_elemwise()
    test_gpujoin_twomatrices_joincolumns()
    test_gpujoin_assert_cndas()
    test_gpujoin_preserves_broadcasting()
    test_gpujoin_twomatrices_badshapes()
