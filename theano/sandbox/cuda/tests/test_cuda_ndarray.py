import time, copy, sys
import theano.sandbox.cuda as cuda_ndarray
import numpy

def test_host_to_device():
    print >>sys.stderr, 'starting test_host_to_dev'
    for shape in ((), (3,), (2,3), (3,4,5,6)):
        a = numpy.asarray(numpy.random.rand(*shape), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        c = numpy.asarray(b)
        assert numpy.all(a == c)

def test_add():
    for shape in ((), (3,), (2,3), (1,10000000),(10,1000000), (100,100000),(1000,10000),(10000,1000)):
        a0 = numpy.asarray(numpy.random.rand(*shape), dtype='float32')
        a1 = a0.copy()
        b0 = cuda_ndarray.CudaNdarray(a0)
        b1 = cuda_ndarray.CudaNdarray(a1)
        t0 = time.time()
        bsum = b0 + b1
        bsum = b0 + b1
        t1 = time.time()
        gpu_dt = t1 - t0
        t0 = time.time()
        asum = a0 + a1
        asum = a0 + a1
        t1 = time.time()
        cpu_dt = t1 - t0
        print shape, 'adding ', a0.size, 'cpu', cpu_dt, 'advantage', cpu_dt / gpu_dt
        assert numpy.allclose(asum,  numpy.asarray(bsum))

        if len(shape)==2:
            #test not contiguous version.
            #should raise not implemented.
            _b = b0[::, ::-1]

            b = numpy.asarray(_b)

            ones = numpy.ones(shape, dtype='float32')
            _ones = cuda_ndarray.CudaNdarray(ones)
            t = False
            try:
                _c = _b+_ones
            except:
                t = True
            assert t



def test_exp():
    print >>sys.stderr, 'starting test_exp'
    for shape in ((), (3,), (2,3), (1,10000000),(10,1000000), (100,100000),(1000,10000),(10000,1000)):
        a0 = numpy.asarray(numpy.random.rand(*shape), dtype='float32')
        a1 = a0.copy()
        b0 = cuda_ndarray.CudaNdarray(a0)
        b1 = cuda_ndarray.CudaNdarray(a1)
        t0 = time.time()
        bsum = b0.exp()
        t1 = time.time()
        gpu_dt = t1 - t0
        t0 = time.time()
        asum = numpy.exp(a1)
        t1 = time.time()
        cpu_dt = t1 - t0
        print shape, 'adding ', a0.size, 'cpu', cpu_dt, 'advantage', cpu_dt / gpu_dt
        #c = numpy.asarray(b0+b1)
        if asum.shape:
            assert numpy.allclose(asum, numpy.asarray(bsum))


def test_copy():
    print >>sys.stderr, 'starting test_copy'
    shape = (5,)
    a = numpy.asarray(numpy.random.rand(*shape), dtype='float32')

    print >>sys.stderr, '.. creating device object'
    b = cuda_ndarray.CudaNdarray(a)

    print >>sys.stderr, '.. copy'
    c = copy.copy(b)
    print >>sys.stderr, '.. deepcopy'
    d = copy.deepcopy(b)

    print >>sys.stderr, '.. comparisons'
    assert numpy.allclose(a, numpy.asarray(b))
    assert numpy.allclose(a, numpy.asarray(c))
    assert numpy.allclose(a, numpy.asarray(d))

def test_dot():
    print >>sys.stderr, 'starting test_dot'
    a0 = numpy.asarray(numpy.random.rand(4, 7), dtype='float32')
    a1 = numpy.asarray(numpy.random.rand(7, 6), dtype='float32')

    b0 = cuda_ndarray.CudaNdarray(a0)
    b1 = cuda_ndarray.CudaNdarray(a1)

    assert numpy.allclose(numpy.dot(a0, a1), cuda_ndarray.dot(b0, b1))

    print >> sys.stderr, 'WARNING test_dot: not testing all 8 transpose cases of dot'

def test_sum():
    shape = (2,3)
    a0 = numpy.asarray(numpy.arange(shape[0]*shape[1]).reshape(shape), dtype='float32')

    b0 = cuda_ndarray.CudaNdarray(a0)

    assert numpy.allclose(a0.sum(), numpy.asarray(b0.reduce_sum([1,1])))

    a0sum = a0.sum(axis=0)
    b0sum = b0.reduce_sum([1,0])

    print 'asum\n',a0sum
    print 'bsum\n',numpy.asarray(b0sum)

    assert numpy.allclose(a0.sum(axis=0), numpy.asarray(b0.reduce_sum([1,0])))
    assert numpy.allclose(a0.sum(axis=1), numpy.asarray(b0.reduce_sum([0,1])))
    assert numpy.allclose(a0, numpy.asarray(b0.reduce_sum([0,0])))

    shape = (3,4,5,6,7,8)
    a0 = numpy.asarray(numpy.arange(3*4*5*6*7*8).reshape(shape), dtype='float32')
    b0 = cuda_ndarray.CudaNdarray(a0)
    assert numpy.allclose(a0.sum(axis=5).sum(axis=3).sum(axis=0), numpy.asarray(b0.reduce_sum([1,0,0,1,0,1])))

    shape = (16,2048)
    a0 = numpy.asarray(numpy.arange(16*2048).reshape(shape), dtype='float32')
    b0 = cuda_ndarray.CudaNdarray(a0)
    assert numpy.allclose(a0.sum(axis=0), numpy.asarray(b0.reduce_sum([1,0])))

    shape = (16,10)
    a0 = numpy.asarray(numpy.arange(160).reshape(shape), dtype='float32')
    b0 = cuda_ndarray.CudaNdarray(a0)
    assert numpy.allclose(a0.sum(), numpy.asarray(b0.reduce_sum([1,1])))

def test_reshape():
    shapelist = [
            ((1,2,3), (1,2,3)),
            ((1,), (1,)),
            ((1,2,3), (3,2,1)),
            ((1,2,3), (6,)),
            ((1,2,3,2), (6,2)),
            ((2,3,2), (6,2))
             ]

    def subtest(shape_1, shape_2):
        #print >> sys.stderr, "INFO: shapes", shape_1, shape_2
        a = numpy.asarray(numpy.random.rand(*shape_1), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)

        aa = a.reshape(shape_2)
        bb = b.reshape(shape_2)

        n_bb = numpy.asarray(bb)

        #print n_bb

        assert numpy.all(aa == n_bb)
   
    # test working shapes
    for shape_1, shape_2 in shapelist:
        subtest(shape_1, shape_2)
        subtest(shape_2, shape_1)

    print >> sys.stderr, "WARN: TODO: test shape combinations that should give error"


def test_getshape():
    shapelist = [
            ((1,2,3), (1,2,3)),
            ((1,), (1,)),
            ((1,2,3), (3,2,1)),
            ((1,2,3), (6,)),
            ((1,2,3,2), (6,2)),
            ((2,3,2), (6,2))
             ]

    def subtest(shape):
        a = numpy.asarray(numpy.random.rand(*shape_1), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        assert b.shape == a.shape

    for shape_1, shape_2 in shapelist:
        subtest(shape_1)
        subtest(shape_2)

def test_stride_manipulation():

    a = numpy.asarray([[0,1,2], [3,4,5]], dtype='float32')
    b = cuda_ndarray.CudaNdarray(a)
    v = b.view()
    v._dev_data += 0
    c = numpy.asarray(v)
    assert numpy.all(a == c)

    sizeof_float = 4
    offset = 0

    b_strides = b._strides
    for i in xrange(len(b.shape)):
        offset += (b.shape[i]-1) * b_strides[i]
        v._set_stride(i, -b_strides[i])

    v._dev_data += offset * sizeof_float
    c = numpy.asarray(v)


    assert numpy.all(c == [[5, 4, 3], [2, 1, 0]])


def test_copy_subtensor0():
    sizeof_float=4
    a = numpy.asarray(numpy.random.rand(30,20,5,5), dtype='float32')
    cuda_a = cuda_ndarray.CudaNdarray(a)
    a_view = cuda_a.view()
    a_view_strides = a_view._strides
    a_view._set_stride(2, -a_view_strides[2])
    a_view._set_stride(3, -a_view_strides[3])
    a_view._dev_data += 24 * sizeof_float

    a_view_copy = copy.deepcopy(a_view)

    assert numpy.all(a[:,:,::-1,::-1] == numpy.asarray(a_view_copy))

def test_mapping_getitem_ellipsis():
    a = numpy.asarray(numpy.random.rand(5,4,3,2), dtype='float32')
    a = cuda_ndarray.CudaNdarray(a)

    b = a[...]
    assert b._dev_data == a._dev_data
    assert b._strides == a._strides
    assert b.shape == a.shape

def test_mapping_getitem_reverse_some_dims():
    dim=(5,4,3,2)
    a = numpy.asarray(numpy.random.rand(*dim), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    _b = _a[:,:,::-1, ::-1]
    
    b = numpy.asarray(_b)
    assert numpy.all(b==a[:,:,::-1,::-1])

def test_mapping_getitem_w_int():
    def _cmp(x,y):
        assert x.shape == y.shape
        if not numpy.all(x == y):
            print x
            print y
        assert numpy.all(x == y)

    dim =(2,)
    a = numpy.asarray(numpy.random.rand(*dim), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)
    _cmp(numpy.asarray(_a[1]), a[1])
    _cmp(numpy.asarray(_a[::1]), a[::1])
    _cmp(numpy.asarray(_a[::-1]), a[::-1])
    _cmp(numpy.asarray(_a[...]), a[...])

    dim =()
    a = numpy.asarray(numpy.random.rand(*dim), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)
    _cmp(numpy.asarray(_a[...]), a[...])



    dim =(5,4,3,2)
    a = numpy.asarray(numpy.random.rand(*dim), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    _cmp(numpy.asarray(_a[:,:,::-1, ::-1]), a[:,:,::-1,::-1])
    _cmp(numpy.asarray(_a[:,:,1,-1]), a[:,:,1,-1])
    _cmp(numpy.asarray(_a[:,:,-1,:]), a[:,:,-1,:])
    _cmp(numpy.asarray(_a[:,::-2,-1,:]), a[:,::-2,-1,:])
    _cmp(numpy.asarray(_a[:,::-2,-1]), a[:,::-2,-1])
    _cmp(numpy.asarray(_a[0,::-2,-1]), a[0,::-2,-1])
    _cmp(numpy.asarray(_a[1]), a[1])
    _cmp(numpy.asarray(_a[...]), a[...])

def test_gemm_vector_vector():
    a = numpy.asarray(numpy.random.rand(5,1), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)
    b = numpy.asarray(numpy.random.rand(1,5), dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)
    
    _c = cuda_ndarray.dot(_a,_b)
    assert _c.shape == (5,5)
    assert numpy.allclose(_c, numpy.dot(a, b))
    
    _c = cuda_ndarray.dot(_b,_a)
    assert _c.shape == (1,1)
    assert numpy.allclose(_c, numpy.dot(b, a))
