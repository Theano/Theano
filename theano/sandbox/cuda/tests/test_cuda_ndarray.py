import time, copy, sys
import theano
import theano.sandbox.cuda as cuda_ndarray
# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
if cuda_ndarray.cuda_available == False:
        raise SkipTest('Optional package cuda disabled')
import numpy

def test_host_to_device():
    print >>sys.stdout, 'starting test_host_to_dev'
    for shape in ((), (3,), (2,3), (3,4,5,6)):
        a = theano._asarray(numpy.random.rand(*shape), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        c = numpy.asarray(b)
        assert numpy.all(a == c)

def test_add():
    for shape in ((), (3,), (2,3), (1,10000000),(10,1000000), (100,100000),(1000,10000),(10000,1000)):
        a0 = theano._asarray(numpy.random.rand(*shape), dtype='float32')
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
    print >>sys.stdout, 'starting test_exp'
    for shape in ((), (3,), (2,3), (1,10000000),(10,1000000), (100,100000),(1000,10000),(10000,1000)):
        a0 = theano._asarray(numpy.random.rand(*shape), dtype='float32')
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
    print >>sys.stdout, 'starting test_copy'
    shape = (5,)
    a = theano._asarray(numpy.random.rand(*shape), dtype='float32')

    print >>sys.stdout, '.. creating device object'
    b = cuda_ndarray.CudaNdarray(a)

    print >>sys.stdout, '.. copy'
    c = copy.copy(b)
    print >>sys.stdout, '.. deepcopy'
    d = copy.deepcopy(b)

    print >>sys.stdout, '.. comparisons'
    assert numpy.allclose(a, numpy.asarray(b))
    assert numpy.allclose(a, numpy.asarray(c))
    assert numpy.allclose(a, numpy.asarray(d))
    b+=b
    assert numpy.allclose(a+a, numpy.asarray(b))
    assert numpy.allclose(a+a, numpy.asarray(c))
    assert numpy.allclose(a, numpy.asarray(d))
    

def test_dot():
    print >>sys.stdout, 'starting test_dot'
    a0 = theano._asarray(numpy.random.rand(4, 7), dtype='float32')
    a1 = theano._asarray(numpy.random.rand(7, 6), dtype='float32')

    b0 = cuda_ndarray.CudaNdarray(a0)
    b1 = cuda_ndarray.CudaNdarray(a1)

    assert numpy.allclose(numpy.dot(a0, a1), cuda_ndarray.dot(b0, b1))

    print >> sys.stderr, 'WARNING TODO test_dot: not testing all 8 transpose cases of dot'

def test_sum():
    shape = (2,3)
    a0 = theano._asarray(numpy.arange(shape[0]*shape[1]).reshape(shape), dtype='float32')

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
    a0 = theano._asarray(numpy.arange(3*4*5*6*7*8).reshape(shape), dtype='float32')
    b0 = cuda_ndarray.CudaNdarray(a0)
    assert numpy.allclose(a0.sum(axis=5).sum(axis=3).sum(axis=0), numpy.asarray(b0.reduce_sum([1,0,0,1,0,1])))

    shape = (16,2048)
    a0 = theano._asarray(numpy.arange(16*2048).reshape(shape), dtype='float32')
    b0 = cuda_ndarray.CudaNdarray(a0)
    assert numpy.allclose(a0.sum(axis=0), numpy.asarray(b0.reduce_sum([1,0])))

    shape = (16,10)
    a0 = theano._asarray(numpy.arange(160).reshape(shape), dtype='float32')
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
        #print >> sys.stdout, "INFO: shapes", shape_1, shape_2
        a = theano._asarray(numpy.random.rand(*shape_1), dtype='float32')
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
        a = theano._asarray(numpy.random.rand(*shape_1), dtype='float32')
        b = cuda_ndarray.CudaNdarray(a)
        assert b.shape == a.shape

    for shape_1, shape_2 in shapelist:
        subtest(shape_1)
        subtest(shape_2)

def test_stride_manipulation():

    a = theano._asarray([[0,1,2], [3,4,5]], dtype='float32')
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
    a = theano._asarray(numpy.random.rand(30,20,5,5), dtype='float32')
    cuda_a = cuda_ndarray.CudaNdarray(a)
    a_view = cuda_a.view()
    a_view_strides = a_view._strides
    a_view._set_stride(2, -a_view_strides[2])
    a_view._set_stride(3, -a_view_strides[3])
    a_view._dev_data += 24 * sizeof_float

    a_view_copy = copy.deepcopy(a_view)

    assert numpy.all(a[:,:,::-1,::-1] == numpy.asarray(a_view_copy))

def test_mapping_getitem_ellipsis():
    a = theano._asarray(numpy.random.rand(5,4,3,2), dtype='float32')
    a = cuda_ndarray.CudaNdarray(a)

    b = a[...]
    assert b._dev_data == a._dev_data
    assert b._strides == a._strides
    assert b.shape == a.shape

def test_mapping_getitem_reverse_some_dims():
    dim=(5,4,3,2)
    a = theano._asarray(numpy.random.rand(*dim), dtype='float32')
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

    def _cmpf(x,*y):
        try:
            x.__getitem__(y)
	except IndexError:
            pass
	else:
            raise Exception("Did not generate out or bound error")

    def _cmpfV(x,*y):
        try:
            if len(y)==1:
                x.__getitem__(*y)
	    else:
                x.__getitem__(y)
	except ValueError:
            pass
	else:
            raise Exception("Did not generate out or bound error")

    dim =(2,)
    a = theano._asarray(numpy.random.rand(*dim), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)
    _cmp(numpy.asarray(_a[1]), a[1])
    _cmp(numpy.asarray(_a[-1]), a[-1])
    _cmp(numpy.asarray(_a[0]), a[0])
    _cmp(numpy.asarray(_a[::1]), a[::1])
    _cmp(numpy.asarray(_a[::-1]), a[::-1])
    _cmp(numpy.asarray(_a[...]), a[...])
    _cmpf(_a,2)

    dim =()
    a = theano._asarray(numpy.random.rand(*dim), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)
    _cmp(numpy.asarray(_a[...]), a[...])
    _cmpf(_a,0)
    _cmpfV(_a,slice(1))
    #TODO: test slice err
    #TODO: test tuple err


    dim =(5,4,3,2)
    a = theano._asarray(numpy.random.rand(*dim), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    _cmpf(_a,slice(-1),slice(-1),10,-10)
    _cmpf(_a,slice(-1),slice(-1),-10,slice(-1))
    _cmpf(_a,0,slice(0,-1,-20),-10)
    _cmpf(_a,10)
    _cmpf(_a,(10,0,0,0))
    _cmpf(_a,-10)

    _cmp(numpy.asarray(_a[:,:,::-1, ::-1]), a[:,:,::-1,::-1])
    _cmp(numpy.asarray(_a[:,:,::-10, ::-10]), a[:,:,::-10,::-10])
    _cmp(numpy.asarray(_a[:,:,1,-1]), a[:,:,1,-1])
    _cmp(numpy.asarray(_a[:,:,-1,:]), a[:,:,-1,:])
    _cmp(numpy.asarray(_a[:,::-2,-1,:]), a[:,::-2,-1,:])
    _cmp(numpy.asarray(_a[:,::-20,-1,:]), a[:,::-20,-1,:])
    _cmp(numpy.asarray(_a[:,::-2,-1]), a[:,::-2,-1])
    _cmp(numpy.asarray(_a[0,::-2,-1]), a[0,::-2,-1])
    _cmp(numpy.asarray(_a[1]), a[1])
    _cmp(numpy.asarray(_a[-1]), a[-1])
    _cmp(numpy.asarray(_a[-1,-1,-1,-2]), a[-1,-1,-1,-2])
    _cmp(numpy.asarray(_a[...]), a[...])

def test_gemm_vector_vector():
    a = theano._asarray(numpy.random.rand(5,1), dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)
    b = theano._asarray(numpy.random.rand(1,5), dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)
    
    _c = cuda_ndarray.dot(_a,_b)
    assert _c.shape == (5,5)
    assert numpy.allclose(_c, numpy.dot(a, b))
    
    _c = cuda_ndarray.dot(_b,_a)
    assert _c.shape == (1,1)
    assert numpy.allclose(_c, numpy.dot(b, a))

# ---------------------------------------------------------------------

def test_setitem_matrixvector1():
    a = theano._asarray([[0,1,2], [3,4,5]], dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([8,9], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)

    # set second column to 8,9
    _a[:,1] = _b

    assert numpy.all(numpy.asarray(_a[:,1]) == b)

def test_setitem_matrix_tensor3():
    a = numpy.arange(27)
    a.resize((3,3,3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([7,8,9], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)

    # set middle row through cube to 7,8,9
    _a[:,1,1] = _b

    assert numpy.all(numpy.asarray(_a[:,1,1]) == b)

def test_setitem_assign_to_slice():
    a = numpy.arange(27)
    a.resize((3,3,3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([7,8,9], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)

    # first get a slice of a
    _c = _a[:,:,1]

    # set middle row through cube to 7,8,9
    # (this corresponds to middle row of matrix _c)
    _c[:,1] = _b

    assert numpy.all(numpy.asarray(_a[:,1,1]) == b)


# this fails for the moment
def test_setitem_broadcast_must_fail():
    a = numpy.arange(27)
    a.resize((3,3,3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([7,8,9], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)

    try:
        # attempt to assign vector to all rows of this submatrix
        _a[:,:,1] = _b
        assert False
    except TypeError:
        assert True

# this also fails for the moment
def test_setitem_rightvalue_ndarray_fails():
    a = numpy.arange(27)
    a.resize((3,3,3))
    a = theano._asarray(a, dtype='float32')
    _a = cuda_ndarray.CudaNdarray(a)

    b = theano._asarray([7,8,9], dtype='float32')
    _b = cuda_ndarray.CudaNdarray(b)

    try:
        # attempt to assign the ndarray b with setitem
        _a[:,:,1] = b
        assert False
    except TypeError, e:
        #print e
        assert True

def test_zeros_basic_3d_tensor():
    _a = cuda_ndarray.CudaNdarray.zeros((3,4,5))
    assert numpy.allclose(numpy.asarray(_a), numpy.zeros((3,4,5)))

def test_zeros_basic_vector():
    _a = cuda_ndarray.CudaNdarray.zeros((300,))
    assert numpy.allclose(numpy.asarray(_a), numpy.zeros((300,)))


if __name__ == '__main__':
    test_zeros_basic_3d_tensor()
    test_zeros_basic_vector()
    test_setitem_matrixvector1()
    test_setitem_matrix_tensor3()
    test_setitem_broadcast_must_fail()
    test_setitem_assign_to_slice()
    test_setitem_rightvalue_ndarray_fails()

