import theano, numpy
import theano.tensor as T

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda
if cuda.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

if theano.config.mode=='FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')

def test_GpuCrossentropySoftmax1HotWithBiasDx():
    """
    This is basic test for GpuCrossentropySoftmaxArgmax1HotWithBias and GpuCrossentropySoftmax1HotWithBiasDx


    We check that we loop when their is too much threads
    TODO: check that we loop when their is too much block(>32*1024)
    """

    n_in = 1000
    batch_size = 4097
    n_out = 1250

    if theano.config.mode!="DEBUG_MODE":
        n_in = 4098
        n_out = 4099

    x = T.fmatrix('x')
    y = T.lvector('y')


    b = T.fvector()
    W = T.fmatrix()

    p_y_given_x = T.nnet.softmax(T.dot(x,W)+b)
    y_pred = T.argmax(p_y_given_x)
    loss = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
    dW = T.grad(loss,W)
    classify = theano.function( inputs = [x,y,b,W], outputs = [loss,y_pred,dW],
                                mode = mode_without_gpu)
    classify_gpu = theano.function( inputs = [x,y,b,W], outputs = [loss,y_pred,dW],
                                    mode = mode_with_gpu)
    
    xx = numpy.asarray(numpy.random.rand(batch_size,n_in),dtype=numpy.float32)
    yy = numpy.ones((batch_size,),dtype='float32')
    b_values = numpy.zeros((n_out,),dtype='float32')
    W_values = numpy.asarray(numpy.random.rand(n_in,n_out),dtype='float32')
    

    assert any([isinstance(node.op,T.nnet.CrossentropySoftmaxArgmax1HotWithBias) for node in classify.maker.env.toposort()])
    assert any([isinstance(node.op,T.nnet.CrossentropySoftmax1HotWithBiasDx) for node in classify.maker.env.toposort()])
    assert any([isinstance(node.op,cuda.nnet.GpuCrossentropySoftmaxArgmax1HotWithBias) for node in classify_gpu.maker.env.toposort()])
    assert any([isinstance(node.op,cuda.nnet.GpuCrossentropySoftmax1HotWithBiasDx) for node in classify_gpu.maker.env.toposort()])

    out=classify(xx,yy,b_values,W_values)
    gout=classify_gpu(xx,yy,b_values,W_values)

    assert len(out)==len(gout)==3
    assert numpy.allclose(out[0],gout[0])
    assert numpy.allclose(out[2],gout[2],atol=3e-6),numpy.absolute(gout-out).max()
    assert numpy.allclose(out[1],gout[1]),[(id,out[1][id],gout[1][id],val) for id,val in enumerate(out[1]-gout[1]) if val!=0]


def test_softmax_with_bias():
    """
    This is basic test for GpuSoftmaxWithBias

    We check that we loop when their is too much block
    TODO: check that we loop when their is too much thread.(THIS IS NOT IMPLEMENTED)
    """
    x = T.fmatrix('x')

    #we need to test n>32*1024 to check that we make the block loop.
    n,m=2<<15,5

    data = numpy.arange(n*m, dtype='float32').reshape(n,m)

    z = T.nnet.softmax_with_bias(x, T.zeros_like(x[0,:]))

    f = theano.function([x],z, mode=mode_without_gpu)
    f_gpu = theano.function([x],z, mode=mode_with_gpu)
    assert f.maker.env.toposort()[-1].op==T.nnet.softmax_with_bias
    assert isinstance(f_gpu.maker.env.toposort()[-2].op,cuda.nnet.GpuSoftmaxWithBias)
    
    out=f(data)
    gout=f_gpu(data)
    assert numpy.allclose(out,gout),numpy.absolute(out-gout)

def test_softmax():
    """
    This is basic test for GpuSoftmax

    We check that we loop when their is too much block
    TODO: check that we loop when their is too much thread.(THIS IS NOT IMPLEMENTED)
    """
    x = T.fmatrix('x')

    #we need to test n>32*1024 to check that we make the block loop.
    n,m=2<<15,5

    data = numpy.arange(n*m, dtype='float32').reshape(n,m)

    z = T.nnet.softmax(x)

    f = theano.function([x],z, mode=mode_without_gpu)
    f_gpu = theano.function([x],z, mode=mode_with_gpu)
    assert f.maker.env.toposort()[-1].op==T.nnet.softmax
    assert isinstance(f_gpu.maker.env.toposort()[-2].op,cuda.nnet.GpuSoftmax)
    
    out=f(data)
    gout=f_gpu(data)
    assert numpy.allclose(out,gout),numpy.absolute(out-gout)
