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
    """

    n_in = 1000
    batch_size = 4097
    n_out = 1250

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

    assert numpy.allclose(classify(xx,yy,b_values,W_values)[0],classify_gpu(xx,yy,b_values,W_values)[0])
    assert numpy.allclose(classify(xx,yy,b_values,W_values)[1],classify_gpu(xx,yy,b_values,W_values)[1])
    assert numpy.allclose(classify(xx,yy,b_values,W_values)[2],classify_gpu(xx,yy,b_values,W_values)[2],atol=2e-6)


