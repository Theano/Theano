from nose.plugins.skip import SkipTest
import numpy 
import numpy as np
import __builtin__
import theano
from theano.gof.python25 import any
import theano.tensor as T
import theano.tests.unittest_tools as utt

# Skip test if cuda_ndarray is not available.
import theano.sandbox.cuda as cuda
if cuda.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')
theano.config.exception_verbosity= 'high'
#theano.config.profile= True

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode(
        'FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')


def test_groupdot():
    x = T.fmatrix('x')
    w = T.tensor3('w',dtype='float32')
    b = T.fmatrix('b')
    c = T.vector('c',dtype='int32')
    z = T.nnet.GroupDot(51)(x, w,b,c)

    f = theano.function([x,w,b,c], z, mode=mode_without_gpu, name='cpu')
    f_gpu = theano.function([x,w,b,c], z, mode=mode_with_gpu, name='gpu')                              
    

    n_batch=50
    n_hid=300
    n_clust=20
    n_classes=7000
    
    def cmp(n_batch, n_hid,    n_clust,n_classes):
        x = numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid)
        w = np.random.rand(n_clust,n_hid,n_classes).astype('float32')
        b = np.random.rand(n_clust, n_classes).astype('float32')
        c = np.random.randint(0, n_clust, size=(n_batch,)).astype('int32')

        output=numpy.zeros(shape=(n_batch, n_classes))
        for i in range(n_batch):
            output[i] = np.dot(x[i,:],w[c[i],:,:])+b[c[i]]
        out=f(x,w,b,c)
        gout=f_gpu(x,w,b,c)
        assert numpy.allclose(out, output)
        assert numpy.allclose(gout, output)
    
    cmp(50,300,20,7000)
    cmp(100,256,51,10000)
    
    #this fails if gpu mem is less than 2gig
    #cmp(1000,512,100,10000)


def test_verify_groupdotgrad():

    def cmp(n_clust,n_batch,n_hid,n_classes):
        x = T.fmatrix('x')
        w = T.tensor3('w',dtype='float32')
        b = T.fmatrix('b')
        h = T.fmatrix('h')
        c = T.vector('c',dtype='int32')
    
        z = T.nnet.GroupDotGrad(n_clust)(x,w,b,c,h)    
        func = theano.function([x,w,b,c,h], z, mode=mode_without_gpu, name='cpu')
        f_gpu = theano.function([x,w,b,c,h], z, mode=mode_with_gpu, name='gpu')    

        c = np.random.randint(0, n_clust, size=(n_batch,)).astype('int32')
        x = numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid)
        w = np.random.rand(n_clust,n_hid,n_classes).astype('float32')
        b = np.random.rand(n_clust, n_classes).astype('float32')
        h = np.random.rand(n_batch,n_classes).astype('float32') 

        out = func(x,w,b,c,h)
        gout=f_gpu(x,w,b,c,h)    
        #print numpy.subtract(out,gout)
        #numpy.testing.assert_array_almost_equal(out,gout,decimal=2)
        #numpy.testing.assert_array_almost_equal(out,gout,decimal=5)
        #numpy.testing.assert_array_almost_equal(out,gout)

        def op_with_fixed_c(x,w,b):
            return T.nnet.GroupDot(n_clust)(x, w, b, c)

        rng = numpy.random.RandomState(42)

        T.verify_grad(op_with_fixed_c, [x,w,b], rng=rng, mode=mode_without_gpu)


    cmp(2,5,10,15)
    cmp(15,11,45,25)
    cmp(20,50,100,100)



    
    
    
   