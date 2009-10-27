import unittest, sys, time
import numpy as N
import theano.tensor as T
from theano.tests import unittest_tools as utt
from theano.sandbox.downsample import DownsampleFactorMax, max_pool
from theano import function, Mode

class TestDownSample(unittest.TestCase):
    def test_maxpool(self):
        # generate flatted images
        maxpoolshps = ((1,1),(2,2),(3,3),(2,3))
        imval = N.random.rand(4,10,64,64)
        do_theano=True
        images = T.dmatrix()
        dmatrix4=T.TensorType('float64', (False, False, False, False))
        images4=dmatrix4()
        tctot, tpytot, ntot, gtot= [],[],[],[]
        for maxpoolshp in maxpoolshps:

            print 'maxpoolshp', maxpoolshp
           
            # numeric verification
            my_output_val = N.zeros((imval.shape[0], imval.shape[1],
                                     imval.shape[2]/maxpoolshp[0],
                                     imval.shape[3]/maxpoolshp[1]))
            
            time1=time.time()
            for n in range(imval.shape[0]):
                for k in range(imval.shape[1]):
                    for i in range(my_output_val.shape[2]):
                        ii =  i*maxpoolshp[0]
                        for j in range(my_output_val.shape[3]):
                            jj = j*maxpoolshp[1]
                            patch = imval[n,k,ii:ii+maxpoolshp[0],jj:jj+maxpoolshp[1]]
                            my_output_val[n,k,i,j] = N.max(patch)
            my_output_val = my_output_val.reshape(imval.shape[0],-1)
            ntot+=[time.time()-time1]

            # symbolic stuff
            if do_theano:
                #### wrapper to DownsampleFactorMax op ####
                output, outshp = max_pool(images, imval.shape[1:], maxpoolshp)
                assert N.prod(my_output_val.shape[1:]) == N.prod(outshp)
                print outshp
                print my_output_val.shape
                assert N.prod(my_output_val.shape[1:]) == N.prod(outshp)
                f = function([images,],[output,])
                imval2=imval.reshape(imval.shape[0],-1)
                output_val = f(imval2)
                assert N.all(output_val == my_output_val)
            else:
                tctot=-1
                output_val = my_output_val.copy()

            #DownsampleFactorMax op
            maxpool_op = DownsampleFactorMax(maxpoolshp, ignore_border=True)(images4)
            f = function([images4],maxpool_op,mode=Mode(linker="py"))
            f2 = function([images4],maxpool_op,mode=Mode(linker="c"))
            time1=time.time()
            output_val2 = f(imval)
            tpytot+=[time.time()-time1]
            assert (N.abs(my_output_val.flatten()-output_val2.flatten())<1e-5).all()
            time1=time.time()
            output_val2 = f2(imval)
            tctot+=[time.time()-time1]
            assert (N.abs(my_output_val.flatten()-output_val2.flatten())<1e-5).all()

            def mp(input):
                output, outshp = max_pool(input, imval.shape[1:], maxpoolshp)
                return output

        print 'Numpy processing time: %.3fs'%sum(ntot),ntot
        print 'c Theano(DownsampleFactorMax) processing time: %.3fs'%sum(tctot),tctot
        print 'py Theano(DownsampleFactorMax) processing time: %.3fs'%sum(tpytot),tpytot
        d=N.asarray(ntot)/tctot
        print 'speed up c theano(DownsampleFactorMax) vs manual: %.3f'%d.mean(),d
        d=N.asarray(ntot)/tpytot
        print 'speed up py theano(DownsampleFactorMax) vs manual: %.3f'%d.mean(),d
        print 'verify_grad time %.3f'%sum(gtot),

    def test_maxpool_grad(self):
        # generate flatted images
        maxpoolshps = ((1,1),(2,2),(3,3),(2,3))
        imval = N.random.rand(3,7,10,10) * 10.0 #more variance means numeric gradient will be more accurate
        do_theano=True
        images = T.dmatrix()
        dmatrix4=T.TensorType('float64', (False, False, False, False))
        images4=dmatrix4()
        for maxpoolshp in maxpoolshps:
            print 'maxpoolshp', maxpoolshp
            def mp(input):
                output, outshp = max_pool(input, imval.shape[1:], maxpoolshp)
                return output
            print >> sys.stderr, 'max_pool verify_grad requires unusually large tolerance... is it correct?'
            utt.verify_grad(mp, [imval.reshape(imval.shape[0],-1)], tol=1e-2)

if __name__ == '__main__':
    t = TestSP("test_convolution")
    t = TestSP("test_maxpool").run()
#    t.test_convolution()
#    t.test_multilayer_conv()
    #t.test_maxpool()
    from theano.tests import main
#    main("test_sp")
