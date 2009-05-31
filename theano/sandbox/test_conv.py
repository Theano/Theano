import sys, time, unittest

import numpy
import numpy as N

from scipy.signal import convolve2d
from theano.tests import unittest_tools as utt

from theano import function, Mode
import theano.tensor as T
from conv import ConvOp, convolve2, getFilterOutShp

def flip(kern, kshp):
    "flip the kernel as scipy.convolv2d do it flipped."
    flip = N.zeros(kern.shape)
    if len(kern.shape)==3:
        kern=kern.reshape(kern.shape[0],-1)
        for k in range(kern.shape[0]):
            it = reversed(kern[k,:])
            for i in range(kshp[0]):
                for j in range(kshp[1]):
                    flip[k,i,j] = it.next()
    elif len(kern.shape)==4:
        kern=kern.reshape(kern.shape[0],kern.shape[1],-1)
        for k in range(kern.shape[0]):
            for m in range(kern.shape[1]):
                it = reversed(kern[k,m,:])
                for i in range(kshp[0]):
                    for j in range(kshp[1]):
                        flip[k,m,i,j] = it.next()
    else:
        raise NotImplementedError()
    
    return flip

class TestConvOp(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_convolution(self):
        print '\n\n*************************************************'
        print '           TEST CONVOLUTION' 
        print '*************************************************'

        # fixed parameters
        bsize = 10     # batch size
        imshp = (28,28)# image shape
        print >> sys.stderr, "WARNING: only square shape tested"
        kshps = [(5,5),(6,7),(12,8)]   # kernel shaped
        nkern = 5      # nb kernel
        ssizes = ((1,1),(2,2),(3,3),(4,4))#step size
        convmodes = ('full','valid')
      
        # TODO: ask Fred about this
        # this combination trigered a bug.
        #        bsize=1
        #        imshp=(9,9)#fail with 9,9
        #        kshp=(2,2)
        #        nkern=5
        #        ssizes=((1,1),)
        # this combination trigered a bug.
        #        bsize = 1     # batch size
        #        imshp = (3,3)# image shape
        #        kshp = (2,3)#(5,5)   # kernel shaped
        #        nkern = 1      # nb kernel
        #        ssizes = ((1,1),)#(2,2),(3,3),(4,4))#step size
        #        convmodes = ('full','valid')

        # symbolic stuff
        bias = T.dvector()
        kerns = T.dmatrix()
        input = T.dmatrix()
        rng = N.random.RandomState(3423489)
        biasvals = rng.randn(nkern)

        #profmode = wraplinker.ProfileMode(OpWiseCLinker(), 'fast_run') 
        tconvop, tscipy, tconv2 = [], [], []

        for conv_mode in convmodes:
            for kshp in kshps:

                filters = rng.randn(nkern,N.prod(kshp))

                for ss in ssizes:

                    # now test with real values
                    img2d = N.arange(bsize*N.prod(imshp)).reshape((bsize,)+imshp)
                    img1d = img2d.reshape(bsize,-1)

                    # create filters (need to be flipped to use convolve2d)
                    filtersflipped = flip(filters.reshape((nkern,)+kshp), kshp)

                    # compute with new convolve2 (no timing info)
                    output4, outshp4  = convolve2(kerns, kshp, nkern, input,\
                            imshp, bsize, (1,1), bias=bias, mode=conv_mode)

                    ttime1 = time.time()
                    f = function([kerns, bias, input], output4)
                    out4 = f(filtersflipped.reshape(nkern,-1), biasvals, img1d)
                    tconv2 += [time.time() - ttime1]
                    out4 = out4.reshape(bsize, nkern, outshp4[1], outshp4[2])
                    out4 = out4[:,:,0::ss[0],0::ss[1]]
                    out4 = out4.reshape(bsize, -1)

                    # compute with ConvOp
                    dmatrix3=T.TensorType('float64', (False, False, False))
                    inputs=dmatrix3()
                    kerns3=dmatrix3()
                    bia=T.dscalar()
                    conv_op = ConvOp(imshp, kshp, nkern, bsize, 1,1, conv_mode)(inputs, kerns3)
                    f2 = function([inputs, kerns3], conv_op, mode=Mode(linker="c"))
                    f3 = function([inputs, kerns3], conv_op, mode=Mode(linker="py"))

                    ttime1 = time.time()
                    out2_ = f2(img2d, filtersflipped)
                    out2__ = out2_[:,:,0::ss[0],0::ss[1]]
                    tconvop += [time.time() - ttime1]
                    out2___ = out2__.copy()
                    out2 = out2___ + biasvals.reshape(1,nkern,1,1)
                    out3_ = f3(img2d, filtersflipped)
                    out3__ = out3_[:,:,0::ss[0],0::ss[1]]
                    out3___ = out3__.copy()
                    out3 = out3___ + biasvals.reshape(1,nkern,1,1)
                    assert (N.abs(out2_-out3_)<1e-5).all()

                    # REFERENCE IMPLEMENTATION: compute output with convolve2d
                    fulloutshp = N.array(imshp) - N.array(kshp) + 1 if conv_mode=='valid'\
                             else N.array(imshp) + N.array(kshp) - 1
                    ntime1 = time.time()
                    refout = N.zeros((bsize,)+tuple(fulloutshp)+(nkern,))
                    for b in range(bsize):
                        for n in range(nkern):
                            refout[b,...,n] = convolve2d(\
                                    img2d[b,:,:], filtersflipped[n,...],conv_mode)
                    tscipy += [time.time() - ntime1]

                    # need to flatten images
                    bench1 = refout[:,0::ss[0],0::ss[1],:].reshape(bsize,-1,nkern)
                    bench1 += biasvals.reshape(1,1,nkern)

                    # swap the last two dimensions (output needs to be nkern x outshp)
                    bench1 = N.swapaxes(bench1,1,2)

                    # compare benchmark with ConvOp
                    temp = bench1.flatten() - out2.flatten()
                    assert (temp < 1e-5).all()
                    # compare benchmark with convolve2
                    temp = bench1.flatten() - out4.flatten()
                    assert (temp < 1e-5).all()
                    
        print '**** Convolution Profiling Results ****'
        print 'Scipy convolve2d processing time: %.3fs'%sum(tscipy),tscipy
        print 'ConvOp processing time: %.3fs'%sum(tconvop),tconvop
        print 'convolve2 processing time: %.3fs'%sum(tconv2),tconv2

        d=N.asarray(tscipy)/tconvop
        print 'speed up ConvOp vs convolve2d: %.3f'%d.mean(),d

    def test_ConvOpGrad(self):
        nkern = 3
        bsize = 2
        imgs  = T.dmatrix('imgs')
        kerns = T.dmatrix('kerns')
     
        for mode in 'valid', 'full':
            for imshp in (2,5,5),(2,10,10): # (12,10), (3,12,11):
                visdim = 1 if len(imshp)!=3 else imshp[0]
                for kshp in (3,3),:# (6,7):
                    imgvals = N.random.random(N.hstack((bsize,imshp)))
                    print 'imgvals.shape = ', imgvals.shape
                    imgvals = imgvals.reshape(bsize,-1)

                    kernvals = N.random.rand(nkern,visdim,kshp[0],kshp[1])
                    print 'kernvals.shape = ', kernvals.shape
                    kernvals = kernvals.reshape(nkern,-1)

                    def testf(imgs, kerns):
                        out, outshp = convolve2(kerns, kshp, nkern, imgs, 
                                                   imshp, bsize, mode=mode)
                        return out
            

                    utt.verify_grad(testf, [imgvals, kernvals])

    def test_ConvOpGrad32(self):
        nkern = 4
        bsize = 3
        imgs  = T.fmatrix('imgs')
        kerns = T.fmatrix('kerns')
     
        def testf(imgs, kerns):
            out, outshp = convolve2(kerns, kshp, nkern, imgs, 
                                       imshp, bsize, mode='valid')
            return out
            
        for mode in 'valid', 'full':
            for imshp in (1,5,5),(2,10,10): # (12,10), (3,12,11):
                visdim = 1 if len(imshp)!=3 else imshp[0]
                for kshp in (3,3),:# (6,7):
                    imgvals = N.random.random(N.hstack((bsize,imshp)))
                    print 'imgvals.shape = ', imgvals.shape
                    imgvals = imgvals.reshape(bsize,-1)

                    kernvals = N.random.rand(nkern,visdim,kshp[0],kshp[1])
                    print 'kernvals.shape = ', kernvals.shape
                    kernvals = kernvals.reshape(nkern,-1)

                    utt.verify_grad(testf, [imgvals, kernvals])
