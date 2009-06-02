import sys, time, unittest

import numpy
import numpy as N

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
        from scipy.signal import convolve2d

        if 0:
            # fixed parameters
            bsize = 10     # batch size
            imshp = (28,28)# image shape
            kshps = [(5,5),(6,7),(12,8)]   # kernel shaped
            nkern = 5      # nb kernel
            ssizes = ((1,1),(2,2),(3,3),(4,4))#step size
            convmodes = ('full','valid')
        elif 0:
            # fixed parameters
            bsize = 10     # batch size
            imshp = (50,50)# image shape
            print >> sys.stderr, "WARNING: only square shape tested"
            kshps = [(12,12), (12,12)]
            nkern = 20      # nb kernel
            ssizes = [(1,1)] #step size
            convmodes = ('full','valid')
        elif 0:
            # fixed parameters
            bsize = 7     # batch size
            imshp = (5,4)# image shape
            print >> sys.stderr, "WARNING: only square shape tested"
            kshps = [(2,3)]
            nkern = 6      # nb kernel
            ssizes = [(1,1)] #step size
            convmodes = ('valid',)
        else:
            # fixed parameters
            bsize = 7     # batch size
            imshp = (5,4)# image shape
            print >> sys.stderr, "WARNING: only square shape tested"
            kshps = [(2,3)]
            nkern = 6      # nb kernel
            ssizes = [(1,1)] #step size
            convmodes = ('valid',)
      
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
                    img2d = 1 + N.arange(bsize*N.prod(imshp)).reshape((bsize,)+imshp)
                    print 'img2d', img2d
                    img1d = img2d.reshape(bsize,-1)

                    # create filters (need to be flipped to use convolve2d)
                    filtersflipped = flip(filters.reshape((nkern,)+kshp), kshp)

                    # compute with new convolve2 (no timing info)
                    output4, outshp4  = convolve2(kerns, kshp, nkern, input,\
                            imshp, bsize, (1,1), bias=bias, mode=conv_mode)
                    print 'output4', output4

                    ttime1 = time.time()
                    f = function([kerns, bias, input], output4)
                    out4 = f(filtersflipped.reshape(nkern,-1), biasvals, img1d)
                    print 'out4', out4, img1d, filtersflipped
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

    def test_multilayer_conv(self):
        # causes an atexit problem
        from scipy.signal.sigtools import _convolve2d
        from scipy.signal.signaltools import  _valfrommode, _bvalfromboundary
        
        # fixed parameters
        bsize = 1 # batch size
        imshp_start = (1,28,28)
        kshps = ([5,6],[7,4])
        nkerns = [20,40] # per output pixel
        ssizes = [(1,1),(2,2)]
        convmodes = ['valid','full']
        do_theano=True

        # TODO: this version show a bug.
        imshp_start = (1,4,4)
        kshps = ([2,2],[2,2])#,[7,4])
        nkerns = [2,2] # per output pixel
        ssizes = [(1,1),(2,2)]#2,2)]

        #test speed
        bsize = 10 # batch size
        imshp_start = (1,50,49)
        kshps = ([11,12],[12,11])
        nkerns = [20,20] # per output pixel
        ssizes = [(1,1),]#(1,1)]#(2,2) bugged
        convmodes = ['valid','full']
        do_theano=False

        N.set_printoptions(threshold=N.nan)

        # symbolic stuff
        kerns = [T.matrix(),T.dmatrix()]
        img = T.dmatrix()
        rng = N.random.RandomState(3423489)
        tctot, tpytot, t2ctot, t2pytot, ntot, convtot = [], [], [], [], [], []

        dmatrix4=T.TensorType('float64', (False, False, False, False))
        inputs4=dmatrix4()
        kerns4=dmatrix4()
        assert len(kshps)==len(nkerns)==len(kerns)

        for conv_mode, n_mode in zip(convmodes,range(len(convmodes))):
            for ss, n_ss in zip(ssizes,range(len(ssizes))):

                # build actual input images
                imgval = rng.rand(bsize, imshp_start[0], imshp_start[1], imshp_start[2])
                imshp=imshp_start

                # for each layer
                for kshp, kern, nkern, n_layer in zip(kshps, kerns, nkerns, range(len(kerns))):

                    print '************* layer %i ***************' % n_layer

                    print conv_mode, ss, n_layer, kshp, nkern

                    # actual values
                    w = rng.random_sample(N.r_[nkern,imshp[0],kshp])
                    w_flip = flip(w,kshp).reshape(w.shape)

                    ## manual implementation
                    # check first stage
                    padimg = imgval
                    if conv_mode == 'full':
                        padimg_shp = N.array(imshp[1:]) + 2*(N.array(kshp) - N.array([1,1]))
                        padimg = N.zeros(N.r_[bsize,imshp[0],padimg_shp])
                        padimg[:, :, kshp[0]-1:-kshp[0]+1, 
                                     kshp[1]-1:-kshp[1]+1] = imgval

                    outshp = N.hstack((nkern, getFilterOutShp(imshp, kshp, ss, conv_mode)))

                    time1 = time.time()
                    outval = N.zeros(N.r_[bsize,outshp])
                    val = _valfrommode(conv_mode)
                    bval = _bvalfromboundary('fill')
                    for b in range(bsize): # loop over batches
                        for n in range(nkern): # loop over filters
                            for i in range(imshp[0]): # loop over input feature maps
                                outval[b,n,...] +=  _convolve2d(\
                                    imgval[b,i,...], w_flip[n,i,...],1,val, bval, 0)[0::ss[0],0::ss[1]]
                    ntot += [time.time() - time1]

                    if do_theano:
                        ####### test with new sp.convolve2 function ######
                        time1 = time.time()
                        hid, outshp2 = convolve2(kern, kshp, nkern, img, imshp,  
                                                 bsize, (1,1), mode=conv_mode)
                        propup = function([kern, img], hid)
                        propup1 = function([kern, img], hid,mode=Mode(linker="py"))
                        
                        hidval  = propup(w_flip.reshape(nkern,-1), imgval.reshape(bsize,-1))
                        hidval  = hidval.reshape(bsize,nkern,outshp2[-2],outshp2[-1])[:,:,::ss[0],::ss[1]]
                        hidval = hidval.reshape(bsize, -1)

                        hidval1 = propup1(w_flip.reshape(nkern,-1), imgval.reshape(bsize,-1))
                        hidval1  = hidval1.reshape(bsize,nkern,outshp2[-2],outshp2[-1])[:,:,::ss[0],::ss[1]]
                        hidval1 = hidval1.reshape(bsize, -1)

                        assert (N.abs(hidval-hidval1)<1e-5).all()
                        temp = N.abs(outval.reshape(bsize,-1) - hidval)
                        assert (temp < 1e-5).all()
 
                    else:
                        hid = img #we don't need it, but it make the flow easier flow
                        convtot+=[-1]
                        tctot+=[-1]
                        tpytot+=[-1]
                        hidval=outval.copy()#to keep the same memory
                        hidval1=outval.copy()
                    
                    # ConvOp
                    conv_op = ConvOp(imshp, kshp, nkern, bsize, 1,1, conv_mode, unroll_kern=10)(inputs4, kerns4)
                    l1shp=N.hstack((nkern,
                                    getFilterOutShp(imshp, kshp, ss, conv_mode)))
                    propup2 = function([inputs4, kerns4], conv_op)
                    propup3 = function([inputs4, kerns4], conv_op, mode=Mode(linker="py"))
                    
                    time1 = time.time()
                    hidval2_ = propup2(imgval,w_flip)
                    hidval2 = hidval2_[:,:,0::ss[0],0::ss[1]]
                    t2ctot += [time.time() - time1]

                    time1 = time.time()
#                    hidval3_ = propup3(imgval,w_flip)
#                    hidval3 = hidval3_[:,:,0::ss[0],0::ss[1]]
                    t2pytot += [time.time() - time1]
#                    assert (N.abs(hidval2-hidval3)<1e-5).all()

                    temp = N.abs(outval - hidval2)
                    assert (temp < 1e-5).all()
#                    temp = N.abs(outval - hidval3)
#                    assert (temp < 1e-5).all()

                    img, imshp = hid, tuple(outshp)
                    imgval = outval.reshape(bsize,outshp[0],outshp[1],outshp[2])

        print '**** Multilayer Convolution Profiling Results ****'
        print 'Numpy convolve2d processing time: %.3fs'%sum(ntot),ntot
        print 'c Theano(ConvOp) processing time: %.3fs'%sum(t2ctot),t2ctot
        print 'py Theano(ConvOp) processing time: %.3fs'%sum(t2pytot),t2pytot
        print 'convolve processing time: %.3fs'%sum(convtot),convtot
        d=N.asarray(ntot)/t2ctot
        print 'speed up c theano(ConvOp) vs convolve2d: %.3f'%d.mean(),d
        d=N.asarray(ntot)/t2pytot
        print 'speed up py theano(ConvOp) vs convolve2d: %.3f'%d.mean(),d


    def test_ConvOpGrad(self):
        nkern = 3
        bsize = 2
        imgs  = T.dmatrix('imgs')
        kerns = T.dmatrix('kerns')
        kshps = [(3,3), (5,5)]

        for mode in 'valid', 'full':

            for imshp in (5,5),(2,3,3),(3,6,6): # (12,10), (3,12,11):
                # 'full' mode should support kernels bigger than the input
                if mode == 'valid' and (kshps[0] > imshp[1]):
                    continue

                visdim = 1 if len(imshp)!=3 else imshp[0]
                for kshp in kshps:
                    imgvals = N.random.random(N.hstack((bsize,imshp)))
                    print 'imgvals.shape = ', imgvals.shape
                    imgvals = imgvals.reshape(bsize,-1)

                    if visdim == 1: 
                        kernvals = N.random.rand(nkern,kshp[0],kshp[1])
                    else:
                        kernvals = N.random.rand(nkern,visdim,kshp[0],kshp[1])
                    kernvals = kernvals.reshape(nkern,-1)

                    def testf(imgs, kerns):
                        out, outshp = convolve2(kerns, kshp, nkern, imgs, 
                                                   imshp, bsize, mode=mode)
                        return out

                    try:
                        utt.verify_grad(testf, [imgvals, kernvals])
                    except NotImplementedError, e:
                        print e

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
