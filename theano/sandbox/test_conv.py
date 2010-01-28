import sys, time, unittest

import numpy
import numpy as N

from theano.tests import unittest_tools as utt

from theano import function, Mode
import theano.tensor as T
from conv import ConvOp, getFilterOutShp

def flip(kern, kshp):
    "flip the kernel as scipy.convolv2d do it flipped."
    flip = N.zeros(kern.shape)
    if len(kern.shape)==2:
        kern=kern.reshape(-1)
        it = reversed(kern)
        for i in range(kshp[0]):
            for j in range(kshp[1]):
                flip[i,j] = it.next()
    elif len(kern.shape)==3:
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

global_rng = N.random.RandomState(3423489)

dmatrix4=T.TensorType('float64', (False, False, False, False))
def exec_multilayer_conv_nnet(conv_mode, ss, bsize, imshp, kshps, nkerns, unroll_batch=0, unroll_kern=0, img=T.dmatrix(), validate=True, conv_op_py=False, do_print=True, repeat=1, unroll_patch=False, unroll_patch_size=False, verbose=0):

        # build actual input images
        imgval = global_rng.rand(bsize, imshp[0], imshp[1], imshp[2])

        a=T.dmatrix()
        kerns = [a for i in nkerns]
        inputs4=dmatrix4()
        kerns4=dmatrix4()

        # for each layer
        ntot=0 
        tctot=0
        tpytot=0

        for kshp, kern, nkern, n_layer in zip(kshps, kerns, nkerns, range(len(nkerns))):
            if do_print:
                print '************* layer %i ***************' % n_layer
                
                print conv_mode, ss, n_layer, kshp, nkern

            # actual values
            w = global_rng.random_sample(N.r_[nkern,imshp[0],kshp])
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
            if validate:
                # causes an atexit problem
                from scipy.signal.sigtools import _convolve2d
                from scipy.signal.signaltools import  _valfrommode, _bvalfromboundary
                val = _valfrommode(conv_mode)
                bval = _bvalfromboundary('fill')
                for b in range(bsize): # loop over batches
                    for n in range(nkern): # loop over filters
                        for i in range(imshp[0]): # loop over input feature maps
                            outval[b,n,...] +=  _convolve2d(\
                                imgval[b,i,...], w_flip[n,i,...],1,val, bval, 0)[0::ss[0],0::ss[1]]
                ntot += time.time() - time1

            # ConvOp
            if unroll_patch and not unroll_patch_size:
                conv_op = ConvOp(dx=ss[0],dy=ss[1], output_mode=conv_mode,
                                 unroll_patch=unroll_patch, verbose=verbose)(inputs4, kerns4)
            else:
                conv_op = ConvOp(imshp, kshp, nkern, bsize, ss[0],ss[1], conv_mode,
                                 unroll_batch=unroll_batch, unroll_kern=unroll_kern, unroll_patch=unroll_patch, verbose=verbose)(inputs4, kerns4)
            l1shp=N.hstack((nkern,
                            getFilterOutShp(imshp, kshp, ss, conv_mode)))
            propup2 = function([inputs4, kerns4], conv_op)
            propup3 = function([inputs4, kerns4], conv_op, mode=Mode(linker="py"))

            time1 = time.time()
            for i in range(repeat):
                hidval2_ = propup2(imgval,w_flip)
            hidval2 = hidval2_#[:,:,0::ss[0],0::ss[1]]
            tctot += time.time() - time1

            if conv_op_py:
                time1 = time.time()
                for i in range(repeat):
                    hidval3_ = propup3(imgval,w_flip)
                hidval3 = hidval3_#[:,:,0::ss[0],0::ss[1]]
                tpytot += time.time() - time1
                assert (N.abs(hidval2-hidval3)<1e-5).all()
            else:
                tpytot += 0

            if validate:
                temp = N.abs(outval - hidval2)
                assert (temp < 1e-5).all()
            if validate and conv_op_py:
                temp = N.abs(outval - hidval3)
                assert (temp < 1e-5).all()

            imshp = tuple(outshp)
            imgval = outval.reshape(bsize,outshp[0],outshp[1],outshp[2])

        return tctot, tpytot, ntot



class TestConvOp(unittest.TestCase):
    """NOTE: we test only when we pass 4d tensor.
    """

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
            imshp = (1,28,28)# image shape
            kshps = [(5,5),(6,7),(12,8)]   # kernel shaped
            nkern = 5      # nb kernel
            ssizes = ((1,1),(2,2),(3,3),(4,4))#step size
            convmodes = ('full','valid')
        elif 0:
            # fixed parameters
            bsize = 10     # batch size
            imshp = (1,50,50)# image shape
            print >> sys.stderr, "WARNING: only square shape tested"
            kshps = [(12,12), (12,12)]
            nkern = 20      # nb kernel
            ssizes = [(1,1)] #step size
            convmodes = ('full','valid')
        elif 0:
            # fixed parameters
            bsize = 7     # batch size
            imshp = (1,5,4)# image shape
            kshps = [(2,3)]
            nkern = 6      # nb kernel
            ssizes = [(1,1)] #step size
            convmodes = ('valid',)
        else:
            # fixed parameters
            bsize = 7     # batch size
            imshp = (1,5,4)# image shape
            kshps = [(2,3)]
            nkern = 6      # nb kernel
            ssizes = [(1,1)] #step size
            convmodes = ('valid',)
      
        # TODO: ask Fred about this
        # this combination trigered a bug.
        #        bsize=1
        #        imshp=(1,9,9)#fail with 9,9
        #        kshp=(2,2)
        #        nkern=5
        #        ssizes=((1,1),)
        # this combination trigered a bug.
        #        bsize = 1     # batch size
        #        imshp = (1,3,3)# image shape
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
#                    print 'img2d', img2d
                    img1d = img2d.reshape(bsize,-1)

                    # create filters
                    filtersflipped = flip(filters.reshape((nkern,)+kshp), kshp)

                    # compute with ConvOp
                    dmatrix3=T.TensorType('float64', (False, False, False))
                    inputs4=dmatrix4()
                    kerns4=dmatrix4()
                    bia=T.dscalar()
                    conv_op = ConvOp(imshp, kshp, nkern, bsize, ss[0],ss[1], conv_mode)(inputs4, kerns4)
                    f2 = function([inputs4, kerns4], conv_op, mode=Mode(linker="c"))
                    f3 = function([inputs4, kerns4], conv_op, mode=Mode(linker="py"))

                    ttime1 = time.time()
                    out2_ = f2(img2d, filtersflipped.reshape(nkern,1,*kshp))
                    out2__ = out2_
                    tconvop += [time.time() - ttime1]
                    out2___ = out2__.copy()
                    out2 = out2___ + biasvals.reshape(1,nkern,1,1)
                    out3_ = f3(img2d, filtersflipped.reshape(nkern,1,*kshp))
                    out3__ = out3_
                    out3___ = out3__.copy()
                    out3 = out3___ + biasvals.reshape(1,nkern,1,1)
                    assert (N.abs(out2_-out3_)<1e-5).all()

                    # REFERENCE IMPLEMENTATION: compute output with convolve2d
		    if conv_mode=='valid':
			fulloutshp = N.array(imshp[1:]) - N.array(kshp) + 1
		    else:
			fulloutshp = N.array(imshp[1:]) + N.array(kshp) - 1
                    ntime1 = time.time()
                    refout = N.zeros((bsize,)+tuple(fulloutshp)+(nkern,))
                    for b in range(bsize):
                        for n in range(nkern):
                            refout[b,...,n] = convolve2d(\
                                    img2d[b,0,:,:], filtersflipped[n,...],conv_mode)
                    tscipy += [time.time() - ntime1]

                    # need to flatten images
                    bench1 = refout[:,0::ss[0],0::ss[1],:].reshape(bsize,-1,nkern)
                    bench1 += biasvals.reshape(1,1,nkern)

                    # swap the last two dimensions (output needs to be nkern x outshp)
                    bench1 = N.swapaxes(bench1,1,2)

                    # compare benchmark with ConvOp
                    temp = bench1.flatten() - out2.flatten()
                    assert (temp < 1e-5).all()
                    
        print '**** Convolution Profiling Results ****'
        print 'Scipy convolve2d processing time: %.3fs'%sum(tscipy),tscipy
        print 'ConvOp processing time: %.3fs'%sum(tconvop),tconvop
        print 'convolve2 processing time: %.3fs'%sum(tconv2),tconv2

        d=N.asarray(tscipy)/tconvop
        print 'speed up ConvOp vs convolve2d: %.3f'%d.mean(),d

    def speed_multilayer_conv(self):
            # calculate the speed up of different combination of unroll
            # put the paramter to the same you will try. 
            
            validate=False# we don't validate the result to have it much faster!
            verbose=1
            unroll_batch = [1,2,4,5,10,20]
            unroll_kern = [1,2,4,5,10,20]
            unroll_batch = [1,4,5]
            unroll_kern = [1,4,5]
            unroll_patch = [True, False]
            
            bsize = 20 # batch size
            imshp_start = (1,48,48)#un square shape to test more corner case.
            kshps = ([11,12],[12,11])#un square shape to test more corner case.
            nkerns = [20,20] # per output pixel
            ssizes = [(1,1),]#(1,1)]#(2,2) bugged
            convmodes = ['valid','full']
            do_convolve2=False
            a=T.dmatrix()
            kerns = [a for i in nkerns]

            assert len(kshps)==len(nkerns)==len(kerns)
        
            timing = N.zeros((len(unroll_batch),len(unroll_kern),3,len(convmodes)*len(ssizes)))
            t_b_k=[]
            #calculate the timing with unrolling

            print 'time unroll batch kern'
            t_=[[ 7.60572791,  3.95069814,  3.74271464], [ 4.05631089,  2.90384555,  2.93613672], [ 3.90551591,  2.92595196,  3.00102282]]
            best=[0.52690219879150391, 2.4266397953033447]
            worst=[0.92042708396911621, 6.8822150230407715]
            best=[]
            worst=[]
            t_=[]
            for unroll_b, n_b in zip(unroll_batch,range(len(unroll_batch))):
                for unroll_k, n_k in zip(unroll_kern,range(len(unroll_kern))):
                    t_b_k.append(str(unroll_b)+"/"+str(unroll_k))
                    if not t_:
                        tctot, tpytot, ntot=[],[],[]
                        for conv_mode, n_mode in zip(convmodes,range(len(convmodes))):
                            for ss, n_ss in zip(ssizes,range(len(ssizes))):
                                tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet(conv_mode, ss, bsize, imshp_start, kshps, nkerns, unroll_batch=unroll_b, unroll_kern=unroll_k, validate=validate, verbose=verbose,do_print=False)
                                tctot+=[tctot_]
                                tpytot+=[tpytot_]
                                ntot+=[ntot_]
                        if unroll_b==4 and unroll_k==4:
                            #print "unroll 4/4",tctot
                            best=tctot
                        if unroll_b==1 and unroll_k==1:
                            #print "unroll 1/1",tctot
                            worst=tctot
                        timing[n_b,n_k]=[tctot, tpytot, ntot]#[sum(tctot), sum(tpytot), sum(ntot)]
            if not t_:
                t=timing[:,:,0,:]#We select only the c timing.
            else:
                t=t_
            t=N.asarray(t)
            #calculate the old timing
            print 'time old version'
            tctot_=[0.52555489540100098, 6.6634182929992676]
            tctot,tpytot,ntot=[],[],[]
            tctot_=[]
            if not tctot_:
                for conv_mode, n_mode in zip(convmodes,range(len(convmodes))):
                    for ss, n_ss in zip(ssizes,range(len(ssizes))):
                        tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet(conv_mode, ss, bsize, imshp_start, kshps, nkerns, unroll_batch=0, unroll_kern=0, validate=validate, verbose=verbose,do_print=False)
                        tctot+=[tctot_]
                        tpytot+=[tpytot_]
                        ntot+=[ntot_]
            else: tctot=N.asarray(tctot_)
            print "old code timing %.3fs"%sum(tctot),tctot
            best=N.asarray(best)
            worst=N.asarray(worst)
            print "timing for unrolled version"
            print t_b_k
            print t
            t_detail=t
            t = t.sum(axis=2)
            print "max %.3fs"%t.max(), "max param(batch unloop size/kernel unloop size)", t_b_k[t.argmax()]
            print "min %.3fs"%t.min(), "min param(batch unloop size/kernel unloop size)", t_b_k[t.argmin()]
            print "speedup vs (1/1)%.3fx, vs old %.3fx"% (t.max()/t.min(),sum(tctot)/t.min())
            print worst/best,tctot/best

            #calculate the timing of unroll_patch
            print 'time unroll_patch'
            tctot_patch = []
            tctot_patch_size = []
            for conv_mode, n_mode in zip(convmodes,range(len(convmodes))):
                for ss, n_ss in zip(ssizes,range(len(ssizes))):
                    tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet(conv_mode, ss, bsize, imshp_start, kshps, nkerns, unroll_batch=0, unroll_kern=0, validate=validate,unroll_patch=True,verbose=verbose,do_print=False)
                    tctot_patch += [tctot_]
                    tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet(conv_mode, ss, bsize, imshp_start, kshps, nkerns, unroll_batch=0, unroll_kern=0, validate=validate,unroll_patch=True,verbose=verbose,do_print=False,unroll_patch_size=True)
                    tctot_patch_size += [tctot_]

            t_patch=sum(tctot_patch)
            print "unroll_patch without shape time", tctot_patch
            print "speedup vs (1/1)%.3fx, vs old %.3fx"% (t.max()/t_patch,sum(tctot)/t_patch)
            print best/tctot_patch, worst/tctot_patch
            t_patch_size=sum(tctot_patch_size)
            print "unroll_patch with shape time", tctot_patch_size
            print "speedup vs (1/1)%.3fx, vs old %.3fx"% (t.max()/t_patch_size,sum(tctot)/t_patch_size)
            print best/tctot_patch_size, worst/tctot_patch_size
            
            return

    def test_multilayer_conv(self):
        print '\n\n*************************************************'
        print '           TEST MULTILAYER CONVOLUTION' 
        print '*************************************************'

        # fixed parameters
        # test multiple configuration at the same time
        bsizes = [6,6] # batch size
        imshp_starts = [(1,13,14),(1,4,5)]
        kshpss = ([[5,6],[7,4]],[[2,2],[2,2]])
        nkernss = [[20,40],[2,2]] # per output pixel
        ssizess = [[(1,1),(1,2)],[(1,1),(2,2)]]
        convmodes = ['valid','full']
        do_convolve2=True
        unroll = [(0,0,True),(0,0,False),(1,1,False),(2,2,False),(3,2,False)]#(batch,kern,patch)

        # TODO: this version show a bug that was fixed
        # the test is included in the upper test.
#        imshp_start = (1,4,4)
#        kshps = ([2,2],[2,2])#,[7,4])
#        nkerns = [2,2] # per output pixel
#        ssizes = [(1,1),(2,2)]#2,2)]

#        bsizes = [1,1] # batch size
#        imshp_starts = [(1,10,10),(1,5,6)]
#        kshpss = ([[2,3],[3,2]],[[2,2],[2,2]])
#        nkernss = [[1,1],[1,1]] # per output pixel

        N.set_printoptions(threshold=N.nan)

        # symbolic stuff
        kerns = [T.matrix(),T.dmatrix()]
        img = T.dmatrix()
        rng = N.random.RandomState(3423489)
        tctot, tpytot, ntot = [], [], []
        for i in range(len(kshpss)):
            assert len(kshpss[i])==len(nkernss[i])==len(kerns)

        for i in range(len(kshpss)):
            for conv_mode, n_mode in zip(convmodes,range(len(convmodes))):
                for ss, n_ss in zip(ssizess[i],range(len(ssizess[i]))):
                    for un_b, un_k, un_p in unroll:
                        tctot_, tpytot_, ntot_ = exec_multilayer_conv_nnet(
                            conv_mode, ss, bsizes[i], imshp_starts[i], 
                            kshpss[i], nkernss[i],
                            img=img, unroll_batch=un_b, unroll_kern=un_k,
                            unroll_patch=un_p,
                            validate=True)
                        tctot+=[tctot_]
                        tpytot+=[tpytot_]
                        ntot+=[ntot_]

        print '**** Multilayer Convolution Profiling Results ****'
        print 'Numpy convolve2d processing time: %.3fs'%sum(ntot),ntot
        print 'c Theano(ConvOp) processing time: %.3fs'%sum(tctot),tctot
        print 'py Theano(ConvOp) processing time: %.3fs'%sum(tpytot),tpytot
        d=N.asarray(ntot)/tctot
        print 'speed up c theano(ConvOp) vs convolve2d: %.3fx'%d.mean(),d
        d=N.asarray(ntot)/tpytot
        print 'speed up py theano(ConvOp) vs convolve2d: %.3fx'%d.mean(),d


    def init_data(self,shape):
        return N.ones(shape)
        return N.random.random(shape)
        
    def test_ConvOpGrad(self):
        """
        test the gradient in float and double
        """
        print '\n\n*************************************************'
        print '           TEST ConvOp.grad' 
        print '*************************************************'

        nkern = 3
        bsize = 2
        types = ["float32", "float64"]
        kshps = [(2,3)]
        imshps = [(2,3,4)]
        modes = ['valid', 'full']
        unroll = [(0,0,True),(1,1,False),(2,3,False),(1,1,False),(0,0,False)]#(batch,kern,patch)
        ssizes = [(1,1)]#,(2,2)]#grad for ss!=(1,1) is currently disabled!

        for typ in types:
            imgs  = T.TensorType(typ, (False, False, False, False),'imgs')
            kerns = T.TensorType(typ, (False, False, False, False),'kerns')
            for mode in modes:
                for imshp in imshps:
		    if len(imshp)!=3:
			visdim = 1
		    else:
		        visdim = imshp[0]
                    imgvals = N.array(N.random.random(N.hstack((bsize,imshp))),dtype=imgs.dtype)
                    for kshp in kshps:
                        t=numpy.array([imshp[1]-kshp[0],imshp[2]-kshp[1]])
                        kernvals = N.array(self.init_data((nkern,visdim,kshp[0],
                                                          kshp[1])),dtype=kerns.dtype)
                        # 'full' mode should support kernels bigger than the input
                        if mode == 'valid' and (t<0).any():
                            continue
                        for un_b,un_k, un_p in unroll:
                                for ss in ssizes:
                                    print 'test_ConvOpGrad'
#                                    print 'mode:',mode,'type:', typ
#                                    print 'imshp:', imshp,
#                                    print 'kshp:', kshp
#                                    print 'un_b:', un_b,
#                                    print 'un_k:', un_k,
#                                    print 'un_p:', un_p
#                                    print 'ss:', ss,
#                                    print 'bsize:', bsize,
#                                    print 'nkern:', nkern

                                    def test_i(imgs):
                                        if un_p and ss[0]==1 and ss[1]==1:
                                            convop = ConvOp(dx=ss[0], dy=ss[1],
                                                            output_mode=mode, unroll_patch=un_p)
                                        else:
                                            convop = ConvOp(imshp, kshp, nkern, bsize, ss[0], ss[1],
                                                            output_mode=mode, unroll_batch=un_b, unroll_kern=un_k, unroll_patch=un_p)
                                        return convop(imgs, kernvals)

                                    def test_k(kerns):
                                        if un_p and ss[0]==1 and ss[1]==1:
                                            convop = ConvOp(dx=ss[0], dy=ss[1],
                                                            output_mode=mode, unroll_patch=un_p)
                                        else:
                                            convop = ConvOp(imshp, kshp, nkern, bsize, ss[0], ss[1],
                                                            output_mode=mode, unroll_batch=un_b, unroll_kern=un_k, unroll_patch=un_p)
                                        return convop(imgvals, kerns)
                                    print mode, imshp, kshp, un_b, un_k, ss
                                    #TODO the tolerance needed to pass is very high for float32(0.17). Is this acceptable? Expected?
				    tol = None
                                    if typ=="float32" and (ss[0]!=1 or ss[1]!=1):
					tol = 0.1
                                    utt.verify_grad(test_i, [imgvals],
                                                    cast_to_output_type=True,
                                                    tol=tol)

                                    utt.verify_grad(test_k, [kernvals],
                                                    cast_to_output_type=True,
                                                    tol=tol)


if __name__ == '__main__':
    t = TestConvOp("test_convolution")
#    t.test_convolution()
    t.test_multilayer_conv()
#    from theano.tests import main
#    main("test_sp")
    if False:
        #used to lanch 8 jobs at the same time.
        bsize = 20 # batch size
        imshp_start = (1,100,100)#un square shape to test more corner case.
        kshps = ([11,12],[12,11])#un square shape to test more corner case.
        nkerns = [20,20] # per output pixel
        ssizes = [(1,1),]#(1,1)]#(2,2) bugged
        convmodes = ['valid','full']
        unroll_batch = 5
        unroll_kern = 2
        ctot=0
        tctot, tpytot, ntot = exec_multilayer_conv_nnet(convmodes[1], ssizes[0], bsize, imshp_start, kshps, nkerns, unroll_batch=unroll_batch, unroll_kern=unroll_kern, validate=False, do_print=False,repeat=5)
        print "total exec time %.3fs"%tctot
        
