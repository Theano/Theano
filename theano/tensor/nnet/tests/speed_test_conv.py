import sys, time, unittest

import numpy
import numpy as N

from theano.tests import unittest_tools as utt

from theano import function, Mode
import theano.tensor as T
from theano.tensor.nnet.conv import ConvOp

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

def exec_multilayer_conv_nnet(conv_mode, ss, bsize, imshp, kshps, nkerns, 
        unroll_batch=0, unroll_kern=0, img=T.dmatrix(), validate=True, 
        conv_op_py=False, do_print=True, repeat=1, 
        unroll_patch=False, unroll_patch_size=False, verbose=0):

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

            outshp = N.hstack((nkern, ConvOp.getOutputShape(imshp, kshp, ss, conv_mode)))

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
                            ConvOp.getOutputShape(imshp, kshp, ss, conv_mode)))
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


def speed_multilayer_conv():
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

if __name__ == '__main__':
    speed_multilayer_conv()
