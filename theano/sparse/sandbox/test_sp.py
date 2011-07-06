import sys

from nose.plugins.skip import SkipTest
import theano.sparse
if not theano.sparse.enable_sparse:
    raise SkipTest('Optional package sparse disabled')

from theano import function, Mode
from theano.gof import OpWiseCLinker
import theano, numpy
import theano.tensor as T
import scipy.sparse

from scipy.signal import convolve2d
import scipy.sparse as sparse
import numpy
import numpy as N
from theano.sparse.sandbox import sp
from theano.tests import unittest_tools as utt

import unittest
import time

class TestSP(unittest.TestCase):
    def test_convolution(self):
        print '\n\n*************************************************'
        print '           TEST CONVOLUTION'
        print '*************************************************'

        # fixed parameters
        bsize = 10     # batch size
        imshp = (28,28)
        kshp = (5,5)
        nkern = 5
        ssizes = ((1,1),(2,2),(3,3),(4,4))
        convmodes = ('full','valid')

        # symbolic stuff
        bias = T.dvector()
        kerns = T.dmatrix()
        input = T.dmatrix()
        rng = N.random.RandomState(3423489)
        filters = rng.randn(nkern,N.prod(kshp))
        biasvals = rng.randn(nkern)

        for mode in ('FAST_COMPILE','FAST_RUN'): #, profmode):
            ttot, ntot = 0, 0
            for conv_mode in convmodes:
                for ss in ssizes:

                    output, outshp  = sp.convolve(kerns, kshp, nkern, input,\
                            imshp, ss, bias=bias, mode=conv_mode)
                    f = function([kerns, bias, input], output, mode=mode)

                    # now test with real values
                    img2d = N.arange(bsize*N.prod(imshp)).reshape((bsize,)+imshp)
                    img1d = img2d.reshape(bsize,-1)

                    # create filters (need to be flipped to use convolve2d)
                    filtersflipped = N.zeros((nkern,)+kshp)
                    for k in range(nkern):
                        it = reversed(filters[k,:])
                        for i in range(kshp[0]):
                            for j in range(kshp[1]):
                                filtersflipped[k,i,j] = it.next()

                    # compute output with convolve2d
                    if conv_mode=='valid':
                        fulloutshp = N.array(imshp) - N.array(kshp) + 1
                    else:
                        fulloutshp = N.array(imshp) + N.array(kshp) - 1
                    ntime1 = time.time()
                    refout = N.zeros((bsize,)+tuple(fulloutshp)+(nkern,))
                    for b in range(bsize):
                        for n in range(nkern):
                            refout[b,...,n] = convolve2d(\
                                    img2d[b,:,:], filtersflipped[n,...],conv_mode)
                    ntot += time.time() - ntime1


                    # need to flatten images
                    bench1 = refout[:,0::ss[0],0::ss[1],:].reshape(bsize,-1,nkern)
                    bench1 += biasvals.reshape(1,1,nkern)

                    # swap the last two dimensions (output needs to be nkern x outshp)
                    bench1 = N.swapaxes(bench1,1,2)
                    ttime1 = time.time()
                    out1 = f(filters, biasvals, img1d)
                    ttot += time.time() - ttime1
                    temp = bench1.flatten() - out1.flatten()

                    assert (temp < 1e-5).all()

                    # test downward propagation -- symbolic stuff
                    #vis = T.grad(output, input, output)
                    #downprop = function([kerns,input], vis, mode=mode)
                    #visval = downprop(filters,img1d)
                    ## test downward propagation -- reference implementation
                    #pshape = (img1d.shape[0],N.prod(outshp[1:]),N.prod(kshp))
                    #patchstack = N.zeros(pshape)
                    #for bi in N.arange(pshape[0]): # batch index
                        #abspos = 0
                        #for outy in N.arange(outshp[1]):
                            #for outx in N.arange(outshp[2]):
                                #for ni in N.arange(nkern):
                                    #print 'filters[n,:].shape = ', filters[n,:].shape
                                    #print 'out1[bi,abspos].shape =',out1[bi,abspos].shape
                                    #patchstack[bi,abspos,:] = filters[n,:]*out1[bi,abspos]
                                    #abspos+=1
                    #patchstack = patchstack.reshape(1,-1)
                    #indices, indptr, spmat_shape, sptype, outshp = \
                            #sp.convolution_indices.conv_eval(imshp,kshp,ss,conv_mode)
                    #spmat = sparse.csc_matrix((N.ones_like(indices),indices,indptr),spmat_shape)
                    #visref = N.dot(patchstack, spmat.todense())

                    #print 'visval = ', visval
                    #print 'visref = ', visref

                    #assert N.all(visref==visval)


            print '**** Convolution Profiling Results (',mode,') ****'
            print 'Numpy processing time: ', ntot
            print 'Theano processing time: ', ttot

        #profmode.print_summary()


    def test_sparse(self):

        print '\n\n*************************************************'
        print '           TEST SPARSE'
        print '*************************************************'

        # fixed parameters
        bsize = 10     # batch size
        imshp = (28,28)
        kshp = (5,5)
        nkern = 1 # per output pixel
        ssizes = ((1,1),(2,2))
        convmodes = ('full','valid',)

        # symbolic stuff
        bias = T.dvector()
        kerns = T.dvector()
        input = T.dmatrix()
        rng = N.random.RandomState(3423489)

        import theano.gof as gof
        #Mode(optimizer='fast_run', linker=gof.OpWiseCLinker(allow_gc=False)),):
        for mode in ('FAST_COMPILE','FAST_RUN'): #,profmode):
            ntot, ttot = 0,0
            for conv_mode in convmodes:
                for ss in ssizes:

                    output, outshp = sp.applySparseFilter(kerns, kshp,\
                            nkern, input, imshp, ss, bias=bias, mode=conv_mode)
                    f = function([kerns, bias, input], output, mode=mode)

                    # build actual input images
                    img2d = N.arange(bsize*N.prod(imshp)).reshape((bsize,)+imshp)
                    img1d = img2d.reshape(bsize,-1)
                    zeropad_img = N.zeros((bsize,\
                                           img2d.shape[1]+2*(kshp[0]-1),\
                                           img2d.shape[2]+2*(kshp[1]-1)))
                    zeropad_img[:, kshp[0]-1:kshp[0]-1+img2d.shape[1],
                                   kshp[1]-1:kshp[1]-1+img2d.shape[2]] = img2d

                    # build kernel matrix -- flatten it for theano stuff
                    filters = N.arange(N.prod(outshp)*N.prod(kshp)).\
                                reshape(nkern,N.prod(outshp[1:]),N.prod(kshp))
                    spfilt = filters.flatten()
                    biasvals = N.arange(N.prod(outshp))

                    # compute output by hand
                    ntime1 = time.time()
                    refout = N.zeros((bsize,nkern,outshp[1],outshp[2]))
                    patch = N.zeros((kshp[0],kshp[1]))
                    for b in xrange(bsize):
                        for k in xrange(nkern):
                            pixi = 0 # pixel index in raster order
                            for j in xrange(outshp[1]):
                                for i in xrange(outshp[2]):
                                    n = j * ss[0]
                                    m = i * ss[1]
                                    patch = zeropad_img[b,n:n+kshp[0],m:m+kshp[1]]
                                    refout[b,k,j,i] = N.dot(filters[k,pixi,:],\
                                                            patch.flatten())
                                    pixi += 1
                    refout = refout.reshape(bsize,-1) + biasvals
                    ntot += time.time() - ntime1

                    # need to flatten images
                    ttime1 = time.time()
                    out1 = f(spfilt, biasvals, img1d)
                    ttot += time.time() - ttime1

                    temp = refout - out1
                    assert (temp < 1e-10).all()

                    # test downward propagation
                    vis = T.grad(0.5*T.sqr(output).sum(), input)
                    downprop = function([kerns,output], vis)
                    temp1 = time.time()
                    for zz in range(100):
                        visval = downprop(spfilt,out1)
                    indices, indptr, spmat_shape, sptype, outshp, kmap = \
                            sp.convolution_indices.sparse_eval(imshp,kshp,nkern,ss,conv_mode)
                    spmat = sparse.csc_matrix((spfilt[kmap],indices,indptr),spmat_shape)
                    visref = N.dot(out1,spmat.todense())
                    assert N.all(visref==visval), (visref, visval)

            print '**** Sparse Profiling Results (',mode,') ****'
            print 'Numpy processing time: ', ntot
            print 'Theano processing time: ', ttot
        #profmode.print_summary()


    def test_multilayer_sparse(self):
        # fixed parameters
        bsize = 10     # batch size
        imshp = (5,5)
        kshp = ((3,3),(2,2))
        nkerns = (10,20) # per output pixel
        ssizes = ((1,1),(2,2))
        convmodes = ('full','valid',)

        # symbolic stuff
        kerns = [T.dvector(),T.dvector()]
        input = T.dmatrix()
        rng = N.random.RandomState(3423489)

        # build actual input images
        img2d = N.arange(bsize*N.prod(imshp)).reshape((bsize,)+imshp)
        img1d = img2d.reshape(bsize,-1)

        for mode in ('FAST_COMPILE','FAST_RUN'):
            for conv_mode in convmodes:
                for ss in ssizes:

                    l1hid, l1outshp = sp.applySparseFilter(kerns[0], kshp[0],\
                            nkerns[0], input, imshp, ss, mode=conv_mode)
                    l2hid, l2outshp = sp.applySparseFilter(kerns[1], kshp[1],\
                            nkerns[1], l1hid, l1outshp, ss, mode=conv_mode)

                    l1propup = function([kerns[0], input], l1hid, mode=mode)
                    l2propup = function([kerns[1], l1hid], l2hid, mode=mode)

                    # actual values
                    l1kernvals = N.arange(N.prod(l1outshp)*N.prod(kshp[0]))
                    l2kernvals = N.arange(N.prod(l2outshp)*N.prod(kshp[1])*nkerns[0])
                    l1hidval = l1propup(l1kernvals,img1d)
                    l2hidval = l2propup(l2kernvals,l1hidval)

    # this doesn't compare the output of anything... but I manually verified that the patches
    # are properly generated
    def test_multilayer_conv(self):
        # fixed parameters
        bsize = 10     # batch size
        imshp = (5,5)
        kshp = ((3,3),(2,2))
        nkerns = (3,6) # per output pixel
        ssizes = (((1,1),(2,2)),)
        convmodes = ('full',)#'valid',)

        # symbolic stuff
        kerns = [T.dmatrix(),T.dmatrix()]
        input = T.dmatrix()
        rng = N.random.RandomState(3423489)

        # build actual input images
        img2d = N.arange(bsize*N.prod(imshp)).reshape((bsize,)+imshp)
        img1d = img2d.reshape(bsize,-1)

        for mode in ('FAST_COMPILE','FAST_RUN'):
            for conv_mode in convmodes:
                for ss in ssizes:

                    l1hid, l1shp = sp.convolve(kerns[0], kshp[0],\
                            nkerns[0], input, imshp, ss[0], mode=conv_mode)
                    l1propup = function([kerns[0], input], l1hid, mode=mode)

                    #l1kernvals = N.random.rand(nkerns[0],N.prod(kshp[0]))
                    l1kernvals = N.arange(nkerns[0]*N.prod(kshp[0])).reshape(nkerns[0],N.prod(kshp[0]))
                    l1hidval = l1propup(l1kernvals,img1d)

                    # actual values
                    l2hid, l2shp = sp.convolve(kerns[1], kshp[1],\
                            nkerns[1], l1hid, l1shp, ss[1], mode=conv_mode)
                    l2propup = function([kerns[1], l1hid], l2hid, mode=mode)

                    #l2kernvals = N.random.rand(nkerns[1],N.prod(kshp[1])*nkerns[0])
                    l2kernvals = N.arange(nkerns[1]*N.prod(kshp[1])*nkerns[0]).reshape(nkerns[1],N.prod(kshp[1])*nkerns[0])
                    # for debugging, we bring things back to integers
                    l1hidval = N.arange(N.size(l1hidval)).reshape(l1hidval.shape)

                    l2hidval = l2propup(l2kernvals,l1hidval)



    def test_maxpool(self):
        # generate flatted images
        maxpoolshps = ((2,2),(3,3),(4,4),(5,5),(6,6))
        imval = N.random.rand(4,5,10,10)

        images = T.dmatrix()
        for maxpoolshp in maxpoolshps:

            # symbolic stuff
            output, outshp = sp.max_pool(images, imval.shape[1:], maxpoolshp)
            f = function([images,],[output,])
            output_val = f(imval.reshape(imval.shape[0],-1))

            # numeric verification
            my_output_val = N.zeros((imval.shape[0], imval.shape[1],
                                     imval.shape[2]/maxpoolshp[0],
                                     imval.shape[3]/maxpoolshp[1]))
            assert N.prod(my_output_val.shape[1:]) == N.prod(N.r_[imval.shape[1],outshp])

            for n in range(imval.shape[0]):
                for k in range(imval.shape[1]):
                    for i in range(imval.shape[2]/maxpoolshp[0]):
                        for j in range(imval.shape[3]/maxpoolshp[1]):
                            ii,jj = i*maxpoolshp[0], j*maxpoolshp[1]
                            patch = imval[n,k,ii:ii+maxpoolshp[0],jj:jj+maxpoolshp[1]]
                            my_output_val[n,k,i,j] = N.max(patch)
            my_output_val = my_output_val.reshape(imval.shape[0],-1)
            assert N.all(output_val == my_output_val)

            def mp(input):
                output, outshp = sp.max_pool(input, imval.shape[1:], maxpoolshp)
                return output
            utt.verify_grad(mp, [imval.reshape(imval.shape[0],-1)])


    def test_CSMGrad(self):
        imshp = (3,3)
        nkern = 1 # per output pixel
        kshp = (2,2)
        #ssizes = ((1,1),(2,2))
        ssizes = ((1,1),)
        #convmodes = ('full','valid',)
        convmodes = ('full',)

        kerns = T.dvector()
        indices = T.ivector()
        indptr = T.ivector()
        spmat_shape = T.ivector()

        for mode in ['FAST_COMPILE','FAST_RUN']:
            for conv_mode in convmodes:
                for ss in ssizes:
                    indvals, indptrvals, spshapevals, sptype, outshp, kmap = \
                            sp.convolution_indices.sparse_eval(imshp,kshp,nkern,ss,conv_mode)
                    kvals = N.random.random(nkern*N.prod(kshp)*N.prod(outshp)).flatten()

                    def d(kerns):
                        return theano.sparse.dense_from_sparse(
                                theano.sparse.CSM(sptype,kmap)(
                                    kerns, indvals, indptrvals, spshapevals))

                    # symbolic stuff
                    utt.verify_grad(d, [kvals])


def test_diagonal():
    for K in 1, 5:

        d = T.ivector()

        sd = sp.square_diagonal(d)

        f = theano.function([d], sd)

        n = N.zeros((K,K), dtype='int32')
        for i in range(K):
            n[i,i] = i

        assert N.all(n == f(range(K)).toarray())

def test_diagonal_grad():
    def d(x):
        return sp.sp_sum(sp.square_diagonal(x), sparse_grad=True)
    utt.verify_grad(d, [[0.0, 0.1, 0.2, 0.3]],
            mode=theano.Mode(linker='py', optimizer='fast_compile'))

def test_row_scale():
    x = theano.sparse.csc_dmatrix()
    s = theano.tensor.dvector()

    def d(x,s):
        return sp.sp_sum(sp.row_scale(x, s), sparse_grad=True)

    rng = numpy.random.RandomState(8723)
    R = 5
    C = 8

    x_val_dense = numpy.zeros((R, C),dtype='d')
    for idx in [(0,0), (4, 1), (2,1), (3, 3), (4, 4), (3, 7), (2, 7)]:
        x_val_dense.__setitem__(idx, rng.randn())
    x_val = scipy.sparse.csc_matrix(x_val_dense)

    s_val = rng.randn(R)

    f = theano.function([x, s], sp.row_scale(x, s))

    print 'A', f(x_val, s_val).toarray()
    print 'B', (x_val_dense.T * s_val).T

    assert numpy.all(f(x_val, s_val).toarray() == (x_val_dense.T * s_val).T)

    if 0:
        T.verify_grad(None, d, [x_val, s_val],
                mode=theano.Mode(linker='py', optimizer='fast_compile'))
    else:
        print >> sys.stderr, "WARNING: skipping gradient test because verify_grad doesn't support sparse arguments"

def test_col_scale():
    x = theano.sparse.csc_dmatrix()
    s = theano.tensor.dvector()

    def d(x,s):
        return sp.sp_sum(sp.col_scale(x, s), sparse_grad=True)

    rng = numpy.random.RandomState(8723)
    R = 5
    C = 8

    x_val_dense = numpy.zeros((R, C),dtype='d')
    for idx in [(0,0), (4, 1), (2,1), (3, 3), (4, 4), (3, 7), (2, 7)]:
        x_val_dense.__setitem__(idx, rng.randn())
    x_val = scipy.sparse.csc_matrix(x_val_dense)

    s_val = rng.randn(C)

    f = theano.function([x, s], sp.col_scale(x, s))

    print 'A', f(x_val, s_val).toarray()
    print 'B', (x_val_dense * s_val)

    assert numpy.all(f(x_val, s_val).toarray() == (x_val_dense * s_val))

    if 0:
        T.verify_grad(None, d, [x_val, s_val],
                mode=theano.Mode(linker='py', optimizer='fast_compile'))
    else:
        print >> sys.stderr, "WARNING: skipping gradient test because verify_grad doesn't support sparse arguments"

if __name__ == '__main__':
    if 0:
        testcase =  TestSP
        suite = unittest.TestLoader()
        suite = suite.loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        unittest.main()
