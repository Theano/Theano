"""
Convolution-like operations with sparse matrix multiplication.

To read about different sparse formats, see U{http://www-users.cs.umn.edu/~saad/software/SPARSKIT/paper.ps}.

@todo: Automatic methods for determining best sparse format?
"""
#### COPIED FROM hpu/icml09/sp.py

import numpy
from scipy import sparse as scipy_sparse

import theano
import theano.sparse
from theano import sparse, gof, Op, tensor
from theano.gof.python25 import all, any

def register_specialize(lopt, *tags, **kwargs):
    theano.compile.optdb['specialize'].register((kwargs and kwargs.pop('name')) or lopt.__name__, lopt, 'fast_run', *tags)


class SpSum(Op):
    """
    Scale each columns of a sparse matrix by the corresponding element of a dense vector
    """
    axis = None
    sparse_grad = False

    def __init__(self, axis, sparse_grad=True):
        """
        :param sparse_grad: if True, this instance ignores the gradient on matrix elements which are implicitly 0.
        """
        super(SpSum, self).__init__()
        self.axis = axis
        self.sparse_grad = sparse_grad
        if self.axis not in (None, 0, 1):
            raise ValueError('illegal value for self.axis')

    def __eq__(self, other):
        #WARNING: judgement call...
        #not using the sparse_grad in the comparison or hashing because it doesn't change the perform method
        #therefore, we *do* want Sums with different sparse_grad values to be merged by the merge optimization.
        # This requires them to compare equal.
        return type(self) == type(other) and self.axis == other.axis

    def __hash__(self):
        return 76324 ^ hash(type(self)) ^ hash(self.axis)

    def __str__(self):
        return self.__class__.__name__+"{axis=%s}" % str(self.axis)

    def make_node(self, x):
        ###
        # At least for small matrices (5x5), the .sum() method of a csc matrix returns a dense matrix
        # as the result whether axis is 0 or 1... weird!
        ###
        assert isinstance(x.type, theano.sparse.SparseType)
        b = ()
        if self.axis is not None:
            b = (False,)
        z = tensor.tensor(broadcastable=b, dtype=x.dtype)
        return gof.Apply(self, [x], [z])

    def infer_shape(self, node, shapes):
        r = None
        if self.axis is None:
            r = [()]
        elif self.axis == 0:
            r = [(shapes[0][1],)]
        else:
            r = [(shapes[0][0],)]
        return r

    def perform(self,node, (x,), (z,)):
        if self.axis is None:
            z[0] = numpy.asarray(x.sum())
        else:
            if self.axis == 0:
                if x.format == 'csc':
                   z[0] = numpy.asarray(x.sum(axis=self.axis)).reshape((x.shape[1],))
                else:
                   z[0] = numpy.asarray(x.asformat(x.format).sum(axis=self.axis)).reshape((x.shape[1],))
            elif self.axis == 1:
               if x.format == 'csr':
                   z[0] = numpy.asarray(x.sum(axis=self.axis)).reshape((x.shape[0],))
               else:
                   z[0] = numpy.asarray(x.asformat(x.format).sum(axis=self.axis)).reshape((x.shape[0],))

    def grad(self,(x,), (gz,)):
        if self.axis is None:
            r = gz * theano.sparse.sp_ones_like(x)
        elif self.axis == 0:
            r = col_scale(theano.sparse.sp_ones_like(x), gz)
        elif self.axis == 1:
            r = row_scale(theano.sparse.sp_ones_like(x), gz)
        else:
            assert False

        if not self.sparse_grad:
            r = theano.sparse.dense_from_sparse(r)

        return [r]

def sp_sum(x, axis=None, sparse_grad=False):
    return SpSum(axis, sparse_grad)(x)

class Diag(Op):
    """
    Extract the diagonal of a square sparse matrix as a dense vector.
    """
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "Diag"

    def make_node(self, x):
        return gof.Apply(self, [x], [tensor.tensor(broadcastable=(False,), dtype=x.dtype)])

    def perform(self, node, (x,), (z,)):
        M, N = x.shape
        if M != N:
            raise ValueError("DenseDiag argument not square. Shape:", x.shape)

        assert x.format == 'csc'

        data = x.data
        indices = x.indices
        indptr = x.indptr

        diag = numpy.zeros(N, x.dtype)

        #TODO: try using ndarrays and then prune() on the result
        # it could be optimized in the case the sparse structure
        # does not allow index duplication

        for j in xrange(0, N):
            for i_idx in xrange(indptr[j], indptr[j+1]):
                if indices[i_idx] == j:
                    diag[j] += data[i_idx]
        z[0] = diag

    def grad(self, (diag,), (gz,)):
        return [square_diagonal(gz)]

    def infer_shape(self, nodes, shapes):
        matrix_shape = shapes[0] 
        diag_length = matrix_shape[0]
        return [(diag_length,)]

diag = Diag()

class SquareDiagonal(Op):
    """
    Return a square sparse (csc) matrix whose diagonal
    is given by the dense vector argument.
    """
    def __eq__(self, other):
        return (type(self) == type(other))

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "SquareDiagonal"

    def make_node(self, diag):
        diag = tensor.as_tensor_variable(diag)
        if diag.type.ndim != 1:
            raise TypeError('data argument must be a vector', diag.type)

        return gof.Apply(self, [diag],
                [sparse.SparseType(dtype=diag.dtype, format='csc')()])

    def perform(self, node, (diag,), (z,)):
        N, = diag.shape
        indptr = range(N + 1)
        indices = indptr[0:N]
        z[0] = scipy_sparse.csc_matrix((diag, indices, indptr),
                                       (N, N), copy=True)

    def grad(self, input, (gz,)):
        return [diag(gz)]

    def infer_shape(self, nodes, shapes):
        diag_length = shapes[0][0]
        return [(diag_length, diag_length)]

square_diagonal = SquareDiagonal()


class ColScaleCSC(Op):
    """
    Scale each columns of a sparse matrix by the corresponding element of a dense vector
    """

    def make_node(self, x, s):
        if x.format != 'csc':
            raise ValueError('x was not a csc matrix')
        return gof.Apply(self, [x, s], [x.type()])

    def perform(self,node, (x,s), (z,)):
        M, N = x.shape
        assert x.format == 'csc'
        assert s.shape == (N,)

        y = x.copy()

        for j in xrange(0, N):
            y.data[y.indptr[j]: y.indptr[j+1]] *= s[j]

        z[0] = y

    def grad(self,(x,s), (gz,)):
        return [col_scale(gz, s), sp_sum(x * gz, axis=0)]

class RowScaleCSC(Op):
    """
    Scale each row of a sparse matrix by the corresponding element of a dense vector
    """

    def make_node(self, x, s):
        return gof.Apply(self, [x, s], [x.type()])

    def perform(self,node, (x,s), (z,)):
        M, N = x.shape
        assert x.format == 'csc'
        assert s.shape == (M,)

        indices = x.indices
        indptr = x.indptr

        y_data = x.data.copy()

        for j in xrange(0, N):
            for i_idx in xrange(indptr[j], indptr[j+1]):
                y_data[i_idx] *= s[indices[i_idx]]

        z[0] = scipy_sparse.csc_matrix((y_data, indices, indptr), (M,N))

    def grad(self,(x,s), (gz,)):
        return [row_scale(gz, s), sp_sum(x * gz, axis=0)]

def col_scale(x, s):
    if x.format == 'csc':
        return ColScaleCSC()(x, s)
    elif x.format == 'csr':
        return RowScaleCSC()(x.T, s).T
    else:
        raise NotImplementedError()

def row_scale(x, s):
    return col_scale(x.T, s).T

class Remove0(Op):
    """
    Remove explicit zeros from a sparse matrix, and resort indices
    """

    def __init__(self, inplace=False, *args, **kwargs):
        Op.__init__(self, *args, **kwargs)
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __eq__(self,other):
        return type(self) == type(other) and self.inplace == other.inplace

    def __hash__(self):
        return 64153 ^ hash(type(self)) ^ hash(self.inplace)

    def __str__(self):
        l = []
        if self.inplace:
            l.append('inplace')
        return self.__class__.__name__+'{%s}'%', '.join(l)

    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])

    def perform(self,node, (x,), (z,)):
        if self.inplace:
            c = x
        else:
            c = x.copy()
        c.eliminate_zeros()
        z[0] = c

    def grad(self, (x,), (gz,)):
        return [gz]

remove0 = Remove0()

class EnsureSortedIndices(Op):
    """
    Remove explicit zeros from a sparse matrix, and resort indices
    """

    def __init__(self, inplace):
        self.inplace = inplace
        if self.inplace:
            self.view_map = {0:[0]}

    def make_node(self, x):
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        if self.inplace:
            x.sort_indices()
            z[0] = x
        else:
            z[0] = x.sorted_indices()

    def grad(self, inputs, output_grad):
        return [output_grad[0]]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def __str__(self):
        if self.inplace:
            return self.__class__.__name__ + "{inplace}"
        else:
            return self.__class__.__name__ + "{no_inplace}"

ensure_sorted_indices = EnsureSortedIndices(inplace=False)

def clean(x):
    return ensure_sorted_indices(remove0(x))

class ConvolutionIndices(Op):
    """Build indices for a sparse CSC matrix that could implement A (convolve) B.

       This generates a sparse matrix M, which generates a stack of image patches
       when computing the dot product of M with image patch. Convolution is then
       simply the dot product of (img x M) and the kernels.
    """

    @staticmethod
    def sparse_eval(inshp, kshp, nkern, (dx,dy)=(1,1), mode='valid'):
        return convolution_indices.evaluate(inshp,kshp,(dx,dy),nkern,mode=mode,ws=False)

    @staticmethod
    def conv_eval(inshp, kshp, (dx,dy)=(1,1), mode='valid'):
        return convolution_indices.evaluate(inshp,kshp,(dx,dy),mode=mode,ws=True)

    # img_shape and ker_shape are (height,width)
    @staticmethod
    def evaluate(inshp, kshp, (dx,dy)=(1,1), nkern=1, mode='valid', ws=True):
        """Build a sparse matrix which can be used for performing...
        * convolution: in this case, the dot product of this matrix with the input
        images will generate a stack of images patches. Convolution is then a
        tensordot operation of the filters and the patch stack.
        * sparse local connections: in this case, the sparse matrix allows us to operate
        the weight matrix as if it were fully-connected. The structured-dot with the
        input image gives the output for the following layer.

        :param ker_shape: shape of kernel to apply (smaller than image)
        :param img_shape: shape of input images
        :param mode: 'valid' generates output only when kernel and image overlap
                     fully. Convolution obtained by zero-padding the input
        :param ws: True if weight sharing, false otherwise
        :param (dx,dy): offset parameter. In the case of no weight sharing,
                        gives the pixel offset between two receptive fields.
                        With weight sharing gives the offset between the
                        top-left pixels of the generated patches

        :rtype: tuple(indices, indptr, logical_shape, sp_type, out_img_shp)
        :returns: the structure of a sparse matrix, and the logical dimensions
                  of the image which will be the result of filtering.
        """
        N = numpy

        # inshp contains either 2 entries (height,width) or 3 (nfeatures,h,w)
        # in the first case, default nfeatures to 1
        if N.size(inshp)==2:
            inshp = (1,)+inshp

        inshp = N.array(inshp)
        kshp  = N.array(kshp)
        ksize = N.prod(kshp)

        kern = ksize-1 - N.arange(ksize)

        # size of output image if doing proper convolution (mode='full',dx=dy=0)
        # outshp is the actual output shape given the parameters
        fulloutshp = inshp[1:] + kshp - 1
        if mode == 'valid':
            s = -1
        else:
            s = 1
        outshp = N.int64(N.ceil((inshp[1:] + s*kshp - s*1) \
                 /N.array([dy,dx], dtype='float')))
        if any(outshp <= 0):
            err = 'Invalid kernel', kshp,'and/or step size',(dx,dy),\
                  'for given input shape', inshp
            raise ValueError(err)

        outsize = N.prod(outshp)
        insize = N.prod(inshp)

        # range of output units over which to iterate
        if mode == 'valid':
            lbound = N.array([kshp[0]-1,kshp[1]-1])
            ubound = lbound + (inshp[1:]-kshp+1)
        else:
            lbound = N.zeros(2)
            ubound = fulloutshp

        # coordinates of image in "fulloutshp" coordinates
        topleft  = N.array([kshp[0]-1,kshp[1]-1])
        botright = topleft + inshp[1:] # bound when counting the receptive field

        # sparse matrix specifics...
        if ws:
            spmatshp = (outsize * N.prod(kshp) * inshp[0], insize)
        else:
            spmatshp = (nkern * outsize, insize)
        spmat = scipy_sparse.lil_matrix(spmatshp)

        # loop over output image pixels
        z,zz = 0,0

        # incremented every time we write something to the sparse matrix
        # this is used to track the ordering of filter tap coefficient in sparse
        # column ordering
        tapi, ntaps = 0, 0

        # Note: looping over the number of kernels could've been done more efficiently
        # as the last step (when writing to spmat). However, this messes up the ordering
        # of the column values (order in which you write the values determines how the
        # vectorized data will get used later one)

        for fmapi in xrange(inshp[0]): # loop over input features
            for n in xrange(nkern): # loop over number of kernels (nkern=1 for weight sharing)

                # FOR EACH OUTPUT PIXEL...
                for oy in N.arange(lbound[0],ubound[0],dy): # loop over output image height
                    for ox in N.arange(lbound[1],ubound[1],dx): # loop over output image width

                        l = 0 # kern[l] is filter value to apply at (oj,oi) for (iy,ix)

                        # ... ITERATE OVER INPUT UNITS IN RECEPTIVE FIELD
                        for ky in oy+N.arange(kshp[0]):
                            for kx in ox+N.arange(kshp[1]):

                                # verify if we are still within image boundaries. Equivalent to
                                # zero-padding of the input image
                                if all((ky,kx) >= topleft) and all((ky,kx) < botright):

                                    # convert to "valid" input space coords
                                    # used to determine column index to write to in sparse mat
                                    iy,ix = N.array((ky,kx)) - topleft
                                    # determine raster-index of input pixel...
                                    col = iy*inshp[2]+ix +\
                                          fmapi*N.prod(inshp[1:]) # taking into account multiple input features

                                    # convert oy,ox values to output space coordinates
                                    if mode == 'full':
                                        (y, x) = (oy, ox)
                                    else:
                                        (y, x) = (oy, ox) - topleft
                                    (y,x) = N.array([y,x]) / (dy,dx) # taking into account step size
                                    # convert to row index of sparse matrix
                                    if ws:
                                        row = (y*outshp[1]+x)*inshp[0]*ksize + l + fmapi*ksize
                                    else:
                                        row = y*outshp[1] + x

                                    # Store something at that location in sparse matrix.
                                    # The written value is only useful for the sparse case. It
                                    # will determine the way kernel taps are mapped onto
                                    # the sparse columns (idea of kernel map)
                                    spmat[row + n*outsize, col] = tapi + 1   # n*... only for sparse

                                    # total number of active taps (used for kmap)
                                    ntaps += 1

                                tapi += 1 # absolute tap index (total number of taps)
                                l+=1 # move on to next filter tap l=(l+1)%ksize

        if spmat.format != 'csc':
            spmat = spmat.tocsc().sorted_indices()
        else:
            # BUG ALERT: scipy0.6 has bug where data and indices are written in
            # reverse column ordering.
            # Explicit call to sorted_indices removes this problem.
            spmat = spmat.sorted_indices()

        if ws:
            kmap = None
        else:
            kmap = N.zeros(ntaps, dtype='int')
            k=0
            #print 'TEMPORARY BUGFIX: REMOVE !!!'
            for j in xrange(spmat.shape[1]):
                for i_idx in xrange(spmat.indptr[j], spmat.indptr[j+1]):
                    if spmat.data[i_idx] != 0:
                        kmap[k] = spmat.data[i_idx] -1 # this is == spmat[i,j] - 1
                        k+=1

        # when in valid mode, it is more efficient to store in sparse row
        # TODO: need to implement structured dot for csr matrix
        assert spmat.format == 'csc'
        sptype = 'csc'
        #sptype = 'csr' if mode=='valid' else 'csc'
        if 0 and mode=='valid':
            spmat = spmat.tocsr()

        rval = (spmat.indices[:spmat.size],
                spmat.indptr, spmatshp, sptype, outshp)
        if kmap is not None:
            rval += (kmap,)

        return rval

    def perform(self, node, (inshp, kshp),\
                (out_indices, out_indptr, spmat_shape)):
        indices, indptr, spmatshp, outshp = self.evaluate(inshp, kshp)
        out_indices[0] = indices
        out_indptr[0] = indptr
        spmat_shape[0] = N.asarray(spmatshp)

convolution_indices = ConvolutionIndices()

def applySparseFilter(kerns, kshp, nkern, images, imgshp, step=(1,1), bias=None, mode='valid'):
    """
    "images" is assumed to be a matrix of shape batch_size x img_size, where the second
    dimension represents each image in raster order

    Output feature map will have shape:

    .. code-block:: python

       batch_size x number of kernels * output_size

    .. note::

        IMPORTANT: note that this means that each feature map is contiguous in memory.
        The memory layout will therefore be:
        [ <feature_map_0> <feature_map_1> ... <feature_map_n>],
        where <feature_map> represents a "feature map" in raster order

    Note that the concept of feature map doesn't really apply to sparse filters without
    weight sharing. Basically, nkern=1 will generate one output img/feature map,
    nkern=2 a second feature map, etc.

    kerns is a 1D tensor, and assume to be of shape:

    .. code-block:: python

       nkern * N.prod(outshp) x N.prod(kshp)

    Each filter is applied seperately to consecutive output pixels.

    :param kerns: nkern*outsize*ksize vector containing kernels
    :param kshp: tuple containing actual dimensions of kernel (not symbolic)
    :param nkern: number of kernels to apply at each pixel in the input image.
                  nkern=1 will apply a single unique filter for each input pixel.
    :param images: bsize x imgsize matrix containing images on which to apply filters
    :param imgshp: tuple containing actual image dimensions (not symbolic)
    :param step: determines number of pixels between adjacent receptive fields
                 (tuple containing dx,dy values)
    :param mode: 'full', 'valid' see CSM.evaluate function for details
    :return: out1, symbolic result
    :return: out2, logical shape of the output img (nkern,height,width)
             (after dot product, not of the sparse matrix!)
    """

    # inshp contains either 2 entries (height,width) or 3 (nfeatures,h,w)
    # in the first case, default nfeatures to 1
    if numpy.size(imgshp)==2:
        imgshp = (1,)+imgshp

    # construct indices and index pointers for sparse matrix
    indices, indptr, spmat_shape, sptype, outshp, kmap = \
        convolution_indices.sparse_eval(imgshp, kshp, nkern, step, mode)

    # build a sparse weight matrix
    sparsew = theano.sparse.CSM(sptype, kmap)(kerns, indices, indptr, spmat_shape)
    output =  sparse.structured_dot(sparsew, images.T).T
    if bias is not None:
        output += bias

    return output, numpy.hstack((nkern,outshp))



def convolve(kerns, kshp, nkern, images, imgshp, step=(1,1), bias=None,\
             mode='valid', flatten=True):
    """Convolution implementation by sparse matrix multiplication.

    :note: For best speed, put the matrix which you expect to be
           smaller as the 'kernel' argument

    "images" is assumed to be a matrix of shape batch_size x img_size, where the second
    dimension represents each image in raster order

    If flatten is "False", the output feature map will have shape:

    .. code-block:: python

        batch_size x number of kernels x output_size

    If flatten is "True", the output feature map will have shape:

    .. code-block:: python

        batch_size x number of kernels * output_size

    .. note::

        IMPORTANT: note that this means that each feature map (image
        generate by each kernel) is contiguous in memory. The memory
        layout will therefore be: [ <feature_map_0> <feature_map_1>
        ... <feature_map_n>], where <feature_map> represents a
        "feature map" in raster order

    kerns is a 2D tensor of shape nkern x N.prod(kshp)

    :param kerns: 2D tensor containing kernels which are applied at every pixel
    :param kshp: tuple containing actual dimensions of kernel (not symbolic)
    :param nkern: number of kernels/filters to apply.
                  nkern=1 will apply one common filter to all input pixels
    :param images: tensor containing images on which to apply convolution
    :param imgshp: tuple containing image dimensions
    :param step: determines number of pixels between adjacent receptive fields
                 (tuple containing dx,dy values)
    :param mode: 'full', 'valid' see CSM.evaluate function for details
    :param sumdims: dimensions over which to sum for the tensordot operation.
                    By default ((2,),(1,)) assumes kerns is a nkern x kernsize
                    matrix and images is a batchsize x imgsize matrix
                    containing flattened images in raster order
    :param flatten: flatten the last 2 dimensions of the output. By default,
                    instead of generating a batchsize x outsize x nkern tensor,
                    will flatten to batchsize x outsize*nkern

    :return: out1, symbolic result
    :return: out2, logical shape of the output img (nkern,heigt,width)

    :TODO: test for 1D and think of how to do n-d convolutions
    """
    N = numpy
    # start by computing output dimensions, size, etc
    kern_size = N.int64(N.prod(kshp))

    # inshp contains either 2 entries (height,width) or 3 (nfeatures,h,w)
    # in the first case, default nfeatures to 1
    if N.size(imgshp)==2:
        imgshp = (1,)+imgshp

    # construct indices and index pointers for sparse matrix, which, when multiplied
    # with input images will generate a stack of image patches
    indices, indptr, spmat_shape, sptype, outshp = \
            convolution_indices.conv_eval(imgshp, kshp, step, mode)

    # build sparse matrix, then generate stack of image patches
    csc = theano.sparse.CSM(sptype)(N.ones(indices.size), indices, indptr, spmat_shape)
    patches = (sparse.structured_dot(csc, images.T)).T

    # compute output of linear classifier
    pshape = tensor.stack(images.shape[0] * tensor.as_tensor(N.prod(outshp)),\
                          tensor.as_tensor(imgshp[0]*kern_size))
    patch_stack = tensor.reshape(patches, pshape, ndim=2);

    # kern is of shape: nkern x ksize*number_of_input_features
    # output is thus of shape: bsize*outshp x nkern
    output = tensor.dot(patch_stack,kerns.T)

    # add bias across each feature map (more efficient to do it now)
    if bias is not None:
        output += bias

    # now to have feature maps in raster order ...
    # go from bsize*outshp x nkern to bsize x nkern*outshp
    newshp = tensor.stack(images.shape[0],\
                          tensor.as_tensor(N.prod(outshp)),\
                          tensor.as_tensor(nkern))
    tensout= tensor.reshape(output, newshp, ndim=3)
    output = tensor.DimShuffle((False,)*tensout.ndim, (0,2,1))(tensout)
    if flatten:
        output = tensor.flatten(output, 2)

    return output, N.hstack((nkern,outshp))


def max_pool(images, imgshp, maxpoolshp):
    """Implements a max pooling layer

    Takes as input a 2D tensor of shape batch_size x img_size and performs max pooling.
    Max pooling downsamples by taking the max value in a given area, here defined by
    maxpoolshp. Outputs a 2D tensor of shape batch_size x output_size.

    :param images: 2D tensor containing images on which to apply convolution.
                   Assumed to be of shape batch_size x img_size
    :param imgshp: tuple containing image dimensions
    :param maxpoolshp: tuple containing shape of area to max pool over

    :return: out1, symbolic result (2D tensor)
    :return: out2, logical shape of the output
    """
    N = numpy
    poolsize = N.int64(N.prod(maxpoolshp))

    # imgshp contains either 2 entries (height,width) or 3 (nfeatures,h,w)
    # in the first case, default nfeatures to 1
    if N.size(imgshp)==2:
        imgshp = (1,)+imgshp

    # construct indices and index pointers for sparse matrix, which, when multiplied
    # with input images will generate a stack of image patches
    indices, indptr, spmat_shape, sptype, outshp = \
            convolution_indices.conv_eval(imgshp, maxpoolshp, maxpoolshp, mode='valid')

    print 'XXXXXXXXXXXXXXXX MAX POOLING LAYER XXXXXXXXXXXXXXXXXXXX'
    print 'imgshp = ', imgshp
    print 'maxpoolshp = ', maxpoolshp
    print 'outshp = ', outshp

    # build sparse matrix, then generate stack of image patches
    csc = theano.sparse.CSM(sptype)(N.ones(indices.size), indices, indptr, spmat_shape)
    patches = sparse.structured_dot(csc, images.T).T

    pshape = tensor.stack(images.shape[0]*\
                            tensor.as_tensor(N.prod(outshp)),
                          tensor.as_tensor(imgshp[0]),
                          tensor.as_tensor(poolsize))
    patch_stack = tensor.reshape(patches, pshape, ndim=3);

    out1 = tensor.max(patch_stack, axis=2)

    pshape = tensor.stack(images.shape[0],
                          tensor.as_tensor(N.prod(outshp)),
                          tensor.as_tensor(imgshp[0]))
    out2 = tensor.reshape(out1, pshape, ndim=3);

    out3 = tensor.DimShuffle(out2.broadcastable, (0,2,1))(out2)

    return tensor.flatten(out3,2), outshp
