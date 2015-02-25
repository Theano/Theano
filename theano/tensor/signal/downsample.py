""" Ops for downsampling images.

Planned:
DownsampleFactorMax, DownsampleAvg, DownsampleSoftmax.

"""
#This file should move along with conv.py
import __builtin__

import numpy

import theano
from theano import gof, Op, tensor, Variable, Apply


def max_pool2D(*args, **kwargs):
    import sys
    print >> sys.stderr, "DEPRECATION: max_pool2D renamed to max_pool_2d"
    return max_pool_2d(*args, **kwargs)


def max_pool_2d(input, ds, ignore_border=False, st=None):
    """
    Takes as input a N-D tensor, where N >= 2. It downscales the input image by
    the specified factor, by keeping only the maximum value of non-overlapping
    patches of size (ds[0],ds[1])

    :type input: N-D theano tensor of input images.
    :param input: input images. Max pooling will be done over the 2 last
        dimensions.
    :type ds: tuple of length 2
    :param ds: factor by which to downscale (vertical ds, horizontal ds).
        (2,2) will halve the image in each dimension.
    :type ignore_border: bool
    :param ignore_border: When True, (5,5) input with ds=(2,2)
        will generate a (2,2) output. (3,3) otherwise.
    :type st: tuple of lenght 2
    :param st: stride size, which is the number of shifts
        over rows/cols to get the the next pool region.
        if st is None, it is considered equal to ds
        (no overlap on pooling regions)

    """
    if input.ndim < 2:
        raise NotImplementedError('max_pool_2d requires a dimension >= 2')
    if input.ndim == 4:
        op = DownsampleFactorMax(ds, ignore_border, st=st)
        output = op(input)
        return output

    # extract image dimensions
    img_shape = input.shape[-2:]

    # count the number of "leading" dimensions, store as dmatrix
    batch_size = tensor.prod(input.shape[:-2])
    batch_size = tensor.shape_padright(batch_size, 1)

    # store as 4D tensor with shape: (batch_size,1,height,width)
    new_shape = tensor.cast(tensor.join(0, batch_size,
                                        tensor.as_tensor([1]),
                                        img_shape), 'int64')
    input_4D = tensor.reshape(input, new_shape, ndim=4)

    # downsample mini-batch of images
    op = DownsampleFactorMax(ds, ignore_border, st=st)
    output = op(input_4D)

    # restore to original shape
    outshp = tensor.join(0, input.shape[:-2], output.shape[-2:])
    return tensor.reshape(output, outshp, ndim=input.ndim)


class DownsampleFactorMax(Op):
    """For N-dimensional tensors, consider that the last two
    dimensions span images.  This Op downsamples these images by a
    factor ds, by taking the max over non- overlapping rectangular
    regions.

    """
    __props__ = ('ds', 'ignore_border', 'st')

    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False, st=None):
        """Return the shape of the output from this op, for input of given
        shape and flags.

        :param imgshape: the shape of a tensor of images. The last two elements
            are interpreted as the number of rows, and the number of cols.
        :type imgshape: tuple, list, or similar of integer or
            scalar Theano variable.

        :param ds: downsample factor over rows and columns
                   this parameter indicates the size of the pooling region
        :type ds: list or tuple of two ints

        :param st: the stride size. This is the distance between the pooling
                   regions. If it's set to None, in which case it equlas ds.
        :type st: list or tuple of two ints

        :param ignore_border: if ds doesn't divide imgshape, do we include an
            extra row/col of partial downsampling (False) or ignore it (True).
        :type ignore_border: bool

        :rtype: list
        :returns: the shape of the output from this op, for input of given
            shape.  This will have the same length as imgshape, but with last
            two elements reduced as per the downsampling & ignore_border flags.
        """
        if len(imgshape) < 2:
            raise TypeError('imgshape must have at least two elements '
                            '(rows, cols)')

        if st is None:
            st = ds
        r, c = imgshape[-2:]

        if ignore_border:
            out_r = (r - ds[0]) // st[0] + 1
            out_c = (c - ds[1]) // st[1] + 1
            if isinstance(r, theano.Variable):
                nr = tensor.maximum(out_r, 0)
            else:
                nr = numpy.maximum(out_r, 0)
            if isinstance(c, theano.Variable):
                nc = tensor.maximum(out_c, 0)
            else:
                nc = numpy.maximum(out_c, 0)
        else:
            if isinstance(r, theano.Variable):
                nr = tensor.switch(tensor.ge(st[0], ds[0]),
                                   (r - 1) // st[0] + 1,
                                   tensor.maximum(0, (r - 1 - ds[0])
                                                  // st[0] + 1) + 1)
            elif st[0] >= ds[0]:
                nr = (r - 1) // st[0] + 1
            else:
                nr = max(0, (r - 1 - ds[0]) // st[0] + 1) + 1

            if isinstance(c, theano.Variable):
                nc = tensor.switch(tensor.ge(st[1], ds[1]),
                                   (c - 1) // st[1] + 1,
                                   tensor.maximum(0, (c - 1 - ds[1])
                                                  // st[1] + 1) + 1)
            elif st[1] >= ds[1]:
                nc = (c - 1) // st[1] + 1
            else:
                nc = max(0, (c - 1 - ds[1]) // st[1] + 1) + 1

        rval = list(imgshape[:-2]) + [nr, nc]
        return rval

    def __init__(self, ds, ignore_border=False, st=None):
        """
        :param ds: downsample factor over rows and column.
                   ds indicates the pool region size.
        :type ds: list or tuple of two ints

        :param ignore_border: if ds doesn't divide imgshape, do we include
            an extra row/col of partial downsampling (False) or
            ignore it (True).
        :type ignore_border: bool

        : param st: stride size, which is the number of shifts
            over rows/cols to get the the next pool region.
            if st is None, it is considered equal to ds
            (no overlap on pooling regions)
        : type st: list or tuple of two ints

        """
        self.ds = tuple(ds)
        if not all([isinstance(d, int) for d in ds]):
            raise ValueError(
                "DownsampleFactorMax downsample parameters must be ints."
                " Got %s" % str(ds))
        if st is None:
            st = ds
        self.st = tuple(st)
        self.ignore_border = ignore_border

    def __str__(self):
        return '%s{%s,%s,%s}' % (self.__class__.__name__,
                                 self.ds, self.st, self.ignore_border)

    def make_node(self, x):
        if x.type.ndim != 4:
            raise TypeError()
        # TODO: consider restrucing the dtype?
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, inp, out):
        """
        """
        x, = inp
        z, = out
        if len(x.shape) != 4:
            raise NotImplementedError(
                'DownsampleFactorMax requires 4D input for now')
        z_shape = self.out_shape(x.shape, self.ds, self.ignore_border, self.st)
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = numpy.empty(self.out_shape(x.shape, self.ds,
                                              self.ignore_border, self.st),
                               dtype=x.dtype)
        zz = z[0]

        #number of pooling output rows
        pr = zz.shape[-2]
        #number of pooling output cols
        pc = zz.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        img_rows = x.shape[-2]
        img_cols = x.shape[-1]

        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = r * st0
                    row_end = __builtin__.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        col_st = c * st1
                        col_end = __builtin__.min(col_st + ds1, img_cols)
                        zz[n, k, r, c] = x[
                            n, k, row_st:row_end, col_st:col_end].max()

    def infer_shape(self, node, in_shapes):
        shp = self.out_shape(in_shapes[0], self.ds,
                             self.ignore_border, self.st)
        return [shp]

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        maxout = self(x)
        return [DownsampleFactorMaxGrad(self.ds,
                                        ignore_border=self.ignore_border,
                                        st=self.st)(
                                            x, maxout, gz)]

    def c_code(self, node, name, inp, out, sub):
        # No implementation is currently for the case where
        # the stride size and the pooling size are different.
        # An exception is raised for such a case.
        if self.ds != self.st:
           raise theano.gof.utils.MethodNotDefined()
        x, = inp
        z, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        return """
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int x_shp0_usable;
        int x_shp1_usable;
        int z_shp0, z_shp1;
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        z_shp0 = PyArray_DIMS(%(x)s)[2] / %(ds0)s;
        z_shp1 = PyArray_DIMS(%(x)s)[3] / %(ds1)s;
        if (%(ignore_border)s)
        {
            x_shp0_usable = z_shp0 * %(ds0)s;
            x_shp1_usable = z_shp1 * %(ds1)s;
        }
        else
        {
            z_shp0 += (PyArray_DIMS(%(x)s)[2] %% %(ds0)s) ? 1 : 0;
            z_shp1 += (PyArray_DIMS(%(x)s)[3] %% %(ds1)s) ? 1 : 0;
            x_shp0_usable = PyArray_DIMS(%(x)s)[2];
            x_shp1_usable = PyArray_DIMS(%(x)s)[3];
        }
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(z)s)[2] != z_shp0)
          ||(PyArray_DIMS(%(z)s)[3] != z_shp1)
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=PyArray_DIMS(%(x)s)[1];
          dims[2]=z_shp0;
          dims[3]=z_shp1;
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }

        if (z_shp0 && z_shp1)
        {
            for(int b=0;b<PyArray_DIMS(%(x)s)[0];b++){
              for(int k=0;k<PyArray_DIMS(%(x)s)[1];k++){
                int mini_i = 0;
                int zi = 0;
                for(int i=0;i< x_shp0_usable; i++){
                  int mini_j = 0;
                  int zj = 0;
                  for(int j=0; j<x_shp1_usable; j++){
                    dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,i,j)))[0];
                    dtype_%(z)s * __restrict__ z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,b,k,zi,zj)));
                    z[0] = (((mini_j|mini_i) == 0) || z[0] < a) ? a : z[0];
                    mini_j = ((mini_j + 1) == %(ds1)s) ? 0 : mini_j+1;
                    zj += (mini_j == 0);
                  }
                  mini_i = ((mini_i + 1) == %(ds0)s) ? 0 : mini_i+1;
                  zi += (mini_i == 0);
                }
              }
            }
        }
        """ % locals()

    def c_code_cache_version(self):
        return (0, 1)


class DownsampleFactorMaxGrad(Op):
    __props__ = ('ds', 'ignore_border', 'st')

    def __init__(self, ds, ignore_border, st=None):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border
        if st is None:
            st = ds
        self.st = tuple(st)

    def __str__(self):
        return '%s{%s,%s,%s}' % (self.__class__.__name__,
                                 self.ds, self.st, self.ignore_border)

    def make_node(self, x, maxout, gz):
        # make_node should only be called by the grad function of
        # DownsampleFactorMax, so these asserts should not fail.
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(maxout, Variable) and maxout.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4

        return Apply(self, [x, maxout, gz], [x.type()])

    def perform(self, node, inp, out):
        x, maxout, gz = inp
        gx_stg, = out
        gx = numpy.zeros_like(x)

        #number of pooling output rows
        pr = maxout.shape[-2]
        #number of pooling output cols
        pc = maxout.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        img_rows = x.shape[-2]
        img_cols = x.shape[-1]

        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = r * st0
                    row_end = __builtin__.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        col_st = c * st1
                        col_end = __builtin__.min(col_st + ds1, img_cols)
                        for row_ind in xrange(row_st, row_end):
                            for col_ind in xrange(col_st, col_end):
                                if (maxout[n, k, r, c] == x[n, k, row_ind, col_ind]):
                                    gx[n, k, row_ind, col_ind] += gz[n, k, r, c]
        gx_stg[0] = gx

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]

    def grad(self, inp, grads):
        x, maxout, gz = inp
        ggx, = grads
        return [theano.tensor.zeros_like(x),
                theano.tensor.zeros_like(maxout),
                DownsampleFactorMaxGradGrad(
                    self.ds, ignore_border=self.ignore_border, st=self.st)(x, maxout, ggx)]

    def c_code(self, node, name, inp, out, sub):
        if self.ds != self.st:
           raise theano.gof.utils.MethodNotDefined()
        x, z, gz = inp
        gx, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        return """
        int x_typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_typenum = PyArray_ObjectType((PyObject*)%(z)s, 0);
        int gz_typenum = PyArray_ObjectType((PyObject*)%(gz)s, 0);
        int x_shp0_usable;
        int x_shp1_usable;
        int z_shp0, z_shp1;
        if ((x_typenum != z_typenum) || (x_typenum != gz_typenum))
        {
            PyErr_SetString(PyExc_ValueError, "input types must all match");
            %(fail)s;
        }
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        if(PyArray_NDIM(%(z)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "z must be a 4d ndarray");
            %(fail)s;
        }
        if(PyArray_NDIM(%(gz)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "gz must be a 4d ndarray");
            %(fail)s;
        }
        z_shp0 = PyArray_DIMS(%(z)s)[2];
        z_shp1 = PyArray_DIMS(%(z)s)[3];
        if (%(ignore_border)s)
        {
            x_shp0_usable = z_shp0 * %(ds0)s;
            x_shp1_usable = z_shp1 * %(ds1)s;
        }
        else
        {
            x_shp0_usable = PyArray_DIMS(%(x)s)[2];
            x_shp1_usable = PyArray_DIMS(%(x)s)[3];
        }
        if ((!%(gx)s)
          || *PyArray_DIMS(%(gx)s)!=4
          ||(PyArray_DIMS(%(gx)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(gx)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(gx)s)[2] != PyArray_DIMS(%(x)s)[2])
          ||(PyArray_DIMS(%(gx)s)[3] != PyArray_DIMS(%(x)s)[3])
          )
        {
          Py_XDECREF(%(gx)s);
          %(gx)s = (PyArrayObject*) PyArray_ZEROS(4, PyArray_DIMS(%(x)s), x_typenum,0);
        }

        for(int b=0;b<PyArray_DIMS(%(x)s)[0];b++){
          for(int k=0;k<PyArray_DIMS(%(x)s)[1];k++){
            int mini_i = 0;
            int zi = 0;
            for(int i=0;i< x_shp0_usable; i++){
               int mini_j = 0;
               int zj = 0;
               for(int j=0; j< x_shp1_usable; j++){
                 dtype_%(x)s * __restrict__ xp = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,i,j)));
                 dtype_%(gx)s * __restrict__ gxp = ((dtype_%(gx)s*)(PyArray_GETPTR4(%(gx)s,b,k,i,j)));
                 dtype_%(z)s * __restrict__ zp = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,b,k,zi,zj)));
                 dtype_%(gz)s * __restrict__ gzp = ((dtype_%(gz)s*)(PyArray_GETPTR4(%(gz)s,b,k,zi,zj)));
                 gxp[0] = (zp[0] == xp[0]) ? gzp[0] : 0;
                 mini_j = (mini_j + 1 == %(ds1)s) ? 0 : mini_j+1;
                 zj += (mini_j == 0);
              }//for j
              mini_i = (mini_i + 1 == %(ds0)s) ? 0 : mini_i+1;
              zi += (mini_i == 0);

              for (int j = x_shp1_usable; j < PyArray_DIMS(%(x)s)[3]; ++j) {
                dtype_%(gx)s * gxp = ((dtype_%(gx)s*)(PyArray_GETPTR4(%(gx)s,b,k,i,j)));
                gxp[0] = 0;
              }
            }//for i

            for(int i = x_shp0_usable; i < PyArray_DIMS(%(x)s)[2]; i++){
                for (int j = 0; j < PyArray_DIMS(%(x)s)[3]; ++j) {
                    dtype_%(gx)s * gxp = ((dtype_%(gx)s*)(PyArray_GETPTR4(%(gx)s,b,k,i,j)));
                    gxp[0] = 0;
                }
            }
          }//for k
        }//for b
        """ % locals()

    def c_code_cache_version(self):
        return (0, 1)


class DownsampleFactorMaxGradGrad(Op):

    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False, st=None):
        """Return the shape of the output from this op, for input of given
        shape and flags.

        :param imgshape: the shape of a tensor of images. The last two elements
            are interpreted as the number of rows, and the number of cols.
        :type imgshape: tuple, list, or similar of integer or
            scalar Theano variable.

        :param ds: downsample factor over rows and columns
                   this parameter indicates the size of the pooling region
        :type ds: list or tuple of two ints

        :param st: the stride size. This is the distance between the pooling
                   regions. If it's set to None, in which case it equlas ds.
        :type st: list or tuple of two ints

        :param ignore_border: if ds doesn't divide imgshape, do we include an
            extra row/col of partial downsampling (False) or ignore it (True).
        :type ignore_border: bool

        :rtype: list
        :returns: the shape of the output from this op, for input of given
            shape.  This will have the same length as imgshape, but with last
            two elements reduced as per the downsampling & ignore_border flags.
        """
        if len(imgshape) < 2:
            raise TypeError('imgshape must have at least two elements '
                            '(rows, cols)')

        if st is None:
            st = ds
        r, c = imgshape[-2:]

        if ignore_border:
            out_r = (r - ds[0]) // st[0] + 1
            out_c = (c - ds[1]) // st[1] + 1
            if isinstance(r, theano.Variable):
                nr = tensor.maximum(out_r, 0)
            else:
                nr = numpy.maximum(out_r, 0)
            if isinstance(c, theano.Variable):
                nc = tensor.maximum(out_c, 0)
            else:
                nc = numpy.maximum(out_c, 0)
        else:
            if isinstance(r, theano.Variable):
                nr = tensor.switch(tensor.ge(st[0], ds[0]),
                                   (r - 1) // st[0] + 1,
                                   tensor.maximum(0, (r - 1 - ds[0])
                                                  // st[0] + 1) + 1)
            elif st[0] >= ds[0]:
                nr = (r - 1) // st[0] + 1
            else:
                nr = max(0, (r - 1 - ds[0]) // st[0] + 1) + 1

            if isinstance(c, theano.Variable):
                nc = tensor.switch(tensor.ge(st[1], ds[1]),
                                   (c - 1) // st[1] + 1,
                                   tensor.maximum(0, (c - 1 - ds[1])
                                                  // st[1] + 1) + 1)
            elif st[1] >= ds[1]:
                nc = (c - 1) // st[1] + 1
            else:
                nc = max(0, (c - 1 - ds[1]) // st[1] + 1) + 1

        rval = list(imgshape[:-2]) + [nr, nc]
        return rval

    def __init__(self, ds, ignore_border, st=None):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border
        if st is None:
            st = ds
        self.st = tuple(st)

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.ds == other.ds
                and self.st == other.st
                and self.ignore_border == other.ignore_border)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.ds) ^ \
            hash(self.st) ^ hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s,%s}' % (self.__class__.__name__,
                                 self.ds, self.st, self.ignore_border)

    def make_node(self, x, maxout, gz):
        # make_node should only be called by the grad function of
        # DownsampleFactorMaxGrad, so these asserts should not fail.
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(maxout, Variable) and maxout.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4

        return Apply(self, [x, maxout, gz], [x.type()])

    def perform(self, node, inp, out):
        x, maxout, ggx = inp
        z, = out

        if len(x.shape) != 4:
            raise NotImplementedError(
                'DownsampleFactorMaxGradGrad requires 4D input for now')
        z_shape = self.out_shape(x.shape, self.ds, self.ignore_border, self.st)
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = numpy.zeros(self.out_shape(x.shape, self.ds,
                                              self.ignore_border, self.st),
                               dtype=x.dtype)
        ggz = z[0]

        #number of pooling output rows
        pr = ggz.shape[-2]
        #number of pooling output cols
        pc = ggz.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        img_rows = x.shape[-2]
        img_cols = x.shape[-1]

        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = r * st0
                    row_end = __builtin__.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        col_st = c * st1
                        col_end = __builtin__.min(col_st + ds1, img_cols)
                        for row_ind in xrange(row_st, row_end):
                            for col_ind in xrange(col_st, col_end):
                                if (maxout[n, k, r, c] == x[n, k, row_ind, col_ind]):
                                    ggz[n, k, r, c] = ggx[n, k, row_ind, col_ind]

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]
