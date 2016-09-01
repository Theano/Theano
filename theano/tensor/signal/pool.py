
"""
Ops for downsampling images.
Planned:
Pool, DownsampleAvg, DownsampleSoftmax.
"""
from __future__ import absolute_import, print_function, division
# This file should move along with conv.py
import warnings

import numpy
from six import integer_types
from six.moves import xrange
import six.moves.builtins as builtins
import theano
from theano import gof, Op, OpenMPOp, tensor, Variable, Apply


def max_pool_2d_same_size(input, patch_size):
    """
    Takes as input a 4-D tensor. It sets all non maximum values
    of non-overlapping patches of size (patch_size[0],patch_size[1]) to zero,
    keeping only the maximum values. The output has the same dimensions as
    the input.

    Parameters
    ----------
    input : 4-D theano tensor of input images
        Input images. Max pooling will be done over the 2 last dimensions.
    patch_size : tuple of length 2
        Size of the patch (patch height, patch width).
        (2,2) will retain only one non-zero value per patch of 4 values.

    """
    output = Pool(patch_size, True)(input)
    outs = MaxPoolGrad(patch_size, True)(input, output, output)
    return outs


def pool_2d(input, ds, ignore_border=None, st=None, padding=(0, 0),
            mode='max'):
    """Downscale the input by a specified factor

    Takes as input a N-D tensor, where N >= 2. It downscales the input image by
    the specified factor, by keeping only the maximum value of non-overlapping
    patches of size (ds[0],ds[1])

    Parameters
    ----------
    input : N-D theano tensor of input images
        Input images. Max pooling will be done over the 2 last dimensions.
    ds : tuple of length 2
        Factor by which to downscale (vertical ds, horizontal ds).
        (2,2) will halve the image in each dimension.
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    st : tuple of two ints
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    padding : tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        Operation executed on each window. `max` and `sum` always exclude
        the padding in the computation. `average` gives you the choice to
        include or exclude it.

    """
    if input.ndim < 2:
        raise NotImplementedError('pool_2d requires a dimension >= 2')
    if ignore_border is None:
        warnings.warn(
            "pool_2d() will have the parameter ignore_border"
            " default value changed to True (currently"
            " False). To have consistent behavior with all Theano"
            " version, explicitly add the parameter ignore_border=True."
            " On the GPU, using ignore_border=True is needed to use cuDNN."
            " When using ignore_border=False and not using cuDNN, the only"
            " GPU combination supported is when"
            " `ds == st and padding == (0, 0) and mode == 'max'`."
            " Otherwise, the convolution will be executed on CPU.",
            stacklevel=2)
        ignore_border = False
    if input.ndim == 4:
        op = Pool(ds, ignore_border, st=st, padding=padding,
                  mode=mode)
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
    op = Pool(ds, ignore_border, st=st, padding=padding,
              mode=mode)
    output = op(input_4D)

    # restore to original shape
    outshp = tensor.join(0, input.shape[:-2], output.shape[-2:])
    return tensor.reshape(output, outshp, ndim=input.ndim)


class Pool(OpenMPOp):
    """
    For N-dimensional tensors, consider that the last two dimensions span
    images. This Op downsamples these images by taking the max, sum or average
    over different patch.

    The constructor takes the max, sum or average or different input patches.

    Parameters
    ----------
    ds : list or tuple of two ints
        Downsample factor over rows and column.
        ds indicates the pool region size.
    ignore_border : bool
        If ds doesn't divide imgshape, do we include an extra row/col
        of partial downsampling (False) or ignore it (True).
    st : list or tuple of two ints or None
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    padding: tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of the images,
        pad_h is the size of the top and bottom margins, and pad_w is the size
        of the left and right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        ('average_inc_pad' excludes the padding from the count,
        'average_exc_pad' include it)

    """

    __props__ = ('ds', 'ignore_border', 'st', 'padding', 'mode')

    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False, st=None, padding=(0, 0)):
        """
        Return the shape of the output from this op, for input of given
        shape and flags.

        Parameters
        ----------
        imgshape : tuple, list, or similar of integer or scalar Theano variable
            The shape of a tensor of images. The last two elements are
            interpreted as the number of rows, and the number of cols.
        ds : list or tuple of two ints
            Downsample factor over rows and columns this parameter indicates
            the size of the pooling region.
        st : list or tuple of two ints
            The stride size. This is the distance between the pooling regions.
            If it's set to None, it equals ds.
        ignore_border : bool
            If ds doesn't divide imgshape, do we include an extra row/col of
            partial downsampling (False) or ignore it (True).
        padding : tuple of two ints
            (pad_h, pad_w), pad zeros to extend beyond four borders
            of the images, pad_h is the size of the top and bottom margins,
            and pad_w is the size of the left and right margins.

        Returns
        -------
        list
            The shape of the output from this op, for input of given shape.
            This will have the same length as imgshape, but with last two
            elements reduced as per the downsampling & ignore_border flags.

        """
        if len(imgshape) < 2:
            raise TypeError('imgshape must have at least two elements '
                            '(rows, cols)')

        if st is None:
            st = ds
        r, c = imgshape[-2:]
        r = tensor.extract_constant(r)
        c = tensor.extract_constant(c)
        if padding[0]:
            r += padding[0] * 2
        if padding[1]:
            c += padding[1] * 2

        if ignore_border:
            if ds[0] == st[0]:
                nr = r // st[0]
            else:
                out_r = (r - ds[0]) // st[0] + 1
                if isinstance(r, theano.Variable):
                    nr = tensor.maximum(out_r, 0)
                else:
                    nr = numpy.maximum(out_r, 0)

            if ds[1] == st[1]:
                nc = c // st[1]
            else:
                out_c = (c - ds[1]) // st[1] + 1
                if isinstance(c, theano.Variable):
                    nc = tensor.maximum(out_c, 0)
                else:
                    nc = numpy.maximum(out_c, 0)
        else:
            if isinstance(r, theano.Variable):
                nr = tensor.switch(tensor.ge(st[0], ds[0]),
                                   (r - 1) // st[0] + 1,
                                   tensor.maximum(0, (r - 1 - ds[0]) //
                                                  st[0] + 1) + 1)
            elif st[0] >= ds[0]:
                nr = (r - 1) // st[0] + 1
            else:
                nr = max(0, (r - 1 - ds[0] + st[0]) // st[0]) + 1

            if isinstance(c, theano.Variable):
                nc = tensor.switch(tensor.ge(st[1], ds[1]),
                                   (c - 1) // st[1] + 1,
                                   tensor.maximum(0, (c - 1 - ds[1]) //
                                                  st[1] + 1) + 1)
            elif st[1] >= ds[1]:
                nc = (c - 1) // st[1] + 1
            else:
                nc = max(0, (c - 1 - ds[1] + st[1]) // st[1]) + 1

        rval = list(imgshape[:-2]) + [nr, nc]
        return rval

    def __init__(self, ds, ignore_border=False, st=None, padding=(0, 0),
                 mode='max', openmp=None):
        super(Pool, self).__init__(openmp=openmp)
        self.ds = tuple(ds)
        if not all([isinstance(d, integer_types) for d in ds]):
            raise ValueError(
                "Pool downsample parameters must be ints."
                " Got %s" % str(ds))
        if st is None:
            st = ds
        assert isinstance(st, (tuple, list))
        self.st = tuple(st)
        self.ignore_border = ignore_border
        self.padding = tuple(padding)
        if self.padding != (0, 0) and not ignore_border:
            raise NotImplementedError(
                'padding works only with ignore_border=True')
        if self.padding[0] >= self.ds[0] or self.padding[1] >= self.ds[1]:
            raise NotImplementedError(
                'padding_h and padding_w must be smaller than strides')
        if mode not in ['max', 'average_inc_pad', 'average_exc_pad', 'sum']:
            raise ValueError(
                "Pool mode parameter only support 'max', 'sum',"
                " 'average_inc_pad' and 'average_exc_pad'. Got %s" % mode)
        self.mode = mode

    def make_node(self, x):
        # TODO: consider restricting the dtype?
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError()
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:2] + (False, False)
        out = tensor.TensorType(x.dtype, broad)
        return gof.Apply(self, [x], [out()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        if len(x.shape) != 4:
            raise NotImplementedError(
                'Pool requires 4D input for now')
        z_shape = self.out_shape(x.shape, self.ds, self.ignore_border, self.st,
                                 self.padding)
        if not self.ignore_border:
            assert z_shape[2] > 0
            assert z_shape[3] > 0
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = numpy.empty(z_shape, dtype=x.dtype)
        zz = z[0]
        # number of pooling output rows
        pr = zz.shape[-2]
        # number of pooling output cols
        pc = zz.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w
        inc_pad = self.mode == 'average_inc_pad'

        # pad the image
        if self.padding != (0, 0):
            y = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype)
            y[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)] = x
        else:
            y = x
        func = numpy.max
        if self.mode == 'sum':
            func = numpy.sum
        elif self.mode != 'max':
            func = numpy.average

        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = r * st0
                    row_end = builtins.min(row_st + ds0, img_rows)
                    if not inc_pad:
                        row_st = builtins.max(row_st, self.padding[0])
                        row_end = builtins.min(row_end, x.shape[-2] + pad_h)
                    for c in xrange(pc):
                        col_st = c * st1
                        col_end = builtins.min(col_st + ds1, img_cols)
                        if not inc_pad:
                            col_st = builtins.max(col_st, self.padding[1])
                            col_end = builtins.min(col_end,
                                                   x.shape[-1] + pad_w)
                        zz[n, k, r, c] = func(y[
                            n, k, row_st:row_end, col_st:col_end])

    def infer_shape(self, node, in_shapes):
        shp = self.out_shape(in_shapes[0], self.ds,
                             self.ignore_border, self.st, self.padding)
        return [shp]

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if self.mode == 'max':
            maxout = self(x)
            return [MaxPoolGrad(self.ds,
                                ignore_border=self.ignore_border,
                                st=self.st, padding=self.padding)(
                                    x, maxout, gz)]
        else:
            return [AveragePoolGrad(self.ds,
                                    ignore_border=self.ignore_border,
                                    st=self.st, padding=self.padding,
                                    mode=self.mode)(
                                        x, gz)]

    def R_op(self, inputs, eval_points):
        if self.mode != 'max':
            return Op.R_op(self, inputs, eval_points)

        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return [None]
        rop = DownsampleFactorMaxRop(self.ds,
                                     ignore_border=self.ignore_border,
                                     st=self.st, padding=self.padding)
        return [rop(inputs[0], eval_points[0])]

    def c_headers(self):
        headers = ['<algorithm>']
        headers += super(Pool, self).c_headers()
        return headers

    def c_code(self, node, name, inp, out, sub):
        if self.mode not in ('max', 'sum', 'average_exc_pad', 'average_inc_pad'):
            raise theano.gof.utils.MethodNotDefined()
        x, = inp
        z, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pd0, pd1 = self.padding
        if self.openmp:
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, c_st, c_end, collector) schedule(static)'
        else:
            omp_parallel = ''
        ccode = """
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_r, z_c; // shape of the output
        int r, c; // shape of the padded_input
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        r += %(pd0)s * 2;
        c += %(pd1)s * 2;
        if (%(pd0)s != 0 && %(pd1)s != 0 && !%(ignore_border)s)
            {
              PyErr_SetString(PyExc_ValueError,
                "padding must be (0,0) when ignore border is False");
              %(fail)s;
            }
        if (%(ignore_border)s)
        {
            // '/' in C is different from '/' in python
            if (r - %(ds0)s < 0)
            {
              z_r = 0;
            }
            else
            {
              z_r = (r - %(ds0)s) / %(st0)s + 1;
            }
            if (c - %(ds1)s < 0)
            {
              z_c = 0;
            }
            else
            {
              z_c = (c - %(ds1)s) / %(st1)s + 1;
            }
        }
        else
        {
            // decide how many rows the output has
            if (%(st0)s >= %(ds0)s)
            {
                z_r = (r - 1) / %(st0)s + 1;
            }
            else
            {
                z_r = std::max(0, (r - 1 - %(ds0)s + %(st0)s) / %(st0)s) + 1;
            }
            // decide how many columns the output has
            if (%(st1)s >= %(ds1)s)
            {
                z_c = (c - 1) / %(st1)s + 1;
            }
            else
            {
                z_c = std::max(0, (c - 1 - %(ds1)s + %(st0)s) / %(st1)s) + 1;
            }
            assert(z_r > 0);
            assert(z_c > 0);
        }
        // memory allocation of z if necessary
        if ((!%(z)s)
          || PyArray_NDIM(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(z)s)[2] != z_r)
          ||(PyArray_DIMS(%(z)s)[3] != z_c)
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=PyArray_DIMS(%(x)s)[1];
          dims[2]=z_r;
          dims[3]=z_c;
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }
        // used for indexing a pool region inside the input
        dtype_%(x)s collector; // temp var for the value in a region
        if (z_r && z_c)
        {
            int r_st, r_end, c_st, c_end;
            %(omp_parallel)s
            for(int t = 0; t < PyArray_DIMS(%(x)s)[0] * PyArray_DIMS(%(x)s)[1]; t++){
                int b = t %% PyArray_DIMS(%(x)s)[0];
                int k = t / PyArray_DIMS(%(x)s)[0];
                for(int i=0; i < z_r; i++){
                  r_st = i * %(st0)s;
                  r_end = r_st + %(ds0)s;
                  // skip the padding
                  r_st = r_st < %(pd0)s ? %(pd0)s : r_st;
                  r_end = r_end > (r - %(pd0)s) ? r - %(pd0)s : r_end;
                  // from padded_img space to img space
                  r_st -= %(pd0)s;
                  r_end -= %(pd0)s;
                  // handle the case where no padding, ignore border is True
                  if (%(ignore_border)s)
                  {
                    r_end = r_end > r ? r : r_end;
                  }
                  for(int j=0; j<z_c; j++){
                    c_st = j * %(st1)s;
                    c_end = c_st + %(ds1)s;
                    // skip the padding
                    c_st = c_st < %(pd1)s ? %(pd1)s : c_st;
                    c_end = c_end > (c - %(pd1)s) ? c - %(pd1)s : c_end;
                    dtype_%(z)s * z = (
                          (dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, b, k, i, j)));
                    // change coordinates from padding_img space into img space
                    c_st -= %(pd1)s;
                    c_end -= %(pd1)s;
                    // handle the case where no padding, ignore border is True
                    if (%(ignore_border)s)
                    {
                      c_end = c_end > c ? c : c_end;
                    }
        """
        if self.mode == 'max':
            ccode += """
                    // use the first element as the initial value of collector
                    collector = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,r_st,c_st)))[0];
                    // go through the pooled region in the unpadded input
                    for(int m=r_st; m<r_end; m++)
                    {
                      for(int n=c_st; n<c_end; n++)
                      {
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,m,n)))[0];
                        collector = (a > collector) ? a : collector;
                      }
                    }
                    z[0] = collector;
            """
        elif self.mode in ('sum', 'average_exc_pad', 'average_inc_pad'):
            ccode += """
                    // initialize the sum at zero
                    collector = ((dtype_%(x)s)(0));
                    // go through the pooled region in the unpadded input
                    for(int m=r_st; m<r_end; m++)
                    {
                      for(int n=c_st; n<c_end; n++)
                      {
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,m,n)))[0];
                        collector += a;
                      }
                    }
            """
            if self.mode == "sum":
                ccode += """
                    z[0] = collector;
                """
            elif self.mode == 'average_inc_pad' and self.ignore_border:
                ccode += """
                    z[0] = collector / (%(ds0)s * %(ds1)s);
                """
            else:
                ccode += """
                    z[0] = collector / ((r_end-r_st)*(c_end-c_st));
                """
        ccode += """
                  }
                }
              }
            }
        """
        return ccode % locals()

    def c_code_cache_version(self):
        return (0, 6, 8, 5, self.openmp)


class PoolGrad(OpenMPOp):
    __props__ = ('ds', 'ignore_border', 'st', 'padding', 'mode')

    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False, st=None, padding=(0, 0)):
        """Return the shape of the output from this op, for input of given
        shape and flags.

        Parameters
        ----------
        imgshape : tuple of integers or scalar Theano variables
            the shape of a tensor of images. The last two elements are
            interpreted as the number of rows, and the number of cols.
        ds : tuple of two ints
            downsample factor over rows and columns this parameter
            indicates the size of the pooling region
        st : tuple of two ints
            the stride size. This is the distance between the pooling
            regions. If it's set to None, in which case it equlas ds.
        ignore_border : bool
            if ds doesn't divide imgshape, do we include an extra
            row/col of partial downsampling (False) or ignore it
            (True).
        padding : tuple of two ints
            (pad_h, pad_w), pad zeros to extend beyond four borders of
            the images, pad_h is the size of the top and bottom
            margins, and pad_w is the size of the left and right
            margins.

        Returns
        -------
        list :
            the shape of the output from this op, for input of given
            shape.  This will have the same length as imgshape, but
            with last two elements reduced as per the downsampling &
            ignore_border flags.

        """
        if len(imgshape) < 2:
            raise TypeError('imgshape must have at least two elements '
                            '(rows, cols)')

        if st is None:
            st = ds
        r, c = imgshape[-2:]
        r += padding[0] * 2
        c += padding[1] * 2

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
                                   tensor.maximum(0, (r - 1 - ds[0]) //
                                                  st[0] + 1) + 1)
            elif st[0] >= ds[0]:
                nr = (r - 1) // st[0] + 1
            else:
                nr = max(0, (r - 1 - ds[0]) // st[0] + 1) + 1

            if isinstance(c, theano.Variable):
                nc = tensor.switch(tensor.ge(st[1], ds[1]),
                                   (c - 1) // st[1] + 1,
                                   tensor.maximum(0, (c - 1 - ds[1]) //
                                                  st[1] + 1) + 1)
            elif st[1] >= ds[1]:
                nc = (c - 1) // st[1] + 1
            else:
                nc = max(0, (c - 1 - ds[1]) // st[1] + 1) + 1

        rval = list(imgshape[:-2]) + [nr, nc]
        return rval

    def __init__(self, ds, ignore_border, st=None, padding=(0, 0), mode='max', openmp=None):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border
        if st is None:
            st = ds
        self.st = tuple(st)
        self.padding = tuple(padding)
        if mode not in ['max', 'sum', 'average_inc_pad', 'average_exc_pad']:
            raise ValueError(
                "Pool mode parameter only support 'max', 'sum',"
                " 'average_inc_pad' and 'average_exc_pad'. Got %s" % mode)
        self.mode = mode
        super(PoolGrad, self).__init__(openmp=openmp)

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]


class MaxPoolGrad(PoolGrad):
    def __init__(self, ds, ignore_border, st=None, padding=(0, 0), openmp=None):
        PoolGrad.__init__(self, ds, ignore_border, st, padding, 'max', openmp)

    def make_node(self, x, maxout, gz):
        # make_node should only be called by the grad function of
        # Pool, so these asserts should not fail.
        x = tensor.as_tensor_variable(x)
        maxout = tensor.as_tensor_variable(maxout)
        gz = tensor.as_tensor_variable(gz)
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(maxout, Variable) and maxout.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4

        return Apply(self, [x, maxout, gz], [x.type()])

    def perform(self, node, inp, out):
        assert self.mode == 'max'
        x, maxout, gz = inp
        gx_stg, = out
        # number of pooling output rows
        pr = maxout.shape[-2]
        # number of pooling output cols
        pc = maxout.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w

        # pad the image
        if self.padding != (0, 0):
            y = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype)
            y[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)] = x
        else:
            y = x
        gx = numpy.zeros_like(y)
        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = builtins.max(r * st0, self.padding[0])
                    row_end = builtins.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        col_st = builtins.max(c * st1, self.padding[1])
                        col_end = builtins.min(col_st + ds1, img_cols)
                        for row_ind in xrange(row_st, row_end):
                            for col_ind in xrange(col_st, col_end):
                                if (maxout[n, k, r, c] == y[n, k, row_ind, col_ind]):
                                    gx[n, k, row_ind, col_ind] += gz[n, k, r, c]
        # unpad the image
        gx = gx[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)]
        gx_stg[0] = gx

    def grad(self, inp, grads):
        x, maxout, gz = inp
        ggx, = grads
        return [theano.tensor.zeros_like(x),
                theano.tensor.zeros_like(maxout),
                DownsampleFactorMaxGradGrad(
                    self.ds, ignore_border=self.ignore_border,
                    st=self.st, padding=self.padding)(x, maxout, ggx)]

    def c_code(self, node, name, inp, out, sub):
        assert self.mode == 'max'
        x, z, gz = inp
        gx, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pd0, pd1 = self.padding
        if self.openmp:
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, c_st, c_end, maximum) schedule(static)'
        else:
            omp_parallel = ''
        return """
        // sanity checks
        int x_typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_typenum = PyArray_ObjectType((PyObject*)%(z)s, 0);
        int gz_typenum = PyArray_ObjectType((PyObject*)%(gz)s, 0);
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
        int z_r, z_c;
        z_r = PyArray_DIMS(%(z)s)[2];
        z_c = PyArray_DIMS(%(z)s)[3];
        int r, c; // shape of the padded_input
        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        r += %(pd0)s * 2;
        c += %(pd1)s * 2;
        // allocating memory for gx
        if ((!%(gx)s)
          || !PyArray_ISCONTIGUOUS(%(gx)s)
          || PyArray_NDIM(%(gx)s)!=4
          ||(PyArray_DIMS(%(gx)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(gx)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(gx)s)[2] != PyArray_DIMS(%(x)s)[2])
          ||(PyArray_DIMS(%(gx)s)[3] != PyArray_DIMS(%(x)s)[3])
          )
        {
          Py_XDECREF(%(gx)s);
          %(gx)s = (PyArrayObject*) PyArray_ZEROS(4, PyArray_DIMS(%(x)s), x_typenum,0);
        }
        else {
          PyArray_FILLWBYTE(%(gx)s, 0);
        }
        dtype_%(z)s maximum; // temp var for maximum value in a region
        if (z_r && z_c)
        {
            int r_st, r_end, c_st, c_end;
            %(omp_parallel)s
            for(int t = 0; t < PyArray_DIMS(%(x)s)[0] * PyArray_DIMS(%(x)s)[1]; t++){
                int b = t %% PyArray_DIMS(%(x)s)[0];
                int k = t / PyArray_DIMS(%(x)s)[0];
                for(int i=0; i < z_r; i++){
                  r_st = i * %(st0)s;
                  r_end = r_st + %(ds0)s;
                  // skip the padding
                  r_st = r_st < %(pd0)s ? %(pd0)s : r_st;
                  r_end = r_end > (r - %(pd0)s) ? r - %(pd0)s : r_end;
                  // from padded_img space to img space
                  r_st -= %(pd0)s;
                  r_end -= %(pd0)s;
                  for(int j=0; j<z_c; j++){
                    c_st = j * %(st1)s;
                    c_end = c_st + %(ds1)s;
                    // skip the padding
                    c_st = c_st < %(pd1)s ? %(pd1)s : c_st;
                    c_end = c_end > (c - %(pd1)s) ? c - %(pd1)s : c_end;
                    // change coordinates from padding_img space into img space
                    c_st -= %(pd1)s;
                    c_end -= %(pd1)s;
                    // the maximum value
                    maximum = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,b,k,i,j)))[0];
                    // the gradient corresponding to this maximum value in z
                    dtype_%(gz)s * gz = (
                          (dtype_%(gz)s*)(PyArray_GETPTR4(%(gz)s, b, k, i, j)));
                    // go through the pooled region in the unpadded input
                    for(int m=r_st; m<r_end; m++)
                    {
                      for(int n=c_st; n<c_end; n++)
                      {
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,m,n)))[0];
                        dtype_%(gx)s * gx = (
                          (dtype_%(gx)s*)(PyArray_GETPTR4(%(gx)s, b, k, m, n)));
                        if (a == maximum){
                          gx[0] = gx[0] + gz[0];
                        }
                      }
                    }
                  }
                }
              }
            }
        """ % locals()

    def c_code_cache_version(self):
        return (0, 8, self.openmp)


class AveragePoolGrad(PoolGrad):
    def __init__(self, ds, ignore_border, st=None, padding=(0, 0),
                 mode='average_inc_pad'):
        assert mode in ['sum', 'average_inc_pad', 'average_exc_pad']
        PoolGrad.__init__(self, ds, ignore_border, st, padding, mode)

    # There is an extra dummy parameter to match the parameter count
    # of MaxPoolGrad.  They have to keep the same interface because of
    # the DownsampleFactorMaxGrad trick to keep old scripts working
    # (see downsample.py for details on this).
    def make_node(self, x, gz, dummy=None):
        # make_node should only be called by the grad function of
        # Pool, so these asserts should not fail.
        x = tensor.as_tensor_variable(x)
        gz = tensor.as_tensor_variable(gz)
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4

        return Apply(self, [x, gz], [x.type()])

    def perform(self, node, inp, out):
        if self.mode == 'average_exc_pad' and self.padding != (0, 0):
            raise NotImplementedError()
        x, gz = inp
        gx_stg, = out
        z_shape = self.out_shape(x.shape, self.ds, self.ignore_border, self.st,
                                 self.padding)
        if (gx_stg[0] is None) or (gx_stg[0].shape != z_shape):
            gx_stg[0] = numpy.empty(z_shape, dtype=x.dtype)
        zz = gx_stg[0]
        # number of pooling output rows
        pr = zz.shape[-2]
        # number of pooling output cols
        pc = zz.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w
        inc_pad = self.mode == 'average_inc_pad'
        sum_mode = self.mode == 'sum'

        # pad the image
        if self.padding != (0, 0):
            y = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype)
            y[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)] = x
        else:
            y = x
        gx = numpy.zeros_like(y)
        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    if sum_mode or inc_pad:
                        row_st = r * st0
                    else:
                        row_st = builtins.max(r * st0, self.padding[0])
                    row_end = builtins.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        if sum_mode or inc_pad:
                            col_st = c * st1
                        else:
                            col_st = builtins.max(c * st1,
                                                  self.padding[1])
                        col_end = builtins.min(col_st + ds1, img_cols)
                        if sum_mode:
                            val = gz[n, k, r, c]
                        else:
                            val = gz[n, k, r, c] / ((row_end - row_st) *
                                                    (col_end - col_st))
                        gx[n, k, row_st:row_end, col_st:col_end] += val
        # unpad the image
        gx = gx[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)]
        gx_stg[0] = gx

    def grad(self, inp, grads):
        x, gz = inp
        ggx, = grads
        return [theano.tensor.zeros_like(x),
                Pool(self.ds, ignore_border=self.ignore_border,
                     st=self.st, padding=self.padding, mode=self.mode)(ggx)]


class DownsampleFactorMaxGradGrad(OpenMPOp):
    __props__ = ('ds', 'ignore_border', 'st', 'padding', 'mode')

    def __init__(self, ds, ignore_border, st=None, padding=(0, 0), mode='max', openmp=None):
        self.ds = tuple(ds)
        if not all([isinstance(d, integer_types) for d in ds]):
            raise ValueError(
                "Pool downsample parameters must be ints."
                " Got %s" % str(ds))
        if st is None:
            st = ds
        assert isinstance(st, (tuple, list))
        self.st = tuple(st)
        self.ignore_border = ignore_border
        self.padding = tuple(padding)
        if self.padding != (0, 0) and not ignore_border:
            raise NotImplementedError(
                'padding works only with ignore_border=True')
        if self.padding[0] >= self.ds[0] or self.padding[1] >= self.ds[1]:
            raise NotImplementedError(
                'padding_h and padding_w must be smaller than strides')
        self.mode = mode
        super(DownsampleFactorMaxGradGrad, self).__init__(openmp=openmp)
        assert self.mode == 'max'

    def make_node(self, x, maxout, gz):
        # make_node should only be called by the grad function of
        # MaxPoolGrad, so these asserts should not fail.
        x = tensor.as_tensor_variable(x)
        maxout = tensor.as_tensor_variable(maxout)
        gz = tensor.as_tensor_variable(gz)
        assert x.ndim == 4
        assert maxout.ndim == 4
        assert gz.ndim == 4

        return Apply(self, [x, maxout, gz], [x.type()])

    def perform(self, node, inp, out):
        x, maxout, ggx = inp
        z, = out
        if len(x.shape) != 4:
            raise NotImplementedError(
                'DownsampleFactorMaxGradGrad requires 4D input for now')
        if (z[0] is None) or (z[0].shape != maxout.shape):
            z[0] = numpy.zeros(maxout.shape, dtype=x.dtype)
        ggz = z[0]  # grad wrt maxout_grad has the same shape as maxout
        # number of pooling output rows
        pr = ggz.shape[-2]
        # number of pooling output cols
        pc = ggz.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pd0, pd1 = self.padding
        img_rows = x.shape[-2] + 2 * pd0
        img_cols = x.shape[-1] + 2 * pd1

        # pad the image and its gradients
        if self.padding != (0, 0):
            y_padded = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype) + x.min() - 1
            y_padded[:, :, pd0:(img_rows - pd0), pd1:(img_cols - pd1)] = x
            ggx_padded = numpy.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype)
            ggx_padded[:, :, pd0:(img_rows - pd0), pd1:(img_cols - pd1)] = ggx

        else:
            y_padded = x
            ggx_padded = ggx
        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = r * st0
                    row_end = builtins.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        col_st = c * st1
                        col_end = builtins.min(col_st + ds1, img_cols)
                        for row_ind in xrange(row_st, row_end):
                            for col_ind in xrange(col_st, col_end):
                                if (maxout[n, k, r, c] == y_padded[n, k, row_ind, col_ind]):
                                    ggz[n, k, r, c] = ggx_padded[n, k, row_ind, col_ind]

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

    def grad(self, inp, grads):
        x, maxout, ggx = inp
        gz, = grads
        return [theano.tensor.zeros_like(x),
                theano.tensor.zeros_like(maxout),
                MaxPoolGrad(
                    self.ds, ignore_border=self.ignore_border,
                    st=self.st, padding=self.padding)(x, maxout, gz)]

    def c_code(self, node, name, inp, out, sub):
        if self.mode != 'max':
            raise theano.gof.utils.MethodNotDefined()
        x, maxout, ggx = inp
        z, = out  # the grad of grad
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pd0, pd1 = self.padding
        if self.openmp:
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, c_st, c_end, maximum) schedule(static)'
        else:
            omp_parallel = ''
        return """
        int z_typenum = PyArray_ObjectType((PyObject*)%(maxout)s, 0);
        int z_r, z_c;
        z_r = PyArray_DIMS(%(maxout)s)[2];
        z_c = PyArray_DIMS(%(maxout)s)[3];
        int r, c; // shape of the padded_input
        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        r += %(pd0)s * 2;
        c += %(pd1)s * 2;
        // allocating memory for output
        if ((!%(z)s)
          || !PyArray_ISCONTIGUOUS(%(z)s)
          || PyArray_NDIM(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(maxout)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(maxout)s)[1])
          ||(PyArray_DIMS(%(z)s)[2] != PyArray_DIMS(%(maxout)s)[2])
          ||(PyArray_DIMS(%(z)s)[3] != PyArray_DIMS(%(maxout)s)[3])
          )
        {
          Py_XDECREF(%(z)s);
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, PyArray_DIMS(%(maxout)s), z_typenum,0);
        }
        else {
          PyArray_FILLWBYTE(%(z)s, 0);
        }
        dtype_%(maxout)s maximum; // temp var for maximum value in a region
        int r_st, r_end, c_st, c_end;
        %(omp_parallel)s
        for(int t = 0; t < PyArray_DIMS(%(x)s)[0] * PyArray_DIMS(%(x)s)[1]; t++){
            int b = t %% PyArray_DIMS(%(x)s)[0];
            int k = t / PyArray_DIMS(%(x)s)[0];
                for(int i=0; i < z_r; i++){
                  r_st = i * %(st0)s;
                  r_end = r_st + %(ds0)s;
                  // skip the padding
                  r_st = r_st < %(pd0)s ? %(pd0)s : r_st;
                  r_end = r_end > (r - %(pd0)s) ? r - %(pd0)s : r_end;
                  // from padded_img space to img space
                  r_st -= %(pd0)s;
                  r_end -= %(pd0)s;
                  for(int j=0; j<z_c; j++){
                    c_st = j * %(st1)s;
                    c_end = c_st + %(ds1)s;
                    // skip the padding
                    c_st = c_st < %(pd1)s ? %(pd1)s : c_st;
                    c_end = c_end > (c - %(pd1)s) ? c - %(pd1)s : c_end;
                    // from padding_img space into img space
                    c_st -= %(pd1)s;
                    c_end -= %(pd1)s;
                    // the maximum value
                    maximum = ((dtype_%(maxout)s*)(PyArray_GETPTR4(%(maxout)s,b,k,i,j)))[0];
                    // z at this position
                    dtype_%(z)s * z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, b, k, i, j)));
                    // go through the pooled region in the unpadded input
                    for(int m=r_st; m<r_end; m++)
                    {
                      for(int n=c_st; n<c_end; n++)
                      {
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,m,n)))[0];
                        dtype_%(ggx)s * ggx = (
                          (dtype_%(ggx)s*)(PyArray_GETPTR4(%(ggx)s, b, k, m, n)));
                        if (a == maximum){
                          z[0] += ggx[0];
                        }
                      }
                    }
                  }
                }
              }
        """ % locals()

    def c_code_cache_version(self):
        return (0, 2, self.openmp)


class DownsampleFactorMaxRop(Op):
    """
    Implements the R-operator for the downsample operation.
    """

    __props__ = ('ds', 'ignore_border', 'st', 'padding', 'mode')

    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False):
        """Return the shape of the output from this op, for input of given shape and flags.
        :param imgshape: the shape of a tensor of images. The last two elements are interpreted
        as the number of rows, and the number of cols.
        :type imgshape: tuple, list, or similar.
        :param ds: downsample factor over rows and columns
        :type ds: list or tuple of two ints
        :param ignore_border: if ds doesn't divide imgshape, do we include an extra row/col of
        partial downsampling (False) or ignore it (True).
        :type ignore_border: bool
        :rtype: list
        :returns: the shape of the output from this op, for input of given shape.  This will
        have the same length as imgshape, but with last two elements reduced as per the
        downsampling & ignore_border flags.
        """
        if len(imgshape) < 2:
            raise TypeError('imgshape must have at least two elements (rows, cols)')
        r, c = imgshape[-2:]
        rval = list(imgshape[:-2]) + [r / ds[0], c / ds[1]]
        if not ignore_border:
            if r % ds[0]:
                rval[-2] += 1
            if c % ds[1]:
                rval[-1] += 1
        return rval

    def __init__(self, ds, ignore_border, st=None, padding=(0, 0), mode='max'):
        """
        :param ds: downsample factor over rows and columns
        :type ds: list or tuple of two ints
        :param ignore_border: if ds doesn't divide imgshape, do we include an extra row/col of
        partial downsampling (False) or ignore it (True).
        :type ignore_border: bool
        TODO: why is poolsize an op parameter here?
        """
        self.ds = tuple(ds)
        self.ignore_border = ignore_border
        if st is None:
            st = ds
        self.st = tuple(st)
        self.padding = tuple(padding)
        self.mode = mode
        assert self.mode == 'max'
        if padding != (0, 0):
            raise NotImplementedError("DownsampleFactorMaxRop do not currently implement pad")
        if st != ds:
            raise NotImplementedError("DownsampleFactorMaxRop do not currently implement strides")

    def __eq__(self, other):
        return type(self) == type(other) and self.ds == other.ds and self.ignore_border == other.ignore_border

    def __hash__(self):
        return hash(type(self)) ^ hash(self.ds) ^ hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s}' % (self.__class__.__name__, self.ds, self.ignore_border)

    def make_node(self, x, eval_point):
        if x.type.ndim != 4:
            raise TypeError('Expected tensor4')
        if x.type.ndim != 4:
            return TypeError('Expected tensor4')
        # TODO: consider restrucing the dtype?
        x = tensor.as_tensor_variable(x)
        eval_point = tensor.as_tensor_variable(eval_point)
        return gof.Apply(self, [x, eval_point], [x.type()])

    def perform(self, node, inp, out):
        x, ex = inp
        z, = out
        if len(x.shape) != 4:
            raise NotImplementedError('DownsampleFactorMax requires 4D input for now')
        z_shape = self.out_shape(x.shape, self.ds, self.ignore_border)
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = numpy.zeros(self.out_shape(x.shape, self.ds, self.ignore_border))
            z[0] = theano._asarray(z[0], dtype=x.dtype)
        zz = z[0]
        fake_zz = numpy.zeros(self.out_shape(x.shape, self.ds,
                                             self.ignore_border))

        # zz needs to be initialized with -inf for the following to work
        zz -= numpy.inf
        fake_zz -= numpy.inf
        ds0, ds1 = self.ds

        if self.ignore_border:
            x_usable2 = (x.shape[2] // ds0 * ds0)
        else:
            x_usable2 = x.shape[2]

        if self.ignore_border:
            x_usable3 = (x.shape[3] // ds1 * ds1)
        else:
            x_usable3 = x.shape[3]

        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for i in xrange(x_usable2):
                    zi = i // ds0
                    for j in xrange(x_usable3):
                        zj = j // ds1
                        if fake_zz[n, k, zi, zj] < x[n, k, i, j]:
                            fake_zz[n, k, zi, zj] = x[n, k, i, j]
                            zz[n, k, zi, zj] = ex[n, k, i, j]

    def c_code(self, node, name, inp, out, sub):
        x, ex = inp
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
          || PyArray_NDIM(%(z)s)!=4
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
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0); //TODO: zeros not necessary
        }
        if (z_shp0 && z_shp1)
        {
            npy_intp fake_dims[4] = {0,0,0,0};
            fake_dims[0]=PyArray_DIMS(%(x)s)[0];
            fake_dims[1]=PyArray_DIMS(%(x)s)[1];
            fake_dims[2]=z_shp0;
            fake_dims[3]=z_shp1;
            PyArrayObject * fake_z = (PyArrayObject*) PyArray_ZEROS(4, fake_dims, typenum, 0);
            for(int b=0;b<PyArray_DIMS(%(x)s)[0];b++){
              for(int k=0;k<PyArray_DIMS(%(x)s)[1];k++){
                int mini_i = 0;
                int zi = 0;
                for(int i=0;i< x_shp0_usable; i++){
                  int mini_j = 0;
                  int zj = 0;
                  for(int j=0; j<x_shp1_usable; j++){
                    dtype_%(x)s  a = ((dtype_%(x)s  *)(PyArray_GETPTR4(%(x)s ,b,k,i,j)))[0];
                    dtype_%(ex)s fa = ((dtype_%(ex)s *)(PyArray_GETPTR4(%(ex)s,b,k,i,j)))[0];
                    dtype_%(z)s * __restrict__ z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,b,k,zi,zj)));
                    dtype_%(z)s * __restrict__ fz = ((dtype_%(z)s*)(PyArray_GETPTR4(fake_z,b,k,zi,zj)));
                    if (((mini_j| mini_i) == 0) || fz[0] < a)
                    {
                      fz[0] = a;
                      z[0] = fa;
                    }
                    mini_j = ((mini_j + 1) == %(ds1)s) ? 0 : mini_j+1;
                    zj += (mini_j == 0);
                  }
                  mini_i = ((mini_i + 1) == %(ds0)s) ? 0 : mini_i+1;
                  zi += (mini_i == 0);
                }
              }
            }
            Py_XDECREF(fake_z);
            //free(fake_z)
        }
        """ % locals()

    def c_code_cache_version(self):
        return (0, 2)
