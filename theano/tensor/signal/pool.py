
"""
Ops for downsampling images.
Planned:
Pool, DownsampleAvg, DownsampleSoftmax.
"""
from __future__ import absolute_import, print_function, division
# This file should move along with conv.py
import warnings

import numpy
from six.moves import xrange
import six.moves.builtins as builtins
import theano
from theano import gof, OpenMPOp, tensor, Variable, Apply
from theano.gradient import DisconnectedType


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
    patch_size : tuple of length 2 or theano vector of ints of size 2.
        Size of the patch (patch height, patch width).
        (2,2) will retain only one non-zero value per patch of 4 values.

    """
    output = Pool(True)(input, patch_size)
    outs = MaxPoolGrad(True)(input, output, output, patch_size)
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
    ds : tuple of length 2 or theano vector of ints of size 2.
        Factor by which to downscale (vertical ds, horizontal ds).
        (2,2) will halve the image in each dimension.
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    st : tuple of two ints or theano vector of ints of size 2.
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    padding : tuple of two ints or theano vector of ints of size 2.
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
        op = Pool(ignore_border, mode=mode)
        output = op(input, ds, st, padding)
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
    op = Pool(ignore_border, mode=mode)
    output = op(input_4D, ds, st, padding)

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

    __props__ = ('ignore_border', 'mode')

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
            r = r + padding[0] * 2
        if padding[1]:
            c = c + padding[1] * 2

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

    def __init__(self, ignore_border=False, mode='max', openmp=None):
        super(Pool, self).__init__(openmp=openmp)
        self.ignore_border = ignore_border
        if mode not in ['max', 'average_inc_pad', 'average_exc_pad', 'sum']:
            raise ValueError(
                "Pool mode parameter only support 'max', 'sum',"
                " 'average_inc_pad' and 'average_exc_pad'. Got %s" % mode)
        self.mode = mode

    def prepare_node(self, node, storage_map, compute_map):
        if len(node.inputs) == 1:
            # Old interface
            self.mode = node.op.mode
            ws = theano.tensor.constant(node.op.ds)
            st = theano.tensor.constant(node.op.st)
            pad = theano.tensor.constant(node.op.padding)
            node.inputs.append(ws)
            node.inputs.append(st)
            node.inputs.append(pad)
            if isinstance(ws, theano.Constant):
                storage_map[ws] = [ws.data]
                compute_map[ws] = [True]
            else:
                storage_map[ws] = [None]
                compute_map[ws] = [False]
            if isinstance(st, theano.Constant):
                storage_map[st] = [st.data]
                compute_map[st] = [True]
            else:
                storage_map[st] = [None]
                compute_map[st] = [False]
            if isinstance(pad, theano.Constant):
                storage_map[pad] = [pad.data]
                compute_map[pad] = [True]
            else:
                storage_map[pad] = [None]
                compute_map[pad] = [False]

    def make_node(self, x, ws, stride=None, pad=(0, 0)):
        # TODO: consider restricting the dtype?
        x = tensor.as_tensor_variable(x)
        if stride is None:
            stride = ws
        if isinstance(pad, (tuple, list)):
            if tuple(pad) != (0, 0) and not self.ignore_border:
                raise NotImplementedError(
                    'padding works only with ignore_border=True')
            if isinstance(ws, (tuple, list)):
                if pad[0] >= ws[0] or pad[1] >= ws[1]:
                    raise NotImplementedError(
                        'padding_h and padding_w must be smaller than strides')
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert ws.ndim == 1
        assert stride.ndim == 1
        assert pad.ndim == 1
        if x.type.ndim != 4:
            raise TypeError()
        if not ws.dtype.startswith('int'):
            raise TypeError('Pool downsample parameters must be ints.')
        if not stride.dtype.startswith('int'):
            raise TypeError('Stride parameters must be ints.')
        if not pad.dtype.startswith('int'):
            raise TypeError('Padding parameters must be ints.')
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:2] + (False, False)
        out = tensor.TensorType(x.dtype, broad)
        return gof.Apply(self, [x, ws, stride, pad], [out()])

    def perform(self, node, inp, out):
        x, ws, stride, pad = inp
        z, = out
        assert ws.shape == stride.shape == pad.shape == (2,)
        if len(x.shape) != 4:
            raise NotImplementedError(
                'Pool requires 4D input for now')
        z_shape = self.out_shape(x.shape, ws, self.ignore_border, stride, pad)
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
        ws0, ws1 = ws
        st0, st1 = stride
        pad_h = pad[0]
        pad_w = pad[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w
        inc_pad = self.mode == 'average_inc_pad'

        # pad the image
        if (pad_h, pad_w) != (0, 0):
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
                    row_end = builtins.min(row_st + ws0, img_rows)
                    if not inc_pad:
                        row_st = builtins.max(row_st, pad_h)
                        row_end = builtins.min(row_end, x.shape[-2] + pad_h)
                    for c in xrange(pc):
                        col_st = c * st1
                        col_end = builtins.min(col_st + ws1, img_cols)
                        if not inc_pad:
                            col_st = builtins.max(col_st, pad_w)
                            col_end = builtins.min(col_end,
                                                   x.shape[-1] + pad_w)
                        zz[n, k, r, c] = func(y[
                            n, k, row_st:row_end, col_st:col_end])

    def infer_shape(self, node, in_shapes):
        ws, stride, pad = [node.inputs[1], node.inputs[2], node.inputs[3]]
        shp = self.out_shape(in_shapes[0], ws, self.ignore_border, stride,
                             pad)
        return [shp]

    def grad(self, inp, grads):
        x, ws, stride, pad = inp
        gz, = grads
        disc = [DisconnectedType()() for i in inp[1:]]
        if self.mode == 'max':
            maxout = self(x, ws, stride, pad)
            return [MaxPoolGrad(ignore_border=self.ignore_border)(
                x, maxout, gz, ws=ws, stride=stride, pad=pad)] + disc
        else:
            return [AveragePoolGrad(ignore_border=self.ignore_border,
                                    mode=self.mode)(
                x, gz, ws=ws, stride=stride, pad=pad)] + disc

    def connection_pattern(self, node):
        return [[1], [0], [0], [0]]

    def c_headers(self):
        headers = ['<algorithm>']
        headers += super(Pool, self).c_headers()
        return headers

    def c_code(self, node, name, inp, out, sub):
        if self.mode not in ('max', 'sum', 'average_exc_pad', 'average_inc_pad'):
            raise theano.gof.utils.MethodNotDefined()
        x, ws, stride, pad = inp
        z, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        if self.openmp:
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, c_st, c_end, collector) schedule(static)'
        else:
            omp_parallel = ''
        ccode = """
        int ws0, ws1, st0, st1, pd0, pd1;
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_r, z_c; // shape of the output
        int r, c; // shape of the padded_input
        if(PyArray_DIM(%(ws)s, 0)!=2)
        {
            PyErr_SetString(PyExc_ValueError, "ws must be a vector of size 2");
            %(fail)s;
        }
        if(PyArray_DIM(%(stride)s, 0)!=2)
        {
            PyErr_SetString(PyExc_ValueError, "stride must be a vector of size 2");
            %(fail)s;
        }
        if(PyArray_DIM(%(pad)s, 0)!=2)
        {
            PyErr_SetString(PyExc_ValueError, "pad must be a vector of size 2");
            %(fail)s;
        }
        // Getting ws, stride and pad
        ws0 = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 0));
        ws1 = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 1));
        st0 = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 0));
        st1 = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 1));
        pd0 = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 0));
        pd1 = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 1));
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        r += pd0 * 2;
        c += pd1 * 2;
        if (pd0 != 0 && pd1 != 0 && !%(ignore_border)s)
            {
              PyErr_SetString(PyExc_ValueError,
                "padding must be (0,0) when ignore border is False");
              %(fail)s;
            }
        if (%(ignore_border)s)
        {
            // '/' in C is different from '/' in python
            if (r - ws0 < 0)
            {
              z_r = 0;
            }
            else
            {
              z_r = (r - ws0) / st0 + 1;
            }
            if (c - ws1 < 0)
            {
              z_c = 0;
            }
            else
            {
              z_c = (c - ws1) / st1 + 1;
            }
        }
        else
        {
            // decide how many rows the output has
            if (st0 >= ws0)
            {
                z_r = (r - 1) / st0 + 1;
            }
            else
            {
                z_r = std::max(0, (r - 1 - ws0 + st0) / st0) + 1;
            }
            // decide how many columns the output has
            if (st1 >= ws1)
            {
                z_c = (c - 1) / st1 + 1;
            }
            else
            {
                z_c = std::max(0, (c - 1 - ws1 + st0) / st1) + 1;
            }
            assert(z_r > 0);
            assert(z_c > 0);
        }
        // memory allocation of z if necessary
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
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
                  r_st = i * st0;
                  r_end = r_st + ws0;
                  // skip the padding
                  r_st = r_st < pd0 ? pd0 : r_st;
                  r_end = r_end > (r - pd0) ? r - pd0 : r_end;
                  // from padded_img space to img space
                  r_st -= pd0;
                  r_end -= pd0;
                  // handle the case where no padding, ignore border is True
                  if (%(ignore_border)s)
                  {
                    r_end = r_end > r ? r : r_end;
                  }
                  for(int j=0; j<z_c; j++){
                    c_st = j * st1;
                    c_end = c_st + ws1;
                    // skip the padding
                    c_st = c_st < pd1 ? pd1 : c_st;
                    c_end = c_end > (c - pd1) ? c - pd1 : c_end;
                    dtype_%(z)s * z = (
                          (dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, b, k, i, j)));
                    // change coordinates from padding_img space into img space
                    c_st -= pd1;
                    c_end -= pd1;
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
                    z[0] = collector / (ws0 * ws1);
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
        return (0, 6, 8, 6, self.openmp)


class PoolGrad(OpenMPOp):
    __props__ = ('ignore_border', 'mode')

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

    def __init__(self, ignore_border, mode='max', openmp=None):
        self.ignore_border = ignore_border
        if mode not in ['max', 'sum', 'average_inc_pad', 'average_exc_pad']:
            raise ValueError(
                "Pool mode parameter only support 'max', 'sum',"
                " 'average_inc_pad' and 'average_exc_pad'. Got %s" % mode)
        self.mode = mode
        super(PoolGrad, self).__init__(openmp=openmp)

    def prepare_node(self, node, storage_map, compute_map):
        if len(node.inputs) < 5:  # 5 for AveragePoolGrad, 6 for MaxPoolGrad
            # Old interface
            self.mode = node.op.mode
            ws = theano.tensor.constant(node.op.ds)
            st = theano.tensor.constant(node.op.st)
            pad = theano.tensor.constant(node.op.padding)
            node.inputs.append(ws)
            node.inputs.append(st)
            node.inputs.append(pad)
            if isinstance(ws, theano.Constant):
                storage_map[ws] = [ws.data]
                compute_map[ws] = [True]
            else:
                storage_map[ws] = [None]
                compute_map[ws] = [False]
            if isinstance(st, theano.Constant):
                storage_map[st] = [st.data]
                compute_map[st] = [True]
            else:
                storage_map[st] = [None]
                compute_map[st] = [False]
            if isinstance(pad, theano.Constant):
                storage_map[pad] = [pad.data]
                compute_map[pad] = [True]
            else:
                storage_map[pad] = [None]
                compute_map[pad] = [False]

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]


class MaxPoolGrad(PoolGrad):
    def __init__(self, ignore_border, openmp=None):
        PoolGrad.__init__(self, ignore_border, mode='max', openmp=openmp)

    def make_node(self, x, maxout, gz, ws, stride=None, pad=(0, 0)):
        # make_node should only be called by the grad function of
        # Pool, so these asserts should not fail.
        x = tensor.as_tensor_variable(x)
        maxout = tensor.as_tensor_variable(maxout)
        gz = tensor.as_tensor_variable(gz)
        if stride is None:
            stride = ws
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(maxout, Variable) and maxout.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4
        assert isinstance(ws, Variable) and ws.ndim == 1
        assert isinstance(stride, Variable) and stride.ndim == 1
        assert isinstance(pad, Variable) and pad.ndim == 1
        if not ws.dtype.startswith('int'):
            raise TypeError('Pool downsample parameters must be ints.')
        if not stride.dtype.startswith('int'):
            raise TypeError('Stride parameters must be ints.')
        if not pad.dtype.startswith('int'):
            raise TypeError('Padding parameters must be ints.')
        return Apply(self, [x, maxout, gz, ws, stride, pad], [x.type()])

    def perform(self, node, inp, out):
        assert self.mode == 'max'
        x, maxout, gz, ws, stride, pad = inp
        gx_stg, = out
        assert ws.shape == stride.shape == pad.shape == (2,)
        # number of pooling output rows
        pr = maxout.shape[-2]
        # number of pooling output cols
        pc = maxout.shape[-1]
        ws0, ws1 = ws
        st0, st1 = stride
        pad_h = pad[0]
        pad_w = pad[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w

        # pad the image
        if (pad_h, pad_w) != (0, 0):
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
                    row_st = builtins.max(r * st0, pad_h)
                    row_end = builtins.min(row_st + ws0, img_rows)
                    for c in xrange(pc):
                        col_st = builtins.max(c * st1, pad_w)
                        col_end = builtins.min(col_st + ws1, img_cols)
                        for row_ind in xrange(row_st, row_end):
                            for col_ind in xrange(col_st, col_end):
                                if (maxout[n, k, r, c] == y[n, k, row_ind, col_ind]):
                                    gx[n, k, row_ind, col_ind] += gz[n, k, r, c]
        # unpad the image
        gx = gx[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)]
        gx_stg[0] = gx

    def grad(self, inp, grads):
        x, maxout, gz, ws, stride, pad = inp
        ggx, = grads
        return ([theano.tensor.zeros_like(x),
                 theano.tensor.zeros_like(maxout),
                 DownsampleFactorMaxGradGrad(ignore_border=self.ignore_border)(
                x, maxout, ggx, ws, stride, pad)] +
                [DisconnectedType()() for i in inp[3:]])

    def connection_pattern(self, node):
        return [[1], [1], [1], [0], [0], [0]]

    def c_code(self, node, name, inp, out, sub):
        assert self.mode == 'max'
        x, z, gz, ws, stride, pad = inp
        gx, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        if self.openmp:
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, c_st, c_end, maximum) schedule(static)'
        else:
            omp_parallel = ''
        return """
        // sanity checks
        int x_typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_typenum = PyArray_ObjectType((PyObject*)%(z)s, 0);
        int gz_typenum = PyArray_ObjectType((PyObject*)%(gz)s, 0);
        int ws0, ws1, st0, st1, pd0, pd1;
        int z_r, z_c;
        int r, c; // shape of the padded_input
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
        if(PyArray_DIM(%(ws)s, 0)!=2)
        {
            PyErr_SetString(PyExc_ValueError, "ws must be a vector of size 2");
            %(fail)s;
        }
        if(PyArray_DIM(%(stride)s, 0)!=2)
        {
            PyErr_SetString(PyExc_ValueError, "stride must be a vector of size 2");
            %(fail)s;
        }
        if(PyArray_DIM(%(pad)s, 0)!=2)
        {
            PyErr_SetString(PyExc_ValueError, "pad must be a vector of size 2");
            %(fail)s;
        }
        // Getting ws, stride and pad
        ws0 = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 0));
        ws1 = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 1));
        st0 = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 0));
        st1 = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 1));
        pd0 = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 0));
        pd1 = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 1));
        z_r = PyArray_DIMS(%(z)s)[2];
        z_c = PyArray_DIMS(%(z)s)[3];
        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        r += pd0 * 2;
        c += pd1 * 2;
        // allocating memory for gx
        if ((!%(gx)s)
          || !PyArray_ISCONTIGUOUS(%(gx)s)
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
                  r_st = i * st0;
                  r_end = r_st + ws0;
                  // skip the padding
                  r_st = r_st < pd0 ? pd0 : r_st;
                  r_end = r_end > (r - pd0) ? r - pd0 : r_end;
                  // from padded_img space to img space
                  r_st -= pd0;
                  r_end -= pd0;
                  for(int j=0; j<z_c; j++){
                    c_st = j * st1;
                    c_end = c_st + ws1;
                    // skip the padding
                    c_st = c_st < pd1 ? pd1 : c_st;
                    c_end = c_end > (c - pd1) ? c - pd1 : c_end;
                    // change coordinates from padding_img space into img space
                    c_st -= pd1;
                    c_end -= pd1;
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
        return (0, 9, self.openmp)


class AveragePoolGrad(PoolGrad):
    def __init__(self, ignore_border, mode='average_inc_pad'):
        assert mode in ['sum', 'average_inc_pad', 'average_exc_pad']
        PoolGrad.__init__(self, ignore_border, mode)

    # There is an extra dummy parameter to match the parameter count
    # of MaxPoolGrad.  They have to keep the same interface because of
    # the DownsampleFactorMaxGrad trick to keep old scripts working
    # (see downsample.py for details on this).
    def make_node(self, x, gz, ws, stride=None, pad=(0, 0), dummy=None):
        # make_node should only be called by the grad function of
        # Pool, so these asserts should not fail.
        x = tensor.as_tensor_variable(x)
        gz = tensor.as_tensor_variable(gz)
        if stride is None:
            stride = ws
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4
        assert isinstance(ws, Variable) and ws.ndim == 1
        assert isinstance(stride, Variable) and stride.ndim == 1
        assert isinstance(pad, Variable) and pad.ndim == 1
        if not ws.dtype.startswith('int'):
            raise TypeError('Pool downsample parameters must be ints.')
        if not stride.dtype.startswith('int'):
            raise TypeError('Stride parameters must be ints.')
        if not pad.dtype.startswith('int'):
            raise TypeError('Padding parameters must be ints.')
        return Apply(self, [x, gz, ws, stride, pad], [x.type()])

    def perform(self, node, inp, out):
        x, gz, ws, stride, pad = inp
        gx_stg, = out
        assert ws.shape == stride.shape == pad.shape == (2,)
        if self.mode == 'average_exc_pad' and pad[0] != 0 and pad[1] != 0:
            raise NotImplementedError()
        z_shape = self.out_shape(x.shape, ws, self.ignore_border, stride, pad)
        if (gx_stg[0] is None) or (gx_stg[0].shape != z_shape):
            gx_stg[0] = numpy.empty(z_shape, dtype=x.dtype)
        zz = gx_stg[0]
        # number of pooling output rows
        pr = zz.shape[-2]
        # number of pooling output cols
        pc = zz.shape[-1]
        ws0, ws1 = ws
        st0, st1 = stride
        pad_h = pad[0]
        pad_w = pad[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w
        inc_pad = self.mode == 'average_inc_pad'
        sum_mode = self.mode == 'sum'

        # pad the image
        if (pad_h, pad_w) != (0, 0):
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
                        row_st = builtins.max(r * st0, pad_h)
                    row_end = builtins.min(row_st + ws0, img_rows)
                    for c in xrange(pc):
                        if sum_mode or inc_pad:
                            col_st = c * st1
                        else:
                            col_st = builtins.max(c * st1, pad_w)
                        col_end = builtins.min(col_st + ws1, img_cols)
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
        x, gz, ws, stride, pad = inp
        ggx, = grads
        return ([theano.tensor.zeros_like(x),
                 Pool(ignore_border=self.ignore_border, mode=self.mode)(ggx,
                ws, stride, pad)] + [DisconnectedType()() for i in inp[2:]])

    def connection_pattern(self, node):
        return [[1], [1], [0], [0], [0]]


class DownsampleFactorMaxGradGrad(OpenMPOp):
    __props__ = ('ignore_border', 'mode')

    def __init__(self, ignore_border, mode='max', openmp=None):
        self.ignore_border = ignore_border
        self.mode = mode
        super(DownsampleFactorMaxGradGrad, self).__init__(openmp=openmp)
        assert self.mode == 'max'

    def make_node(self, x, maxout, gz, ws, stride=None, pad=(0, 0)):
        # make_node should only be called by the grad function of
        # MaxPoolGrad, so these asserts should not fail.
        x = tensor.as_tensor_variable(x)
        maxout = tensor.as_tensor_variable(maxout)
        gz = tensor.as_tensor_variable(gz)
        assert x.ndim == 4
        assert maxout.ndim == 4
        assert gz.ndim == 4
        if stride is None:
            stride = ws
        if isinstance(pad, (tuple, list)):
            if tuple(pad) != (0, 0) and not self.ignore_border:
                raise NotImplementedError(
                    'padding works only with ignore_border=True')
            if isinstance(ws, (tuple, list)):
                if pad[0] >= ws[0] or pad[1] >= ws[1]:
                    raise NotImplementedError(
                        'padding_h and padding_w must be smaller than strides')
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert ws.ndim == 1
        assert stride.ndim == 1
        assert pad.ndim == 1
        if not ws.dtype.startswith('int'):
            raise TypeError('Pool downsample parameters must be ints.')
        if not stride.dtype.startswith('int'):
            raise TypeError('Stride parameters must be ints.')
        if not pad.dtype.startswith('int'):
            raise TypeError('Padding parameters must be ints.')
        return Apply(self, [x, maxout, gz, ws, stride, pad], [x.type()])

    def perform(self, node, inp, out):
        x, maxout, ggx, ws, stride, pad = inp
        z, = out
        assert ws.shape == stride.shape == pad.shape == (2,)
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
        ws0, ws1 = ws
        st0, st1 = stride
        pd0, pd1 = pad
        img_rows = x.shape[-2] + 2 * pd0
        img_cols = x.shape[-1] + 2 * pd1

        # pad the image and its gradients
        if pd0 != 0 and pd1 != 0:
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
                    row_end = builtins.min(row_st + ws0, img_rows)
                    for c in xrange(pc):
                        col_st = c * st1
                        col_end = builtins.min(col_st + ws1, img_cols)
                        for row_ind in xrange(row_st, row_end):
                            for col_ind in xrange(col_st, col_end):
                                if (maxout[n, k, r, c] == y_padded[n, k, row_ind, col_ind]):
                                    ggz[n, k, r, c] = ggx_padded[n, k, row_ind, col_ind]

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

    def grad(self, inp, grads):
        x, maxout, ggx, ws, stride, pad = inp
        gz, = grads
        return [theano.tensor.zeros_like(x),
                theano.tensor.zeros_like(maxout),
                MaxPoolGrad(ignore_border=self.ignore_border)(x, maxout, gz,
                                                              ws, stride, pad),
                DisconnectedType()(),
                DisconnectedType()(),
                DisconnectedType()()]

    def connection_pattern(self, node):
        return [[1], [1], [1], [0], [0], [0]]

    def c_code(self, node, name, inp, out, sub):
        if self.mode != 'max':
            raise theano.gof.utils.MethodNotDefined()
        x, maxout, ggx, ws, stride, pad = inp
        z, = out  # the grad of grad
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        if self.openmp:
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, c_st, c_end, maximum) schedule(static)'
        else:
            omp_parallel = ''
        return """
        int ws0, ws1, st0, st1, pd0, pd1;
        int z_typenum = PyArray_ObjectType((PyObject*)%(maxout)s, 0);
        int z_r, z_c;
        int r, c; // shape of the padded_input
        if(PyArray_DIM(%(ws)s, 0)!=2)
        {
            PyErr_SetString(PyExc_ValueError, "ws must be a vector of size 2");
            %(fail)s;
        }
        if(PyArray_DIM(%(stride)s, 0)!=2)
        {
            PyErr_SetString(PyExc_ValueError, "stride must be a vector of size 2");
            %(fail)s;
        }
        if(PyArray_DIM(%(pad)s, 0)!=2)
        {
            PyErr_SetString(PyExc_ValueError, "pad must be a vector of size 2");
            %(fail)s;
        }
        // Getting ws, stride and pad
        ws0 = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 0));
        ws1 = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 1));
        st0 = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 0));
        st1 = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 1));
        pd0 = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 0));
        pd1 = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 1));
        z_r = PyArray_DIMS(%(maxout)s)[2];
        z_c = PyArray_DIMS(%(maxout)s)[3];
        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        r += pd0 * 2;
        c += pd1 * 2;
        // allocating memory for output
        if ((!%(z)s)
          || !PyArray_ISCONTIGUOUS(%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
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
                  r_st = i * st0;
                  r_end = r_st + ws0;
                  // skip the padding
                  r_st = r_st < pd0 ? pd0 : r_st;
                  r_end = r_end > (r - pd0) ? r - pd0 : r_end;
                  // from padded_img space to img space
                  r_st -= pd0;
                  r_end -= pd0;
                  for(int j=0; j<z_c; j++){
                    c_st = j * st1;
                    c_end = c_st + ws1;
                    // skip the padding
                    c_st = c_st < pd1 ? pd1 : c_st;
                    c_end = c_end > (c - pd1) ? c - pd1 : c_end;
                    // from padding_img space into img space
                    c_st -= pd1;
                    c_end -= pd1;
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
        return (0, 3, self.openmp)
