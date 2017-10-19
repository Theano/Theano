
"""
Ops for downsampling images.
Planned:
Pool, DownsampleAvg, DownsampleSoftmax.
"""
from __future__ import absolute_import, print_function, division
# This file should move along with conv.py
import warnings
import itertools

import numpy as np
from six.moves import xrange
import six.moves.builtins as builtins
import theano
from theano import gof, OpenMPOp, tensor, Variable, Apply
from theano.gof import ParamsType, EnumList
from theano.gradient import DisconnectedType
from theano.scalar import bool as bool_t


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


def pool_2d(input, ws=None, ignore_border=None, stride=None, pad=(0, 0),
            mode='max', ds=None, st=None, padding=None):
    """Downscale the input by a specified factor

    Takes as input a N-D tensor, where N >= 2. It downscales the input image by
    the specified factor, by keeping only the maximum value of non-overlapping
    patches of size (ws[0],ws[1])

    Parameters
    ----------
    input : N-D theano tensor of input images
        Input images. Max pooling will be done over the 2 last dimensions.
    ws : tuple of length 2 or theano vector of ints of size 2.
        Factor by which to downscale (vertical ws, horizontal ws).
        (2,2) will halve the image in each dimension.
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5) input with ws=(2,2) will generate a (2,2) output.
        (3,3) otherwise.
    stride : tuple of two ints or theano vector of ints of size 2.
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If stride is None, it is considered equal to ws
        (no overlap on pooling regions), eg: stride=(1,1) will shifts over
        one row and one col for every iteration.
    pad : tuple of two ints or theano vector of ints of size 2.
        (pad_h, pad_w), pad zeros to extend beyond four borders of the
        images, pad_h is the size of the top and bottom margins, and
        pad_w is the size of the left and right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        Operation executed on each window. `max` and `sum` always exclude
        the padding in the computation. `average` gives you the choice to
        include or exclude it.
    ds
        *deprecated*, use parameter ws instead.
    st
        *deprecated*, use parameter stride instead.
    padding
        *deprecated*, use parameter pad instead.

    """
    # check for deprecated parameter names
    if ds is not None:
        if ws is not None:
            raise ValueError(
                "You can't provide a tuple value to both 'ws' and 'ds'."
                " Please provide a value only to 'ws'."
            )
        else:
            warnings.warn(
                "DEPRECATION: the 'ds' parameter is not going to exist"
                " anymore as it is going to be replaced by the parameter"
                " 'ws'.",
                stacklevel=2
            )
            ws = ds
    elif ds is None and ws is None:
        raise ValueError(
            "You must provide a tuple value for the window size."
        )

    if st is not None:
        if stride is not None:
            raise ValueError(
                "You can't provide a tuple value to both 'st and 'stride'."
                " Please provide a value only to 'stride'."
            )
        else:
            warnings.warn(
                "DEPRECATION: the 'st' parameter is not going to exist"
                " anymore as it is going to be replaced by the parameter"
                " 'stride'.",
                stacklevel=2
            )
            stride = st

    if padding is not None:
        if pad not in {None, (0, 0)}:
            raise ValueError(
                "You can't provide a tuple value to both 'padding' and pad."
                "  Please provide a value only to pad."
            )
        else:
            warnings.warn(
                "DEPRECATION: the 'padding' parameter is not going to exist"
                " anymore as it is going to be replaced by the parameter"
                " 'pad'.",
                stacklevel=2
            )
            pad = padding

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
            " `ws == stride and pad == (0, 0) and mode == 'max'`."
            " Otherwise, the convolution will be executed on CPU.",
            stacklevel=2)
        ignore_border = False
    op = Pool(ignore_border, ndim=2, mode=mode)
    output = op(input, ws, stride, pad)
    return output


def pool_3d(input, ws=None, ignore_border=None, stride=None, pad=(0, 0, 0),
            mode='max', ds=None, st=None, padding=None):
    """Downscale the input by a specified factor

    Takes as input a N-D tensor, where N >= 3. It downscales the input image by
    the specified factor, by keeping only the maximum value of non-overlapping
    patches of size (ws[0],ws[1],ws[2])

    Parameters
    ----------
    input : N-D theano tensor of input images
        Input images. Max pooling will be done over the 3 last dimensions.
    ws : tuple of length 3 or theano vector of ints of size 3
        Factor by which to downscale (vertical ws, horizontal ws, depth ws).
        (2,2,2) will halve the image in each dimension.
    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5,5) input with ws=(2,2,2) will generate a (2,2,2) output.
        (3,3,3) otherwise.
    st : tuple of three ints or theano vector of ints of size 3
        Stride size, which is the number of shifts over rows/cols/slices to get
        the next pool region. If st is None, it is considered equal to ws
        (no overlap on pooling regions).
    pad : tuple of two ints or theano vector of ints of size 3
        (pad_h, pad_w, pad_d), pad zeros to extend beyond six borders of the
        images, pad_h is the size of the top and bottom margins,
        pad_w is the size of the left and right margins, and pad_d is the size
        of the front and back margins
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        Operation executed on each window. `max` and `sum` always exclude
        the padding in the computation. `average` gives you the choice to
        include or exclude it.
    ds
        *deprecated*, use parameter ws instead.
    st
        *deprecated*, use parameter st instead.
    padding
        *deprecated*, use parameter pad instead.

    """
    # check for deprecated parameter names
    if ds is not None:
        if ws is not None:
            raise ValueError(
                "You can't provide a tuple value to both 'ws' and 'ds'."
                " Please provide a value only to 'ws'."
            )
        else:
            warnings.warn(
                "DEPRECATION: the 'ds' parameter is not going to exist"
                " anymore as it is going to be replaced by the parameter"
                " 'ws'.",
                stacklevel=2
            )
            ws = ds
    elif ds is None and ws is None:
        raise ValueError(
            "You must provide a tuple value for the window size."
        )

    if st is not None:
        if stride is not None:
            raise ValueError(
                "You can't provide a tuple value to both 'st and 'stride'."
                " Please provide a value only to 'stride'."
            )
        else:
            warnings.warn(
                "DEPRECATION: the 'st' parameter is not going to exist"
                " anymore as it is going to be replaced by the parameter"
                " 'stride'.",
                stacklevel=2
            )
            stride = st

    if padding is not None:
        if pad not in {None, (0, 0, 0)}:
            raise ValueError(
                "You can't provide a tuple value to both 'padding' and pad."
                "  Please provide a value only to pad."
            )
        else:
            warnings.warn(
                "DEPRECATION: the 'padding' parameter is not going to exist"
                " anymore as it is going to be replaced by the parameter"
                " 'pad'.",
                stacklevel=2
            )
            pad = padding

    if input.ndim < 3:
        raise NotImplementedError('pool_3d requires a dimension >= 3')
    if ignore_border is None:
        warnings.warn(
            "pool_3d() will have the parameter ignore_border"
            " default value changed to True (currently"
            " False). To have consistent behavior with all Theano"
            " version, explicitly add the parameter ignore_border=True."
            " On the GPU, using ignore_border=True is needed to use cuDNN."
            " When using ignore_border=False and not using cuDNN, the only"
            " GPU combination supported is when"
            " `ws == stride and pad == (0, 0, 0) and mode == 'max'`."
            " Otherwise, the convolution will be executed on CPU.",
            stacklevel=2)
        ignore_border = False
    op = Pool(ignore_border, ndim=3, mode=mode)
    output = op(input, ws, stride, pad)
    return output


# NB: This enum type is currently used in gpuarray/pool.py.
# It may be used later as op param in this current file.
# Enum name and constants names are inspired from cuDNN type `cudnnPoolingMode_t`
# (cf. `theano/gpuarray/cudnn_defs.py`).
PoolingMode_t = EnumList(('POOLING_MAX', 'max'),
                         ('POOLING_SUM', 'sum'),
                         ('POOLING_AVERAGE_COUNT_INCLUDE_PADDING', 'average_inc_pad'),
                         ('POOLING_AVERAGE_COUNT_EXCLUDE_PADDING', 'average_exc_pad'))


class Pool(OpenMPOp):
    """
    sum or average over different patches.

    Parameters
    ----------
    ws : list or tuple of N ints
        Downsample factor over rows, columns etc.
        ws indicates the size of the pooling region.
    ignore_border : bool
        If ws doesn't divide imgshape, do we include an extra row/col/slice
        of partial downsampling (False) or ignore it (True).
    stride : list or tuple of N ints or None
        Stride size, which is the number of shifts over rows/cols/slices to get the
        next pool region. If stride is None, it is considered equal to ws
        (no overlap on pooling regions).
    pad : tuple of N ints or None
        For each downsampling dimension, this specifies the number of zeros to
        add as padding on both sides. For 2D and (pad_h, pad_w), pad_h specifies the
        size of the top and bottom margins, pad_w specifies the size of the left and
        right margins. No padding is added if pad is None.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        ('average_inc_pad' excludes the padding from the count,
        'average_exc_pad' include it)
    ndim : int
        The number of pooling dimensions N.
        The default is 2.
    ds
        *deprecated*, use parameter ws instead.
    st
        *deprecated*, use parameter st instead.
    padding
        *deprecated*, use parameter pad instead.


    """

    __props__ = ('ignore_border', 'mode', 'ndim')
    params_type = ParamsType(ignore_border=bool_t,)

    @staticmethod
    def out_shape(imgshape, ws=None, ignore_border=False, stride=None, pad=None,
                  ndim=2, ds=None, st=None, padding=None):
        """
        Return the shape of the output from this op, for input of given
        shape and flags.

        Parameters
        ----------
        imgshape : tuple, list, or similar of integer or scalar Theano variable
            The shape of a tensor of images. The last N elements are
            interpreted as the number of rows, and the number of cols.
        ws : list or tuple of N ints
            Downsample factor over rows and column.
            ws indicates the pool region size.
        ignore_border : bool
            If ws doesn't divide imgshape, do we include an extra row/col/slice
            of partial downsampling (False) or ignore it (True).
        stride : list or tuple of N ints or None
            Stride size, which is the number of shifts over rows/cols/slices to get the
            next pool region. If stride is None, it is considered equal to ws
            (no overlap on pooling regions).
        pad : tuple of N ints or None
            For each downsampling dimension, this specifies the number of zeros to
            add as padding on both sides. For 2D and (pad_h, pad_w), pad_h specifies the
            size of the top and bottom margins, pad_w specifies the size of the left and
            right margins. No padding is added if pad is None.
        ndim : int
            The number of pooling dimensions N.
            The default is 2.
        ds
            *deprecated*, use parameter ws instead.
        st
            *deprecated*, use parameter st instead.
        padding
            *deprecated*, use parameter pad instead.

        Returns
        -------
        list
            The shape of the output from this op, for input of given shape.
            This will have the same length as imgshape, but with last N
            elements reduced as per the downsampling & ignore_border flags.

        """
        # check for deprecated parameter names
        if ds is not None:
            if ws is not None:
                raise ValueError(
                    "You can't provide a tuple value to both 'ws' and 'ds'."
                    " Please provide a value only to 'ws'."
                )
            else:
                warnings.warn(
                    "DEPRECATION: the 'ds' parameter is not going to exist"
                    " anymore as it is going to be replaced by the parameter"
                    " 'ws'.",
                    stacklevel=2
                )
                ws = ds
        elif ds is None and ws is None:
            raise ValueError(
                "You must provide a tuple value for the window size."
            )

        if st is not None:
            if stride is not None:
                raise ValueError(
                    "You can't provide a tuple value to both 'st and 'stride'."
                    " Please provide a value only to 'stride'."
                )
            else:
                warnings.warn(
                    "DEPRECATION: the 'st' parameter is not going to exist"
                    " anymore as it is going to be replaced by the parameter"
                    " 'stride'.",
                    stacklevel=2
                )
                stride = st

        if padding is not None:
            zero_pad = (0,) * ndim
            if pad not in {None, zero_pad}:
                raise ValueError(
                    "You can't provide a tuple value to both 'padding' and pad."
                    "  Please provide a value only to pad."
                )
            else:
                warnings.warn(
                    "DEPRECATION: the 'padding' parameter is not going to"
                    " exist anymore as it is going to be replaced by the"
                    " parameter 'pad'.",
                    stacklevel=2
                )
                pad = padding

        if ndim is None:
            ndim = 2
        assert ndim > 0
        if len(imgshape) < ndim:
            raise TypeError('imgshape must have at least {} dimensions'.format(ndim))

        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * ndim
        patch_shape = tuple(tensor.extract_constant(imgshape[-ndim + i]) + pad[i] * 2
                            for i in xrange(ndim))

        def compute_out(v, downsample, stride):
            if ignore_border:
                if downsample == stride:
                    return v // stride
                else:
                    out = (v - downsample) // stride + 1
                    if isinstance(out, theano.Variable):
                        return tensor.maximum(out, 0)
                    else:
                        return np.maximum(out, 0)
            else:
                if isinstance(v, theano.Variable):
                    return tensor.switch(tensor.ge(stride, downsample),
                                         (v - 1) // stride + 1,
                                         tensor.maximum(0, (v - 1 - downsample) //
                                                        stride + 1) + 1)
                elif stride >= downsample:
                    return (v - 1) // stride + 1
                else:
                    return max(0, (v - 1 - downsample + stride) // stride) + 1

        out_shape = [compute_out(patch_shape[i], ws[i], stride[i]) for i in xrange(ndim)]

        rval = list(imgshape[:-ndim]) + out_shape
        return rval

    def __init__(self, ignore_border=False, mode='max', ndim=2, openmp=None):
        super(Pool, self).__init__(openmp=openmp)
        self.ndim = ndim
        self.ignore_border = ignore_border
        if mode == 'max_deterministic':
            # It seems max pool algo is already deterministic in CPU.
            mode = 'max'
        if mode not in ['max', 'average_inc_pad', 'average_exc_pad', 'sum']:
            raise ValueError(
                "Pool mode parameter only support 'max', 'sum',"
                " 'average_inc_pad' and 'average_exc_pad'. Got %s" % mode)
        self.mode = mode

    def prepare_node(self, node, storage_map, compute_map, impl):
        if len(node.inputs) == 1:
            # Old interface
            self.ndim = len(node.op.ds)
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

    def make_node(self, x, ws, stride=None, pad=None):
        # TODO: consider restricting the dtype?
        x = tensor.as_tensor_variable(x)
        nd = self.ndim
        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        elif isinstance(pad, (tuple, list)):
            if max(pad) != 0 and not self.ignore_border:
                raise NotImplementedError(
                    'padding works only with ignore_border=True')
            if isinstance(ws, (tuple, list)):
                if any(pad[i] >= ws[i] for i in range(nd)):
                    raise NotImplementedError(
                        'padding must be smaller than strides')
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert ws.ndim == 1
        assert stride.ndim == 1
        assert pad.ndim == 1
        if x.type.ndim < nd:
            raise TypeError()
        if ws.dtype not in tensor.int_dtypes:
            raise TypeError('Pool downsample parameters must be ints.')
        if stride.dtype not in tensor.int_dtypes:
            raise TypeError('Stride parameters must be ints.')
        if pad.dtype not in tensor.int_dtypes:
            raise TypeError('Padding parameters must be ints.')
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:-nd] + (False,) * nd
        out = tensor.TensorType(x.dtype, broad)
        return gof.Apply(self, [x, ws, stride, pad], [out()])

    def perform(self, node, inp, out, params):
        x, ws, stride, pad = inp
        z, = out
        nd = self.ndim
        assert ws.shape == stride.shape == pad.shape == (nd,)
        if len(x.shape) < nd:
            raise NotImplementedError(
                'Pool requires input with {} or more dimensions'.format(nd))
        z_shape = self.out_shape(x.shape, ws, params.ignore_border, stride, pad, nd)
        if not params.ignore_border:
            assert all(z > 0 for z in z_shape[-nd:])
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = np.empty(z_shape, dtype=x.dtype)
        zz = z[0]
        # size of pooling output
        pool_out_shp = zz.shape[-nd:]
        img_shp = tuple(x.shape[-nd + i] + 2 * pad[i] for i in xrange(nd))
        inc_pad = self.mode == 'average_inc_pad'

        # pad the image
        if max(pad) != 0:
            y = np.zeros(x.shape[:-nd] + img_shp, dtype=x.dtype)
            y[(slice(None),) * (len(x.shape) - nd) +
              tuple(slice(pad[i], img_shp[i] - pad[i]) for i in xrange(nd))] = x
        else:
            y = x
        func = np.max
        if self.mode == 'sum':
            func = np.sum
        elif self.mode != 'max':
            func = np.average

        # precompute the region boundaries for each dimension
        region_slices = [[] for i in xrange(nd)]
        for i in xrange(nd):
            for j in xrange(pool_out_shp[i]):
                start = j * stride[i]
                end = builtins.min(start + ws[i], img_shp[i])
                if not inc_pad:
                    start = builtins.max(start, pad[i])
                    end = builtins.min(end, img_shp[i] - pad[i])
                region_slices[i].append(slice(start, end))

        # iterate over non-pooling dimensions
        for k in np.ndindex(*x.shape[:-nd]):
            zzk = zz[k]
            yk = y[k]
            # iterate over pooling regions
            for r in np.ndindex(*pool_out_shp):
                zzk[r] = func(
                    yk[[region_slices[i][r[i]] for i in xrange(nd)]])

    def infer_shape(self, node, in_shapes):
        ws, stride, pad = [node.inputs[1], node.inputs[2], node.inputs[3]]
        shp = self.out_shape(in_shapes[0], ws, self.ignore_border, stride,
                             pad, self.ndim)
        return [shp]

    def L_op(self, inputs, outputs, grads):
        x, ws, stride, pad = inputs
        gz, = grads
        disc = [DisconnectedType()() for i in inputs[1:]]
        if self.mode == 'max':
            return [MaxPoolGrad(ndim=self.ndim,
                                ignore_border=self.ignore_border)(
                x, outputs[0], gz, ws=ws, stride=stride, pad=pad)] + disc
        else:
            return [AveragePoolGrad(ndim=self.ndim,
                                    ignore_border=self.ignore_border,
                                    mode=self.mode)(
                x, gz, ws=ws, stride=stride, pad=pad)] + disc

    def connection_pattern(self, node):
        return [[1], [0], [0], [0]]

    def R_op(self, inputs, eval_points):
        if self.mode != 'max':
            # Rop for average or sum is simply pooling evaluated at eval point
            eval_inputs = [eval_points[0]] + inputs[1:]
            return [self(*eval_inputs)]

        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return [None]
        z = self(*inputs)
        x, ws, stride, pad = inputs
        return [
            DownsampleFactorMaxGradGrad(self.ignore_border, self.mode,
                                        self.ndim)(x, z, eval_points[0], ws,
                                                   stride, pad)
        ]

    def c_headers(self):
        headers = ['<algorithm>']
        headers += super(Pool, self).c_headers()
        return headers

    def c_code(self, node, name, inp, out, sub):
        if self.mode not in ('max', 'sum', 'average_exc_pad', 'average_inc_pad'):
            raise theano.gof.utils.MethodNotDefined()
        x, ws, stride, pad = inp
        z, = out
        nd = self.ndim
        total_ndim = node.inputs[0].ndim
        non_pool_ndim = total_ndim - nd
        fail = sub['fail']
        params = sub['params']
        if self.openmp:
            # run in parallel over each pooling block
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, r_idx, i_idx, o_idx, collector) schedule(static)'
        else:
            omp_parallel = ''
        ccode = """
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        if(PyArray_NDIM(%(x)s)!=%(total_ndim)s)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a %(total_ndim)sD ndarray");
            %(fail)s;
        }
        if(PyArray_DIM(%(ws)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "ws must be a vector of size %(nd)s");
            %(fail)s;
        }
        if(PyArray_DIM(%(stride)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "stride must be a vector of size %(nd)s");
            %(fail)s;
        }
        if(PyArray_DIM(%(pad)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "pad must be a vector of size %(nd)s");
            %(fail)s;
        }
        npy_intp z[%(nd)s]; // shape of the output
        npy_intp r[%(nd)s]; // shape of the padded_input
        npy_intp ws[%(nd)s];
        npy_intp st[%(nd)s];
        npy_intp pd[%(nd)s];
        int nonzero_padding;
        nonzero_padding = 0;
        for (int i=0; i<%(nd)s; i++)
        {
            ws[i] = *((dtype_%(ws)s*)PyArray_GETPTR1(%(ws)s, i));
            st[i] = *((dtype_%(stride)s*)PyArray_GETPTR1(%(stride)s, i));
            pd[i] = *((dtype_%(pad)s*)PyArray_GETPTR1(%(pad)s, i));
            r[i] = PyArray_DIMS(%(x)s)[%(non_pool_ndim)s + i] + 2 * pd[i];
            if (pd[i]>0)
                nonzero_padding = 1;
        }
        if (!%(params)s->ignore_border && nonzero_padding)
        {
            PyErr_SetString(PyExc_ValueError,
              "padding must be zero when ignore border is False");
            %(fail)s;
        }
        if (%(params)s->ignore_border)
        {
            for (int i=0; i<%(nd)s; i++)
            {
                // '/' in C is different from '/' in python
                if (r[i] - ws[i] < 0)
                {
                  z[i] = 0;
                }
                else
                {
                  z[i] = (r[i] - ws[i]) / st[i] + 1;
                }
            }
        }
        else
        {
            for (int i=0; i<%(nd)s; i++)
            {
                // decide how many rows/cols the output has
                if (st[i] >= ws[i])
                {
                    z[i] = (r[i] - 1) / st[i] + 1;
                }
                else
                {
                    z[i] = std::max((npy_intp)0, (r[i] - 1 - ws[i] + st[i]) / st[i]) + 1;
                }
                assert(z[i] > 0);
            }
        }
        // memory allocation of z if necessary
        int mem_nec;
        mem_nec = 0;
        if ((!%(z)s) || *PyArray_DIMS(%(z)s)!=%(total_ndim)s)
        {
            mem_nec = 1;
        }
        if (!mem_nec)
        {
            for (int i=0; i<%(non_pool_ndim)s; i++)
            {
                if (PyArray_DIMS(%(z)s)[i] != PyArray_DIMS(%(x)s)[i])
                {
                    mem_nec = 1;
                    break;
                }
            }
        }
        if (!mem_nec)
        {
            for (int i=0; i<%(nd)s; i++)
            {
                if (PyArray_DIMS(%(z)s)[%(non_pool_ndim)s + i] != z[i])
                {
                    mem_nec = 1;
                    break;
                }
            }
        }
        if (mem_nec)
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[%(total_ndim)s];
          for (int i=0; i<%(non_pool_ndim)s; i++)
          {
              dims[i] = PyArray_DIMS(%(x)s)[i];
          }
          for (int i=0; i<%(nd)s; i++)
          {
              dims[%(non_pool_ndim)s + i] = z[i];
          }
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(%(total_ndim)s, dims, typenum,0);
        }
        // initialize temp var for the value in a region
        dtype_%(x)s collector;
        npy_intp z_prod;
        // do not run if any z[i] is zero
        z_prod = 1;
        for (int i=0; i<%(nd)s; i++)
        {
            z_prod *= z[i];
        }
        if (z_prod)
        {
            // will be used to hold start and end index of a region
            npy_intp r_st[%(nd)s];
            npy_intp r_end[%(nd)s];
            // index for iterating over the pooling regions
            npy_intp r_idx[%(nd)s];
            // placeholder for PyArray indexing (output)
            npy_intp o_idx[%(total_ndim)s];
            // placeholder for PyArray indexing (input)
            npy_intp i_idx[%(total_ndim)s];
            // loop over non-pooling dimensions
            npy_intp non_pooling_prod = 1;
            for (int i=0; i<%(non_pool_ndim)s; i++)
            {
                non_pooling_prod *= PyArray_DIMS(%(x)s)[i];
            }
            %(omp_parallel)s
            // first loop over non-pooling dimensions
            for (npy_intp t=0; t<non_pooling_prod; t++)
            {
                // compute the non-pooling index in each dimension
                if (%(non_pool_ndim)s!=0)
                {
                    o_idx[0] = t;
                    i_idx[0] = t;
                    for (int i=1; i<%(non_pool_ndim)s; i++)
                    {
                        o_idx[i] = o_idx[i - 1] / PyArray_DIMS(%(x)s)[i - 1];
                        o_idx[i - 1] = o_idx[i - 1] %% PyArray_DIMS(%(x)s)[i - 1];
                        i_idx[i] = o_idx[i];
                        i_idx[i - 1] = o_idx[i - 1];
                    }
                }

                // then loop over each region in each pooling dimension
        """

        for i in xrange(nd):
            ccode += """
                for (r_idx[%(i)s]=0; r_idx[%(i)s] < z[%(i)s]; r_idx[%(i)s]++) {
                  r_st[%(i)s] = r_idx[%(i)s] * st[%(i)s];
                  r_end[%(i)s] = r_st[%(i)s] + ws[%(i)s];
                  // skip the padding
                  r_st[%(i)s] = r_st[%(i)s] < pd[%(i)s] ? pd[%(i)s] : r_st[%(i)s];
                  r_end[%(i)s] = r_end[%(i)s] > (r[%(i)s] - pd[%(i)s]) ? r[%(i)s] - pd[%(i)s] : r_end[%(i)s];
                  // from padded_img space to img space
                  r_st[%(i)s] -= pd[%(i)s];
                  r_end[%(i)s] -= pd[%(i)s];
                  // handle the case where no padding, ignore border is True
                  if (%(params)s->ignore_border)
                  {
                    r_end[%(i)s] = r_end[%(i)s] > r[%(i)s] ? r[%(i)s] : r_end[%(i)s];
                  }
                  // use the index to find the correct position in the output
                  o_idx[%(non_pool_ndim)s + %(i)s] = r_idx[%(i)s];
            """ % dict(i=i, non_pool_ndim=non_pool_ndim, params=sub['params'])

        ccode += """
                  // get a pointer to the correct position in the output
                  dtype_%(z)s * z;
                  if (%(total_ndim)s == 4)
                    z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, o_idx[0], o_idx[1], o_idx[2], o_idx[3])));
                  else
                    z = ((dtype_%(z)s*)(PyArray_GetPtr(%(z)s, o_idx)));
        """

        if self.mode == 'max':
            for i in xrange(nd):
                ccode += """
                  // set the first index of dimension %(i)s
                  i_idx[%(non_pool_ndim)s + %(i)s] = r_st[%(i)s];
                """ % dict(i=i, non_pool_ndim=non_pool_ndim)
            ccode += """
                  // use the first element as the initial value of collector
                  if (%(total_ndim)s == 4)
                    collector = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,i_idx[0],i_idx[1],i_idx[2],i_idx[3])))[0];
                  else
                    collector = ((dtype_%(x)s*)(PyArray_GetPtr(%(x)s,i_idx)))[0];
            """
            for i in xrange(nd):
                ccode += """
                  // go through the pooled region in the unpadded input
                  for(npy_intp m%(i)s=r_st[%(i)s]; m%(i)s<r_end[%(i)s]; m%(i)s++)
                  {
                    i_idx[%(non_pool_ndim)s + %(i)s] = m%(i)s;
                """ % dict(i=i, non_pool_ndim=non_pool_ndim)
            ccode += """
                    // update maximum
                    dtype_%(x)s a;
                    if (%(total_ndim)s == 4)
                      a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,i_idx[0],i_idx[1],i_idx[2],i_idx[3])))[0];
                    else
                      a = ((dtype_%(x)s*)(PyArray_GetPtr(%(x)s,i_idx)))[0];
                    collector = (a > collector) ? a : collector;
            """
            for i in xrange(nd):
                ccode += """
                  } // for loop over region
                """
            ccode += """
                  z[0] = collector;
            """
        elif self.mode in ('sum', 'average_exc_pad', 'average_inc_pad'):
            ccode += """
                  // initialize the sum at zero
                  collector = ((dtype_%(x)s)(0));
            """
            for i in xrange(nd):
                ccode += """
                  // go through the pooled region in the unpadded input
                  for(npy_intp m%(i)s=r_st[%(i)s]; m%(i)s<r_end[%(i)s]; m%(i)s++)
                  {
                    i_idx[%(non_pool_ndim)s + %(i)s] = m%(i)s;
                """ % dict(i=i, non_pool_ndim=non_pool_ndim)
            ccode += """
                    // update sum
                    dtype_%(x)s a;
                    if (%(total_ndim)s == 4)
                      a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,i_idx[0],i_idx[1],i_idx[2],i_idx[3])))[0];
                    else
                      a = ((dtype_%(x)s*)(PyArray_GetPtr(%(x)s,i_idx)))[0];
                    collector += a;
            """
            for i in xrange(nd):
                ccode += """
                  } // for loop over region
                """
            if self.mode == "sum":
                ccode += """
                  z[0] = collector;
                """
            elif self.mode == 'average_inc_pad' and self.ignore_border:
                # region size = product over all pooling dimensions
                region_size = ' * '.join('ws[%d]' % i for i in xrange(nd))
                ccode += """
                  z[0] = collector / (%(region_size)s);
                """ % dict(region_size=region_size)
            else:
                # region size = number elements of in this region
                region_size = ' * '.join('(r_end[%d]-r_st[%d])' % (i, i) for i in xrange(nd))
                ccode += """
                  z[0] = collector / (%(region_size)s);
                """ % dict(region_size=region_size)
        for i in xrange(nd):
            ccode += """
            } // loop over pooling dimension
            """

        ccode += """
          } // for loop over non-pooling dimensions
        } // if z_prod
        """
        return ccode % locals()

    def c_code_cache_version(self):
        return (10, self.openmp)


class PoolGrad(OpenMPOp):
    __props__ = ('ignore_border', 'mode', 'ndim')

    @staticmethod
    def out_shape(imgshape, ws=None, ignore_border=False, stride=None, pad=None, ndim=2,
                  ds=None, st=None, padding=None):
        """Return the shape of the output from this op, for input of given
        shape and flags.

        Parameters
        ----------
        imgshape : tuple of integers or scalar Theano variables
            the shape of a tensor of images. The last N elements are
            interpreted as the downsampling dimensions.
        ws : tuple of N ints
            downsample factor over rows and columns this parameter
            indicates the size of the pooling region
        ignore_border : bool
            If ws doesn't divide imgshape, do we include an extra row/col/slice
            of partial downsampling (False) or ignore it (True).
        stride : list or tuple of N ints or None
            Stride size, which is the number of shifts over rows/cols/slices to get the
            next pool region. If stride is None, it is considered equal to ws
            (no overlap on pooling regions).
        pad : tuple of N ints or None
            For each downsampling dimension, this specifies the number of zeros to
            add as padding on both sides. For 2D and (pad_h, pad_w), pad_h specifies the
            size of the top and bottom margins, pad_w specifies the size of the left and
            right margins. No padding is added if pad is None.
        ndim : int
            The number of pooling dimensions N.
            The default is 2.
        ds
            *deprecated*, use parameter ws instead.
        st
            *deprecated*, use parameter st instead.
        padding
            *deprecated*, use parameter pad instead.

        Returns
        -------
        list :
            the shape of the output from this op, for input of given
            shape.  This will have the same length as imgshape, but
            with last N elements reduced as per the downsampling &
            ignore_border flags.

        """
        # check for deprecated parameter names
        if ds is not None:
            if ws is not None:
                raise ValueError(
                    "You can't provide a tuple value to both 'ws' and 'ds'."
                    " Please provide a value only to 'ws'."
                )
            else:
                warnings.warn(
                    "DEPRECATION: the 'ds' parameter in PoolGrad is not going"
                    " to exist anymore as it is going to be replaced by the"
                    " parameter 'ws'.",
                    stacklevel=2
                )
                ws = ds
        elif ds is None and ws is None:
            raise ValueError(
                "You must provide a tuple value for the window size."
            )

        if st is not None:
            if stride is not None:
                raise ValueError(
                    "You can't provide a tuple value to both 'st and 'stride'."
                    " Please provide a value only to 'stride'."
                )
            else:
                warnings.warn(
                    "DEPRECATION: the 'st' parameter in PoolGrad is not going"
                    " to exist anymore as it is going to be replaced by the"
                    " parameter 'stride'.",
                    stacklevel=2
                )
                stride = st

        if padding is not None:
            if pad is not None:
                raise ValueError(
                    "You can't provide a tuple value to both 'padding' and pad."
                    "  Please provide a value only to pad."
                )
            else:
                warnings.warn(
                    "DEPRECATION: the 'padding' parameter in PoolGrad is not"
                    " going to exist anymore as it is going to be replaced"
                    " by the parameter 'pad'.",
                    stacklevel=2
                )
                pad = padding

        if len(imgshape) < ndim:
            raise TypeError('imgshape must have at least {} dimensions'.format(ndim))

        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * ndim
        patch_shape = tuple(tensor.extract_constant(imgshape[-ndim + i]) + pad[i] * 2
                            for i in xrange(ndim))

        def compute_out(v, downsample, stride):
            if ignore_border:
                out = (v - downsample) // stride + 1
                if isinstance(out, theano.Variable):
                    return tensor.maximum(out, 0)
                else:
                    return np.maximum(out, 0)
            else:
                if isinstance(v, theano.Variable):
                    return tensor.switch(tensor.ge(stride, downsample),
                                         (v - 1) // stride + 1,
                                         tensor.maximum(0, (v - 1 - downsample) //
                                                        stride + 1) + 1)
                elif stride >= downsample:
                    return (v - 1) // stride + 1
                else:
                    return max(0, (v - 1 - downsample) // stride + 1) + 1

        out_shape = [compute_out(patch_shape[i], ws[i], stride[i]) for i in xrange(ndim)]

        rval = list(imgshape[:-ndim]) + out_shape
        return rval

    def __init__(self, ignore_border, mode='max', ndim=2, openmp=None):
        self.ndim = ndim
        self.ignore_border = ignore_border
        if mode == 'max_deterministic':
            # It seems max pool grad algo is already deterministic in CPU.
            mode = 'max'
        if mode not in ['max', 'sum', 'average_inc_pad', 'average_exc_pad']:
            raise ValueError(
                "Pool mode parameter only support 'max', 'sum',"
                " 'average_inc_pad' and 'average_exc_pad'. Got %s" % mode)
        self.mode = mode
        super(PoolGrad, self).__init__(openmp=openmp)

    def prepare_node(self, node, storage_map, compute_map, impl):
        if len(node.inputs) < 5:  # 5 for AveragePoolGrad, 6 for MaxPoolGrad
            # Old interface
            self.ndim = len(node.op.ds)
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
    # params_type ignore_border don't change c code

    def __init__(self, ignore_border, ndim=2, openmp=None):
        PoolGrad.__init__(self, ignore_border, mode='max', ndim=ndim, openmp=openmp)

    def make_node(self, x, maxout, gz, ws, stride=None, pad=None):
        # make_node should only be called by the grad function of
        # Pool, so these asserts should not fail.
        x = tensor.as_tensor_variable(x)
        maxout = tensor.as_tensor_variable(maxout)
        gz = tensor.as_tensor_variable(gz)
        nd = self.ndim
        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert isinstance(x, Variable) and x.ndim >= nd
        assert isinstance(maxout, Variable) and maxout.ndim >= nd
        assert isinstance(gz, Variable) and gz.ndim >= nd
        assert isinstance(ws, Variable) and ws.ndim == 1
        assert isinstance(stride, Variable) and stride.ndim == 1
        assert isinstance(pad, Variable) and pad.ndim == 1
        assert x.ndim == maxout.ndim == gz.ndim >= nd
        if ws.dtype not in tensor.int_dtypes:
            raise TypeError('Pool downsample parameters must be ints.')
        if stride.dtype not in tensor.int_dtypes:
            raise TypeError('Stride parameters must be ints.')
        if pad.dtype not in tensor.int_dtypes:
            raise TypeError('Padding parameters must be ints.')
        return Apply(self, [x, maxout, gz, ws, stride, pad], [x.type()])

    def perform(self, node, inp, out):
        assert self.mode == 'max'
        x, maxout, gz, ws, stride, pad = inp
        gx_stg, = out
        nd = self.ndim
        assert ws.shape == stride.shape == pad.shape == (nd,)
        if len(x.shape) < nd:
            raise NotImplementedError(
                'MaxPoolGrad requires input with {} or more dimensions'.format(nd))
        pool_out_shp = maxout.shape[-nd:]
        img_shp = tuple(x.shape[-nd + i] + 2 * pad[i] for i in xrange(nd))

        # pad the image
        if max(pad) != 0:
            y = np.zeros(x.shape[:-nd] + img_shp, dtype=x.dtype)
            y[(slice(None),) * (len(x.shape) - nd) +
              tuple(slice(pad[i], img_shp[i] - pad[i]) for i in xrange(nd))] = x
        else:
            y = x
        gx = np.zeros_like(y)

        # precompute the region boundaries for each dimension
        region_ranges = [[] for i in xrange(nd)]
        for i in xrange(nd):
            for j in xrange(pool_out_shp[i]):
                start = builtins.max(j * stride[i], pad[i])
                end = builtins.min(start + ws[i], img_shp[i])
                region_ranges[i].append(xrange(start, end))

        # iterate over non-pooling dimensions
        for k in np.ndindex(*x.shape[:-nd]):
            gxk = gx[k]
            gzk = gz[k]
            yk = y[k]
            maxoutk = maxout[k]
            # iterate over pooling regions
            for r in np.ndindex(*pool_out_shp):
                maxout_value = maxoutk[r]
                # iterate inside region
                for c in itertools.product(*[region_ranges[i][r[i]]
                                             for i in xrange(nd)]):
                    if maxout_value == yk[c]:
                        gxk[c] += gzk[r]

        # unpad the image
        gx = gx[(slice(None),) * (len(x.shape) - nd) +
                tuple(slice(pad[i], img_shp[i] - pad[i]) for i in xrange(nd))]
        gx_stg[0] = gx

    def grad(self, inp, grads):
        x, maxout, gz, ws, stride, pad = inp
        ggx, = grads
        return ([theano.tensor.zeros_like(x),
                 theano.tensor.zeros_like(maxout),
                 DownsampleFactorMaxGradGrad(ndim=self.ndim,
                                             ignore_border=self.ignore_border)(
                x, maxout, ggx, ws, stride, pad)] +
                [DisconnectedType()() for i in inp[3:]])

    def connection_pattern(self, node):
        return [[1], [1], [1], [0], [0], [0]]

    def c_code(self, node, name, inp, out, sub):
        assert self.mode == 'max'
        x, z, gz, ws, stride, pad = inp
        gx, = out
        nd = self.ndim
        total_ndim = node.inputs[0].ndim
        non_pool_ndim = total_ndim - nd
        fail = sub['fail']

        if self.openmp:
            # run in parallel over each pooling block
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, r_idx, i_idx, o_idx, maximum) schedule(static)'
        else:
            omp_parallel = ''

        ccode = """
        // sanity checks
        int x_typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_typenum = PyArray_ObjectType((PyObject*)%(z)s, 0);
        int gz_typenum = PyArray_ObjectType((PyObject*)%(gz)s, 0);
        if ((x_typenum != z_typenum) || (x_typenum != gz_typenum))
        {
            PyErr_SetString(PyExc_ValueError, "input types must all match");
            %(fail)s;
        }
        if(PyArray_NDIM(%(x)s)!=%(total_ndim)s)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a %(total_ndim)sD ndarray");
            %(fail)s;
        }
        if(PyArray_NDIM(%(z)s)!=%(total_ndim)s)
        {
            PyErr_SetString(PyExc_ValueError, "z must be a %(total_ndim)sD ndarray");
            %(fail)s;
        }
        if(PyArray_NDIM(%(gz)s)!=%(total_ndim)s)
        {
            PyErr_SetString(PyExc_ValueError, "gz must be a %(total_ndim)sD ndarray");
            %(fail)s;
        }
        if(PyArray_DIM(%(ws)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "ws must be a vector of size %(nd)s");
            %(fail)s;
        }
        if(PyArray_DIM(%(stride)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "stride must be a vector of size %(nd)s");
            %(fail)s;
        }
        if(PyArray_DIM(%(pad)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "pad must be a vector of size %(nd)s");
            %(fail)s;
        }
        npy_intp z[%(nd)s]; // shape of the output
        npy_intp r[%(nd)s]; // shape of the padded_input
        npy_intp ws[%(nd)s];
        npy_intp st[%(nd)s];
        npy_intp pd[%(nd)s];
        int nonzero_padding;
        nonzero_padding = 0;
        for (int i=0; i<%(nd)s; i++)
        {
            ws[i] = *((dtype_%(ws)s*)PyArray_GETPTR1(%(ws)s, i));
            st[i] = *((dtype_%(stride)s*)PyArray_GETPTR1(%(stride)s, i));
            pd[i] = *((dtype_%(pad)s*)PyArray_GETPTR1(%(pad)s, i));
            z[i] = PyArray_DIMS(%(z)s)[%(non_pool_ndim)s + i];
            r[i] = PyArray_DIMS(%(x)s)[%(non_pool_ndim)s + i] + 2 * pd[i];
            if (pd[i]>0)
                nonzero_padding = 1;
        }
        // allocating memory for output, if necessary
        int mem_nec;
        mem_nec = 0;
        if ((!%(gx)s) || !PyArray_ISCONTIGUOUS(%(gx)s)
            || *PyArray_DIMS(%(gx)s)!=%(total_ndim)s)
        {
            mem_nec = 1;
        }
        if (!mem_nec)
        {
            for (int i=0; i<%(total_ndim)s; i++)
            {
                if (PyArray_DIMS(%(gx)s)[i] != PyArray_DIMS(%(x)s)[i])
                {
                    mem_nec = 1;
                    break;
                }
            }
        }
        if (mem_nec)
        {
          Py_XDECREF(%(gx)s);
          %(gx)s = (PyArrayObject*) PyArray_ZEROS(%(total_ndim)s, PyArray_DIMS(%(x)s), x_typenum,0);
        }
        else {
          PyArray_FILLWBYTE(%(gx)s, 0);
        }
        dtype_%(z)s maximum; // temp var for maximum value in a region
        npy_intp z_prod;
        // do not run if any z[i] is zero
        z_prod = 1;
        for (int i=0; i<%(nd)s; i++)
        {
            z_prod *= z[i];
        }
        if (z_prod)
        {
            // will be used to hold start and end index of a region
            npy_intp r_st[%(nd)s];
            npy_intp r_end[%(nd)s];
            // index for iterating over the pooling regions
            npy_intp r_idx[%(nd)s];
            // placeholder for PyArray indexing (output)
            npy_intp o_idx[%(total_ndim)s];
            // placeholder for PyArray indexing (input)
            npy_intp i_idx[%(total_ndim)s];
            // loop over non-pooling dimensions
            npy_intp non_pooling_prod = 1;
            for (int i=0; i<%(non_pool_ndim)s; i++)
            {
                non_pooling_prod *= PyArray_DIMS(%(x)s)[i];
            }
            %(omp_parallel)s
            // first loop over non-pooling dimensions
            for (npy_intp t=0; t<non_pooling_prod; t++)
            {
                // compute the non-pooling index in each dimension
                if (%(non_pool_ndim)s!=0)
                {
                    o_idx[0] = t;
                    i_idx[0] = t;
                    for (int i=1; i<%(non_pool_ndim)s; i++)
                    {
                        o_idx[i] = o_idx[i - 1] / PyArray_DIMS(%(x)s)[i - 1];
                        o_idx[i - 1] =o_idx[i - 1] %% PyArray_DIMS(%(x)s)[i - 1];
                        i_idx[i] = o_idx[i];
                        i_idx[i - 1] = o_idx[i - 1];
                    }
                }

                // then loop over each region in each pooling dimension
        """

        for i in xrange(nd):
            ccode += """
                for (r_idx[%(i)s]=0; r_idx[%(i)s] < z[%(i)s]; r_idx[%(i)s]++) {
                  r_st[%(i)s] = r_idx[%(i)s] * st[%(i)s];
                  r_end[%(i)s] = r_st[%(i)s] + ws[%(i)s];
                  // skip the padding
                  r_st[%(i)s] = r_st[%(i)s] < pd[%(i)s] ? pd[%(i)s] : r_st[%(i)s];
                  r_end[%(i)s] = r_end[%(i)s] > (r[%(i)s] - pd[%(i)s]) ? r[%(i)s] - pd[%(i)s] : r_end[%(i)s];
                  // from padded_img space to img space
                  r_st[%(i)s] -= pd[%(i)s];
                  r_end[%(i)s] -= pd[%(i)s];
                  // use the index to find the correct position in the output
                  o_idx[%(non_pool_ndim)s + %(i)s] = r_idx[%(i)s];
            """ % dict(i=i, non_pool_ndim=non_pool_ndim)

        ccode += """
                  dtype_%(gz)s * gz;
                  if (%(total_ndim)s == 4)
                  {
                    // the maximum value
                    maximum = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,o_idx[0],o_idx[1],o_idx[2],o_idx[3])))[0];
                    // the gradient corresponding to this maximum value in z
                    gz = ((dtype_%(gz)s*)(PyArray_GETPTR4(%(gz)s, o_idx[0],o_idx[1],o_idx[2],o_idx[3])));
                  }
                  else
                  {
                    // the maximum value
                    maximum = ((dtype_%(z)s*)(PyArray_GetPtr(%(z)s,o_idx)))[0];
                    // the gradient corresponding to this maximum value in z
                    gz = ((dtype_%(gz)s*)(PyArray_GetPtr(%(gz)s, o_idx)));
                  }
        """
        for i in xrange(nd):
            ccode += """
                  // go through the pooled region in the unpadded input
                  for(npy_intp m%(i)s=r_st[%(i)s]; m%(i)s<r_end[%(i)s]; m%(i)s++)
                  {
                    i_idx[%(non_pool_ndim)s + %(i)s] = m%(i)s;
                """ % dict(i=i, non_pool_ndim=non_pool_ndim)
        ccode += """
                    dtype_%(x)s a;
                    dtype_%(gx)s * gx;
                    if (%(total_ndim)s == 4)
                    {
                      a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,i_idx[0],i_idx[1],i_idx[2],i_idx[3])))[0];
                      gx = ((dtype_%(gx)s*)(PyArray_GETPTR4(%(gx)s, i_idx[0],i_idx[1],i_idx[2],i_idx[3])));
                    }
                    else
                    {
                      a = ((dtype_%(x)s*)(PyArray_GetPtr(%(x)s,i_idx)))[0];
                      gx = ((dtype_%(gx)s*)(PyArray_GetPtr(%(gx)s, i_idx)));
                    }
                    if (a == maximum){
                      gx[0] = gx[0] + gz[0];
                    }
        """
        for i in xrange(nd):
            ccode += """
                  } // for loop over region
                """
        for i in xrange(nd):
            ccode += """
                } // loop over pooling dimension
            """

        ccode += """
            } // for loop over non-pooling dimensions
        } // if z_prod
        """
        return ccode % locals()

    def c_code_cache_version(self):
        return (0, 11, self.openmp)


class AveragePoolGrad(PoolGrad):
    # ignore_border is used for perform, but not c code. No need in params_type

    def __init__(self, ignore_border, mode='average_inc_pad', ndim=2):
        assert mode in ['sum', 'average_inc_pad', 'average_exc_pad']
        PoolGrad.__init__(self, ignore_border, mode, ndim)

    # There is an extra dummy parameter to match the parameter count
    # of MaxPoolGrad.  They have to keep the same interface because of
    # the DownsampleFactorMaxGrad trick to keep old scripts working
    # (see downsample.py for details on this).
    def make_node(self, x, gz, ws, stride=None, pad=None, dummy=None):
        # make_node should only be called by the grad function of
        # Pool, so these asserts should not fail.
        x = tensor.as_tensor_variable(x)
        gz = tensor.as_tensor_variable(gz)
        nd = self.ndim
        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert isinstance(x, Variable) and x.ndim >= nd
        assert isinstance(gz, Variable) and gz.ndim >= nd
        assert isinstance(ws, Variable) and ws.ndim == 1
        assert isinstance(stride, Variable) and stride.ndim == 1
        assert x.ndim == gz.ndim >= nd
        assert isinstance(pad, Variable) and pad.ndim == 1
        if ws.dtype not in tensor.int_dtypes:
            raise TypeError('Pool downsample parameters must be ints.')
        if stride.dtype not in tensor.int_dtypes:
            raise TypeError('Stride parameters must be ints.')
        if pad.dtype not in tensor.int_dtypes:
            raise TypeError('Padding parameters must be ints.')
        return Apply(self, [x, gz, ws, stride, pad], [x.type()])

    def perform(self, node, inp, out):
        x, gz, ws, stride, pad = inp
        gx_stg, = out
        nd = self.ndim
        assert ws.shape == stride.shape == pad.shape == (nd,)
        if len(x.shape) < nd:
            raise NotImplementedError(
                'AveragePoolGrad requires input with {} or more dimensions'.format(nd))
        if self.mode == 'average_exc_pad' and max(pad) != 0:
            raise NotImplementedError()
        z_shape = self.out_shape(x.shape, ws, self.ignore_border, stride, pad, nd)
        if (gx_stg[0] is None) or (gx_stg[0].shape != z_shape):
            gx_stg[0] = np.empty(z_shape, dtype=x.dtype)
        zz = gx_stg[0]
        # size of pooling output
        pool_out_shp = zz.shape[-nd:]
        img_shp = tuple(x.shape[-nd + i] + 2 * pad[i] for i in xrange(nd))
        inc_pad = self.mode == 'average_inc_pad'
        sum_mode = self.mode == 'sum'

        # initialize the padded output
        gx = np.zeros((x.shape[:-nd] + img_shp), dtype=x.dtype)

        # precompute the region boundaries and sizes for each dimension
        region_slices = [[] for i in xrange(nd)]
        region_sizes = [[] for i in xrange(nd)]
        for i in xrange(nd):
            for j in xrange(pool_out_shp[i]):
                if sum_mode or inc_pad:
                    start = j * stride[i]
                else:
                    start = builtins.max(j * stride[i], pad[i])
                end = builtins.min(start + ws[i], img_shp[i])
                region_slices[i].append(slice(start, end))
                region_sizes[i].append(end - start)

        # iterate over non-pooling dimensions
        region_slice = [None] * nd
        for k in np.ndindex(*x.shape[:-nd]):
            gzk = gz[k]
            gxk = gx[k]
            # iterate over pooling regions
            for r in np.ndindex(*pool_out_shp):
                region_size = 1
                for i in xrange(nd):
                    region_slice[i] = region_slices[i][r[i]]
                    region_size *= region_sizes[i][r[i]]
                if sum_mode:
                    val = gzk[r]
                else:
                    # divide by region size
                    val = gzk[r] / region_size
                gxk[region_slice] += val

        # unpad the image
        gx = gx[(slice(None),) * (len(x.shape) - nd) +
                tuple(slice(pad[i], img_shp[i] - pad[i]) for i in xrange(nd))]
        gx_stg[0] = gx

    def grad(self, inp, grads):
        x, gz, ws, stride, pad = inp
        ggx, = grads
        return ([theano.tensor.zeros_like(x),
                 Pool(ignore_border=self.ignore_border,
                      ndim=self.ndim, mode=self.mode)(ggx,
                ws, stride, pad)] + [DisconnectedType()() for i in inp[2:]])

    def connection_pattern(self, node):
        return [[1], [1], [0], [0], [0]]

    def c_code(self, node, name, inp, out, sub):
        x, gz, ws, stride, pad = inp
        gx, = out
        nd = self.ndim
        total_ndim = node.inputs[0].ndim
        non_pool_ndim = total_ndim - nd
        fail = sub['fail']
        inc_pad = int(self.mode == 'average_inc_pad')
        sum_mode = int(self.mode == 'sum')
        if self.openmp:
            # run in parallel over each pooling block
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, r_pad_width, r_idx, i_idx, o_idx) schedule(static)'
        else:
            omp_parallel = ''

        ccode = """
        // sanity checks
        int x_typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int gz_typenum = PyArray_ObjectType((PyObject*)%(gz)s, 0);
        if (x_typenum != gz_typenum)
        {
            PyErr_SetString(PyExc_ValueError, "input types must all match");
            %(fail)s;
        }
        if(PyArray_NDIM(%(x)s)!=%(total_ndim)s)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a %(total_ndim)sD ndarray");
            %(fail)s;
        }
        if(PyArray_NDIM(%(gz)s)!=%(total_ndim)s)
        {
            PyErr_SetString(PyExc_ValueError, "gz must be a %(total_ndim)sD ndarray");
            %(fail)s;
        }
        if(PyArray_DIM(%(ws)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "ws must be a vector of size %(nd)s");
            %(fail)s;
        }
        if(PyArray_DIM(%(stride)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "stride must be a vector of size %(nd)s");
            %(fail)s;
        }
        if(PyArray_DIM(%(pad)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "pad must be a vector of size %(nd)s");
            %(fail)s;
        }
        npy_intp z[%(nd)s]; // shape of the output
        npy_intp r[%(nd)s]; // shape of the padded_input
        npy_intp ws[%(nd)s];
        npy_intp st[%(nd)s];
        npy_intp pd[%(nd)s];
        int nonzero_padding;
        nonzero_padding = 0;
        for (int i=0; i<%(nd)s; i++)
        {
            ws[i] = *((dtype_%(ws)s*)PyArray_GETPTR1(%(ws)s, i));
            st[i] = *((dtype_%(stride)s*)PyArray_GETPTR1(%(stride)s, i));
            pd[i] = *((dtype_%(pad)s*)PyArray_GETPTR1(%(pad)s, i));
            z[i] = PyArray_DIMS(%(gz)s)[%(non_pool_ndim)s + i];
            r[i] = PyArray_DIMS(%(x)s)[%(non_pool_ndim)s + i] + 2 * pd[i];
            if (pd[i]>0)
                nonzero_padding = 1;
        }
        if (!%(inc_pad)s && !%(sum_mode)s && nonzero_padding)
        {
            PyErr_SetString(PyExc_ValueError,
              "padding must be zero for average_exc_pad");
            %(fail)s;
        }
        // allocating memory for output, if necessary
        int mem_nec;
        mem_nec = 0;
        if ((!%(gx)s) || !PyArray_ISCONTIGUOUS(%(gx)s)
            || *PyArray_DIMS(%(gx)s)!=%(total_ndim)s)
        {
            mem_nec = 1;
        }
        if (!mem_nec)
        {
            for (int i=0; i<%(total_ndim)s; i++)
            {
                if (PyArray_DIMS(%(gx)s)[i] != PyArray_DIMS(%(x)s)[i])
                {
                    mem_nec = 1;
                    break;
                }
            }
        }
        if (mem_nec)
        {
          Py_XDECREF(%(gx)s);
          %(gx)s = (PyArrayObject*) PyArray_ZEROS(%(total_ndim)s, PyArray_DIMS(%(x)s), x_typenum,0);
        }
        else {
          PyArray_FILLWBYTE(%(gx)s, 0);
        }
        npy_intp z_prod;
        // do not run if any z[i] is zero
        z_prod = 1;
        for (int i=0; i<%(nd)s; i++)
        {
            z_prod *= z[i];
        }
        if (z_prod)
        {
            // will be used to hold start and end index of a region
            npy_intp r_st[%(nd)s];
            npy_intp r_end[%(nd)s];
            // padded region size
            npy_intp r_pad_width[%(nd)s];
            // index for iterating over the pooling regions
            npy_intp r_idx[%(nd)s];
            // placeholder for PyArray indexing (output)
            npy_intp o_idx[%(total_ndim)s];
            // placeholder for PyArray indexing (input)
            npy_intp i_idx[%(total_ndim)s];
            // loop over non-pooling dimensions
            npy_intp non_pooling_prod = 1;
            for (int i=0; i<%(non_pool_ndim)s; i++)
            {
                non_pooling_prod *= PyArray_DIMS(%(x)s)[i];
            }
            %(omp_parallel)s
            // first loop over non-pooling dimensions
            for (npy_intp t=0; t<non_pooling_prod; t++)
            {
                // compute the non-pooling index in each dimension
                if (%(non_pool_ndim)s!=0)
                {
                    o_idx[0] = t;
                    i_idx[0] = t;
                    for (int i=1; i<%(non_pool_ndim)s; i++)
                    {
                        o_idx[i] = o_idx[i - 1] / PyArray_DIMS(%(x)s)[i - 1];
                        o_idx[i - 1] =o_idx[i - 1] %% PyArray_DIMS(%(x)s)[i - 1];
                        i_idx[i] = o_idx[i];
                        i_idx[i - 1] = o_idx[i - 1];
                    }
                }

                // then loop over each region in each pooling dimension
        """

        for i in xrange(nd):
            ccode += """
                for (r_idx[%(i)s]=0; r_idx[%(i)s] < z[%(i)s]; r_idx[%(i)s]++) {
                  r_st[%(i)s] = r_idx[%(i)s] * st[%(i)s];
                  if (!%(sum_mode)s && !%(inc_pad)s && r_st[%(i)s] < pd[%(i)s])
                  {
                    r_st[%(i)s] = pd[%(i)s];
                  }
                  r_end[%(i)s] = r_st[%(i)s] + ws[%(i)s];
                  r_end[%(i)s] = r_end[%(i)s] > r[%(i)s] ? r[%(i)s] : r_end[%(i)s];
                  r_pad_width[%(i)s] = r_end[%(i)s] - r_st[%(i)s];
                  // from padded_img space to img space
                  r_st[%(i)s] = r_st[%(i)s] - pd[%(i)s] > 0 ? r_st[%(i)s] - pd[%(i)s] : 0;
                  r_end[%(i)s] = r_end[%(i)s] > r[%(i)s] - pd[%(i)s] ? r[%(i)s] - 2 * pd[%(i)s] : r_end[%(i)s] - pd[%(i)s];

                  // use the index to find the correct position in the output
                  o_idx[%(non_pool_ndim)s + %(i)s] = r_idx[%(i)s];
            """ % dict(i=i, sum_mode=sum_mode, inc_pad=inc_pad, non_pool_ndim=non_pool_ndim)

        ccode += """
                  dtype_%(gz)s * gz;
                  dtype_%(gz)s val;
                  if (%(total_ndim)s == 4)
                  {
                    // the gradient for this region
                    gz = ((dtype_%(gz)s*)(PyArray_GETPTR4(%(gz)s, o_idx[0],o_idx[1],o_idx[2],o_idx[3])));
                  }
                  else
                  {
                    // the gradient for this region
                    gz = ((dtype_%(gz)s*)(PyArray_GetPtr(%(gz)s, o_idx)));
                  }
                  // compute the contribution
                  if (%(sum_mode)s)
                  {
                    val = gz[0];
                  }
                  else
                  {
                    val = gz[0] / (%(region_size)s);
                  }
        """
        region_size = ' * '.join('r_pad_width[%d]' % i for i in xrange(nd))
        for i in xrange(nd):
            ccode += """
                  // go through the pooled region in the unpadded input
                  for(npy_intp m%(i)s=r_st[%(i)s]; m%(i)s<r_end[%(i)s]; m%(i)s++)
                  {
                    i_idx[%(non_pool_ndim)s + %(i)s] = m%(i)s;
                """ % dict(i=i, non_pool_ndim=non_pool_ndim)
        ccode += """
                    dtype_%(gx)s * gx;
                    if (%(total_ndim)s == 4)
                    {
                      gx = ((dtype_%(gx)s*)(PyArray_GETPTR4(%(gx)s, i_idx[0],i_idx[1],i_idx[2],i_idx[3])));
                    }
                    else
                    {
                      gx = ((dtype_%(gx)s*)(PyArray_GetPtr(%(gx)s, i_idx)));
                    }
                    gx[0] = gx[0] + val;
        """
        for i in xrange(nd):
            ccode += """
                  } // for loop over region
                """
        for i in xrange(nd):
            ccode += """
                } // loop over pooling dimension
            """

        ccode += """
            } // for loop over non-pooling dimensions
        } // if z_prod
        """
        return ccode % locals()

    def c_code_cache_version(self):
        return (0, 4, self.openmp)


class DownsampleFactorMaxGradGrad(OpenMPOp):
    __props__ = ('ignore_border', 'mode', 'ndim')

    def __init__(self, ignore_border, mode='max', ndim=2, openmp=None):
        self.ndim = ndim
        self.ignore_border = ignore_border
        self.mode = mode
        super(DownsampleFactorMaxGradGrad, self).__init__(openmp=openmp)
        assert self.mode == 'max'

    def make_node(self, x, maxout, gz, ws, stride=None, pad=None):
        # make_node should only be called by the grad function of
        # MaxPoolGrad, so these asserts should not fail.
        x = tensor.as_tensor_variable(x)
        maxout = tensor.as_tensor_variable(maxout)
        gz = tensor.as_tensor_variable(gz)
        nd = self.ndim
        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        elif isinstance(pad, (tuple, list)):
            if max(pad) != 0 and not self.ignore_border:
                raise NotImplementedError(
                    'padding works only with ignore_border=True')
            if isinstance(ws, (tuple, list)):
                if any(pad[i] >= ws[i] for i in range(nd)):
                    raise NotImplementedError(
                        'padding must be smaller than strides')
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert ws.ndim == 1
        assert stride.ndim == 1
        assert pad.ndim == 1
        assert x.ndim == maxout.ndim == gz.ndim >= nd
        if ws.dtype not in tensor.int_dtypes:
            raise TypeError('Pool downsample parameters must be ints.')
        if stride.dtype not in tensor.int_dtypes:
            raise TypeError('Stride parameters must be ints.')
        if pad.dtype not in tensor.int_dtypes:
            raise TypeError('Padding parameters must be ints.')
        return Apply(self, [x, maxout, gz, ws, stride, pad], [x.type()])

    def perform(self, node, inp, out):
        x, maxout, ggx, ws, stride, pad = inp
        z, = out
        nd = self.ndim
        assert ws.shape == stride.shape == pad.shape == (nd,)
        if len(x.shape) < nd:
            raise NotImplementedError(
                'DownsampleFactorMaxGradGrad requires input '
                'with {} or more dimensions'.format(nd))
        if (z[0] is None) or (z[0].shape != maxout.shape):
            z[0] = np.zeros(maxout.shape, dtype=x.dtype)
        ggz = z[0]  # grad wrt maxout_grad has the same shape as maxout
        # size of pooling output
        pool_out_shp = ggz.shape[-nd:]
        img_shp = tuple(x.shape[-nd + i] + 2 * pad[i] for i in xrange(nd))

        # pad the image and its gradients
        if max(pad) > 0:
            y_padded = np.zeros(x.shape[:-nd] + img_shp, dtype=x.dtype)
            y_padded[(slice(None),) * (len(x.shape) - nd) +
                     tuple(slice(pad[i], img_shp[i] - pad[i]) for i in xrange(nd))] = x
            ggx_padded = np.zeros(x.shape[:-nd] + img_shp, dtype=x.dtype)
            ggx_padded[(slice(None),) * (len(x.shape) - nd) +
                       tuple(slice(pad[i], img_shp[i] - pad[i]) for i in xrange(nd))] = ggx

        else:
            y_padded = x
            ggx_padded = ggx

        # precompute the region boundaries for each dimension
        region_ranges = [[] for i in xrange(nd)]
        for i in xrange(nd):
            for j in xrange(pool_out_shp[i]):
                start = j * stride[i]
                end = builtins.min(start + ws[i], img_shp[i])
                region_ranges[i].append(xrange(start, end))

        # iterate over non-pooling dimensions
        for k in np.ndindex(*x.shape[:-nd]):
            ggxk = ggx_padded[k]
            ggzk = ggz[k]
            yk = y_padded[k]
            maxoutk = maxout[k]
            # iterate over pooling regions
            for r in np.ndindex(*pool_out_shp):
                # iterate inside region
                maxout_value = maxoutk[r]
                for c in itertools.product(*[region_ranges[i][r[i]]
                                             for i in xrange(nd)]):
                    if maxout_value == yk[c]:
                        ggzk[r] += ggxk[c]

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

    def grad(self, inp, grads):
        x, maxout, ggx, ws, stride, pad = inp
        gz, = grads
        return [theano.tensor.zeros_like(x),
                theano.tensor.zeros_like(maxout),
                MaxPoolGrad(ignore_border=self.ignore_border,
                            ndim=self.ndim)(x, maxout, gz,
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
        nd = self.ndim
        total_ndim = node.inputs[0].ndim
        non_pool_ndim = total_ndim - nd
        fail = sub['fail']

        if self.openmp:
            # run in parallel over each pooling block
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, r_idx, i_idx, o_idx, maximum) schedule(static)'
        else:
            omp_parallel = ''
        ccode = """
        int z_typenum = PyArray_ObjectType((PyObject*)%(maxout)s, 0);
        npy_intp z[%(nd)s]; // shape of the output
        npy_intp r[%(nd)s]; // shape of the padded_input
        npy_intp ws[%(nd)s];
        npy_intp st[%(nd)s];
        npy_intp pd[%(nd)s];
        if(PyArray_DIM(%(ws)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "ws must be a vector of size %(nd)s");
            %(fail)s;
        }
        if(PyArray_DIM(%(stride)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "stride must be a vector of size %(nd)s");
            %(fail)s;
        }
        if(PyArray_DIM(%(pad)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "pad must be a vector of size %(nd)s");
            %(fail)s;
        }
        for (int i=0; i<%(nd)s; i++)
        {
            ws[i] = *((dtype_%(ws)s*)PyArray_GETPTR1(%(ws)s, i));
            st[i] = *((dtype_%(stride)s*)PyArray_GETPTR1(%(stride)s, i));
            pd[i] = *((dtype_%(pad)s*)PyArray_GETPTR1(%(pad)s, i));
            z[i] = PyArray_DIMS(%(maxout)s)[%(non_pool_ndim)s + i];
            r[i] = PyArray_DIMS(%(x)s)[%(non_pool_ndim)s + i] + 2 * pd[i];
        }
        // allocating memory for output, if necessary
        int mem_nec;
        mem_nec = 0;
        if ((!%(z)s) || !PyArray_ISCONTIGUOUS(%(z)s)
            || *PyArray_DIMS(%(z)s)!=%(total_ndim)s)
        {
            mem_nec = 1;
        }
        if (!mem_nec)
        {
            for (int i=0; i<%(total_ndim)s; i++)
            {
                if (PyArray_DIMS(%(z)s)[i] != PyArray_DIMS(%(maxout)s)[i])
                {
                    mem_nec = 1;
                    break;
                }
            }
        }
        if (mem_nec)
        {
          Py_XDECREF(%(z)s);
          %(z)s = (PyArrayObject*) PyArray_ZEROS(%(total_ndim)s, PyArray_DIMS(%(maxout)s), z_typenum,0);
        }
        else {
          PyArray_FILLWBYTE(%(z)s, 0);
        }
        dtype_%(maxout)s maximum; // temp var for maximum value in a region
        // will be used to hold start and end index of a region
        npy_intp r_st[%(nd)s];
        npy_intp r_end[%(nd)s];
        // index for iterating over the pooling regions
        npy_intp r_idx[%(nd)s];
        // placeholder for PyArray indexing (output)
        npy_intp o_idx[%(total_ndim)s];
        // placeholder for PyArray indexing (input)
        npy_intp i_idx[%(total_ndim)s];
        // loop over non-pooling dimensions
        npy_intp non_pooling_prod;
        non_pooling_prod = 1;
        for (int i=0; i<%(non_pool_ndim)s; i++)
        {
            non_pooling_prod *= PyArray_DIMS(%(x)s)[i];
        }
        %(omp_parallel)s
        // first loop over non-pooling dimensions
        for (npy_intp t=0; t<non_pooling_prod; t++)
        {
            // compute the non-pooling index in each dimension
            if (%(non_pool_ndim)s!=0)
            {
                o_idx[0] = t;
                i_idx[0] = t;
                for (int i=1; i<%(non_pool_ndim)s; i++)
                {
                    o_idx[i] = o_idx[i - 1] / PyArray_DIMS(%(x)s)[i - 1];
                    o_idx[i - 1] = o_idx[i - 1] %% PyArray_DIMS(%(x)s)[i - 1];
                    i_idx[i] = o_idx[i];
                    i_idx[i - 1] = o_idx[i - 1];
                }
            }

            // then loop over each region in each pooling dimension
        """

        for i in xrange(nd):
            ccode += """
                for (r_idx[%(i)s]=0; r_idx[%(i)s] < z[%(i)s]; r_idx[%(i)s]++) {
                  r_st[%(i)s] = r_idx[%(i)s] * st[%(i)s];
                  r_end[%(i)s] = r_st[%(i)s] + ws[%(i)s];
                  // skip the padding
                  r_st[%(i)s] = r_st[%(i)s] < pd[%(i)s] ? pd[%(i)s] : r_st[%(i)s];
                  r_end[%(i)s] = r_end[%(i)s] > (r[%(i)s] - pd[%(i)s]) ? r[%(i)s] - pd[%(i)s] : r_end[%(i)s];
                  // from padded_img space to img space
                  r_st[%(i)s] -= pd[%(i)s];
                  r_end[%(i)s] -= pd[%(i)s];
                  // use the index to find the correct position in the output
                  o_idx[%(non_pool_ndim)s + %(i)s] = r_idx[%(i)s];
            """ % dict(i=i, non_pool_ndim=non_pool_ndim)

        ccode += """
                  dtype_%(z)s * z;
                  if (%(total_ndim)s == 4)
                  {
                    // the maximum value
                    maximum = ((dtype_%(maxout)s*)(PyArray_GETPTR4(%(maxout)s,o_idx[0],o_idx[1],o_idx[2],o_idx[3])))[0];
                    // z at this position
                    z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,o_idx[0],o_idx[1],o_idx[2],o_idx[3])));
                  }
                  else
                  {
                    // the maximum value
                    maximum = ((dtype_%(maxout)s*)(PyArray_GetPtr(%(maxout)s,o_idx)))[0];
                    // z at this position
                    z = ((dtype_%(z)s*)(PyArray_GetPtr(%(z)s,o_idx)));
                  }
        """
        for i in xrange(nd):
            ccode += """
                  // go through the pooled region in the unpadded input
                  for(npy_intp m%(i)s=r_st[%(i)s]; m%(i)s<r_end[%(i)s]; m%(i)s++)
                  {
                    i_idx[%(non_pool_ndim)s + %(i)s] = m%(i)s;
                """ % dict(i=i, non_pool_ndim=non_pool_ndim)
        ccode += """
                    dtype_%(x)s a;
                    dtype_%(ggx)s * ggx;
                    if (%(total_ndim)s == 4)
                    {
                      a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,i_idx[0],i_idx[1],i_idx[2],i_idx[3])))[0];
                      ggx = ((dtype_%(ggx)s*)(PyArray_GETPTR4(%(ggx)s,i_idx[0],i_idx[1],i_idx[2],i_idx[3])));
                    }
                    else
                    {
                      a = ((dtype_%(x)s*)(PyArray_GetPtr(%(x)s,i_idx)))[0];
                      ggx = ((dtype_%(ggx)s*)(PyArray_GetPtr(%(ggx)s,i_idx)));
                    }
                    if (a == maximum){
                      z[0] += ggx[0];
                    }
        """
        for i in xrange(nd):
            ccode += """
                  } // for loop over region
                """
        for i in xrange(nd):
            ccode += """
              } // loop over pooling dimension
            """

        ccode += """
          } // for loop over non-pooling dimensions
        """
        return ccode % locals()

    def c_code_cache_version(self):
        return (0, 5, self.openmp)


class MaxPoolRop(OpenMPOp):
    """
    Implements the R-operator for the downsample operation.

    Parameters
    ----------
    ws : list or tuple of N ints
        Downsample factor over rows, columns etc.
        ws indicates the size of the pooling region.
    ignore_border : bool
        If ws doesn't divide imgshape, do we include an extra row/col/slice
        of partial downsampling (False) or ignore it (True).
    stride : list or tuple of N ints or None
        Stride size, which is the number of shifts over rows/cols/slices to get the
        next pool region. If stride is None, it is considered equal to ws
        (no overlap on pooling regions).
    pad : tuple of N ints or None
        For each downsampling dimension, this specifies the number of zeros to
        add as padding on both sides. For 2D and (pad_h, pad_w), pad_h specifies the
        size of the top and bottom margins, pad_w specifies the size of the left and
        right margins. No padding is added if pad is None.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        ('average_inc_pad' excludes the padding from the count,
        'average_exc_pad' include it)
    ndim : int
        The number of pooling dimensions N.
        The default is 2.
    """

    __props__ = ('ignore_border', 'mode', 'ndim')
    params_type = ParamsType(ignore_border=bool_t,)

    def __init__(self, ignore_border=False, mode='max', ndim=2, openmp=None):
        super(MaxPoolRop, self).__init__(openmp=openmp)
        self.ndim = ndim
        self.ignore_border = ignore_border
        self.mode = mode
        assert mode == 'max'

    def make_node(self, x, eval_point, ws, stride=None, pad=None):
        # TODO: consider restricting the dtype?
        x = tensor.as_tensor_variable(x)
        eval_point = tensor.as_tensor_variable(eval_point)
        nd = self.ndim
        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        elif isinstance(pad, (tuple, list)):
            if max(pad) != 0 and not self.ignore_border:
                raise NotImplementedError(
                    'padding works only with ignore_border=True')
            if isinstance(ws, (tuple, list)):
                if any(pad[i] >= ws[i] for i in range(nd)):
                    raise NotImplementedError(
                        'padding must be smaller than strides')
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert ws.ndim == 1
        assert stride.ndim == 1
        assert pad.ndim == 1
        if x.type.ndim < nd:
            raise TypeError()
        if not ws.dtype.startswith('int'):
            raise TypeError('Pool downsample parameters must be ints.')
        if not stride.dtype.startswith('int'):
            raise TypeError('Stride parameters must be ints.')
        if not pad.dtype.startswith('int'):
            raise TypeError('Padding parameters must be ints.')
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:-nd] + (False,) * nd
        out = tensor.TensorType(eval_point.dtype, broad)
        return gof.Apply(self, [x, eval_point, ws, stride, pad], [out()])

    def perform(self, node, inp, out, params):
        x, ex, ws, stride, pad = inp
        z, = out
        nd = self.ndim
        assert ws.shape == stride.shape == pad.shape == (nd,)
        if len(x.shape) < nd:
            raise NotImplementedError(
                'Pool requires input with {} or more dimensions'.format(nd))
        z_shape = Pool.out_shape(x.shape, ws, params.ignore_border, stride, pad, nd)
        if not self.ignore_border:
            assert all(z > 0 for z in z_shape[-nd:])
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = np.empty(z_shape, dtype=x.dtype)
        zz = z[0]
        # size of pooling output
        pool_out_shp = zz.shape[-nd:]
        img_shp = tuple(x.shape[-nd + i] + 2 * pad[i] for i in xrange(nd))
        inc_pad = self.mode == 'average_inc_pad'

        # pad the image and the eval point
        if max(pad) != 0:
            y = np.zeros(x.shape[:-nd] + img_shp, dtype=x.dtype)
            y[(slice(None),) * (len(x.shape) - nd) +
              tuple(slice(pad[i], img_shp[i] - pad[i]) for i in xrange(nd))] = x
            ey = np.zeros(ex.shape[:-nd] + img_shp, dtype=ex.dtype)
            ey[(slice(None),) * (len(ex.shape) - nd) +
               tuple(slice(pad[i], img_shp[i] - pad[i]) for i in xrange(nd))] = ex
        else:
            y = x
            ey = ex

        # precompute the region boundaries for each dimension
        region_slices = [[] for i in xrange(nd)]
        for i in xrange(nd):
            for j in xrange(pool_out_shp[i]):
                start = j * stride[i]
                end = builtins.min(start + ws[i], img_shp[i])
                if not inc_pad:
                    start = builtins.max(start, pad[i])
                    end = builtins.min(end, img_shp[i] - pad[i])
                region_slices[i].append(slice(start, end))

        # iterate over non-pooling dimensions
        for k in np.ndindex(*x.shape[:-nd]):
            zzk = zz[k]
            yk = y[k]
            eyk = ey[k]
            # iterate over pooling regions
            for r in np.ndindex(*pool_out_shp):
                # current slice in padded input
                ykslice = yk[[region_slices[i][r[i]] for i in xrange(nd)]]
                # current slice in eval points
                eykslice = eyk[[region_slices[i][r[i]] for i in xrange(nd)]]
                # indices of maximum
                idx = np.unravel_index(np.argmax(ykslice), ykslice.shape)
                zzk[r] = eykslice[idx]

    def c_headers(self):
        headers = ['<algorithm>']
        headers += super(MaxPoolRop, self).c_headers()
        return headers

    def c_code(self, node, name, inp, out, sub):
        if self.mode != 'max':
            raise theano.gof.utils.MethodNotDefined()
        x, ex, ws, stride, pad = inp
        z, = out
        nd = self.ndim
        total_ndim = node.inputs[0].ndim
        non_pool_ndim = total_ndim - nd
        fail = sub['fail']
        params = sub['params']

        if self.openmp:
            # run in parallel over each pooling block
            omp_parallel = '#pragma omp parallel for private(r_st, r_end, r_idx, i_idx, o_idx, collector, eval_collector) schedule(static)'
        else:
            omp_parallel = ''
        ccode = """
        int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        if(PyArray_NDIM(%(x)s)!=%(total_ndim)s)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a %(total_ndim)sD ndarray");
            %(fail)s;
        }
        if(PyArray_NDIM(%(ex)s)!=%(total_ndim)s)
        {
            PyErr_SetString(PyExc_ValueError, "eval_point must be a %(total_ndim)sD ndarray");
            %(fail)s;
        }
        if(PyArray_DIM(%(ws)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "ws must be a vector of size %(nd)s");
            %(fail)s;
        }
        if(PyArray_DIM(%(stride)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "stride must be a vector of size %(nd)s");
            %(fail)s;
        }
        if(PyArray_DIM(%(pad)s, 0)!=%(nd)s)
        {
            PyErr_SetString(PyExc_ValueError, "pad must be a vector of size %(nd)s");
            %(fail)s;
        }
        npy_intp z[%(nd)s]; // shape of the output
        npy_intp r[%(nd)s]; // shape of the padded_input
        npy_intp ws[%(nd)s];
        npy_intp st[%(nd)s];
        npy_intp pd[%(nd)s];
        int nonzero_padding;
        nonzero_padding = 0;
        for (int i=0; i<%(nd)s; i++)
        {
            ws[i] = *((dtype_%(ws)s*)PyArray_GETPTR1(%(ws)s, i));
            st[i] = *((dtype_%(stride)s*)PyArray_GETPTR1(%(stride)s, i));
            pd[i] = *((dtype_%(pad)s*)PyArray_GETPTR1(%(pad)s, i));
            r[i] = PyArray_DIMS(%(x)s)[%(non_pool_ndim)s + i] + 2 * pd[i];
            if (pd[i]>0)
                nonzero_padding = 1;
        }
        if (!%(params)s->ignore_border && nonzero_padding)
        {
            PyErr_SetString(PyExc_ValueError,
              "padding must be zero when ignore border is False");
            %(fail)s;
        }
        if (%(params)s->ignore_border)
        {
            for (int i=0; i<%(nd)s; i++)
            {
                // '/' in C is different from '/' in python
                if (r[i] - ws[i] < 0)
                {
                  z[i] = 0;
                }
                else
                {
                  z[i] = (r[i] - ws[i]) / st[i] + 1;
                }
            }
        }
        else
        {
            for (int i=0; i<%(nd)s; i++)
            {
                // decide how many rows/cols the output has
                if (st[i] >= ws[i])
                {
                    z[i] = (r[i] - 1) / st[i] + 1;
                }
                else
                {
                    z[i] = std::max((npy_intp)0, (r[i] - 1 - ws[i] + st[i]) / st[i]) + 1;
                }
                assert(z[i] > 0);
            }
        }
        // memory allocation of z if necessary
        int mem_nec;
        mem_nec = 0;
        if ((!%(z)s) || *PyArray_DIMS(%(z)s)!=%(total_ndim)s)
        {
            mem_nec = 1;
        }
        if (!mem_nec)
        {
            for (int i=0; i<%(non_pool_ndim)s; i++)
            {
                if (PyArray_DIMS(%(z)s)[i] != PyArray_DIMS(%(x)s)[i])
                {
                    mem_nec = 1;
                    break;
                }
            }
        }
        if (!mem_nec)
        {
            for (int i=0; i<%(nd)s; i++)
            {
                if (PyArray_DIMS(%(z)s)[%(non_pool_ndim)s + i] != z[i])
                {
                    mem_nec = 1;
                    break;
                }
            }
        }
        if (mem_nec)
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[%(total_ndim)s];
          for (int i=0; i<%(non_pool_ndim)s; i++)
          {
              dims[i] = PyArray_DIMS(%(x)s)[i];
          }
          for (int i=0; i<%(nd)s; i++)
          {
              dims[%(non_pool_ndim)s + i] = z[i];
          }
          //TODO: zeros not necessary
          %(z)s = (PyArrayObject*) PyArray_ZEROS(%(total_ndim)s, dims, typenum,0);
        }
        // initialize temp var for the value in a region
        dtype_%(x)s collector;
        dtype_%(ex)s eval_collector;
        npy_intp z_prod;
        // do not run if any z[i] is zero
        z_prod = 1;
        for (int i=0; i<%(nd)s; i++)
        {
            z_prod *= z[i];
        }
        if (z_prod)
        {
            // will be used to hold start and end index of a region
            npy_intp r_st[%(nd)s];
            npy_intp r_end[%(nd)s];
            // index for iterating over the pooling regions
            npy_intp r_idx[%(nd)s];
            // placeholder for PyArray indexing (output)
            npy_intp o_idx[%(total_ndim)s];
            // placeholder for PyArray indexing (input)
            npy_intp i_idx[%(total_ndim)s];
            // loop over non-pooling dimensions
            npy_intp non_pooling_prod = 1;
            for (int i=0; i<%(non_pool_ndim)s; i++)
            {
                non_pooling_prod *= PyArray_DIMS(%(x)s)[i];
            }
            %(omp_parallel)s
            // first loop over non-pooling dimensions
            for (npy_intp t=0; t<non_pooling_prod; t++)
            {
                // compute the non-pooling index in each dimension
                if (%(non_pool_ndim)s!=0)
                {
                    o_idx[0] = t;
                    i_idx[0] = t;
                    for (int i=1; i<%(non_pool_ndim)s; i++)
                    {
                        o_idx[i] = o_idx[i - 1] / PyArray_DIMS(%(x)s)[i - 1];
                        o_idx[i - 1] = o_idx[i - 1] %% PyArray_DIMS(%(x)s)[i - 1];
                        i_idx[i] = o_idx[i];
                        i_idx[i - 1] = o_idx[i - 1];
                    }
                }

                // then loop over each region in each pooling dimension
        """

        for i in xrange(nd):
            ccode += """
                for (r_idx[%(i)s]=0; r_idx[%(i)s] < z[%(i)s]; r_idx[%(i)s]++) {
                  r_st[%(i)s] = r_idx[%(i)s] * st[%(i)s];
                  r_end[%(i)s] = r_st[%(i)s] + ws[%(i)s];
                  // skip the padding
                  r_st[%(i)s] = r_st[%(i)s] < pd[%(i)s] ? pd[%(i)s] : r_st[%(i)s];
                  r_end[%(i)s] = r_end[%(i)s] > (r[%(i)s] - pd[%(i)s]) ? r[%(i)s] - pd[%(i)s] : r_end[%(i)s];
                  // from padded_img space to img space
                  r_st[%(i)s] -= pd[%(i)s];
                  r_end[%(i)s] -= pd[%(i)s];
                  // handle the case where no padding, ignore border is True
                  if (%(params)s->ignore_border)
                  {
                    r_end[%(i)s] = r_end[%(i)s] > r[%(i)s] ? r[%(i)s] : r_end[%(i)s];
                  }
                  // use the index to find the correct position in the output
                  o_idx[%(non_pool_ndim)s + %(i)s] = r_idx[%(i)s];
            """ % dict(i=i, params=sub['params'], non_pool_ndim=non_pool_ndim)

        ccode += """
                  // get a pointer to the correct position in the output
                  dtype_%(z)s * z;
                  if (%(total_ndim)s == 4)
                    z = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, o_idx[0], o_idx[1], o_idx[2], o_idx[3])));
                  else
                    z = ((dtype_%(z)s*)(PyArray_GetPtr(%(z)s, o_idx)));
        """

        for i in xrange(nd):
            ccode += """
              // set the first index of dimension %(i)s
              i_idx[%(non_pool_ndim)s + %(i)s] = r_st[%(i)s];
            """ % dict(i=i, non_pool_ndim=non_pool_ndim)
        ccode += """
              // use the first element as the initial value of collector
              if (%(total_ndim)s == 4) {
                collector = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,i_idx[0],i_idx[1],i_idx[2],i_idx[3])))[0];
                eval_collector = ((dtype_%(ex)s*)(PyArray_GETPTR4(%(ex)s,i_idx[0],i_idx[1],i_idx[2],i_idx[3])))[0];
              } else {
                collector = ((dtype_%(x)s*)(PyArray_GetPtr(%(x)s,i_idx)))[0];
                eval_collector = ((dtype_%(ex)s*)(PyArray_GetPtr(%(ex)s,i_idx)))[0];
              }
        """
        for i in xrange(nd):
            ccode += """
              // go through the pooled region in the unpadded input
              for(npy_intp m%(i)s=r_st[%(i)s]; m%(i)s<r_end[%(i)s]; m%(i)s++)
              {
                i_idx[%(non_pool_ndim)s + %(i)s] = m%(i)s;
            """ % dict(i=i, non_pool_ndim=non_pool_ndim)
        ccode += """
                // update maximum
                dtype_%(x)s a;
                dtype_%(ex)s ea;
                if (%(total_ndim)s == 4) {
                  a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,i_idx[0],i_idx[1],i_idx[2],i_idx[3])))[0];
                  ea = ((dtype_%(ex)s*)(PyArray_GETPTR4(%(ex)s,i_idx[0],i_idx[1],i_idx[2],i_idx[3])))[0];
                }
                else {
                  a = ((dtype_%(x)s*)(PyArray_GetPtr(%(x)s,i_idx)))[0];
                  ea = ((dtype_%(ex)s*)(PyArray_GetPtr(%(ex)s,i_idx)))[0];
                }
                if (a > collector) {
                  collector = a;
                  eval_collector = ea;
                }
        """
        for i in xrange(nd):
            ccode += """
              } // for loop over region
            """
        ccode += """
              z[0] = eval_collector;
        """
        for i in xrange(nd):
            ccode += """
            } // loop over pooling dimension
            """

        ccode += """
          } // for loop over non-pooling dimensions
        } // if z_prod
        """
        return ccode % locals()

    def c_code_cache_version(self):
        return (2, self.openmp)
