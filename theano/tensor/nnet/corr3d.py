from __future__ import absolute_import, print_function, division
import os
import logging

from six import integer_types

import theano
from theano import Apply
from theano import gof
from theano.gof import ParamsType, EnumList
from theano.scalar import int64
from theano.tensor import as_tensor_variable, TensorType
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor import blas_headers
from theano.tensor.blas import ldflags, blas_header_version

_logger = logging.getLogger(__name__)


class BaseCorr3dMM(gof.OpenMPOp):
    """
    Base class for `Corr3dMM`, `Corr3dMM_gradWeights` and
    `Corr3dMM_gradInputs`. Cannot be used directly.

    Every sub-class must define internal attribute ``_direction`` out of __init__().
    ``_direction`` must take one of following values:

     - "forward" to correlate bottom with weights and store results in top.
     - "backprop weights" to do a valid convolution of bottom with top
       (swapping the first two dimensions) and store results in weights.
     - "backprop inputs" to do a full convolution of top with weights
       (swapping the first two dimensions) and store results in bottom.

    Parameters
    ----------
    border_mode : {'valid', 'full', 'half'}
        Additionally, the padding size could be directly specified by an integer
        or a tuple of three of integers
    subsample
        Perform subsampling of the output (default: (1, 1, 1)).
    filter_dilation
        Perform dilated correlation (default: (1, 1, 1))
    num_groups
        Perform grouped convolutions (default: 1)
    """
    check_broadcast = False
    __props__ = ('border_mode', 'subsample', 'filter_dilation', 'num_groups')

    _direction = None

    params_type = ParamsType(direction=EnumList(('DIRECTION_FORWARD', 'forward'),  # 0
                                                ('DIRECTION_BACKPROP_WEIGHTS', 'backprop weights'),  # 1
                                                ('DIRECTION_BACKPROP_INPUTS', 'backprop inputs')),  # 2
                             dH=int64, dW=int64, dD=int64,
                             dilH=int64, dilW=int64, dilD=int64,
                             padH=int64, padW=int64, padD=int64,
                             num_groups=int64)

    def __init__(self, border_mode="valid", subsample=(1, 1, 1),
                 filter_dilation=(1, 1, 1), openmp=None, num_groups=1):
        super(BaseCorr3dMM, self).__init__(openmp=openmp)
        if isinstance(border_mode, integer_types):
            if border_mode < 0:
                raise ValueError(
                    'invalid border_mode {}, which must be a '
                    'non-negative integer'.format(border_mode))
            border_mode = (border_mode, border_mode, border_mode)
        if isinstance(border_mode, tuple):
            if len(border_mode) != 3 or min(border_mode) < 0:
                raise ValueError(
                    'invalid border_mode {}, which must be a tuple of '
                    'three non-negative integers'.format(border_mode))
            pad_h, pad_w, pad_d = map(int, border_mode)
            border_mode = (pad_h, pad_w, pad_d)
        if not ((isinstance(border_mode, tuple) and min(border_mode) >= 0) or
                border_mode in ('valid', 'full', 'half')):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a tuple of three'
                ' integers'.format(border_mode))
        self.border_mode = border_mode
        if len(subsample) != 3:
            raise ValueError("subsample must have three elements")
        if len(filter_dilation) != 3:
            raise ValueError("filter_dilation must have three elements")
        self.subsample = tuple(subsample)
        self.filter_dilation = tuple(filter_dilation)
        if num_groups < 1:
            raise ValueError("Number of groups should be greater than 0")
        self.num_groups = num_groups

        if not theano.config.blas.ldflags:
            # Theano will use a NumPy C implementation of [sd]gemm_ instead.
            self.blas_type = ''
        else:
            if 'openblas' in theano.config.blas.ldflags:
                self.blas_type = 'openblas'
            elif 'mkl' in theano.config.blas.ldflags:
                self.blas_type = 'mkl'
            else:
                self.blas_type = ''

        if self._direction not in ["forward", "backprop weights", "backprop inputs"]:
            raise ValueError("_direction must be one of 'forward', "
                             "'backprop weights', 'backprop inputs'")

    @property
    def pad(self):
        if self.border_mode == "half":
            return (-1, -1, -1)
        elif self.border_mode == "full":
            return (-2, -2, -2)
        elif isinstance(self.border_mode, tuple):
            return self.border_mode
        else:
            assert self.border_mode == "valid"
            return (0, 0, 0)

    # Direction should be converted to real enum value,
    # as it is compared to integer later in c_code_helper().
    direction = property(lambda self: self.params_type.enum_from_alias(self._direction))

    dH = property(lambda self: self.subsample[0])
    dW = property(lambda self: self.subsample[1])
    dD = property(lambda self: self.subsample[2])

    dilH = property(lambda self: self.filter_dilation[0])
    dilW = property(lambda self: self.filter_dilation[1])
    dilD = property(lambda self: self.filter_dilation[2])

    padH = property(lambda self: self.pad[0])
    padW = property(lambda self: self.pad[1])
    padD = property(lambda self: self.pad[2])

    def __str__(self):
        return '%s{%s, %s, %s, %s}' % (
            self.__class__.__name__,
            self.border_mode,
            str(self.subsample),
            str(self.filter_dilation),
            str(self.num_groups))

    @staticmethod
    def as_common_dtype(in1, in2):
        """
        Upcast input variables if necessary.
        """
        dtype = theano.scalar.upcast(in1.dtype, in2.dtype)
        return in1.astype(dtype), in2.astype(dtype)

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'num_groups'):
            self.num_groups = 1

    def c_support_code(self):
        ccodes = blas_headers.blas_header_text()
        if self.blas_type == 'openblas':
            ccodes += blas_headers.openblas_threads_text()
        elif self.blas_type == 'mkl':
            ccodes += blas_headers.mkl_threads_text()
        return ccodes

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(BaseCorr3dMM, self).c_compile_args()
        return compile_args

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_headers(self):
        headers = ['<stdio.h>']
        headers += super(BaseCorr3dMM, self).c_headers()
        return headers

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (8, self.openmp, blas_header_version())

    def c_support_code_apply(self, node, nodename):
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        sub = {}
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')
        if dtype == 'float32':
            sub['gemm'] = 'sgemm_'
            sub['float_type'] = 'npy_float'
            sub['float_typenum'] = 'NPY_FLOAT'
            sub['n_bytes'] = 4
            sub['c_float_type'] = 'float'
        else:
            sub['gemm'] = 'dgemm_'
            sub['float_type'] = 'npy_double'
            sub['float_typenum'] = 'NPY_DOUBLE'
            sub['n_bytes'] = 8
            sub['c_float_type'] = 'double'

        if self.openmp:
            sub['omp_flags'] = '#pragma omp parallel for schedule(static)'
            sub['omp_get_max_threads'] = 'omp_get_max_threads()'
            sub['omp_get_thread_num'] = 'omp_get_thread_num()'

            if self.blas_type == 'openblas':
                sub['blas_set_num_threads'] = 'openblas_set_num_threads'
                sub['blas_get_num_threads'] = 'openblas_get_num_threads()'
            elif self.blas_type == 'mkl':
                sub['blas_set_num_threads'] = 'mkl_set_num_threads'
                sub['blas_get_num_threads'] = 'mkl_get_max_threads()'
            else:
                sub['blas_set_num_threads'] = ''
                sub['blas_get_num_threads'] = '0'
        else:
            sub['omp_flags'] = ''
            sub['omp_get_max_threads'] = '1'
            sub['omp_get_thread_num'] = '0'
            sub['blas_set_num_threads'] = ''
            sub['blas_get_num_threads'] = '0'

        files = [os.path.join('c_code', 'corr3d_gemm.c')]
        codes = [open(os.path.join(os.path.split(__file__)[0], f)).read()
                 for f in files]
        final_code = ''
        for code in codes:
            final_code += code
        return final_code % sub

    def c_code_helper(self, bottom, weights, top, sub,
                      height=None, width=None, depth=None):
        """
        This generates the C code for Corr3dMM (direction="forward"),
        Corr3dMM_gradWeights (direction="backprop weights"), and
        Corr3dMM_gradInputs (direction="backprop inputs").
        Depending on the direction, one of bottom, weights, top will
        receive the output, while the other two serve as inputs.

        :param bottom: Variable name of the input images in the forward pass,
            or the gradient of the input images in backprop wrt. inputs
        :param weights: Variable name of the filters in the forward pass,
            or the gradient of the filters in backprop wrt. weights
        :param top: Variable name of the output images / feature maps in the
            forward pass, or the gradient of the outputs in the backprop passes
        :param sub: Dictionary of substitutions useable to help generating the
            C code.
        :param height: If self.subsample[0] != 1, a variable giving the height
            of the filters for direction="backprop weights" or the height of
            the input images for direction="backprop inputs".

            If self.border_mode == 'half', a variable giving the height of the
            filters for direction="backprop weights".  Ignored otherwise.
        :param width: If self.subsample[1] != 1, a variable giving the width
            of the filters for direction="backprop weights" or the width of the
            input images for direction="backprop inputs".

            If self.border_mode == 'half', a variable giving the width of the
            filters for direction="backprop weights".  Ignored otherwise.
        :param depth: If self.subsample[1] != 1, a variable giving the depth
            of the filters for direction="backprop weights" or the depth of the
            input images for direction="backprop inputs".

            If self.border_mode == 'half', a variable giving the depth of the
            filters for direction="backprop weights".  Ignored otherwise.
        """

        # When subsampling, we cannot unambiguously infer the height and width
        # of bottom and weights from top, so we require them to be given.
        # Similarly, when border_mode="half", we cannot infer the weight size.
        if height:
            height = '(*(npy_int64 *)(PyArray_DATA(%s)))' % height
        else:
            if ((self.direction != 0) and (self.dH != 1)) or ((self.direction == 1) and (self.padH == -1)):
                raise ValueError("height must be given for backprop with vertical sampling or border_mode='half'")
            height = '-1'
        if width:
            width = '(*(npy_int64 *)(PyArray_DATA(%s)))' % width
        else:
            if ((self.direction != 0) and (self.dW != 1)) or ((self.direction == 1) and (self.padW == -1)):
                raise ValueError("width must be given for backprop with horizontal sampling or border_mode='half'")
            width = '-1'
        if depth:
            depth = '(*(npy_int64 *)(PyArray_DATA(%s)))' % depth
        else:
            if ((self.direction != 0) and (self.dD != 1)) or ((self.direction == 1) and (self.padD == -1)):
                raise ValueError("depth must be given for backprop with depth sampling or border_mode='half'")
            depth = '-1'

        return """
    // Mandatory args
    int direction = %(params)s->direction;  // forward, bprop weights, bprop inputs

    // Optional args
    int dH = %(params)s->dH;
    int dW = %(params)s->dW;
    int dD = %(params)s->dD;
    int dilH = %(params)s->dilH;
    int dilW = %(params)s->dilW;
    int dilD = %(params)s->dilD;
    int padH = %(params)s->padH;
    int padW = %(params)s->padW;
    int padD = %(params)s->padD;
    int numgroups = %(params)s->num_groups;

    PyArrayObject * bottom = %(bottom)s;
    PyArrayObject * weights = %(weights)s;
    PyArrayObject * top = %(top)s;
    PyArrayObject * out2 = NULL;
    PyArrayObject **out = NULL;

    switch(%(params)s->direction) {
        case DIRECTION_FORWARD:
            out = &%(top)s;
            break;
        case DIRECTION_BACKPROP_WEIGHTS:
            out = &%(weights)s;
            break;
        case DIRECTION_BACKPROP_INPUTS:
            out = &%(bottom)s;
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "CPU Corr3dMM: Invalid direction.");
            {%(fail)s}
            break;
    }

    // Obtain or infer kernel width, height and depth
    // (we need to know it early to be able to handle auto-padding)
    int kH, kW, kD, dil_kH, dil_kW, dil_kD;
    if (direction != 1) {
        // weight is an input variable, we can just read its shape
        kH = PyArray_DIMS(weights)[2];
        kW = PyArray_DIMS(weights)[3];
        kD = PyArray_DIMS(weights)[4];
    }
    else {
        if (%(height)s != -1) {
            // kernel height is specified (perhaps vertical subsampling or half padding)
            kH = %(height)s;
        }
        else if (padH == -2) {
            // vertical full padding, we can infer the kernel height
            kH = (2 - PyArray_DIMS(bottom)[2] + (PyArray_DIMS(top)[2] - 1) * dH - 1)/ dilH + 1;
        }
        else {
            // explicit padding, we can infer the kernel height
            kH = (PyArray_DIMS(bottom)[2] + 2*padH - (PyArray_DIMS(top)[2] - 1) * dH - 1) / dilH +1;
        }
        if (%(width)s != -1) {
            kW = %(width)s;
        }
        else if (padW == -2) {
            kW = (2 - PyArray_DIMS(bottom)[3] + (PyArray_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
        }
        else {
            kW = (PyArray_DIMS(bottom)[3] + 2*padW - (PyArray_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
        }
        if (%(depth)s != -1) {
            kD = %(depth)s;
        }
        else if (padD == -2) {
            kD = (2 - PyArray_DIMS(bottom)[4] + (PyArray_DIMS(top)[4] - 1) * dD - 1) / dilD + 1;
        }
        else {
            kD = (PyArray_DIMS(bottom)[4] + 2*padD - (PyArray_DIMS(top)[4] - 1) * dD - 1) / dilD + 1;
        }
    }

    // Implicit dilated kernel size
    dil_kH = (kH - 1) * dilH + 1;
    dil_kW = (kW - 1) * dilW + 1;
    dil_kD = (kD - 1) * dilD + 1;

    // Auto-padding if requested
    if (padH == -1) {  // vertical half padding
        padH = dil_kH / 2;
    }
    else if (padH == -2) {  // vertical full padding
        padH = dil_kH - 1;
    }
    else if (padH < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseCorr3dMM: padH must be >= -2");
        %(fail)s
    }
    if (padW == -1) {  // horizontal half padding
        padW = dil_kW / 2;
    }
    else if (padW == -2) {  // horizontal full padding
        padW = dil_kW - 1;
    }
    else if (padW < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseCorr3dMM: padW must be >= -2");
        %(fail)s
    }
    if (padD == -1) {  // depth half padding
        padD = dil_kD / 2;
    }
    else if (padD == -2) {  // depth full padding
        padD = dil_kD - 1;
    }
    else if (padD < 0) {
        PyErr_SetString(PyExc_ValueError, "BaseCorr3dMM: padD must be >= -2");
        %(fail)s
    }

    // Infer output shape
    npy_intp out_dim[5];
    switch(direction) {
    case 0:  // forward pass
        // output is top: (batchsize, num_filters, height, width, depth)
        // height and width: top = (bottom + 2*pad - ((weight-1)*dil + 1)) / sample + 1
        out_dim[0] = (npy_intp)PyArray_DIMS(bottom)[0];
        out_dim[1] = (npy_intp)PyArray_DIMS(weights)[0];
        out_dim[2] = (npy_intp)((PyArray_DIMS(bottom)[2] + 2*padH - ((PyArray_DIMS(weights)[2]-1)*dilH + 1)) / dH + 1);
        out_dim[3] = (npy_intp)((PyArray_DIMS(bottom)[3] + 2*padW - ((PyArray_DIMS(weights)[3]-1)*dilW + 1)) / dW + 1);
        out_dim[4] = (npy_intp)((PyArray_DIMS(bottom)[4] + 2*padD - ((PyArray_DIMS(weights)[4]-1)*dilD + 1)) / dD + 1);
        if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0 || out_dim[4] <= 0)
        {
            PyErr_Format(PyExc_ValueError,
                         "Corr3dMM: impossible output shape\\n"
                         "  bottom shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                         "  weights shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                         "  top shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n",
                         (long int)PyArray_DIMS(bottom)[0], (long int)PyArray_DIMS(bottom)[1],
                         (long int)PyArray_DIMS(bottom)[2], (long int)PyArray_DIMS(bottom)[3],
                         (long int)PyArray_DIMS(bottom)[4],
                         (long int)PyArray_DIMS(weights)[0], (long int)PyArray_DIMS(weights)[1],
                         (long int)PyArray_DIMS(weights)[2], (long int)PyArray_DIMS(weights)[3],
                         (long int)PyArray_DIMS(weights)[4],
                         (long int)out_dim[0], (long int)out_dim[1], (long int)out_dim[2],
                         (long int)out_dim[3], (long int)out_dim[4]);
            %(fail)s
        }
        break;
    case 1:  // backprop wrt. weights
        // output is weights: (num_filters, num_channels, height, width, depth)
        // height and width: weights = (bottom + 2*pad - (top - 1) * sample - 1) / dil + 1
        out_dim[0] = (npy_intp)PyArray_DIMS(top)[1];
        out_dim[1] = (npy_intp)PyArray_DIMS(bottom)[1] / numgroups;
        out_dim[2] = (npy_intp)kH;  // already inferred further above
        out_dim[3] = (npy_intp)kW;  // how convenient
        out_dim[4] = (npy_intp)kD;
        if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0 || out_dim[4] <= 0)
        {
            PyErr_Format(PyExc_ValueError,
                         "Corr3dMM backprop wrt. weights: impossible output shape\\n"
                         "  bottom shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                         "  weights shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                         "  top shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n",
                         (long int)PyArray_DIMS(bottom)[0], (long int)PyArray_DIMS(bottom)[1],
                         (long int)PyArray_DIMS(bottom)[2], (long int)PyArray_DIMS(bottom)[3],
                         (long int)PyArray_DIMS(bottom)[4],
                         (long int)out_dim[0], (long int)out_dim[1], (long int)out_dim[2],
                         (long int)out_dim[3], (long int)out_dim[4],
                         (long int)PyArray_DIMS(top)[0], (long int)PyArray_DIMS(top)[1],
                         (long int)PyArray_DIMS(top)[2], (long int)PyArray_DIMS(top)[3],
                         (long int)PyArray_DIMS(top)[4]);
            %(fail)s
        }
        break;
    case 2:  // backprop wrt. inputs
        // output is bottom: (batchsize, num_channels, height, width, depth)
        // height and width: bottom = (top - 1) * sample + (weights-1)*dil + 1 - 2*pad
        out_dim[0] = (npy_intp)PyArray_DIMS(top)[0];
        out_dim[1] = (npy_intp)PyArray_DIMS(weights)[1] * numgroups;
        out_dim[2] = (npy_intp)((%(height)s != -1) ? %(height)s : (PyArray_DIMS(top)[2] - 1) * dH + (PyArray_DIMS(weights)[2]-1)*dilH + 1 - 2*padH);
        out_dim[3] = (npy_intp)((%(width)s != -1) ? %(width)s : (PyArray_DIMS(top)[3] - 1) * dW + (PyArray_DIMS(weights)[3]-1)*dilW + 1 - 2*padW);
        out_dim[4] = (npy_intp)((%(depth)s != -1) ? %(depth)s : (PyArray_DIMS(top)[4] - 1) * dD + (PyArray_DIMS(weights)[4]-1)*dilD + 1 - 2*padD);
        if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0 || out_dim[4] <= 0)
        {
            PyErr_Format(PyExc_ValueError,
                         "Corr3dMM backprop wrt. inputs: impossible output shape\\n"
                         "  bottom shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                         "  weights shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                         "  top shape: %%ld x %%ld x %%ld x %%ld x %%ld\\n",
                         (long int)out_dim[0], (long int)out_dim[1], (long int)out_dim[2],
                         (long int)out_dim[3], (long int)out_dim[4],
                         (long int)PyArray_DIMS(weights)[0], (long int)PyArray_DIMS(weights)[1],
                         (long int)PyArray_DIMS(weights)[2], (long int)PyArray_DIMS(weights)[3],
                         (long int)PyArray_DIMS(weights)[4],
                         (long int)PyArray_DIMS(top)[0], (long int)PyArray_DIMS(top)[1],
                         (long int)PyArray_DIMS(top)[2], (long int)PyArray_DIMS(top)[3],
                         (long int)PyArray_DIMS(top)[4]);
            %(fail)s
        }
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "BaseCorr3dMM: direction must be 0, 1, or 2\\n");
        %(fail)s
    }

    // Prepare output array
    int typenum;
    if ( !(*out
           && PyArray_NDIM(*out)==4
           && PyArray_IS_C_CONTIGUOUS(*out)
           && PyArray_DIMS(*out)[0]==out_dim[0]
           && PyArray_DIMS(*out)[1]==out_dim[1]
           && PyArray_DIMS(*out)[2]==out_dim[2]
           && PyArray_DIMS(*out)[3]==out_dim[3]
           && PyArray_DIMS(*out)[4]==out_dim[4]))
    {
        Py_XDECREF(*out);
        if (direction != 1) {
          typenum = PyArray_TYPE(weights);
        }
        else {
          typenum = PyArray_TYPE(bottom);
        }
        //Change to PyArray_ZEROS which is faster than PyArray_EMPTY.
        *out = (PyArrayObject*)PyArray_ZEROS(5,
                                          out_dim,
                                          typenum,
                                          0);
        if (NULL == *out)
        {
            PyErr_Format(PyExc_RuntimeError,
                    "BaseCorr3dMM: Failed to allocate output of %%lld x %%lld x %%lld x %%lld x %%lld",
                    (long long)out_dim[0], (long long)out_dim[1],
                    (long long)out_dim[2], (long long)out_dim[3], (long long)out_dim[4]);
            %(fail)s
        }
    }

    // Call corr3dMM code
    out2 = corr3dMM(%(bottom)s, %(weights)s, %(top)s, direction,
                    dH, dW, dD, dilH, dilW, dilD, padH, padW, padD,
                    numgroups);
    if (out2==NULL){
       %(fail)s
    }
    assert (out2 == *out);

""" % dict(bottom=bottom, weights=weights, top=top,
           height=height, width=width, depth=depth,
           fail=sub['fail'], params=sub['params'])


class Corr3dMM(BaseCorr3dMM):
    """
    CPU correlation implementation using Matrix Multiplication.

    Parameters
    ----------
    border_mode
        The width of a border of implicit zeros to pad the
        input with. Must be a tuple with 3 elements giving the width of
        the padding on each side, or a single integer to pad the same
        on all sides, or a string shortcut setting the padding at runtime:
        ``'valid'`` for ``(0, 0, 0)`` (valid convolution, no padding), ``'full'``
        for ``(kernel_rows - 1, kernel_columns - 1, kernel_depth - 1)``
        (full convolution), ``'half'`` for ``(kernel_rows // 2,
        kernel_columns // 2, kernel_depth // 2)`` (same convolution for
        odd-sized kernels). Note that the three widths are each
        applied twice, once per side (left and right, top and bottom, front
        and back).
    subsample
        The subsample operation applied to each output image. Should be a tuple
        with 3 elements. Set to `(1, 1, 1)` to disable subsampling.
    filter_dilation
        The filter dilation operation applied to each input image.
        Should be a tuple with 3 elements.
        Set to `(1, 1, 1)` to disable filter dilation.
    num_groups
        Perform grouped convolutions (default: 1)
    """

    _direction = "forward"

    def make_node(self, img, kern):
        img = as_tensor_variable(img)
        kern = as_tensor_variable(kern)
        img, kern = self.as_common_dtype(img, kern)
        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')
        if kern.type.ndim != 5:
            raise TypeError('kern must be 5D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False, False]
        dtype = img.type.dtype
        return Apply(self, [img, kern], [TensorType(dtype, broadcastable)()])

    def infer_shape(self, node, input_shape):
        imshp = input_shape[0]
        kshp = input_shape[1]
        res = get_conv_output_shape(
            imshp,
            kshp,
            self.border_mode,
            self.subsample,
            self.filter_dilation)
        return [res]

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, weights = inp
        top, = out_
        return super(Corr3dMM, self).c_code_helper(bottom, weights, top, sub)

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        d_bottom = Corr3dMM_gradInputs(self.border_mode,
                                       self.subsample,
                                       self.filter_dilation,
                                       num_groups=self.num_groups)(weights, top,
                                                                   bottom.shape[-3:])
        d_weights = Corr3dMM_gradWeights(self.border_mode,
                                         self.subsample,
                                         self.filter_dilation,
                                         num_groups=self.num_groups)(bottom, top,
                                                                     weights.shape[-3:])
        return d_bottom, d_weights


class Corr3dMM_gradWeights(BaseCorr3dMM):
    """
    Gradient wrt. filters for `Corr3dMM`.

    Notes
    -----
    You will not want to use this directly, but rely on
    Theano's automatic differentiation or graph optimization to
    use it as needed.

    """

    _direction = "backprop weights"

    def make_node(self, img, topgrad, shape=None):
        img = as_tensor_variable(img)
        topgrad = as_tensor_variable(topgrad)
        img, topgrad = self.as_common_dtype(img, topgrad)
        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')
        if topgrad.type.ndim != 5:
            raise TypeError('topgrad must be 5D tensor')
        if shape is None:
            if self.subsample != (1, 1, 1) or self.border_mode == "half":
                raise ValueError('shape must be given if subsample != (1, 1, 1)'
                                 ' or border_mode == "half"')
            height_width_depth = []
        else:
            height_width_depth = [as_tensor_variable(shape[0]).astype('int64'),
                                  as_tensor_variable(shape[1]).astype('int64'),
                                  as_tensor_variable(shape[2]).astype('int64')]

        broadcastable = [topgrad.type.broadcastable[1], img.type.broadcastable[1],
                         False, False, False]
        dtype = img.type.dtype
        return Apply(self, [img, topgrad] + height_width_depth,
                     [TensorType(dtype, broadcastable)()])

    def infer_shape(self, node, input_shape):
        if self.border_mode == "half":
            padH = padW = padD = -1
        elif self.border_mode == "full":
            padH = padW = padD = -2
        elif isinstance(self.border_mode, tuple):
            padH, padW, padD = self.border_mode
        else:
            assert self.border_mode == "valid"
            padH = padW = padD = 0
        dH, dW, dD = self.subsample
        imshp = input_shape[0]
        topshp = input_shape[1]
        ssize, imshp = imshp[1], list(imshp[2:])
        ssize = ssize // self.num_groups
        nkern, topshp = topshp[1], list(topshp[2:])
        height_width_depth = node.inputs[-3:]
        if ((dH != 1) or (padH == -1)):
            # vertical subsampling or half padding, kernel height is specified
            kH = height_width_depth[0]
        elif padH == -2:
            # vertical full padding, we can infer the kernel height
            kH = 2 - imshp[0] + (topshp[0] - 1) * dH
        else:
            # explicit padding, we can infer the kernel height
            kH = imshp[0] + 2 * padH - (topshp[0] - 1) * dH
        if ((dW != 1) or (padW == -1)):
            kW = height_width_depth[1]
        elif (padW == -2):
            kW = 2 - imshp[1] + (topshp[1] - 1) * dW
        else:
            kW = imshp[1] + 2 * padW - (topshp[1] - 1) * dW
        if ((dD != 1) or (padD == -1)):
            kD = height_width_depth[2]
        elif (padD == -2):
            kD = 2 - imshp[2] + (topshp[2] - 1) * dD
        else:
            kD = imshp[2] + 2 * padD - (topshp[2] - 1) * dD
        return [(nkern, ssize, kH, kW, kD)]

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, top = inp[:2]
        height, width, depth = inp[2:] or (None, None, None)
        weights, = out_
        return super(Corr3dMM_gradWeights,
                     self).c_code_helper(bottom, weights, top,
                                         sub, height, width, depth)

    def grad(self, inp, grads):
        bottom, top = inp[:2]
        weights, = grads
        d_bottom = Corr3dMM_gradInputs(self.border_mode,
                                       self.subsample,
                                       self.filter_dilation,
                                       num_groups=self.num_groups)(weights, top,
                                                                   bottom.shape[-3:])
        d_top = Corr3dMM(self.border_mode,
                         self.subsample,
                         self.filter_dilation,
                         num_groups=self.num_groups)(bottom, weights)
        d_height_width_depth = ((theano.gradient.DisconnectedType()(),) * 3
                                if len(inp) == 5 else ())
        return (d_bottom, d_top) + d_height_width_depth

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0], [0]]  # no connection to height, width, depth


class Corr3dMM_gradInputs(BaseCorr3dMM):
    """
    Gradient wrt. inputs for `Corr3dMM`.

    Notes
    -----
    You will not want to use this directly, but rely on
    Theano's automatic differentiation or graph optimization to
    use it as needed.

    """

    _direction = "backprop inputs"

    def make_node(self, kern, topgrad, shape=None):
        kern = as_tensor_variable(kern)
        topgrad = as_tensor_variable(topgrad)
        kern, topgrad = self.as_common_dtype(kern, topgrad)
        if kern.type.ndim != 5:
            raise TypeError('kern must be 5D tensor')
        if topgrad.type.ndim != 5:
            raise TypeError('topgrad must be 5D tensor')
        if shape is None:
            if self.subsample != (1, 1, 1):
                raise ValueError('shape must be given if subsample != (1, 1, 1)')
            height_width_depth = []
        else:
            height_width_depth = [as_tensor_variable(shape[0]).astype('int64'),
                                  as_tensor_variable(shape[1]).astype('int64'),
                                  as_tensor_variable(shape[2]).astype('int64')]

        if self.num_groups > 1:
            broadcastable = [topgrad.type.broadcastable[0], False,
                             False, False, False]
        else:
            broadcastable = [topgrad.type.broadcastable[0], kern.type.broadcastable[1],
                             False, False, False]
        dtype = kern.type.dtype
        return Apply(self, [kern, topgrad] + height_width_depth,
                     [TensorType(dtype, broadcastable)()])

    def infer_shape(self, node, input_shape):
        if self.border_mode == "half":
            padH = padW = padD = -1
        elif self.border_mode == "full":
            padH = padW = padD = -2
        elif isinstance(self.border_mode, tuple):
            padH, padW, padD = self.border_mode
        else:
            assert self.border_mode == "valid"
            padH = padW = padD = 0
        dH, dW, dD = self.subsample
        kshp = input_shape[0]
        topshp = input_shape[1]
        ssize, kshp = kshp[1], list(kshp[2:])
        ssize = ssize * self.num_groups
        bsize, topshp = topshp[0], list(topshp[2:])
        height_width_depth = node.inputs[-3:]
        if padH == -1:
            padH = kshp[0] // 2
        elif padH == -2:
            padH = kshp[0] - 1
        elif padH < -2:
            raise ValueError('Corr3dMM_gradInputs: border_mode must be >= 0.')
        if padW == -1:
            padW = kshp[1] // 2
        elif padW == -2:
            padW = kshp[1] - 1
        elif padW < -2:
            raise ValueError('Corr3dMM_gradInputs: border_mode must be >= 0.')
        if padD == -1:
            padD = kshp[2] // 2
        elif padD == -2:
            padD = kshp[2] - 1
        elif padD < -2:
            raise ValueError('Corr3dMM_gradInputs: border_mode must be >= 0.')

        if dH != 1:
            out_shp0 = height_width_depth[0]
        else:
            out_shp0 = (topshp[0] - 1) * dH + kshp[0] - 2 * padH
        if dW != 1:
            out_shp1 = height_width_depth[1]
        else:
            out_shp1 = (topshp[1] - 1) * dW + kshp[1] - 2 * padW
        if dD != 1:
            out_shp2 = height_width_depth[2]
        else:
            out_shp2 = (topshp[2] - 1) * dD + kshp[2] - 2 * padD
        out_shp = (out_shp0, out_shp1, out_shp2)
        return [(bsize, ssize) + out_shp]

    def c_code(self, node, nodename, inp, out_, sub):
        weights, top = inp[:2]
        height, width, depth = inp[2:] or (None, None, None)
        bottom, = out_
        return super(Corr3dMM_gradInputs,
                     self).c_code_helper(bottom, weights, top, sub,
                                         height, width, depth)

    def grad(self, inp, grads):
        weights, top = inp[:2]
        bottom, = grads
        d_weights = Corr3dMM_gradWeights(self.border_mode,
                                         self.subsample,
                                         self.filter_dilation,
                                         num_groups=self.num_groups)(bottom,
                                                                     top,
                                                                     weights.shape[-3:])
        d_top = Corr3dMM(self.border_mode,
                         self.subsample,
                         self.filter_dilation,
                         num_groups=self.num_groups)(bottom, weights)
        d_height_width_depth = ((theano.gradient.DisconnectedType()(),) * 3
                                if len(inp) == 5 else ())
        return (d_weights, d_top) + d_height_width_depth

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0], [0]]  # no connection to height, width, depth
