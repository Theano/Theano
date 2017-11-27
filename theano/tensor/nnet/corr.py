from __future__ import absolute_import, print_function, division
import os
import logging

from six import integer_types

import theano
from theano import Apply
from theano import gof
from theano.gof import ParamsType, EnumList
from theano.scalar import int64, int8
from theano.tensor import as_tensor_variable, TensorType
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor import blas_headers
from theano.tensor.blas import ldflags, blas_header_version

_logger = logging.getLogger(__name__)


class BaseCorrMM(gof.OpenMPOp):
    """
    Base class for `CorrMM`, `CorrMM_gradWeights` and
    `CorrMM_gradInputs`. Cannot be used directly.

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
        Additionally, the padding size could be directly specified by an integer,
        a pair of integers, or two pairs of integers.
    subsample
        Perform subsampling of the output (default: (1, 1)).
    filter_dilation
        Perform dilated correlation (default: (1,1))
    num_groups
        Perform grouped convolutions (default: 1)
    unshared
        Perform unshared correlation (default: False)
    """
    check_broadcast = False
    __props__ = ('border_mode', 'subsample', 'filter_dilation', 'num_groups', 'unshared')

    _direction = None

    params_type = ParamsType(direction=EnumList(('DIRECTION_FORWARD', 'forward'),  # 0
                                                ('DIRECTION_BACKPROP_WEIGHTS', 'backprop weights'),  # 1
                                                ('DIRECTION_BACKPROP_INPUTS', 'backprop inputs')),  # 2
                             dH=int64, dW=int64,
                             dilH=int64, dilW=int64,
                             padH_l=int64, padH_r=int64,
                             padW_l=int64, padW_r=int64,
                             num_groups=int64, unshared=int8)

    def __init__(self, border_mode="valid", subsample=(1, 1),
                 filter_dilation=(1, 1), num_groups=1, unshared=False, openmp=None):
        super(BaseCorrMM, self).__init__(openmp=openmp)
        if isinstance(border_mode, integer_types):
            if border_mode < 0:
                raise ValueError(
                    'invalid border_mode {}, which must be a '
                    'non-negative integer'.format(border_mode))
            border_mode = ((border_mode, border_mode),) * 2
        elif isinstance(border_mode, tuple):
            if len(border_mode) != 2:
                raise ValueError(
                    'invalid border_mode {} which must be a '
                    'tuple of length 2'.format(border_mode))
            border = ()
            for mode in border_mode:
                if isinstance(mode, tuple) and len(mode) == 2 and \
                        min(mode) >= 0:
                    border += ((int(mode[0]), int(mode[1])),)
                elif mode >= 0:
                    border += ((int(mode), int(mode)),)
                else:
                    raise ValueError(
                        'invalid border mode {}. The tuple can only contain '
                        'integers or tuples of length 2'.format(border_mode))
            border_mode = border
        elif border_mode not in ('valid', 'full', 'half'):
            raise ValueError(
                'invalid border_mode {}, which must be either '
                '"valid", "full", "half", an integer or a tuple '
                'of two integers or a pair of integers'.format(border_mode))
        self.border_mode = border_mode
        if len(subsample) != 2:
            raise ValueError("subsample must have two elements")
        if len(filter_dilation) != 2:
            raise ValueError("filter_dilation must have two elements")
        self.subsample = tuple(subsample)
        self.filter_dilation = tuple(filter_dilation)
        self.unshared = unshared

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
        if num_groups < 1:
            raise ValueError("Number of groups should be greater than 0")
        self.num_groups = num_groups

    @property
    def pad(self):
        if self.border_mode == "half":
            return ((-1, -1),) * 2
        elif self.border_mode == "full":
            return ((-2, -2),) * 2
        elif isinstance(self.border_mode, tuple):
            return self.border_mode
        else:
            assert self.border_mode == "valid"
            return ((0, 0),) * 2

    # Direction should be converted to real enum value,
    # as it is compared to integer later in c_code_helper().
    direction = property(lambda self: self.params_type.enum_from_alias(self._direction))

    dH = property(lambda self: self.subsample[0])
    dW = property(lambda self: self.subsample[1])

    dilH = property(lambda self: self.filter_dilation[0])
    dilW = property(lambda self: self.filter_dilation[1])

    padH_l = property(lambda self: self.pad[0][0])
    padH_r = property(lambda self: self.pad[0][1])
    padW_l = property(lambda self: self.pad[1][0])
    padW_r = property(lambda self: self.pad[1][1])

    def __str__(self):
        return '%s{%s, %s, %s, %s %s}' % (
            self.__class__.__name__,
            self.border_mode,
            str(self.subsample),
            str(self.filter_dilation),
            str(self.num_groups),
            str(self.unshared))

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
        compile_args += super(BaseCorrMM, self).c_compile_args()
        return compile_args

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_headers(self):
        headers = ['<stdio.h>']
        headers += super(BaseCorrMM, self).c_headers()
        return headers

    def c_code_cache_version(self):
        # raise this whenever modifying any of the support_code_files
        return (10, self.openmp, blas_header_version())

    def c_support_code_apply(self, node, nodename):
        # REMEMBER TO RAISE c_code_cache_version when changing any of
        # these files
        sub = {}
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')
        if dtype == 'float32':
            sub['gemm'] = 'sgemm_'
            sub['gemv'] = 'sgemv_'
            sub['float_type'] = 'npy_float'
            sub['float_typenum'] = 'NPY_FLOAT'
            sub['n_bytes'] = 4
            sub['c_float_type'] = 'float'
        else:
            sub['gemm'] = 'dgemm_'
            sub['gemv'] = 'dgemv_'
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

        files = [os.path.join('c_code', 'corr_gemm.c')]
        codes = [open(os.path.join(os.path.split(__file__)[0], f)).read()
                 for f in files]
        final_code = ''
        for code in codes:
            final_code += code
        return final_code % sub

    def c_code_helper(self, bottom, weights, top, sub, height=None, width=None):
        """
        This generates the C code for CorrMM (direction="forward"),
        CorrMM_gradWeights (direction="backprop weights"), and
        CorrMM_gradInputs (direction="backprop inputs").
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
        """

        # When subsampling, we cannot unambiguously infer the height and width
        # of bottom and weights from top, so we require them to be given.
        # Similarly, when border_mode="half", we cannot infer the weight size.
        if height:
            height = '(*(npy_int64 *)(PyArray_DATA(%s)))' % height
        else:
            if ((self.direction != 0) and (self.dH != 1)) or ((self.direction == 1) and (self.padH_l == -1 or self.padH_r == -1)):
                raise ValueError("height must be given for backprop with vertical sampling or border_mode='half'")
            height = '-1'
        if width:
            width = '(*(npy_int64 *)(PyArray_DATA(%s)))' % width
        else:
            if ((self.direction != 0) and (self.dW != 1)) or ((self.direction == 1) and (self.padW_l == -1 or self.padW_r == -1)):
                raise ValueError("width must be given for backprop with horizontal sampling or border_mode='half'")
            width = '-1'

        return """
    // Mandatory args
    int direction = %(params)s->direction;  // forward, bprop weights, bprop inputs

    // Optional args
    int dH = %(params)s->dH;
    int dW = %(params)s->dW;
    int dilH = %(params)s->dilH;
    int dilW = %(params)s->dilW;
    int padH_l = %(params)s->padH_l;
    int padH_r = %(params)s->padH_r;
    int padW_l = %(params)s->padW_l;
    int padW_r = %(params)s->padW_r;
    int numgroups = %(params)s->num_groups;
    int unshared = %(params)s->unshared;

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
            PyErr_SetString(PyExc_ValueError, "CPU CorrMM: Invalid direction.");
            {%(fail)s}
            break;
    }

    int wdim, odim;
    wdim = unshared ? 6 : 4;
    odim = 4; //Can be set to 6 later for unshared backprop wrt weights

    // Obtain or infer kernel width and height
    // (we need to know it early to be able to handle auto-padding)
    int kH, kW, dil_kH, dil_kW;
    if (direction != 1) {
        // weight is an input variable, we can just read its shape
        kH = PyArray_DIMS(weights)[wdim-2];
        kW = PyArray_DIMS(weights)[wdim-1];
    }
    else {
        if (%(height)s != -1) {
            // kernel height is specified (perhaps vertical subsampling or half padding)
            kH = %(height)s;
        }
        else if (padH_l == -2 || padH_r == -2) {
            // vertical full padding, we can infer the kernel height
            kH = (2 - PyArray_DIMS(bottom)[2] + (PyArray_DIMS(top)[2] - 1) * dH - 1)/ dilH + 1;
        }
        else {
            // explicit padding, we can infer the kernel height
            kH = (PyArray_DIMS(bottom)[2] + padH_l + padH_r - (PyArray_DIMS(top)[2] - 1) * dH - 1) / dilH +1;
        }
        if (%(width)s != -1) {
            // kernel width is specified (perhaps horizontal subsampling or half padding)
            kW = %(width)s;
        }
        else if (padW_l == -2 || padW_r == -2) {
            kW = (2 - PyArray_DIMS(bottom)[3] + (PyArray_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
        }
        else {
            kW = (PyArray_DIMS(bottom)[3] + padW_l + padW_r - (PyArray_DIMS(top)[3] - 1) * dW - 1) / dilW + 1;
        }
    }

    // Implicit dilated kernel size
    dil_kH = (kH - 1) * dilH + 1;
    dil_kW = (kW - 1) * dilW + 1;

    // Auto-padding if requested
    if (padH_l == -1 || padH_r == -1) {  // vertical half padding
        padH_l = padH_r = dil_kH / 2;
    }
    else if (padH_l == -2 || padH_r == -2) {  // vertical full padding
        padH_l = padH_r = dil_kH - 1;
    }
    else if (padH_l < -2 || padH_r < -2) {
        PyErr_SetString(PyExc_ValueError, "BaseCorrMM: padH_l and padH_r must be >= -2");
        %(fail)s
    }
    if (padW_l == -1 || padW_r == -1) {  // horizontal half padding
        padW_l = padW_r = dil_kW / 2;
    }
    else if (padW_l == -2 || padW_r == -2) {  // horizontal full padding
        padW_l = padW_r = dil_kW - 1;
    }
    else if (padW_l < -2 || padW_r < -2) {
        PyErr_SetString(PyExc_ValueError, "BaseCorrMM: padW_l and padW_r must be >= -2");
        %(fail)s
    }

    // Infer output shape
    npy_intp out_dim[6];
    out_dim[4] = out_dim[5] = 0; //Only used for unshared backprop wrt weights
    switch(direction) {
    case 0:  // forward pass
        // output is top: (batchsize, num_filters, height, width)
        // height and width: top = (bottom + pad_l + pad_r - ((weight-1)*dil + 1)) / sample + 1
        out_dim[0] = (npy_intp)PyArray_DIMS(bottom)[0];
        out_dim[1] = (npy_intp)PyArray_DIMS(weights)[0];
        out_dim[2] = (npy_intp)((PyArray_DIMS(bottom)[2] + padH_l + padH_r - ((PyArray_DIMS(weights)[wdim-2]-1)*dilH + 1)) / dH + 1);
        out_dim[3] = (npy_intp)((PyArray_DIMS(bottom)[3] + padW_l + padW_r - ((PyArray_DIMS(weights)[wdim-1]-1)*dilW + 1)) / dW + 1);
        if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0)
        {
            if (unshared) {
                PyErr_Format(PyExc_ValueError,
                             "CorrMM: impossible output shape\\n"
                             "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  weights shape: %%ld x %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                             "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                             (long int)PyArray_DIMS(bottom)[0], (long int)PyArray_DIMS(bottom)[1],
                             (long int)PyArray_DIMS(bottom)[2], (long int)PyArray_DIMS(bottom)[3],
                             (long int)PyArray_DIMS(weights)[0], (long int)PyArray_DIMS(weights)[1],
                             (long int)PyArray_DIMS(weights)[2], (long int)PyArray_DIMS(weights)[3],
                             (long int)PyArray_DIMS(weights)[4], (long int)PyArray_DIMS(weights)[5],
                             (long int)out_dim[0], (long int)out_dim[1], (long int)out_dim[2],
                             (long int)out_dim[3]);
            }
            else {
                PyErr_Format(PyExc_ValueError,
                             "CorrMM: impossible output shape\\n"
                             "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  weights shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                             (long int)PyArray_DIMS(bottom)[0], (long int)PyArray_DIMS(bottom)[1],
                             (long int)PyArray_DIMS(bottom)[2], (long int)PyArray_DIMS(bottom)[3],
                             (long int)PyArray_DIMS(weights)[0], (long int)PyArray_DIMS(weights)[1],
                             (long int)PyArray_DIMS(weights)[2], (long int)PyArray_DIMS(weights)[3],
                             (long int)out_dim[0], (long int)out_dim[1], (long int)out_dim[2],
                             (long int)out_dim[3]);
            }
            %(fail)s
        }
        break;
    case 1:  // backprop wrt. weights
        // output is weights: (num_filters, num_channels, height, width)
        // height and width: weights = (bottom + pad_l + pad_r - (top - 1) * sample - 1) / dil + 1
        out_dim[0] = (npy_intp)PyArray_DIMS(top)[1];
        if (unshared){
            odim = 6;
            out_dim[1] = (npy_intp)PyArray_DIMS(top)[2];
            out_dim[2] = (npy_intp)PyArray_DIMS(top)[3];
        }
        out_dim[wdim-3] = (npy_intp)PyArray_DIMS(bottom)[1] / numgroups;
        out_dim[wdim-2] = (npy_intp)kH;  // already inferred further above
        out_dim[wdim-1] = (npy_intp)kW;  // how convenient
        if (unshared) {
            if (out_dim[0] < 0 || out_dim[1] <= 0 || out_dim[2] <= 0 || out_dim[3] < 0
                    || out_dim[4] <= 0 || out_dim[5] <= 0){
                PyErr_Format(PyExc_ValueError,
                             "CorrMM backprop wrt. weights: impossible output shape\\n"
                             "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  weights shape: %%ld x %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                             "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                             (long int)PyArray_DIMS(bottom)[0], (long int)PyArray_DIMS(bottom)[1],
                             (long int)PyArray_DIMS(bottom)[2], (long int)PyArray_DIMS(bottom)[3],
                             (long int)out_dim[0], (long int)out_dim[1], (long int)out_dim[2],
                             (long int)out_dim[3], (long int)out_dim[4], (long int)out_dim[5],
                             (long int)PyArray_DIMS(top)[0], (long int)PyArray_DIMS(top)[1],
                             (long int)PyArray_DIMS(top)[2], (long int)PyArray_DIMS(top)[3]);
                %(fail)s
            }
        }
        else {
            if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0)
            {
                PyErr_Format(PyExc_ValueError,
                             "CorrMM backprop wrt. weights: impossible output shape\\n"
                             "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  weights shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                             (long int)PyArray_DIMS(bottom)[0], (long int)PyArray_DIMS(bottom)[1],
                             (long int)PyArray_DIMS(bottom)[2], (long int)PyArray_DIMS(bottom)[3],
                             (long int)out_dim[0], (long int)out_dim[1], (long int)out_dim[2],
                             (long int)out_dim[3],
                             (long int)PyArray_DIMS(top)[0], (long int)PyArray_DIMS(top)[1],
                             (long int)PyArray_DIMS(top)[2], (long int)PyArray_DIMS(top)[3]);
                %(fail)s
            }
        }
        break;
    case 2:  // backprop wrt. inputs
        // output is bottom: (batchsize, num_channels, height, width)
        // height and width: bottom = (top - 1) * sample + (weights-1)*dil + 1 - 2*pad
        out_dim[0] = (npy_intp)PyArray_DIMS(top)[0];
        out_dim[1] = (npy_intp)PyArray_DIMS(weights)[wdim-3] * numgroups;
        out_dim[2] = (npy_intp)((%(height)s != -1) ? %(height)s : (PyArray_DIMS(top)[2] - 1) * dH + (PyArray_DIMS(weights)[wdim-2]-1)*dilH + 1 - padH_l - padH_r);
        out_dim[3] = (npy_intp)((%(width)s != -1) ? %(width)s : (PyArray_DIMS(top)[3] - 1) * dW + (PyArray_DIMS(weights)[wdim-1]-1)*dilW + 1 - padW_l - padW_r);
        if (unshared) {
            if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0)
            {
                PyErr_Format(PyExc_ValueError,
                             "CorrMM backprop wrt. inputs: impossible output shape\\n"
                             "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  weights shape: %%ld x %%ld x %%ld x %%ld x %%ld x %%ld\\n"
                             "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                             (long int)out_dim[0], (long int)out_dim[1], (long int)out_dim[2],
                             (long int)out_dim[3],
                             (long int)PyArray_DIMS(weights)[0], (long int)PyArray_DIMS(weights)[1],
                             (long int)PyArray_DIMS(weights)[2], (long int)PyArray_DIMS(weights)[3],
                             (long int)PyArray_DIMS(weights)[4], (long int)PyArray_DIMS(weights)[5],
                             (long int)PyArray_DIMS(top)[0], (long int)PyArray_DIMS(top)[1],
                             (long int)PyArray_DIMS(top)[2], (long int)PyArray_DIMS(top)[3]);
                %(fail)s
            }
        }
        else {
            if (out_dim[0] < 0 || out_dim[1] < 0 || out_dim[2] <= 0 || out_dim[3] <= 0)
            {
                PyErr_Format(PyExc_ValueError,
                             "CorrMM backprop wrt. inputs: impossible output shape\\n"
                             "  bottom shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  weights shape: %%ld x %%ld x %%ld x %%ld\\n"
                             "  top shape: %%ld x %%ld x %%ld x %%ld\\n",
                             (long int)out_dim[0], (long int)out_dim[1], (long int)out_dim[2],
                             (long int)out_dim[3],
                             (long int)PyArray_DIMS(weights)[0], (long int)PyArray_DIMS(weights)[1],
                             (long int)PyArray_DIMS(weights)[2], (long int)PyArray_DIMS(weights)[3],
                             (long int)PyArray_DIMS(top)[0], (long int)PyArray_DIMS(top)[1],
                             (long int)PyArray_DIMS(top)[2], (long int)PyArray_DIMS(top)[3]);
                %(fail)s
            }
        }
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "BaseCorrMM: direction must be 0, 1, or 2\\n");
        %(fail)s
    }

    // Prepare output array
    int typenum;
    int failure;
    failure = !(*out
           && PyArray_NDIM(*out)==odim
           && PyArray_IS_C_CONTIGUOUS(*out)
           && PyArray_DIMS(*out)[0]==out_dim[0]
           && PyArray_DIMS(*out)[1]==out_dim[1]
           && PyArray_DIMS(*out)[2]==out_dim[2]
           && PyArray_DIMS(*out)[3]==out_dim[3]);
    if (odim == 6){
        failure = failure || !(PyArray_DIMS(*out)[4]==out_dim[4]
                && PyArray_DIMS(*out)[5]==out_dim[5]);
    }
    if ( failure )
    {
        Py_XDECREF(*out);
        if (direction != 1) {
          typenum = PyArray_TYPE(weights);
        }
        else {
          typenum = PyArray_TYPE(bottom);
        }
        //Change to PyArray_ZEROS which is faster than PyArray_EMPTY.
        *out = (PyArrayObject*)PyArray_ZEROS(odim,
                                          out_dim,
                                          typenum,
                                          0);
        if (NULL == *out)
        {
            if (odim == 4) {
                PyErr_Format(PyExc_RuntimeError,
                        "BaseCorrMM: Failed to allocate output of %%lld x %%lld x %%lld x %%lld",
                        (long long)out_dim[0], (long long)out_dim[1], (long long)out_dim[2], (long long)out_dim[3]);
            }
            if (odim == 6) {
                PyErr_Format(PyExc_RuntimeError,
                        "BaseCorrMM: Failed to allocate output of %%lld x %%lld x %%lld x %%lld %%lld %%lld",
                        (long long)out_dim[0], (long long)out_dim[1], (long long)out_dim[2], (long long)out_dim[3],
                        (long long)out_dim[4], (long long)out_dim[5]);
            }
            %(fail)s
        }
    }

    // Call corrMM code
    out2 = corrMM(%(bottom)s, %(weights)s, %(top)s, direction, dH, dW, dilH, dilW,
                padH_l, padH_r, padW_l, padW_r, numgroups, unshared);
    if (out2==NULL){
       %(fail)s
    }
    assert (out2 == *out);

""" % dict(bottom=bottom, weights=weights, top=top, height=height, width=width,
           fail=sub['fail'], params=sub['params'])


class CorrMM(BaseCorrMM):
    """
    CPU correlation implementation using Matrix Multiplication.

    Parameters
    ----------
    border_mode
        The width of a border of implicit zeros to pad the
        input with. Must be a tuple with 2 elements giving the numbers of rows
        and columns to pad on each side, or a single integer to pad the same
        on all sides, or a string shortcut setting the padding at runtime:
        ``'valid'`` for ``(0, 0)`` (valid convolution, no padding), ``'full'``
        for ``(kernel_rows - 1, kernel_columns - 1)`` (full convolution),
        ``'half'`` for ``(kernel_rows // 2, kernel_columns // 2)`` (same
        convolution for odd-sized kernels).
        If it is a tuple containing 2 pairs of integers, then these specify
        the padding to be applied on each side ((left, right), (top, bottom)).
        Otherwise, each width is applied twice, once per side (left and right,
        top and bottom).
    subsample
        The subsample operation applied to each output image.
        Should be a tuple with 2 elements.
        `(sv, sh)` is equivalent to `CorrMM(...)(...)[:,:,::sv, ::sh]`,
        but faster.
        Set to `(1, 1)` to disable subsampling.
    filter_dilation
        The filter dilation operation applied to each input image.
        Should be a tuple with 2 elements.
        Set to `(1, 1)` to disable filter dilation.
    num_groups
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately.
        Should be an integer.
    unshared
        Boolean value. If true, then a different filter will be applied to
        each region of the input image.

    """

    _direction = "forward"

    def make_node(self, img, kern):
        img = as_tensor_variable(img)
        kern = as_tensor_variable(kern)
        img, kern = self.as_common_dtype(img, kern)
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if self.unshared is True:
            if kern.type.ndim != 6:
                raise TypeError('kern must be 6D tensor')
        else:
            if kern.type.ndim != 4:
                raise TypeError('kern must be 4D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0],
                         False, False]
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
        return super(CorrMM, self).c_code_helper(bottom, weights, top, sub)

    def grad(self, inp, grads):
        bottom, weights = inp
        top, = grads
        d_bottom = CorrMM_gradInputs(self.border_mode,
                                     self.subsample,
                                     self.filter_dilation,
                                     self.num_groups,
                                     self.unshared)(weights, top,
                                                    bottom.shape[-2:])
        d_weights = CorrMM_gradWeights(self.border_mode,
                                       self.subsample,
                                       self.filter_dilation,
                                       self.num_groups,
                                       self.unshared)(bottom, top,
                                                      weights.shape[-2:])
        return d_bottom, d_weights


class CorrMM_gradWeights(BaseCorrMM):
    """
    Gradient wrt. filters for `CorrMM`.

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
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if shape is None:
            if self.subsample != (1, 1) or self.border_mode == "half":
                raise ValueError('shape must be given if subsample != (1, 1)'
                                 ' or border_mode == "half"')
            height_width = []
        else:
            height_width = [as_tensor_variable(shape[0]).astype('int64'), as_tensor_variable(shape[1]).astype('int64')]

        if self.unshared is True:
            broadcastable = [topgrad.type.broadcastable[1], False, False,
                             img.type.broadcastable[1], False, False]
        else:
            broadcastable = [topgrad.type.broadcastable[1], img.type.broadcastable[1],
                             False, False]
        dtype = img.type.dtype
        return Apply(self, [img, topgrad] + height_width,
                     [TensorType(dtype, broadcastable)()])

    def infer_shape(self, node, input_shape):
        if self.border_mode == "half":
            padH_l = padH_r = padW_l = padW_r = -1
        elif self.border_mode == "full":
            padH_l = padH_r = padW_l = padW_r = -2
        elif isinstance(self.border_mode, tuple):
            border = ()
            for mode in self.border_mode:
                if isinstance(mode, tuple):
                    border += ((int(mode[0]), int(mode[1])),)
                else:
                    border += ((int(mode), int(mode)),)
            (padH_l, padH_r), (padW_l, padW_r) = border
        else:
            assert self.border_mode == "valid"
            padH_l = padH_r = padW_l = padW_r = 0
        dH, dW = self.subsample
        imshp = input_shape[0]
        topshp = input_shape[1]
        ssize, imshp = imshp[1], list(imshp[2:])
        ssize = ssize // self.num_groups
        nkern, topshp = topshp[1], list(topshp[2:])
        height_width = node.inputs[-2:]
        if ((dH != 1) or (padH_l == -1) or (padH_r == -1)):
            # vertical subsampling or half padding, kernel height is specified
            kH = height_width[0]
        elif (padH_l == -2) or (padH_r == -2):
            # vertical full padding, we can infer the kernel height
            kH = 2 - imshp[0] + (topshp[0] - 1) * dH
        else:
            # explicit padding, we can infer the kernel height
            kH = imshp[0] + padH_l + padH_r - (topshp[0] - 1) * dH
        if ((dW != 1) or (padW_l == -1) or (padW_r == -1)):
            kW = height_width[1]
        elif (padW_l == -2) or (padW_r == -2):
            kW = 2 - imshp[1] + (topshp[1] - 1) * dW
        else:
            kW = imshp[1] + padW_l + padW_r - (topshp[1] - 1) * dW
        if self.unshared is True:
            return [(nkern, topshp[0], topshp[1], ssize, kH, kW)]
        else:
            return [(nkern, ssize, kH, kW)]

    def c_code(self, node, nodename, inp, out_, sub):
        bottom, top = inp[:2]
        height, width = inp[2:] or (None, None)
        weights, = out_
        return super(CorrMM_gradWeights,
                     self).c_code_helper(bottom, weights, top,
                                         sub, height, width)

    def grad(self, inp, grads):
        bottom, top = inp[:2]
        weights, = grads
        d_bottom = CorrMM_gradInputs(self.border_mode,
                                     self.subsample,
                                     self.filter_dilation,
                                     self.num_groups,
                                     self.unshared)(weights, top,
                                                    bottom.shape[-2:])
        d_top = CorrMM(self.border_mode,
                       self.subsample,
                       self.filter_dilation,
                       self.num_groups,
                       self.unshared)(bottom, weights)
        d_height_width = ((theano.gradient.DisconnectedType()(),) * 2
                          if len(inp) == 4 else ())
        return (d_bottom, d_top) + d_height_width

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0]]  # no connection to height, width


class CorrMM_gradInputs(BaseCorrMM):
    """
    Gradient wrt. inputs for `CorrMM`.

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
        if self.unshared is True:
            if kern.type.ndim != 6:
                raise TypeError('kern must be 6D tensor')
        else:
            if kern.type.ndim != 4:
                raise TypeError('kern must be 4D tensor')
        if topgrad.type.ndim != 4:
            raise TypeError('topgrad must be 4D tensor')
        if shape is None:
            if self.subsample != (1, 1):
                raise ValueError('shape must be given if subsample != (1, 1)')
            height_width = []
        else:
            height_width = [as_tensor_variable(shape[0]).astype('int64'),
                            as_tensor_variable(shape[1]).astype('int64')]

        if self.num_groups > 1:
            broadcastable = [topgrad.type.broadcastable[0], False,
                             False, False]
        else:
            broadcastable = [topgrad.type.broadcastable[0], kern.type.broadcastable[-3],
                             False, False]
        dtype = kern.type.dtype
        return Apply(self, [kern, topgrad] + height_width,
                     [TensorType(dtype, broadcastable)()])

    def infer_shape(self, node, input_shape):
        if self.border_mode == "half":
            padH_l = padH_r = padW_l = padW_r = -1
        elif self.border_mode == "full":
            padH_l = padH_r = padW_l = padW_r = -2
        elif isinstance(self.border_mode, tuple):
            border = ()
            for mode in self.border_mode:
                if isinstance(mode, tuple):
                    border += ((int(mode[0]), int(mode[1])),)
                else:
                    border += ((int(mode), int(mode)),)
            (padH_l, padH_r), (padW_l, padW_r) = border
        else:
            assert self.border_mode == "valid"
            padH_l = padH_r = padW_l = padW_r = 0
        dH, dW = self.subsample
        kshp = input_shape[0]
        topshp = input_shape[1]
        ssize, kshp = kshp[-3], list(kshp[-2:])
        ssize = ssize * self.num_groups
        bsize, topshp = topshp[0], list(topshp[2:])
        height_width = node.inputs[-2:]
        if padH_l == -1 or padH_r == -1:
            padH_l = padH_r = kshp[0] // 2
        elif padH_l == -2 or padH_r == -2:
            padH_l = padH_r = kshp[0] - 1
        elif padH_l < -2 or padH_r < -2:
            raise ValueError('CorrMM_gradInputs: border_mode must be >= 0.')
        if padW_l == -1 or padW_r == -1:
            padW_l = padW_r = kshp[1] // 2
        elif padW_l == -2 or padW_r == -2:
            padW_l = padW_r = kshp[1] - 1
        elif padW_l < -2 or padW_r < -2:
            raise ValueError('CorrMM_gradInputs: border_mode must be >= 0.')

        if dH != 1:
            out_shp0 = height_width[0]
        else:
            out_shp0 = (topshp[0] - 1) * dH + kshp[0] - padH_l - padH_r
        if dW != 1:
            out_shp1 = height_width[1]
        else:
            out_shp1 = (topshp[1] - 1) * dW + kshp[1] - padW_l - padW_r
        out_shp = (out_shp0, out_shp1)
        return [(bsize, ssize) + out_shp]

    def c_code(self, node, nodename, inp, out_, sub):
        weights, top = inp[:2]
        height, width = inp[2:] or (None, None)
        bottom, = out_
        return super(CorrMM_gradInputs,
                     self).c_code_helper(bottom, weights, top, sub,
                                         height,
                                         width)

    def grad(self, inp, grads):
        weights, top = inp[:2]
        bottom, = grads
        d_weights = CorrMM_gradWeights(self.border_mode,
                                       self.subsample,
                                       self.filter_dilation,
                                       self.num_groups,
                                       self.unshared)(bottom,
                                                      top,
                                                      weights.shape[-2:])
        d_top = CorrMM(self.border_mode,
                       self.subsample,
                       self.filter_dilation,
                       self.num_groups,
                       self.unshared)(bottom, weights)
        d_height_width = ((theano.gradient.DisconnectedType()(),) *
                          2 if len(inp) == 4 else ())
        return (d_weights, d_top) + d_height_width

    def connection_pattern(self, node):
        if node.nin == 2:
            return [[1], [1]]
        else:
            return [[1], [1], [0], [0]]  # no connection to height, width
