"""
This module is just a collection of definitions to be used by `check_dnn.py`.

Following classes, functions and definitions are used to check if
tests fail as expected when conditions listed into cuDNN documentation are not verified.
I have currently implemented checking only for 2D/3D FWD algorithms in cuDNN V5.1,
and in practice, many tests pass even when they don't follow cuDNN doc conditions.
So, I think we should better just run all tests and check ourselves
which tests pass, which fail, and why they fail.


Reminder:
    N: batch number
    C: number of feature maps
    D: depth
    H: height
    W: width

NB: We assume that we **always** use NC(D)HW tensors in Theano.

"""

from __future__ import absolute_import, print_function, division
import theano
from ..cudnn_defs import HALF, FLOAT, DOUBLE, get_definitions
from ..dnn import version

UNKNOWN, TRUE_HALF_CONFIG, PSEUDO_HALF_CONFIG, FLOAT_CONFIG, DOUBLE_CONFIG = -1, 0, 1, 2, 3

cudnn = get_definitions(version(raises=False))
cudnnConvolutionFwdAlgo_t = cudnn.cudnnConvolutionFwdAlgo_t


class Success:
    ok = True
    messages = []

    def __init__(self, messages=[]):
        self.messages = list(messages)

    def add_message(self, *parts):
        self.messages.append(''.join(str(part) for part in parts))


class Failure(Success):
    ok = False


def _and(*tests):
    # `tests` is a list of tuples with format (lambda test, test description)
    messages = []
    for test_lambda, message in tests:
        if not test_lambda():
            messages.append(message)
    return Failure(messages) if messages else Success()


def _or(*tests):
    messages = []
    ok = False
    for test_lambda, message in tests:
        if test_lambda():
            ok = True
            break
        else:
            messages.append(message)
    return Success() if ok else Failure(messages)


def type_conf(precision):
    # All Op's input tensors are floatX tensors.
    floatX = theano.config.floatX
    if floatX == precision == HALF:
        return TRUE_HALF_CONFIG
    if floatX == HALF and precision == FLOAT:
        return PSEUDO_HALF_CONFIG
    if floatX == precision == FLOAT:
        return FLOAT_CONFIG
    if floatX == precision == DOUBLE:
        return DOUBLE_CONFIG
    return UNKNOWN
    # raise ValueError('Unknown data type configuration (%s %s)' % (floatX, precision))


def type_conf_to_string(conf):
    if conf == -1:
        return 'UNKNOWN'
    if conf == 0:
        return 'TRUE_HALF_CONFIG'
    if conf == 1:
        return 'PSEUDO_HALF_CONFIG'
    if conf == 2:
        return 'FLOAT_CONFIG'
    if conf == 3:
        return 'DOUBLE_CONFIG'


def strideof(tensor, i):
    return tensor.strides[i] // tensor.itemsize


def tensor_is_partially_packed(tensor, packed_dim_names):
    if tensor.ndim == 4:
        dim_names = 'NCHW'
    else:
        dim_names = 'NCDHW'
    packed_dims = []
    unpacked_dims = []
    for i in range(tensor.ndim - 1):
        if dim_names[i] in packed_dim_names:
            packed_dims.append(i)
        else:
            unpacked_dims.append(i)
    if dim_names[tensor.ndim - 1] in packed_dim_names and strideof(tensor, -1) != 1:
        # We won't put last dimension in the list of packed dims.
        # We just need to check if stride of that dimension is 1.
        return False
    return (all(strideof(tensor, i) >= tensor.shape[i + 1] * strideof(tensor, i + 1) for i in unpacked_dims) and
            all(strideof(tensor, i) == tensor.shape[i + 1] * strideof(tensor, i + 1) for i in packed_dims))


def tensor_is_fully_packed(tensor):
    return strideof(tensor, -1) == 1 and all(strideof(tensor, i) == tensor.shape[i + 1] * strideof(tensor, i + 1)
                                             for i in range(tensor.ndim - 1))


def check_fwd_algorithm(img, kern, out, desc_op, algo, precision, subsample, dilation):
    # Based on cuDNN v5.1 user guide.

    ndim = img.ndim - 2
    if ndim == 2:
        # rD won't be used.
        rD, rH, rW = -1, 0, 1
    else:
        rD, rH, rW = 0, 1, 2

    algo = cudnnConvolutionFwdAlgo_t.fromalias(algo)

    kern_shape = kern.shape[2:]
    kern_shape = tuple((kern_shape[i] - 1) * dilation[i] + 1 for i in range(len(dilation)))

    pad = (desc_op.pad0, desc_op.pad1, desc_op.pad2)[:len(kern_shape)]
    if desc_op.bmode == 'full':
        pad = tuple(kern_shape[i] - 1 for i in range(len(pad)))
    elif desc_op.bmode == 'half':
        pad = tuple(kern_shape[i] // 2 for i in range(len(pad)))

    img_shape = img.shape[2:]

    img_with_borders = tuple(img_shape[i] + 2 * pad[i] for i in range(len(pad)))

    def check_algo():
        if algo == cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
            return _and((lambda: type_conf(precision) != TRUE_HALF_CONFIG,
                         "Data Type Config Support: All except TRUE_HALF_CONFIG"))

        # CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: 2D: everything supported.
        if ndim == 3 and algo == cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
            return _and(
                (lambda: type_conf(precision) != TRUE_HALF_CONFIG,
                 "Data Type Config Support: All except TRUE_HALF_CONFIG"),
                (lambda: tensor_is_fully_packed(img),
                 "xDesc Format Support: NCDHW-fully-packed"),
                (lambda: tensor_is_fully_packed(out),
                 "yDesc Format Support: NCDHW-fully-packed"),
            )

        if algo == cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            return _and(
                (lambda: type_conf(precision) != TRUE_HALF_CONFIG,
                 "Data Type Config Support: All except TRUE_HALF_CONFIG"),
                (lambda: ndim == 2,
                 "Only for conv2d")
            )

        # CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: not implemented.

        if algo == cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_FFT:
            return _and(
                (lambda: type_conf(precision) in (PSEUDO_HALF_CONFIG, FLOAT_CONFIG),
                 "Data Type Config Support: PSEUDO_HALF_CONFIG, FLOAT_CONFIG"),
                (lambda: ndim == 2,
                 "Only for conv2d"),
                (lambda: tensor_is_partially_packed(img, 'HW'),
                 "xDesc Format Support: NCHW HW-packed"),
                (lambda: tensor_is_partially_packed(out, 'HW'),
                 "yDesc Format Support: NCHW HW-packed"),
                (lambda: img_with_borders[rH] <= 256,
                 "xDesc 's feature map height + 2 * convDesc 's zero-padding height must equal 256 or less"),
                (lambda: img_with_borders[rW] <= 256,
                 "xDesc 's feature map width + 2 * convDesc 's zero-padding width must equal 256 or less"),
                (lambda: subsample[rH] == subsample[rW] == 1,
                 "convDesc 's vertical and horizontal filter stride must equal 1"),
                (lambda: kern_shape[rH] > pad[rH],
                 "wDesc 's filter height must be greater than convDesc 's zero-padding height"),
                (lambda: kern_shape[rW] > pad[rW],
                 "wDesc 's filter width must be greater than convDesc 's zero-padding width")
            )

        if algo == cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
            if ndim == 2:
                return _and(
                    (lambda: type_conf(precision) in (PSEUDO_HALF_CONFIG, FLOAT_CONFIG),
                     "Data Type Config Support: PSEUDO_HALF_CONFIG, FLOAT_CONFIG"),
                    (lambda: tensor_is_partially_packed(img, 'HW'),
                     "xDesc Format Support: NCHW HW-packed"),
                    (lambda: tensor_is_partially_packed(out, 'HW'),
                     "yDesc Format Support: NCHW HW-packed"),
                    (lambda: kern_shape[rH] <= 32,
                     "wDesc 's filter height must equal 32 or less"),
                    (lambda: kern_shape[rW] <= 32,
                     "wDesc 's filter width must equal 32 or less"),
                    (lambda: subsample[rH] == subsample[rW] == 1,
                     "convDesc 's vertical and horizontal filter stride must equal 1"),
                    (lambda: pad[rH] < kern_shape[rH],
                     "wDesc 's filter height must be greater than convDesc 's zero-padding height"),
                    (lambda: pad[rW] < kern_shape[rW],
                     "wDesc 's filter width must be greater than convDesc 's zero-padding width"),
                )
            if ndim == 3:
                return _and(
                    (lambda: type_conf(precision) != TRUE_HALF_CONFIG,
                     "Data Type Config Support: All except TRUE_HALF_CONFIG"),
                    (lambda: tensor_is_partially_packed(img, 'DHW'),
                     "xDesc Format Support: NCDHW DHW-packed"),
                    (lambda: tensor_is_partially_packed(out, 'DHW'),
                     "yDesc Format Support: NCDHW DHW-packed"),
                    (lambda: kern_shape[rH] <= 16,
                     "wDesc 's filter height must equal 16 or less"),
                    (lambda: kern_shape[rW] <= 16,
                     "wDesc 's filter width must equal 16 or less"),
                    (lambda: kern_shape[rD] <= 16,
                     "wDesc 's filter depth must equal 16 or less"),
                    (lambda: all(s == 1 for s in subsample),
                     "convDesc 's must have all filter strides equal to 1"),
                    (lambda: pad[rH] < kern_shape[rH],
                     "wDesc 's filter height must be greater than convDesc 's zero-padding height"),
                    (lambda: pad[rW] < kern_shape[rW],
                     "wDesc 's filter width must be greater than convDesc 's zero-padding width"),
                    (lambda: pad[rW] < kern_shape[rD],
                     "wDesc 's filter depth must be greater than convDesc 's zero-padding width"),
                )

        if algo == cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            return _and(
                (lambda: type_conf(precision) in (PSEUDO_HALF_CONFIG, FLOAT_CONFIG, DOUBLE_CONFIG),
                 "Data Type Config Support: PSEUDO_HALF_CONFIG, FLOAT_CONFIG"),
                (lambda: ndim == 2,
                 "Only for conv2d"),
                (lambda: subsample[rH] == subsample[rW] == 1,
                 "convDesc 's vertical and horizontal filter stride must equal 1"),
                (lambda: kern_shape[rH] == 3,
                 "wDesc 's filter height must be 3"),
                (lambda: kern_shape[rW] == 3,
                 "wDesc 's filter width must be 3"),
            )

        if algo == cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
            data_type_conf = type_conf(precision)
            return _and(
                (lambda: data_type_conf != DOUBLE_CONFIG,
                 "Data Type Config Support: All except DOUBLE_CONFIG"),
                (lambda: ndim == 2,
                 "Only for conv2d"),
                (lambda: subsample[rH] == subsample[rW] == 1,
                 "convDesc 's vertical and horizontal filter stride must equal 1"),
                (lambda: kern_shape[rH] == kern_shape[rW] and kern_shape[rH] in (3, 5),
                 "wDesc 's filter (height, width) must be (3,3) or (5,5)"),
                (lambda: kern_shape[rH] == 3 or data_type_conf != TRUE_HALF_CONFIG,
                 "If wDesc 's filter (height, width) is (5,5), "
                 "data type config TRUE_HALF_CONFIG is not supported")
            )

    checking = check_algo()
    if not checking.ok:
        messages = checking.messages
        checking.messages = []
        checking.add_message('config             : ', type_conf_to_string(type_conf(precision)))
        checking.add_message('computed borders   : ', pad)
        checking.add_message('img with borders   : ', img_with_borders)
        checking.add_message('computed kern shape: ', kern_shape)
        checking.add_message('== why should fail ==')
        checking.messages += messages
    return checking
