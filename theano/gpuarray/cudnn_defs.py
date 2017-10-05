"""
Declarations of cuDNN types and constants used in Theano gpuarray DNN module.

For every cuDNN API supported by Theano, this module defines a class that
provides the set of cuDNN definitions to be used in Theano Ops.

Use :func:`get_definitions` to get the right cuDNN definitions
for a given cuDNN version.

Currently supported cuDNN APIs:

 - v5.1*
 - v6.0*
 - v7.0*

"""

from __future__ import absolute_import, print_function, division

from theano.gof import CEnumType

HALF, FLOAT, DOUBLE = ('float16', 'float32', 'float64')
TRUE_HALF_CONFIG = (HALF, HALF)
PSEUDO_HALF_CONFIG = (HALF, FLOAT)
FLOAT_CONFIG = (FLOAT, FLOAT)
DOUBLE_CONFIG = (DOUBLE, DOUBLE)


def is_true_half_config(dtype, precision):
    return dtype == precision == HALF


def is_pseudo_half_config(dtype, precision):
    return dtype == HALF and precision == FLOAT


def is_float_config(dtype, precision):
    return dtype == precision == FLOAT


def is_double_config(dtype, precision):
    return dtype == precision == DOUBLE


# NB: Some cuDNN algorithms are listed in cuDNN enums but not implemented.
# We still register them here because we try to exactly copy cuDNN enums
# in Python side, but they will have no aliases associated, to help
# exclude them from lists of supported algorithms.


class CuDNNV51(object):
    version = 5

    cudnnConvolutionMode_t = CEnumType(('CUDNN_CONVOLUTION', 'conv'),
                                       ('CUDNN_CROSS_CORRELATION', 'cross'),
                                       ctype='cudnnConvolutionMode_t')

    cudnnDataType_t = CEnumType(('CUDNN_DATA_FLOAT', 'float32'),
                                ('CUDNN_DATA_DOUBLE', 'float64'),
                                ('CUDNN_DATA_HALF', 'float16'),
                                ctype='cudnnDataType_t')

    cudnnConvolutionFwdAlgo_t = CEnumType(('CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM', 'none'),
                                          ('CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM', 'small'),
                                          ('CUDNN_CONVOLUTION_FWD_ALGO_GEMM', 'large'),
                                          # not implemented:
                                          ('CUDNN_CONVOLUTION_FWD_ALGO_DIRECT'),
                                          ('CUDNN_CONVOLUTION_FWD_ALGO_FFT', 'fft'),
                                          ('CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING', 'fft_tiling'),
                                          ('CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD', 'winograd'),
                                          # TODO: Not yet tested/documented:
                                          ('CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED', 'winograd_non_fused'),
                                          ctype='cudnnConvolutionFwdAlgo_t')

    conv3d_fwd_algorithms = ('none', 'small', 'fft_tiling')

    deterministic_fwd_algorithms = cudnnConvolutionFwdAlgo_t.get_aliases()

    cudnnConvolutionBwdFilterAlgo_t = CEnumType(('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0', 'none'),
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1', 'deterministic'),
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT', 'fft'),
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3', 'small'),
                                                # TODO: not yet tested/documented:
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED', 'winograd_non_fused'),
                                                ctype='cudnnConvolutionBwdFilterAlgo_t')

    conv3d_bwd_filter_algorithms = ('none', 'small')

    deterministic_bwd_filter_algorithms = ('deterministic', 'fft', 'winograd_non_fused')

    cudnnConvolutionBwdDataAlgo_t = CEnumType(('CUDNN_CONVOLUTION_BWD_DATA_ALGO_0', 'none'),
                                              ('CUDNN_CONVOLUTION_BWD_DATA_ALGO_1', 'deterministic'),
                                              ('CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT', 'fft'),
                                              ('CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING', 'fft_tiling'),
                                              ('CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD', 'winograd'),
                                              # TODO: not yet tested/documented:
                                              ('CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED', 'winograd_non_fused'),
                                              ctype='cudnnConvolutionBwdDataAlgo_t')

    conv3d_bwd_data_algorithms = ('none', 'deterministic', 'fft_tiling')

    deterministic_bwd_data_algorithms = ('deterministic', 'fft', 'fft_tiling', 'winograd', 'winograd_non_fused')

    cudnnPoolingMode_t = CEnumType(('CUDNN_POOLING_MAX', 'max'),
                                   ('CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING', 'average_inc_pad'),
                                   ('CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING', 'average_exc_pad'),
                                   ctype='cudnnPoolingMode_t')

    cudnnSoftmaxAlgorithm_t = CEnumType(('CUDNN_SOFTMAX_FAST', 'fast'),
                                        ('CUDNN_SOFTMAX_ACCURATE', 'accurate'),
                                        ('CUDNN_SOFTMAX_LOG', 'log'),
                                        ctype='cudnnSoftmaxAlgorithm_t')

    cudnnSoftmaxMode_t = CEnumType(('CUDNN_SOFTMAX_MODE_INSTANCE', 'instance'),
                                   ('CUDNN_SOFTMAX_MODE_CHANNEL', 'channel'),
                                   ctype='cudnnSoftmaxMode_t')

    cudnnBatchNormMode_t = CEnumType(('CUDNN_BATCHNORM_PER_ACTIVATION', 'per-activation'),
                                     ('CUDNN_BATCHNORM_SPATIAL', 'spatial'),
                                     ctype='cudnnBatchNormMode_t')
    # It was introduced in cudnnv6, but we need to define it with an
    # empty list of enum to don't crash with cudnn 5
    cudnnReduceTensorOp_t = CEnumType()

    def get_supported_dtype_configs(self, check_runtime=None):
        """
        Return the tuple of data type configurations supported by this version of cuDNN.
        This is currently convenient for all supported cuDNN versions, as Theano does not
        yet support new data types (like INT8, INT8x4, etc.).

        ``check_runtime`` may be a function that tests if a data type configuration is supported.::

            is_supported = check_runtime(dtype, precision)

        .. warning::

            From documentation for cudnnConvolutionForward (for both v5.1 and v6):

            .. code-block::

                TRUE_HALF_CONFIG is only supported on architectures with true fp16 support
                (compute capability 5.3 and 6.0)

            This seems to be a general remark about f16 support (not only for FWD).
            It can be checked at runtime only.

        """

        if check_runtime is None or check_runtime(*TRUE_HALF_CONFIG):
            return (TRUE_HALF_CONFIG, PSEUDO_HALF_CONFIG, FLOAT_CONFIG, DOUBLE_CONFIG)
        return (PSEUDO_HALF_CONFIG, FLOAT_CONFIG, DOUBLE_CONFIG)

    def fwd_algo_supports_dtype_config(self, algo, dtype, precision, ndim):
        algorithms = self.cudnnConvolutionFwdAlgo_t
        algo = algorithms.fromalias(algo)
        if algo == algorithms.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
            return not is_true_half_config(dtype, precision)
        if algo == algorithms.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
            return ndim == 2 or not is_true_half_config(dtype, precision)
        if algo == algorithms.CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            return ndim == 2 and not is_true_half_config(dtype, precision)
        # CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: not implemented.
        if algo == algorithms.CUDNN_CONVOLUTION_FWD_ALGO_FFT:
            return ndim == 2 and (is_pseudo_half_config(dtype, precision) or is_float_config(dtype, precision))
        if algo == algorithms.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
            if ndim == 2:
                return is_pseudo_half_config(dtype, precision) or is_float_config(dtype, precision)
            if ndim == 3:
                return not is_true_half_config(dtype, precision)
        if algo == algorithms.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            return ndim == 2 and (is_pseudo_half_config(dtype, precision) or is_float_config(dtype, precision))
        if algo == algorithms.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
            # NB: "If wDesc 's filter (height, width) is (5,5), data type config TRUE_HALF_CONFIG is not supported".
            # We could not check it before being in C code.
            return ndim == 2 and not is_double_config(dtype, precision)
        return False

    def bwd_filter_algo_supports_dtype_config(self, algo, dtype, precision, ndim):
        # NB: Theano does not support float16 precision anymore for backward cuDNN convolutions.
        if is_true_half_config(dtype, precision):
            return False
        algorithms = self.cudnnConvolutionBwdFilterAlgo_t
        algo = algorithms.fromalias(algo)
        if algo == algorithms.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
            return not is_true_half_config(dtype, precision)
        if algo == algorithms.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
            return ndim == 2
        if algo == algorithms.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
            return ndim == 2 and (is_pseudo_half_config(dtype, precision) or is_float_config(dtype, precision))
        if algo == algorithms.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
            return not is_true_half_config(dtype, precision)
        if algo == algorithms.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
            # NB: "If wDesc 's filter (height, width) is (5,5), data type config TRUE_HALF_CONFIG is not supported".
            # We could not check it before being in C code.
            return ndim == 2 and not is_double_config(dtype, precision)
        return False

    def bwd_data_algo_supports_dtype_config(self, algo, dtype, precision, ndim):
        # NB: Theano does not support float16 precision anymore for backward cuDNN convolutions.
        if is_true_half_config(dtype, precision):
            return False
        algorithms = self.cudnnConvolutionBwdDataAlgo_t
        algo = algorithms.fromalias(algo)
        if algo == algorithms.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
            return not is_true_half_config(dtype, precision)
        if algo == algorithms.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
            # CUDNN_CONVOLUTION_BWD_DATA_ALGO_1: all data type configs supported.
            # NB: Let's avoid float16 precision, as some strange errors may be encountered
            # with that precision ( see https://github.com/Theano/Theano/pull/5932/ )
            return not is_true_half_config(dtype, precision)
        if algo == algorithms.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
            return ndim == 2 and (is_pseudo_half_config(dtype, precision) or is_float_config(dtype, precision))
        if algo == algorithms.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
            if ndim == 2:
                return is_pseudo_half_config(dtype, precision) or is_float_config(dtype, precision)
            if ndim == 3:
                return not is_true_half_config(dtype, precision)
        if algo == algorithms.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
            return ndim == 2 and (is_pseudo_half_config(dtype, precision) or is_float_config(dtype, precision))
        if algo == algorithms.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
            # NB: "If wDesc 's filter (height, width) is (5,5), data type config TRUE_HALF_CONFIG is not supported".
            # We could not check it before being in C code.
            return ndim == 2 and not is_double_config(dtype, precision)
        return False


class CuDNNV6(CuDNNV51):
    version = 6

    cudnnDataType_t = CEnumType(('CUDNN_DATA_FLOAT', 'float32'),
                                ('CUDNN_DATA_DOUBLE', 'float64'),
                                ('CUDNN_DATA_HALF', 'float16'),
                                # new in v6
                                ('CUDNN_DATA_INT8', 'int8'),
                                ('CUDNN_DATA_INT32', 'int32'),
                                # ('CUDNN_DATA_INT8X4', 'int8x4'),
                                ctype='cudnnDataType_t')

    cudnnPoolingMode_t = CEnumType(('CUDNN_POOLING_MAX', 'max'),
                                   ('CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING', 'average_inc_pad'),
                                   ('CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING', 'average_exc_pad'),
                                   # new in v6:
                                   ('CUDNN_POOLING_MAX_DETERMINISTIC', 'max_deterministic'),
                                   ctype='cudnnPoolingMode_t')

    cudnnConvolutionBwdFilterAlgo_t = CEnumType(('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0', 'none'),
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1', 'deterministic'),
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT', 'fft'),
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3', 'small'),
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD'),
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED', 'winograd_non_fused'),
                                                # new in v6:
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING', 'fft_tiling'),
                                                ctype='cudnnConvolutionBwdFilterAlgo_t')

    deterministic_bwd_filter_algorithms = CuDNNV51.deterministic_bwd_filter_algorithms + ('fft_tiling',)

    cudnnReduceTensorOp_t = CEnumType(('CUDNN_REDUCE_TENSOR_ADD', 'add'),
                                      ('CUDNN_REDUCE_TENSOR_MUL', 'mul'),
                                      ('CUDNN_REDUCE_TENSOR_MIN', 'minimum'),
                                      ('CUDNN_REDUCE_TENSOR_MAX', 'maximum'),
                                      ('CUDNN_REDUCE_TENSOR_AMAX', 'absmax'),
                                      ('CUDNN_REDUCE_TENSOR_AVG', 'avg'),
                                      ('CUDNN_REDUCE_TENSOR_NORM1', 'norm1'),
                                      ('CUDNN_REDUCE_TENSOR_NORM2', 'norm2'),
                                      ctype='cudnnReduceTensorOp_t')

    def fwd_algo_supports_dtype_config(self, algo, dtype, precision, ndim):
        is_supported = super(CuDNNV6, self).fwd_algo_supports_dtype_config(algo, dtype, precision, ndim)
        if not is_supported:
            algorithms = self.cudnnConvolutionFwdAlgo_t
            algo = algorithms.fromalias(algo)
            if algo == algorithms.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
                # NB: For cuDNN V6:
                # "Data Type Config Support: PSEUDO_HALF_CONFIG, FLOAT_CONFIG
                # (DOUBLE_CONFIG is also supported when the task can be handled by 1D FFT,
                # ie, one of the filter dimension, width or height is 1)"
                # Could be checked only in C code. By default, let's allow DOUBLE_CONFIG.
                return ndim == 2 and (is_pseudo_half_config(dtype, precision) or
                                      is_float_config(dtype, precision) or
                                      is_double_config(dtype, precision))
        return is_supported

    def bwd_filter_algo_supports_dtype_config(self, algo, dtype, precision, ndim):
        is_supported = super(CuDNNV6, self).bwd_filter_algo_supports_dtype_config(algo, dtype, precision, ndim)
        if not is_supported:
            algorithms = self.cudnnConvolutionBwdFilterAlgo_t
            algo = algorithms.fromalias(algo)
            if algo == algorithms.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
                return ndim == 2 and (is_pseudo_half_config(dtype, precision) or
                                      is_float_config(dtype, precision) or
                                      is_double_config(dtype, precision))
        return is_supported

    def bwd_data_algo_supports_dtype_config(self, algo, dtype, precision, ndim):
        is_supported = super(CuDNNV6, self).bwd_data_algo_supports_dtype_config(algo, dtype, precision, ndim)
        if not is_supported:
            algorithms = self.cudnnConvolutionBwdDataAlgo_t
            algo = algorithms.fromalias(algo)
            if algo == algorithms.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
                # NB: For cuDNN V6:
                # "Data Type Config Support: PSEUDO_HALF_CONFIG, FLOAT_CONFIG
                # (DOUBLE_CONFIG is also supported when the task can be handled by 1D FFT,
                # ie, one of the filter dimension, width or height is 1)"
                # Could be checked only in C code. By default, let's allow DOUBLE_CONFIG.
                return ndim == 2 and (is_pseudo_half_config(dtype, precision) or
                                      is_float_config(dtype, precision) or
                                      is_double_config(dtype, precision))
        return is_supported


class CuDNNV7(CuDNNV6):
    version = 7
    cudnnMathType_t = CEnumType(('CUDNN_DEFAULT_MATH', 'non_tensor_op'),
                                ('CUDNN_TENSOR_OP_MATH', 'tensor_op'),
                                ctype='cudnnMathType_t')
    cudnnDeterminism_t = CEnumType(('CUDNN_NON_DETERMINISTIC', 'non_deterministic'),
                                   ('CUDNN_DETERMINISTIC', 'deterministic'),
                                   ctype='cudnnDeterminism_t')


def get_definitions(cudnn_version=None):
    """
    Return cuDNN definitions to be used by Theano for the given cuDNN version.

    ``cudnn_version`` must be None or an integer
    (typically the version returned by :func:`theano.gpuarray.dnn.version`).
    if None, return definitions for the  most recent supported cuDNN version.

    """
    if cudnn_version is not None:
        if cudnn_version // 1000 == 5:
            return CuDNNV51()
        if cudnn_version // 1000 == 6:
            return CuDNNV6()
    # By default, we use definitions for the last supported cuDNN version.
    return CuDNNV7()
