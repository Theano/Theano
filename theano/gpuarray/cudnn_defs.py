"""
Declarations of cuDNN types and constants used in Theano gpuarray DNN module.

For every cuDNN API supported by Theano, this module defines a class that
provides the set of cuDNN definitions to be used in Theano Ops.

Use :func:`get_definitions` to get the right cuDNN definitions
for a given cuDNN version.

Currently supported cuDNN APIs:

 - v5.1
 - v6.0
 - v7.0

"""

from __future__ import absolute_import, print_function, division

from theano.gof import CEnumType

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

    cudnnConvolutionBwdFilterAlgo_t = CEnumType(('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0', 'none'),
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1', 'deterministic'),
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT', 'fft'),
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3', 'small'),
                                                # TODO: not yet tested/documented:
                                                ('CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED', 'winograd_non_fused'),
                                                ctype='cudnnConvolutionBwdFilterAlgo_t')

    conv3d_bwd_filter_algorithms = ('none', 'small')

    cudnnConvolutionBwdDataAlgo_t = CEnumType(('CUDNN_CONVOLUTION_BWD_DATA_ALGO_0', 'none'),
                                              ('CUDNN_CONVOLUTION_BWD_DATA_ALGO_1', 'deterministic'),
                                              ('CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT', 'fft'),
                                              ('CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING', 'fft_tiling'),
                                              ('CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD', 'winograd'),
                                              # TODO: not yet tested/documented:
                                              ('CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED', 'winograd_non_fused'),
                                              ctype='cudnnConvolutionBwdDataAlgo_t')

    conv3d_bwd_data_algorithms = ('none', 'deterministic', 'fft_tiling')

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

    cudnnReduceTensorOp_t = CEnumType(('CUDNN_REDUCE_TENSOR_ADD', 'add'),
                                      ('CUDNN_REDUCE_TENSOR_MUL', 'mul'),
                                      ('CUDNN_REDUCE_TENSOR_MIN', 'minimum'),
                                      ('CUDNN_REDUCE_TENSOR_MAX', 'maximum'),
                                      ('CUDNN_REDUCE_TENSOR_AMAX', 'absmax'),
                                      ('CUDNN_REDUCE_TENSOR_AVG', 'avg'),
                                      ('CUDNN_REDUCE_TENSOR_NORM1', 'norm1'),
                                      ('CUDNN_REDUCE_TENSOR_NORM2', 'norm2'),
                                      ctype='cudnnReduceTensorOp_t')


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
