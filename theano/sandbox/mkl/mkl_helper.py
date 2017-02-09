from __future__ import absolute_import, print_function, division
import logging

_logger = logging.getLogger('theano.tensor.nnet.mkl_helper')


def header_text():
    """
        C header for mkl dnn primitive interface,
        Build date: 20170206
    """
    header = """
    extern "C"
    {
        /* MKL Version type */
        typedef
        struct {
            int    MajorVersion;
            int    MinorVersion;
            int    UpdateVersion;
            char * ProductStatus;
            char * Build;
            char * Processor;
            char * Platform;
        } MKLVersion;

        void    MKL_Get_Version(MKLVersion *ver); /* Returns information about the version of the Intel MKL software */
        #define mkl_get_version             MKL_Get_Version

        struct _uniPrimitive_s {};
        struct _dnnLayout_s {};

        typedef struct _uniPrimitive_s* dnnPrimitive_t;
        typedef struct _dnnLayout_s* dnnLayout_t;
        typedef void* dnnPrimitiveAttributes_t;

        #define DNN_MAX_DIMENSION       32
        #define DNN_QUERY_MAX_LENGTH    128

        typedef enum {
            E_SUCCESS                   =  0,
            E_INCORRECT_INPUT_PARAMETER = -1,
            E_UNEXPECTED_NULL_POINTER   = -2,
            E_MEMORY_ERROR              = -3,
            E_UNSUPPORTED_DIMENSION     = -4,
            E_UNIMPLEMENTED             = -127
        } dnnError_t;

        typedef enum {
            /** GEMM base convolution (unimplemented) */
            dnnAlgorithmConvolutionGemm,
            /** Direct convolution */
            dnnAlgorithmConvolutionDirect,
            /** FFT based convolution (unimplemented) */
            dnnAlgorithmConvolutionFFT,
            /** Maximum pooling */
            dnnAlgorithmPoolingMax,
            /** Minimum pooling */
            dnnAlgorithmPoolingMin,
            /** Average pooling (padded values are not taken into account) */
            dnnAlgorithmPoolingAvgExcludePadding,
            /** Alias for average pooling (padded values are not taken into account) */
            dnnAlgorithmPoolingAvg = dnnAlgorithmPoolingAvgExcludePadding,
            /** Average pooling (padded values are taken into account) */
            dnnAlgorithmPoolingAvgIncludePadding
        } dnnAlgorithm_t;

        typedef enum {
            dnnResourceSrc            = 0,
            dnnResourceFrom           = 0,
            dnnResourceDst            = 1,
            dnnResourceTo             = 1,
            dnnResourceFilter         = 2,
            dnnResourceScaleShift     = 2,
            dnnResourceBias           = 3,
            dnnResourceMean           = 3,
            dnnResourceDiffSrc        = 4,
            dnnResourceDiffFilter     = 5,
            dnnResourceDiffScaleShift = 5,
            dnnResourceDiffBias       = 6,
            dnnResourceVariance       = 6,
            dnnResourceDiffDst        = 7,
            dnnResourceWorkspace      = 8,
            dnnResourceMultipleSrc    = 16,
            dnnResourceMultipleDst    = 24,
            dnnResourceNumber         = 32
        } dnnResourceType_t;

        typedef enum {
            dnnBorderZeros          = 0x0,
            dnnBorderZerosAsymm     = 0x100,
            dnnBorderExtrapolation  = 0x3
        } dnnBorder_t;

        typedef enum {
            dnnUseInputMeanVariance = 0x1U,
            dnnUseScaleShift        = 0x2U
        } dnnBatchNormalizationFlag_t;

        /*******************************************************************************
         * F32 section: single precision
         ******************************************************************************/

        dnnError_t dnnLayoutCreate_F32(
                dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]);
        dnnError_t dnnLayoutCreateFromPrimitive_F32(
                dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type);

        /** Returns the size of buffer required to serialize dnnLayout_t structure. */
        size_t dnnLayoutSerializationBufferSize_F32();

        /** Serializes given @p layout into buffer @p buf. User-provided buffer @p buf
         * should have enough space to store dnnLayout_t structure.
         * @sa dnnLayoutSerializationBufferSize_F32 */
        dnnError_t dnnLayoutSerialize_F32(const dnnLayout_t layout, void *buf);

        /** Creates new layout restored from previously serialized one. */
        dnnError_t dnnLayoutDeserialize_F32(dnnLayout_t *pLayout, const void *buf);

        size_t dnnLayoutGetMemorySize_F32(
                const dnnLayout_t layout);
        int dnnLayoutCompare_F32(
                const dnnLayout_t l1, const dnnLayout_t l2);
        dnnError_t dnnAllocateBuffer_F32(
                void **pPtr, dnnLayout_t layout);
        dnnError_t dnnReleaseBuffer_F32(
                void *ptr);
        dnnError_t dnnLayoutDelete_F32(
                dnnLayout_t layout);

        dnnError_t dnnPrimitiveAttributesCreate_F32(
                dnnPrimitiveAttributes_t *attributes);
        dnnError_t dnnPrimitiveAttributesDestroy_F32(
                dnnPrimitiveAttributes_t attributes);
        dnnError_t dnnPrimitiveGetAttributes_F32(
                dnnPrimitive_t primitive,
                dnnPrimitiveAttributes_t *attributes);

        dnnError_t dnnExecute_F32(
                dnnPrimitive_t primitive, void *resources[]);
        dnnError_t dnnExecuteAsync_F32(
                dnnPrimitive_t primitive, void *resources[]);
        dnnError_t dnnWaitFor_F32(
                dnnPrimitive_t primitive);
        dnnError_t dnnDelete_F32(
                dnnPrimitive_t primitive);

        dnnError_t dnnConversionCreate_F32(
                dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to);
        dnnError_t dnnConversionExecute_F32(
                dnnPrimitive_t conversion, void *from, void *to);

        dnnError_t dnnSumCreate_F32(
                dnnPrimitive_t *pSum, dnnPrimitiveAttributes_t attributes, const size_t nSummands,
                dnnLayout_t layout, float *coefficients);
        dnnError_t dnnConcatCreate_F32(
                dnnPrimitive_t* pConcat, dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors, dnnLayout_t *src);
        dnnError_t dnnSplitCreate_F32(
                dnnPrimitive_t *pSplit, dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
                dnnLayout_t layout, size_t dstChannelSize[]);
        dnnError_t dnnScaleCreate_F32(
                dnnPrimitive_t *pScale,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, float alpha);

        dnnError_t dnnConvolutionCreateForward_F32(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnConvolutionCreateForwardBias_F32(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnConvolutionCreateBackwardData_F32(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnConvolutionCreateBackwardFilter_F32(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnConvolutionCreateBackwardBias_F32(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t dimension, const size_t dstSize[]);

        dnnError_t dnnGroupsConvolutionCreateForward_F32(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnGroupsConvolutionCreateForwardBias_F32(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnGroupsConvolutionCreateBackwardData_F32(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnGroupsConvolutionCreateBackwardFilter_F32(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnGroupsConvolutionCreateBackwardBias_F32(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t groups, size_t dimension, const size_t dstSize[]);

        dnnError_t dnnReLUCreateForward_F32(
                dnnPrimitive_t* pRelu,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, float negativeSlope);
        dnnError_t dnnReLUCreateBackward_F32(
                dnnPrimitive_t* pRelu,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, float negativeSlope);

        dnnError_t dnnLRNCreateForward_F32(
                dnnPrimitive_t* pLrn,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k);
        dnnError_t dnnLRNCreateBackward_F32(
                dnnPrimitive_t* pLrn,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, float alpha, float beta, float k);

        dnnError_t dnnBatchNormalizationCreateForward_F32(
                dnnPrimitive_t* pBatchNormalization,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, float eps);
        dnnError_t dnnBatchNormalizationCreateBackwardScaleShift_F32(
                dnnPrimitive_t* pBatchNormalization,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, float eps);
        dnnError_t dnnBatchNormalizationCreateBackwardData_F32(
                dnnPrimitive_t* pBatchNormalization,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, float eps);

        dnnError_t dnnBatchNormalizationCreateForward_v2_F32(
                dnnPrimitive_t* pBatchNormalization,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, float eps,
                unsigned int flags);
        dnnError_t dnnBatchNormalizationCreateBackward_v2_F32(
                dnnPrimitive_t* pBatchNormalization,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, float eps,
                unsigned int flags);

        dnnError_t dnnPoolingCreateForward_F32(
                dnnPrimitive_t* pPooling,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t op,
                const dnnLayout_t srcLayout,
                const size_t kernelSize[], const size_t kernelStride[],
                const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnPoolingCreateBackward_F32(
                dnnPrimitive_t* pPooling,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t op,
                const dnnLayout_t srcLayout,
                const size_t kernelSize[], const size_t kernelStride[],
                const int inputOffset[], const dnnBorder_t borderType);

        dnnError_t dnnInnerProductCreateForward_F32(
                dnnPrimitive_t *pInnerProduct,
                dnnPrimitiveAttributes_t attributes,
                size_t dimensions,
                const size_t srcSize[],
                size_t outputChannels);
        dnnError_t dnnInnerProductCreateForwardBias_F32(
                dnnPrimitive_t *pInnerProduct,
                dnnPrimitiveAttributes_t attributes,
                size_t dimensions,
                const size_t srcSize[],
                size_t outputChannels);
        dnnError_t dnnInnerProductCreateBackwardData_F32(
                dnnPrimitive_t *pInnerProduct,
                dnnPrimitiveAttributes_t attributes,
                size_t dimensions,
                const size_t srcSize[],
                size_t outputChannels);
        dnnError_t dnnInnerProductCreateBackwardFilter_F32(
                dnnPrimitive_t *pInnerProduct,
                dnnPrimitiveAttributes_t attributes,
                size_t dimensions,
                const size_t srcSize[],
                size_t outputChannels);
        dnnError_t dnnInnerProductCreateBackwardBias_F32(
                dnnPrimitive_t *pInnerProduct,
                dnnPrimitiveAttributes_t attributes,
                size_t dimensions,
                const size_t dstSize[]);

        /*******************************************************************************
         * F64 section: double precision
         ******************************************************************************/

        dnnError_t dnnLayoutCreate_F64 (
                dnnLayout_t *pLayout, size_t dimension, const size_t size[], const size_t strides[]);
        dnnError_t dnnLayoutCreateFromPrimitive_F64(
                dnnLayout_t *pLayout, const dnnPrimitive_t primitive, dnnResourceType_t type);

        /** Returns the size of buffer required to serialize dnnLayout_t structure. */
        size_t dnnLayoutSerializationBufferSize_F64();

        /** Serializes given @p layout into buffer @p buf. User-provided buffer @p buf
         * should have enough space to store dnnLayout_t structure.
         * @sa dnnLayoutSerializationBufferSize_F64 */
        dnnError_t dnnLayoutSerialize_F64(const dnnLayout_t layout, void *buf);

        /** Creates new layout restored from previously serialized one. */
        dnnError_t dnnLayoutDeserialize_F64(dnnLayout_t *pLayout, const void *buf);

        size_t dnnLayoutGetMemorySize_F64(
                const dnnLayout_t layout);
        int dnnLayoutCompare_F64(
                const dnnLayout_t l1, const dnnLayout_t l2);
        dnnError_t dnnAllocateBuffer_F64(
                void **pPtr, dnnLayout_t layout);
        dnnError_t dnnReleaseBuffer_F64(
                void *ptr);
        dnnError_t dnnLayoutDelete_F64(
                dnnLayout_t layout);

        dnnError_t dnnPrimitiveAttributesCreate_F64(
                dnnPrimitiveAttributes_t *attributes);
        dnnError_t dnnPrimitiveAttributesDestroy_F64(
                dnnPrimitiveAttributes_t attributes);
        dnnError_t dnnPrimitiveGetAttributes_F64(
                dnnPrimitive_t primitive,
                dnnPrimitiveAttributes_t *attributes);

        dnnError_t dnnExecute_F64(
                dnnPrimitive_t primitive, void *resources[]);
        dnnError_t dnnExecuteAsync_F64(
                dnnPrimitive_t primitive, void *resources[]);
        dnnError_t dnnWaitFor_F64(
                dnnPrimitive_t primitive);
        dnnError_t dnnDelete_F64(
                dnnPrimitive_t primitive);

        dnnError_t dnnConversionCreate_F64(
                dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to);
        dnnError_t dnnConversionExecute_F64(
                dnnPrimitive_t conversion, void *from, void *to);

        dnnError_t dnnSumCreate_F64(
                dnnPrimitive_t *pSum, dnnPrimitiveAttributes_t attributes, const size_t nSummands,
                dnnLayout_t layout, double *coefficients);
        dnnError_t dnnConcatCreate_F64(
                 dnnPrimitive_t* pConcat, dnnPrimitiveAttributes_t attributes, const size_t nSrcTensors, dnnLayout_t *src);
        dnnError_t dnnSplitCreate_F64(
                dnnPrimitive_t *pSplit, dnnPrimitiveAttributes_t attributes, const size_t nDstTensors,
                dnnLayout_t layout, size_t dstChannelSize[]);
        dnnError_t dnnScaleCreate_F64(
                dnnPrimitive_t *pScale,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, double alpha);

        dnnError_t dnnConvolutionCreateForward_F64(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnConvolutionCreateForwardBias_F64(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnConvolutionCreateBackwardData_F64(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnConvolutionCreateBackwardFilter_F64(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnConvolutionCreateBackwardBias_F64(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t dimension, const size_t dstSize[]);

        dnnError_t dnnGroupsConvolutionCreateForward_F64(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnGroupsConvolutionCreateForwardBias_F64(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnGroupsConvolutionCreateBackwardData_F64(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnGroupsConvolutionCreateBackwardFilter_F64(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t groups, size_t dimension, const size_t srcSize[], const size_t dstSize[], const size_t filterSize[],
                const size_t convolutionStrides[], const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnGroupsConvolutionCreateBackwardBias_F64(
                dnnPrimitive_t* pConvolution,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t algorithm,
                size_t groups, size_t dimension, const size_t dstSize[]);

        dnnError_t dnnReLUCreateForward_F64(
                dnnPrimitive_t* pRelu,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, double negativeSlope);
        dnnError_t dnnReLUCreateBackward_F64(
                dnnPrimitive_t* pRelu,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, double negativeSlope);

        dnnError_t dnnLRNCreateForward_F64(
                dnnPrimitive_t* pLrn,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta, double k);
        dnnError_t dnnLRNCreateBackward_F64(
                dnnPrimitive_t* pLrn,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t diffLayout, const dnnLayout_t dataLayout, size_t kernel_size, double alpha, double beta, double k);

        dnnError_t dnnBatchNormalizationCreateForward_F64(
                dnnPrimitive_t* pBatchNormalization,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, double eps);
        dnnError_t dnnBatchNormalizationCreateBackwardScaleShift_F64(
                dnnPrimitive_t* pBatchNormalization,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, double eps);
        dnnError_t dnnBatchNormalizationCreateBackwardData_F64(
                dnnPrimitive_t* pBatchNormalization,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, double eps);

        dnnError_t dnnBatchNormalizationCreateForward_v2_F64(
                dnnPrimitive_t* pBatchNormalization,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, double eps,
                unsigned int flags);
        dnnError_t dnnBatchNormalizationCreateBackward_v2_F64(
                dnnPrimitive_t* pBatchNormalization,
                dnnPrimitiveAttributes_t attributes,
                const dnnLayout_t dataLayout, double eps,
                unsigned int flags);

        dnnError_t dnnPoolingCreateForward_F64(
                dnnPrimitive_t* pPooling,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t op,
                const dnnLayout_t srcLayout,
                const size_t kernelSize[], const size_t kernelStride[],
                const int inputOffset[], const dnnBorder_t borderType);
        dnnError_t dnnPoolingCreateBackward_F64(
                dnnPrimitive_t* pPooling,
                dnnPrimitiveAttributes_t attributes,
                dnnAlgorithm_t op,
                const dnnLayout_t srcLayout,
                const size_t kernelSize[], const size_t kernelStride[],
                const int inputOffset[], const dnnBorder_t borderType);

        dnnError_t dnnInnerProductCreateForward_F64(
                dnnPrimitive_t *pInnerProduct,
                dnnPrimitiveAttributes_t attributes,
                size_t dimensions,
                const size_t srcSize[],
                size_t outputChannels);
        dnnError_t dnnInnerProductCreateForwardBias_F64(
                dnnPrimitive_t *pInnerProduct,
                dnnPrimitiveAttributes_t attributes,
                size_t dimensions,
                const size_t srcSize[],
                size_t outputChannels);
        dnnError_t dnnInnerProductCreateBackwardData_F64(
                dnnPrimitive_t *pInnerProduct,
                dnnPrimitiveAttributes_t attributes,
                size_t dimensions,
                const size_t srcSize[],
                size_t outputChannels);
        dnnError_t dnnInnerProductCreateBackwardFilter_F64(
                dnnPrimitive_t *pInnerProduct,
                dnnPrimitiveAttributes_t attributes,
                size_t dimensions,
                const size_t srcSize[],
                size_t outputChannels);
        dnnError_t dnnInnerProductCreateBackwardBias_F64(
                dnnPrimitive_t *pInnerProduct,
                dnnPrimitiveAttributes_t attributes,
                size_t dimensions,
                const size_t dstSize[]);
    }
    """
    return header
