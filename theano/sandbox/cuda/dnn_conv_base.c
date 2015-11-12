#section support_code_struct
cudnnTensorDescriptor_t APPLY_SPECIFIC(input);
cudnnTensorDescriptor_t APPLY_SPECIFIC(output);
cudnnFilterDescriptor_t APPLY_SPECIFIC(kerns);

/* Keep track, from one execution to another, of the dimension of the data
and the algorithms, if any, that were selected according to these dimensions
and according to the amount of memory available at that time.

Note : Implementation selection for backward convolution only exists starting
at V3.
*/
int APPLY_SPECIFIC(previous_input_shape)[5];
int APPLY_SPECIFIC(previous_kerns_shape)[5];
int APPLY_SPECIFIC(previous_output_shape)[5];
bool APPLY_SPECIFIC(previous_algo_set);
cudnnConvolutionFwdAlgo_t APPLY_SPECIFIC(previous_algo);
cudnnConvolutionBwdFilterAlgo_t APPLY_SPECIFIC(previous_bwd_f_algo);
cudnnConvolutionBwdDataAlgo_t APPLY_SPECIFIC(previous_bwd_d_algo);

#section init_code_struct

cudnnStatus_t APPLY_SPECIFIC(err);
APPLY_SPECIFIC(input) = NULL;
APPLY_SPECIFIC(output) = NULL;
APPLY_SPECIFIC(kerns) = NULL;
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(input))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
	       "(inp): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(output))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor descriptor "
               "(out): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateFilterDescriptor(&APPLY_SPECIFIC(kerns))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate filter descriptor: %s",
	       cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}

for (int i = 0; i < 5; i++)
{
  APPLY_SPECIFIC(previous_input_shape)[i] = 0;
  APPLY_SPECIFIC(previous_kerns_shape)[i] = 0;
  APPLY_SPECIFIC(previous_output_shape)[i] = 0;
}

APPLY_SPECIFIC(previous_algo_set) = false;

// Select default implementations for the case where the convolution
// implementations should be selected based on the size of the data.
APPLY_SPECIFIC(previous_algo) = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
APPLY_SPECIFIC(previous_bwd_f_algo) = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
APPLY_SPECIFIC(previous_bwd_d_algo) = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

#section cleanup_code_struct

if (APPLY_SPECIFIC(input) != NULL)
  cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(input));
if (APPLY_SPECIFIC(output) != NULL)
  cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(output));
if (APPLY_SPECIFIC(kerns) != NULL)
  cudnnDestroyFilterDescriptor(APPLY_SPECIFIC(kerns));
