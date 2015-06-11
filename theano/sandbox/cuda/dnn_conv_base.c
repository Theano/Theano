#section support_code_struct
cudnnTensorDescriptor_t APPLY_SPECIFIC(input);
cudnnTensorDescriptor_t APPLY_SPECIFIC(output);
cudnnFilterDescriptor_t APPLY_SPECIFIC(kerns);

/* Keep track, from one execution to another, of the dimension of the inputs
and the algorithm, if any, that was selected according to these dimensions
and the amount of memory available at that time.
*/
int APPLY_SPECIFIC(previous_input_shape)[4];
int APPLY_SPECIFIC(previous_kerns_shape)[4];
cudnnConvolutionFwdAlgo_t APPLY_SPECIFIC(previous_algo);


#section init_code_struct

cudnnStatus_t APPLY_SPECIFIC(err);
APPLY_SPECIFIC(input) = NULL;
APPLY_SPECIFIC(output) = NULL;
APPLY_SPECIFIC(kerns) = NULL;
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(input))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor4d descriptor "
	       "(inp): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateTensorDescriptor(&APPLY_SPECIFIC(output))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor4d descriptor "
               "(out): %s", cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}
if ((APPLY_SPECIFIC(err) = cudnnCreateFilterDescriptor(&APPLY_SPECIFIC(kerns))) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate filter descriptor: %s",
	       cudnnGetErrorString(APPLY_SPECIFIC(err)));
  FAIL;
}

APPLY_SPECIFIC(previous_input_shape)[0] = 0;
APPLY_SPECIFIC(previous_input_shape)[1] = 0;
APPLY_SPECIFIC(previous_input_shape)[2] = 0;
APPLY_SPECIFIC(previous_input_shape)[3] = 0;
APPLY_SPECIFIC(previous_kerns_shape)[0] = 0;
APPLY_SPECIFIC(previous_kerns_shape)[1] = 0;
APPLY_SPECIFIC(previous_kerns_shape)[2] = 0;
APPLY_SPECIFIC(previous_kerns_shape)[3] = 0;
APPLY_SPECIFIC(previous_algo) = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

#section cleanup_code_struct

if (APPLY_SPECIFIC(input) != NULL)
  cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(input));
if (APPLY_SPECIFIC(output) != NULL)
  cudnnDestroyTensorDescriptor(APPLY_SPECIFIC(output));
if (APPLY_SPECIFIC(kerns) != NULL)
  cudnnDestroyFilterDescriptor(APPLY_SPECIFIC(kerns));
