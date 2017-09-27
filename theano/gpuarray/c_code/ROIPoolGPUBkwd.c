
#section kernels

#kernel ROIPoolGPUBkwd_kernel : int32, int32, *, *, *, float32, int32, int32, int32, int32, int32, * :
KERNEL void ROIPoolGPUBkwd_kernel(
  const ga_int nthreads, const ga_int num_rois, GLOBAL_MEM DTYPE_INPUT_1 *bottom_rois, GLOBAL_MEM DTYPE_INPUT_2 *argmax_data,
  GLOBAL_MEM DTYPE_INPUT_3 *top_diff, const DTYPE_INPUT_1 spatial_scale, const ga_int channels, const ga_int height,
  const ga_int width, const ga_int pooled_height, const ga_int pooled_width, GLOBAL_MEM DTYPE_OUTPUT_0 *bottom_diff) {

  // (n, c, h, w) coords in bottom data
  // Accumulate gradient over all ROIs that pooled this element
  for (ga_size index = GID_0 * LDIM_0 + LID_0; index < nthreads; index += LDIM_0 * GDIM_0) {
    ga_int w = index % width;
    ga_int h = (index / width) % height;
    ga_int c = (index / width / height) % channels;
    ga_int n = index / width / height / channels;

    DTYPE_OUTPUT_0 gradient = 0;
    for (ga_int roi_n = 0; roi_n < num_rois; ++roi_n) {

      const DTYPE_INPUT_1* offset_bottom_rois = bottom_rois + roi_n * 5;
      ga_int roi_batch_ind = static_cast<ga_int> (offset_bottom_rois[0]);

      if (n != roi_batch_ind) {
        continue;
      }

      ga_int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      ga_int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      ga_int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      ga_int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      ga_int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const DTYPE_INPUT_3* offset_top_diff = top_diff + offset;
      const DTYPE_INPUT_2* offset_argmax_data = argmax_data + offset;

      ga_int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      ga_int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      DTYPE_INPUT_1 bin_size_h = static_cast<DTYPE_INPUT_1>(roi_height) / static_cast<DTYPE_INPUT_1>(pooled_height);
      DTYPE_INPUT_1 bin_size_w = static_cast<DTYPE_INPUT_1>(roi_width) / static_cast<DTYPE_INPUT_1>(pooled_width);

      ga_int phstart = static_cast<ga_int>(floor(static_cast<DTYPE_INPUT_1>(h - roi_start_h) / bin_size_h));
      ga_int phend = static_cast<ga_int>(ceil(static_cast<DTYPE_INPUT_1>(h - roi_start_h + 1) / bin_size_h));
      ga_int pwstart = static_cast<ga_int>(floor(static_cast<DTYPE_INPUT_1>(w - roi_start_w) / bin_size_w));
      ga_int pwend = static_cast<ga_int>(ceil(static_cast<DTYPE_INPUT_1>(w - roi_start_w + 1) / bin_size_w));

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (ga_int ph = phstart; ph < phend; ++ph) {
        for (ga_int pw = pwstart; pw < pwend; ++pw) {
          if (static_cast<ga_int>(offset_argmax_data[ph * pooled_width + pw]) == (h * width + w)) {
            gradient += static_cast<DTYPE_OUTPUT_0>(offset_top_diff[ph * pooled_width + pw]);
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}


#section support_code_struct

bool vector_same_shape(PyGpuArrayObject* arr1, PyGpuArrayObject* arr2){
  return (PyGpuArray_DIMS(arr1)[0] == PyGpuArray_DIMS(arr2)[0]);
  }


int APPLY_SPECIFIC(ROIPoolGPUBkwd)(PyGpuArrayObject *data,
                           PyGpuArrayObject *rois,
                           PyGpuArrayObject *argmaxes,
                           PyGpuArrayObject *out_grad,
                           PyGpuArrayObject **out,
                           PyGpuContextObject *ctx) {

    size_t num_rois = PyGpuArray_DIMS(rois)[0];
    size_t channels = PyGpuArray_DIMS(data)[1];
    size_t height = PyGpuArray_DIMS(data)[2];
    size_t width = PyGpuArray_DIMS(data)[3];
    int err;
    size_t batch_size = PyGpuArray_DIMS(data)[0];
    size_t dim[4];

    if (!GpuArray_IS_C_CONTIGUOUS(&data->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&rois->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&argmaxes->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&out_grad->ga))
    {
      PyErr_Format(PyExc_ValueError,
                   "GpuRoIPoolGradOp: requires data to be C-contiguous");
      return 1;
    }

  if (*out == NULL || !vector_same_shape(data, *out)){
    Py_XDECREF(*out);
    dim[0] = batch_size;
    dim[1] = channels;
    dim[2] = height;
    dim[3] = width;
    *out = (PyGpuArrayObject*) pygpu_zeros(4, dim, data->ga.typecode, GA_C_ORDER, ctx, Py_None);
    if (*out == NULL) {
      PyErr_Format(PyExc_ValueError, "Could not allocate output storage");
      return 1;
    }
  }

  size_t nthreads = batch_size * channels * height * width;

  err = ROIPoolGPUBkwd_kernel_scall(1, &nthreads, 0, nthreads, num_rois,
    rois->ga.data, argmaxes->ga.data, out_grad->ga.data, SPATIAL_SCALE, channels,
    height, width, POOLED_HEIGHT, POOLED_WIDTH, (*out)->ga.data);

  if (err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError,
                 "gpuarray error: ROIPoolGPUBkwd_kernel: %s.",
                 GpuKernel_error(&k_ROIPoolGPUBkwd_kernel, err));
    return -1;
  }
  return 0;
}