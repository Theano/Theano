
#section kernels

#kernel ROIPoolGPUBkwd_kernel : *, *, *, *, int32, int32, float32, int32, int32, int32, int32, int32 :
KERNEL void ROIPoolGPUBkwd_kernel(
  DTYPE_i3* top_diff, DTYPE_i2* argmax_data, DTYPE_o0* bottom_diff, DTYPE_i1* bottom_rois,
  ga_int batch_n, ga_int num_rois, ga_float spatial_scale,
  ga_int channels, ga_int height, ga_int width,
  ga_int pooled_height, ga_int pooled_width) {

  // (n, c, h, w) coords in bottom data
  // Accumulate gradient over all ROIs that pooled this element
  for (ga_int bn = 0; bn < batch_n; ++bn){
    const ga_int inp_bn = bn * channels * height * width;
    const ga_int out_bn = bn * num_rois * channels * pooled_width * pooled_height;
    // Incrementing the input and output pointers by a batch
    DTYPE_o0* batch_grad = bottom_diff + inp_bn;
    DTYPE_i3* batch_out = top_diff + out_bn;
    DTYPE_i2* batch_argmax = argmax_data + out_bn;
    for (ga_int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const ga_int out_inc = roi_n * channels * pooled_width * pooled_height;
      // Incrementing the pointers by respective ROI channel.
      batch_out += out_inc;
      batch_argmax += out_inc;
      DTYPE_i1* batch_roi = bottom_rois + roi_n * 5;
      ga_int roi_start_w = floorf(batch_roi[1] * spatial_scale + 0.5);
      ga_int roi_start_h = floorf(batch_roi[2] * spatial_scale + 0.5);
      ga_int roi_end_w = floorf(batch_roi[3] * spatial_scale + 0.5);
      ga_int roi_end_h = floorf(batch_roi[4] * spatial_scale + 0.5);
      ga_int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      ga_int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      ga_float bin_size_h = static_cast<ga_float>(roi_height) / static_cast<ga_float>(pooled_height);
      ga_float bin_size_w = static_cast<ga_float>(roi_width) / static_cast<ga_float>(pooled_width);
      for (ga_int c = 0; c < channels; ++c) {
        const ga_int data_inc = c * height * width;
        const ga_int out_channel_inc = c * pooled_height * pooled_width;
        // incrementing the output dimension pointers
        DTYPE_i3* channel_out = batch_out + out_channel_inc;
        DTYPE_i2* channel_argmax = batch_argmax + out_channel_inc;
        // increment input dimension pointers
        DTYPE_o0* channel_grad = batch_grad + data_inc;
        for (ga_int h = 0; h < height; ++h){
          for(ga_int w = 0; w < width; ++w){
            ga_int bottom_index = h * width + w;
            // Skip if ROI doesn't include (h, w)
            const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                               h >= roi_start_h && h <= roi_end_h);
            if (!in_roi) {
              continue;
            }
            // Compute feasible set of pooled units that could have pooled
            // this bottom unit
            // Force malformed ROIs to be 1x1
            ga_int phstart = static_cast<ga_int>(floor(static_cast<ga_float>(h - roi_start_h) / bin_size_h));
            ga_int phend = static_cast<ga_int>(ceil(static_cast<ga_float>(h - roi_start_h + 1) / bin_size_h));
            ga_int pwstart = static_cast<ga_int>(floor(static_cast<ga_float>(w - roi_start_w) / bin_size_w));
            ga_int pwend = static_cast<ga_int>(ceil(static_cast<ga_float>(w - roi_start_w + 1) / bin_size_w));
            phstart = min(max(phstart, 0), pooled_height);
            phend = min(max(phend, 0), pooled_height);
            pwstart = min(max(pwstart, 0), pooled_width);
            pwend = min(max(pwend, 0), pooled_width);
            for (ga_int ph = phstart; ph < phend; ++ph) {
              for (ga_int pw = pwstart; pw < pwend; ++pw) {
                ga_int pool_index = ph * pooled_width + pw;
                if (static_cast<ga_int>(channel_argmax[pool_index]) == 
                    bottom_index) {
                  channel_grad[bottom_index] += channel_out[pool_index];
                }
              }
            }
          }
        }
      }
    }
  }
}

#section support_code_struct

int APPLY_SPECIFIC(ROIPoolGPUBkwd)(PyGpuArrayObject *data,
                           PyGpuArrayObject *rois,
                           PyGpuArrayObject *argmaxes,
                           PyGpuArrayObject *out_grad,
                           PyGpuArrayObject **out,
                           PyGpuContextObject *ctx) {
    size_t num_kernel = PyGpuArray_SIZE(data);
    size_t num_rois = PyGpuArray_DIMS(rois)[0];
    size_t channels = PyGpuArray_DIMS(data)[1];
    size_t height = PyGpuArray_DIMS(data)[2];
    size_t width = PyGpuArray_DIMS(data)[3];
    int err;
    size_t batch_size = PyGpuArray_DIMS(data)[0];

    if (!GpuArray_IS_C_CONTIGUOUS(&data->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&rois->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&argmaxes->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&out_grad->ga))
    {
      PyErr_Format(PyExc_ValueError,
                   "GpuRoIPoolGradOp: requires data to be C-contiguous");
      return 1;
    }

    if (theano_prep_output(out, PyGpuArray_NDIM(data), PyGpuArray_DIMS(data),
                         data->ga.typecode, GA_C_ORDER, ctx) != 0)
    {
      PyErr_SetString(PyExc_RuntimeError,
                      "GpuRoIPoolGradOp: failed to allocate memory");
      return 1;
    }

  err = ROIPoolGPUBkwd_kernel_scall(1, &num_kernel, 0,
    out_grad->ga.data, argmaxes->ga.data, (*out)->ga.data, rois->ga.data, batch_size, num_rois,
    SPATIAL_SCALE, channels, height, width, POOLED_HEIGHT, POOLED_WIDTH);
  if (err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError,
                 "gpuarray error: ROIPoolGPUBkwd_kernel: %s.",
                 GpuKernel_error(&k_ROIPoolGPUBkwd_kernel, err));
    return -1;
  }
  return 0;
}