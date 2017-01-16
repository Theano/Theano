
#section kernels

#kernel ROIPoolGPUBkwd_kernel : size, *, *, size, size, size, size, size, size, size, *, * :
KERNEL void ROIPoolGPUBkwd_kernel(
    ga_size nloops, DTYPE_i0* top_diff,
    DTYPE_i0* argmax_data, ga_size num_rois, DTYPE_i0 spatial_scale,
    ga_size channels, ga_size height, ga_size width,
    ga_size pooled_height, ga_size pooled_width,
    DTYPE_i0* bottom_diff,
    DTYPE_i0* bottom_rois) {
    for (ga_size index = 0; index < nloops; ++index) {
        // (n, c, h, w) coords in bottom data
        ga_size w = index % width;
        ga_size h = (index / width) % height;
        ga_size c = (index / width / height) % channels;
        ga_size n = index / width / height / channels;

        DTYPE_i0 gradient = 0;
        // Accumulate gradient over all ROIs that pooled this element
        for (ga_size roi_n = 0; roi_n < num_rois; ++roi_n) {
            const DTYPE_i0* offset_bottom_rois = bottom_rois + roi_n * 5;
            ga_size roi_batch_index = offset_bottom_rois[0];
            if (n != roi_batch_index) {
            continue;
            }

            ga_size roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
            ga_size roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
            ga_size roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
            ga_size roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

            // Skip if ROI doesn't include (h, w)
            const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                               h >= roi_start_h && h <= roi_end_h);
            if (!in_roi) {
            continue;
            }

            ga_size offset = (roi_n * channels + c) * pooled_height * pooled_width;
            const DTYPE_i0* offset_top_diff = top_diff + offset;
            const DTYPE_i0* offset_argmax_data = argmax_data + offset;

            // Compute feasible set of pooled units that could have pooled
            // this bottom unit

            // Force malformed ROIs to be 1x1
            ga_size roi_width = (ga_size) (fmax((DTYPE_i0) (roi_end_w - roi_start_w + 1) * 1.0, 1));
            ga_size roi_height =  (ga_size) (fmax((DTYPE_i0) (roi_end_h - roi_start_h + 1) * 1.0, 1));

            DTYPE_i0 bin_size_h = (DTYPE_i0)roi_height
                             / (DTYPE_i0)pooled_height;
            DTYPE_i0 bin_size_w = (DTYPE_i0)roi_width
                             / (DTYPE_i0)pooled_width;

            ga_size phstart = floor((DTYPE_i0)(h - roi_start_h) / bin_size_h);
            ga_size phend = ceil((DTYPE_i0)(h - roi_start_h + 1) / bin_size_h);
            ga_size pwstart = floor((DTYPE_i0)(w - roi_start_w) / bin_size_w);
            ga_size pwend = ceil((DTYPE_i0)(w - roi_start_w + 1) / bin_size_w);

            phstart = fmin(fmax((DTYPE_i0)phstart * 1.0, 0), (DTYPE_i0) pooled_height * 1.0);
            phend = fmin(fmax((DTYPE_i0) phend * 1.0, 0), (DTYPE_i0) pooled_height * 1.0);
            pwstart = fmin(fmax((DTYPE_i0) pwstart * 1.0, 0), (DTYPE_i0) pooled_width * 1.0);
            pwend = fmin(fmax((DTYPE_i0) pwend * 1.0, 0), (DTYPE_i0) pooled_width * 1.0);

            for (ga_size ph = phstart; ph < phend; ++ph) {
                for (ga_size pw = pwstart; pw < pwend; ++pw) {
                  if ((ga_size)(offset_argmax_data[ph * pooled_width + pw]) == 
                      (h * width + w)) {
                    gradient += offset_top_diff[ph * pooled_width + pw];
                  }
                }
            }
        }

        bottom_diff[index] = gradient;
    }
}

#section support_code_struct

int APPLY_SPECIFIC(ROIPoolGPUBkwd)(PyGpuArrayObject *data,
                           PyGpuArrayObject *rois,
                           PyGpuArrayObject *argmaxes,
                           PyGpuArrayObject **out_grad,
                           PyGpuArrayObject **out,
                           PyGpuContextObject *ctx) {
    size_t num_kernel = PyGpuArray_SIZE(data);
    size_t batch_size = PyGpuArray_DIMS(rois)[0];
    size_t channels = PyGpuArray_DIMS(data)[1];
    size_t height = PyGpuArray_DIMS(data)[2];
    size_t width = PyGpuArray_DIMS(data)[3];
    int err;

    if (!GpuArray_IS_C_CONTIGUOUS(&data->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&rois->ga)
      || !GpuArray_IS_C_CONTIGUOUS(&argmaxes->ga))
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
    num_kernel, (*out_grad)->ga.data, argmaxes->ga.data, batch_size,
    SPATIAL_SCALE, channels, height, width, POOLED_HEIGHT, POOLED_WIDTH, (*out)->ga.data, rois->ga.data);
  if (err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError,
                 "gpuarray error: ROIPoolGPUBkwd_kernel: %s.",
                 GpuKernel_error(&k_ROIPoolGPUBkwd_kernel, err));
    return -1;
  }
  return 0;
}