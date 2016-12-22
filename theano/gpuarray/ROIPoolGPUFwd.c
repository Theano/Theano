#section kernels

#kernel ROIPoolGPUFwd_kernel : size, *, size, size, size, size, size, size, *, *, * :
KERNEL void ROIPoolGPUFwd_kernel(
    ga_size nloops, DTYPE_i0 * bottom_data,
    DTYPE_i0 spatial_scale, ga_size channels, ga_size height,
    ga_size width, ga_size pooled_height, ga_size pooled_width,
    DTYPE_i0* bottom_rois, DTYPE_i0* top_data, DTYPE_i0* argmax_data) {
    
    for (ga_size index = 0; index < nloops; ++index) {

        ga_size pw = index % pooled_width;
        ga_size ph = (index / pooled_width) % pooled_height;
        ga_size c = (index / pooled_width / pooled_height) % channels;
        ga_size n = index / pooled_width / pooled_height / channels;
        bottom_rois += n * 5;
        ga_size roi_batch_index = bottom_rois[0];
        ga_size roi_start_w = round(bottom_rois[1] * spatial_scale);
        ga_size roi_start_h = round(bottom_rois[2] * spatial_scale);
        ga_size roi_end_w = round(bottom_rois[3] * spatial_scale);
        ga_size roi_end_h = round(bottom_rois[4] * spatial_scale);
        ga_size roi_width = fmax((DTYPE_i0) (roi_end_w - roi_start_w + 1) * 1.0, 1.0);
        ga_size roi_height = fmax((DTYPE_i0) (roi_end_h - roi_start_h + 1) * 1.0, 1.0);
        DTYPE_i0 bin_size_h = (DTYPE_i0)(roi_height)
                           / (DTYPE_i0)(pooled_height);
        DTYPE_i0 bin_size_w = (DTYPE_i0)(roi_width)
                           / (DTYPE_i0)(pooled_width);

        ga_size hstart = (ga_size)(floor((DTYPE_i0)ph * bin_size_h));
        ga_size wstart = (ga_size)(floor((DTYPE_i0)pw * bin_size_w));
        ga_size hend = (ga_size)(ceil((DTYPE_i0)(ph + 1)
                                         * bin_size_h));
        ga_size wend = (ga_size)(ceil((DTYPE_i0)(pw + 1)
                                         * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = fmin(fmax((DTYPE_i0) (hstart + roi_start_h) * 1.0, 0), (DTYPE_i0) height * 1.0);
        hend = fmin(fmax((DTYPE_i0) (hend + roi_start_h) * 1.0, 0), (DTYPE_i0) height * 1.0);
        wstart = fmin(fmax((DTYPE_i0) (wstart + roi_start_w) * 1.0, 0), (DTYPE_i0) width * 1.0);
        wend = fmin(fmax((DTYPE_i0) (wend + roi_start_w) * 1.0, 0), (DTYPE_i0) width * 1.0);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        DTYPE_i0 maxval = is_empty ? 0 : - 100000.0;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        ga_ssize maxidx = -1;
        bottom_data += (roi_batch_index * channels + c) * height * width;
        for (ga_size h = hstart; h < hend; ++h) {
            for (ga_size w = wstart; w < wend; ++w) {
            ga_size bottom_index = h * width + w;
                if (bottom_data[bottom_index] > maxval) {
                  maxval = bottom_data[bottom_index];
                  maxidx = bottom_index;
                }
            }
        }
        top_data[index] = maxval;
        argmax_data[index] = maxidx;
    }
}

#section support_code_struct

int APPLY_SPECIFIC(ROIPoolGPUFwd)(PyGpuArrayObject* features,
                      PyGpuArrayObject* rois,
                      PyGpuArrayObject** out,
                      PyGpuArrayObject** argmaxes,
                      ) {
  size_t batch_size = PyGpuArray_DIMS(rois)[0];
  size_t channels = PyGpuArray_DIMS(features)[1];
  size_t height = PyGpuArray_DIMS(features)[2];
  size_t width = PyGpuArray_DIMS(features)[3];

  // Prepare outputs.
  size_t dims[] = {0, 0, 0, 0};
  dims[0] = batch_size;
  dims[1] = channels;
  dims[2] = POOLED_HEIGHT;
  dims[3] = POOLED_WIDTH;

  size_t num_kernel = batch_size * channels * POOLED_HEIGHT * POOLED_WIDTH;
  int err;

  err = ROIPoolGPUFwd_kernel_scall(1, &num_kernel, 0,
          num_kernel, features->ga.data, SPATIAL_SCALE, channels, height, width,
          POOLED_HEIGHT, POOLED_WIDTH, rois->ga.data, (*out)->ga.data,
          (*argmaxes)->ga.data);
  if (err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError,
                 "gpuarray error: ROIPoolGPUFwd_kernel: %s.",
                 GpuKernel_error(&k_ROIPoolGPUFwd_kernel, err));
    return -1;
  }

  return 0;
}
