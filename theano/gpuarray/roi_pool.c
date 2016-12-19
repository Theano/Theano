#section kernels

#kernel ROIPoolGPUFwd_kernel : size, *, size, size, size, size, size, size, *, *, * :

KERNEL void ROIPoolGPUFwd_kernel(
    const ga_size nloops, GLOBAL_MEM const DTYPE_i0 * bottom_data,
    const DTYPE_i0 spatial_scale, const ga_size channels, const ga_size height,
    const ga_size width, const ga_size pooled_height, const ga_size pooled_width,
    const DTYPE_i0* bottom_rois, DTYPE_i0* top_data, DTYPE_i0* argmax_data) {
    
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
        ga_bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        DTYPE_i0 maxval = is_empty ? 0 : - 100000;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        ga_size maxidx = -1;
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

#kernel ROIPoolGPUBkwd_kernel : size, *, *, size, size, size, size, size, size, size, *, * :

KERNEL void ROIPoolGPUBkwd_kernel(
    const ga_size nloops, const DTYPE_i0* top_diff,
    const DTYPE_i0* argmax_data, const ga_size num_rois, const DTYPE_i0 spatial_scale,
    const ga_size channels, const ga_size height, const ga_size width,
    const ga_size pooled_height, const ga_size pooled_width, DTYPE_i0* bottom_diff,
    const DTYPE_i0* bottom_rois) {
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
            const ga_bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                               h >= roi_start_h && h <= roi_end_h);
            if (!in_roi) {
            continue;
            }

            ga_size offset = (roi_n * channels + c) * pooled_height * pooled_width;
            const ga_size* offset_top_diff = top_diff + offset;
            const ga_size* offset_argmax_data = argmax_data + offset;

            // Compute feasible set of pooled units that could have pooled
            // this bottom unit

            // Force malformed ROIs to be 1x1
            ga_size roi_width = fmax((DTYPE_i0) (roi_end_w - roi_start_w + 1) * 1.0, 1);
            ga_size roi_height =  fmax((DTYPE_i0) (roi_end_h - roi_start_h + 1) * 1.0, 1);

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

int APPLY_SPECIFIC(ROIPoolGPUFwd)(PyGpuArrayObject* data,
                      PyGpuArrayObject* rois,
                      PyGpuArrayObject** out,
                      PyGpuArrayObject** argmaxes
                      ) {
  size_t batch_size = PyGpuArray_DIMS(rois)[0];
  size_t channels = PyGpuArray_DIMS(data)[1];
  size_t height = PyGpuArray_DIMS(data)[2];
  size_t width = PyGpuArray_DIMS(data)[3];

  // Prepare outputs.
  size_t dims[] = {0, 0, 0, 0};
  dims[0] = batch_size;
  dims[1] = channels;
  dims[2] = POOLED_HEIGHT;
  dims[3] = POOLED_WIDTH;

  size_t num_kernel = batch_size * channels * POOLED_HEIGHT * POOLED_WIDTH;
  //Getting maximum number of threads per block
  
  size_t max_threads_dim;
  err = gpucontext_property(data->context->ctx, GA_CTX_PROP_MAXLSIZE, &max_threads_dim);
  if (err != GA_NO_ERROR){
    PyErr_Format(PyExc_RuntimeError,
        "Could not fetch max_threads_dim.");
    return NULL;
    }

  size_t threads_per_block = max_threads_dim;

  size_t n_blocks = (size_t) ((num_kernel + threads_per_block - 1) / threads_per_block);

  err = ROIPoolGPUFwd_kernel_call(1, &threads_per_block, &n_blocks, 0,
          num_kernel, data->ga.data, SPATIAL_SCALE, channels, height, width,
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

int APPLY_SPECIFIC(GPUBackward)(PyGpuArrayObject* data,
                           PyGpuArrayObject* rois,
                           PyGpuArrayObject** argmaxes,
                           PyGpuArrayObject* out_grad,
                           PyGpuArrayObject** data_grad) {
    size_t num_kernel = gpuarray_get_elsize(data);
    size_t batch_size = PyGpuArray_DIMS(rois)[0];
    size_t channels = PyGpuArray_DIMS(data)[1];
    size_t height = PyGpuArray_DIMS(data)[2];
    size_t width = PyGpuArray_DIMS(data)[3];

    size_t max_threads_dim;
    err = gpucontext_property(data->context->ctx, GA_CTX_PROP_MAXLSIZE, &max_threads_dim);
    if (err != GA_NO_ERROR){
        PyErr_Format(PyExc_RuntimeError,
            "Could not fetch max_threads_dim.");
        return NULL;
    }

  size_t threads_per_block = max_threads_dim;

  size_t n_blocks = (size_t) ((num_kernel + threads_per_block - 1) / threads_per_block);

  err = ROIPoolGPUBkwd_kernel_call(1, &threads_per_block, &n_blocks, 0,
    num_kernel, out_grad->ga.data, argmaxes->ga.data, batch_size,
    SPATIAL_SCALE, channels, height, width, POOLED_HEIGHT, POOLED_WIDTH,
    (*data_grad)->ga.data, rois->ga.data);
  if (err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError,
                 "gpuarray error: ROIPoolGPUFwd_kernel: %s.",
                 GpuKernel_error(&k_ROIPoolGPUBkwd_kernel, err));
    return -1;
  }
  return 0;
}