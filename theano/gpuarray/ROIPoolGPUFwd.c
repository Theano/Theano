#section kernels

#kernel ROIPoolGPUFwd_kernel : ssize, *, ssize, ssize, ssize, ssize, ssize, ssize, *, *, * :
KERNEL void ROIPoolGPUFwd_kernel(
    const ga_int num_rois, GLOBAL_MEM DTYPE_i0 *bottom_data,
    const DTYPE_i0 spatial_scale, const ga_int channels, const ga_int height,
    const ga_int width, const ga_int pooled_height, const ga_int pooled_width,
    GLOBAL_MEM DTYPE_i0 *bottom_rois, GLOBAL_MEM DTYPE_i0 *top_data, GLOBAL_MEM DTYPE_i0 *argmax_data) {
    
    printf("spatial_scale is %lf\n", spatial_scale);
    for (ga_int index = 0; index < num_rois; ++index) {
        ga_int out_inc = index * channels * pooled_width * pooled_height;
        // Incrementing the pointers by respective ROI channel.
        top_data += out_inc;
        argmax_data += out_inc;
        bottom_rois += index * 5;
        ga_int roi_start_w = floor(bottom_rois[1] * spatial_scale + 0.5);
        ga_int roi_start_h = floor(bottom_rois[2] * spatial_scale + 0.5);
        ga_int roi_end_w = floor(bottom_rois[3] * spatial_scale + 0.5);
        ga_int roi_end_h = floor(bottom_rois[4] * spatial_scale + 0.5);
        ga_int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        ga_int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        DTYPE_i0 bin_size_h = static_cast<DTYPE_i0>(roi_height) / static_cast<DTYPE_i0>(pooled_height);
        DTYPE_i0 bin_size_w = static_cast<DTYPE_i0>(roi_width) / static_cast<DTYPE_i0>(pooled_width);
        for (ga_int c = 0; c < channels; ++c) {
            ga_int data_inc = c * height * width;
            ga_int out_channel_inc = c * pooled_height * pooled_width;
            DTYPE_i0* channel_data = bottom_data + data_inc;
            DTYPE_i0* channel_out = top_data + out_channel_inc;
            DTYPE_i0* channel_argmax = argmax_data + out_channel_inc;
            for (ga_int ph = 0; ph < pooled_height; ++ph) {
                for (ga_int pw = 0; pw < pooled_width; ++pw) {
                    ga_int hstart = static_cast<ga_int>(floor(static_cast<DTYPE_i0>(ph) * bin_size_h));
                    ga_int wstart = static_cast<ga_int>(floor(static_cast<DTYPE_i0>(pw) * bin_size_w));
                    ga_int hend = static_cast<ga_int>(ceil(static_cast<DTYPE_i0>(ph + 1) * bin_size_h));
                    ga_int wend = static_cast<ga_int>(ceil(static_cast<DTYPE_i0>(pw + 1) * bin_size_w));
                    // Add roi offsets and clip to input boundaries
                    hstart = min(max(hstart + roi_start_h, 0), height);
                    hend = min(max(hend + roi_start_h, 0), height);
                    wstart = min(max(wstart + roi_start_w, 0), width);
                    wend = min(max(wend + roi_start_w, 0), width);
                    bool is_empty = (hend <= hstart) || (wend <= wstart);
                    ga_int pool_index = ph * pooled_width + pw;
                    if (is_empty) {
                        channel_out[pool_index] = 0;
                        channel_argmax[pool_index] = -1;
                    }
                    // Define an empty pooling region to be zero
                    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
                    for (ga_int h = hstart; h < hend; ++h) {
                        for (ga_int w = wstart; w < wend; ++w) {
                            ga_int bottom_index = h * width + w;
                            if (channel_data[bottom_index] > channel_out[pool_index]) {
                                channel_out[pool_index] = channel_data[bottom_index];
                                channel_argmax[pool_index] = bottom_index;
                            }
                        }
                    }
                }
            }
        }
    }
}

#section support_code_struct

bool vector_same_shape(PyGpuArrayObject* arr1, PyGpuArrayObject* arr2){
  return (PyGpuArray_DIMS(arr1)[0] == PyGpuArray_DIMS(arr2)[0]);
  }

int APPLY_SPECIFIC(ROIPoolGPUFwd)(PyGpuArrayObject *data,
                      PyGpuArrayObject *rois,
                      PyGpuArrayObject **argmaxes,
		              PyGpuArrayObject **out,
                      PyGpuContextObject *ctx) {
    size_t address = 1;
    int num_rois = PyGpuArray_DIMS(rois)[0];
    int batch_size =  PyGpuArray_DIMS(data)[0];
    int channels = PyGpuArray_DIMS(data)[1];
    int height = PyGpuArray_DIMS(data)[2];
    int width = PyGpuArray_DIMS(data)[3];

    // Prepare outputs.
    int err;

    if (!GpuArray_IS_C_CONTIGUOUS(&data->ga) || !GpuArray_IS_C_CONTIGUOUS(&rois->ga)){
        PyErr_Format(PyExc_ValueError, "GpuRoIPoolOp: requires data to be C-contiguous");
        return 1;
    }

    if (*out != NULL || *argmaxes != NULL || (!vector_same_shape(data, *out)) || (!vector_same_shape(data, *argmaxes))){
    Py_XDECREF(*out);
    Py_XDECREF(*argmaxes);
    size_t dim[4];
    dim[0] = batch_size;
    dim[1] = num_rois;
    dim[2] = channels;
    dim[3] = POOLED_HEIGHT * POOLED_WIDTH;
    *out = (PyGpuArrayObject*) pygpu_zeros(4, dim, data->ga.typecode, GA_C_ORDER, ctx, Py_None);
    *argmaxes = (PyGpuArrayObject*) pygpu_zeros(4, dim, data->ga.typecode, GA_C_ORDER, ctx, Py_None);
    if (*out == NULL || *argmaxes == NULL) {
      PyErr_Format(PyExc_ValueError, "Could not allocate output storage");
      return 1;
    }
  }

  err = ROIPoolGPUFwd_kernel_scall(1, &address, 0,
          num_rois, data->ga.data, SPATIAL_SCALE, channels, height, width,
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
