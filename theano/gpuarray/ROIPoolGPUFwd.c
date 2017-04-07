#section kernels

#kernel ROIPoolGPUFwd_kernel : int32, int32, int32, *, float32, int32, int32, int32, int32, int32, *, *, * :
KERNEL void ROIPoolGPUFwd_kernel(
    const ga_int nthreads,
    const ga_int batch_n, const ga_int num_rois, GLOBAL_MEM DTYPE_INPUT_0 *bottom_data,
    const DTYPE_INPUT_1 spatial_scale, const ga_int channels, const ga_int height,
    const ga_int width, const ga_int pooled_height, const ga_int pooled_width,
    GLOBAL_MEM DTYPE_INPUT_1 *bottom_rois, GLOBAL_MEM DTYPE_OUTPUT_0 *top_data, GLOBAL_MEM DTYPE_OUTPUT_1 *argmax_data) {
    
    for (ga_size index = GID_0 * LDIM_0 + LID_0; index < nthreads; index += LDIM_0 * GDIM_0) {

        ga_int pw = index % pooled_width;
        ga_int ph = (index / pooled_width) % pooled_height;
        ga_int c = (index / pooled_width / pooled_height) % channels;
        ga_int n = index / pooled_width / pooled_height / channels;
        bottom_rois += n * 5;
        ga_int roi_batch_ind = bottom_rois[0];
        ga_int roi_start_w = round(bottom_rois[1] * spatial_scale);
        ga_int roi_start_h = round(bottom_rois[2] * spatial_scale);
        ga_int roi_end_w = round(bottom_rois[3] * spatial_scale);
        ga_int roi_end_h = round(bottom_rois[4] * spatial_scale);
        ga_int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        ga_int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        DTYPE_INPUT_1 bin_size_h = static_cast<DTYPE_INPUT_1>(roi_height) / static_cast<DTYPE_INPUT_1>(pooled_height);
        DTYPE_INPUT_1 bin_size_w = static_cast<DTYPE_INPUT_1>(roi_width) / static_cast<DTYPE_INPUT_1>(pooled_width);
        ga_int hstart = static_cast<ga_int>(floor(static_cast<DTYPE_INPUT_1>(ph) * bin_size_h)) + roi_start_h;
        ga_int wstart = static_cast<ga_int>(floor(static_cast<DTYPE_INPUT_1>(pw) * bin_size_w)) + roi_start_w;
        ga_int hend = static_cast<ga_int>(ceil(bin_size_h)) + hstart;
        ga_int wend = static_cast<ga_int>(ceil(bin_size_w)) + wstart;
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);

        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        ga_int maxidx;
        DTYPE_INPUT_0 maxval = -99999999.0;
        if (is_empty) {
            maxval = 0;
            maxidx = -1;
        }
        bottom_data += (roi_batch_ind * channels + c) * height * width;
        for (ga_int h = hstart; h < hend; ++h) {
            for (ga_int w = wstart; w < wend; ++w) {
                ga_int bottom_index = h * width + w;
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

bool vector_same_shape(PyGpuArrayObject* arr1, PyGpuArrayObject* arr2){
  return (PyGpuArray_DIMS(arr1)[0] == PyGpuArray_DIMS(arr2)[0]);
  }

int APPLY_SPECIFIC(ROIPoolGPUFwd)(PyGpuArrayObject *data,
                      PyGpuArrayObject *rois,
                      PyGpuArrayObject **out,
		              PyGpuArrayObject **argmaxes,
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

  if (*out == NULL || !vector_same_shape(data, *out)){
    Py_XDECREF(*out);
    size_t dim[4];
    dim[0] = num_rois;
    dim[1] = channels;
    dim[2] = POOLED_HEIGHT;
    dim[3] = POOLED_WIDTH;
    *out = (PyGpuArrayObject*) pygpu_zeros(4, dim, data->ga.typecode, GA_C_ORDER, ctx, Py_None);
    if (*out == NULL) {
      PyErr_Format(PyExc_ValueError, "Could not allocate output storage");
      return 1;
    }
  }

  if (*argmaxes == NULL || !vector_same_shape(data, *argmaxes)){
    Py_XDECREF(*argmaxes);
    size_t dim[4];
    dim[0] = num_rois;
    dim[1] = channels;
    dim[2] = POOLED_HEIGHT;
    dim[3] = POOLED_WIDTH;
    *argmaxes = (PyGpuArrayObject*) pygpu_zeros(4, dim, data->ga.typecode, GA_C_ORDER, ctx, Py_None);
    if (*argmaxes == NULL) {
      PyErr_Format(PyExc_ValueError, "Could not allocate output storage");
      return 1;
    }
  }

  size_t nthreads = num_rois * channels * POOLED_HEIGHT * POOLED_WIDTH;

  err = ROIPoolGPUFwd_kernel_scall(1, &nthreads, 0, nthreads, batch_size,
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
