#section support_code_apply


#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <stdlib.h>


#define max(a,b) (a>b?a:b)
#define min(a,b) (a<b?a:b)


bool vector_same_shape(PyArrayObject* arr1, PyArrayObject* arr2){
  return (PyArray_DIMS(arr1)[0] == PyArray_DIMS(arr2)[0]);
  }

void APPLY_SPECIFIC(ROIPoolForward)(
    int num_rois, float* bottom_data,
    float spatial_scale, int channels, int height,
    int width, int pooled_height, int pooled_width,
    float* bottom_rois, float* top_data, float* argmax_data) {

  for (int index = 0; index < num_rois; ++index) {

    int roi_batch_index = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    float bin_size_h = (float)((roi_height) / (pooled_height));
    float bin_size_w = (float)((roi_width) / (pooled_width));
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = static_cast<int>(floor((float)ph * bin_size_h));
          int wstart = static_cast<int>(floor((float)pw * bin_size_w));
          int hend = static_cast<int>(ceil((float)(ph + 1)* bin_size_h));
          int wend = static_cast<int>(ceil((float)(pw + 1)* bin_size_w));

          // Add roi offsets and clip to input boundaries
          hstart = min(max(hstart + roi_start_h, 0), height);
          hend = min(max(hend + roi_start_h, 0), height);
          wstart = min(max(wstart + roi_start_w, 0), width);
          wend = min(max(wend + roi_start_w, 0), width);
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }
          // Define an empty pooling region to be zero
          // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int bottom_index = h * width + w;
              if (bottom_data[bottom_index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[bottom_index];
                  argmax_data[pool_index] = bottom_index;
              }
            }
          }
        }
      }
      bottom_data += 1;
      top_data += 1;
      argmax_data += 1;

    }
    bottom_rois += 1;
  }
}

int APPLY_SPECIFIC(CPUFwd)(PyArrayObject* data,
                      PyArrayObject* rois,
                      PyArrayObject** out,
                      PyArrayObject** argmaxes) {
  int num_rois =  PyArray_DIMS(rois)[0];
  int channels = PyArray_DIMS(data)[1];
  int height = PyArray_DIMS(data)[2];
  int width = PyArray_DIMS(data)[3];
  int data_typenum = PyArray_ObjectType((PyObject*)(data), 0);

  // Prepare outputs.
  int dims[] = {0, 0, 0, 0};
  dims[0] = num_rois;
  dims[1] = channels;
  dims[2] = POOLED_HEIGHT;
  dims[3] = POOLED_WIDTH;
  int total_ndim = 4;
  int mem_nc;
  mem_nc = 0;
  // Checking if contiguous
  if(!PyArray_ISCONTIGUOUS(data) || !PyArray_ISCONTIGUOUS(rois)){
    PyErr_Format(PyExc_ValueError, "RoIPoolGradOp: requires data to be C-contiguous");
    return 1;
  }
  if (*out != NULL || *argmaxes != NULL || (!vector_same_shape(data, *out)) || (!vector_same_shape(data, *argmaxes))){
    Py_XDECREF(*out);
    Py_XDECREF(*argmaxes);
    npy_intp dim[4];
    for (int i=0; i<4; i++){
      dim[i] = PyArray_DIMS(data)[i];
    }
    *out = (PyArrayObject*) PyArray_ZEROS(PyArray_NDIM(data), dim, data_typenum, 0);
    *argmaxes = (PyArrayObject*) PyArray_ZEROS(PyArray_NDIM(data), dim, data_typenum, 0);
    if (!*out || !*argmaxes) {
      PyErr_Format(PyExc_ValueError, "Could not allocate output storage");
      return 1;
    }
  }

  APPLY_SPECIFIC(ROIPoolForward)(
          num_rois, (float *)PyArray_DATA(data), SPATIAL_SCALE, channels, height, width,
          POOLED_HEIGHT, POOLED_WIDTH, (float *)PyArray_DATA(rois), (float *)PyArray_DATA(*out), (float *)PyArray_DATA(*argmaxes));

  return 0;
}


void APPLY_SPECIFIC(ROIPoolBackward)(
    int nloops, float* top_diff,
    float* argmax_data, int num_rois, float spatial_scale,
    int channels, int height, int width,
    int pooled_height, int pooled_width, float* bottom_diff,
    float* bottom_rois) {
    for (int index = 0; index < nloops; ++index) {
        // (n, c, h, w) coords in bottom data
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height / channels;

        float gradient = 0;
        // Accumulate gradient over all ROIs that pooled this element
        for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
            const float* offset_bottom_rois = bottom_rois + roi_n * 5;
            int roi_batch_index = offset_bottom_rois[0];
            if (n != roi_batch_index) {
            continue;
            }

            int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
            int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
            int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
            int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

            // Skip if ROI doesn't include (h, w)
            const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                               h >= roi_start_h && h <= roi_end_h);
            if (!in_roi) {
            continue;
            }

            int offset = (roi_n * channels + c) * pooled_height * pooled_width;
            const float* offset_top_diff = top_diff + offset;
            const float* offset_argmax_data = argmax_data + offset;

            // Compute feasible set of pooled units that could have pooled
            // this bottom unit

            // Force malformed ROIs to be 1x1
            int roi_width = max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);

            float bin_size_h = (float)roi_height
                             / (float)pooled_height;
            float bin_size_w = (float)roi_width
                             / (float)pooled_width;

            int phstart = floor((float)(h - roi_start_h) / bin_size_h);
            int phend = ceil((float)(h - roi_start_h + 1) / bin_size_h);
            int pwstart = floor((float)(w - roi_start_w) / bin_size_w);
            int pwend = ceil((float)(w - roi_start_w + 1) / bin_size_w);

            phstart = min(max(phstart, 0), pooled_height);
            phend = min(max(phend, 0), pooled_height);
            pwstart = min(max(pwstart, 0), pooled_width);
            pwend = min(max(pwend, 0), pooled_width);

            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                  if ((int)(offset_argmax_data[ph * pooled_width + pw]) == 
                      (h * width + w)) {
                    gradient += offset_top_diff[ph * pooled_width + pw];
                  }
                }
            }
        }
        bottom_diff[index] = gradient;
    }
}


int APPLY_SPECIFIC(CPUBackward)(PyArrayObject* data,
                           PyArrayObject* rois,
                           PyArrayObject* argmaxes,
                           PyArrayObject* out_grad,
                           PyArrayObject** data_grad) {
  int count = PyArray_SIZE(data);
  int batch_size = PyArray_DIMS(rois)[0];
  int channels = PyArray_DIMS(data)[1];
  int height = PyArray_DIMS(data)[2];
  int width = PyArray_DIMS(data)[3];
  int data_typenum = PyArray_ObjectType((PyObject*)(data), 0);

  int mem_nc;
  mem_nc = 0;
  int total_ndim = 4;
  //Checking Arrays are continious
  if(!PyArray_ISCONTIGUOUS(data) || !PyArray_ISCONTIGUOUS(rois) || !PyArray_ISCONTIGUOUS(argmaxes)){
    PyErr_Format(PyExc_ValueError, "RoIPoolGradOp: requires data to be C-contiguous");
    return 1;
  }
  if (*data_grad == NULL || (!vector_same_shape(data, *data_grad))){
    Py_XDECREF(*data_grad);
    npy_intp dim[4];
    for (int i=0; i<4; i++){
      dim[i] = PyArray_DIMS(data)[i];
    }
    *data_grad = (PyArrayObject*) PyArray_ZEROS(PyArray_NDIM(data), dim, data_typenum, 0);
    if (!*data_grad) {
      PyErr_Format(PyExc_ValueError, "Could not allocate output storage");
      return 1;
    }
  }

  APPLY_SPECIFIC(ROIPoolBackward)(
      count, (float *)PyArray_DATA(out_grad), (float *)PyArray_DATA(argmaxes), batch_size , 
      SPATIAL_SCALE, channels, height, width, POOLED_HEIGHT, POOLED_WIDTH, 
      (float *)PyArray_DATA(*data_grad), (float *)PyArray_DATA(rois));

  return 0;
}