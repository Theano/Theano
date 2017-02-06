#section support_code_apply


#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <stdlib.h>
#include <limits.h>


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

    const int out_inc = index * channels * pooled_width * pooled_height;
    // Incrementing the output pointers and the ROI by respective ROI channel.
    top_data += out_inc;
    argmax_data += out_inc;
    bottom_rois += index * 5;

    int roi_start_w = floorf(bottom_rois[1] * spatial_scale + 0.5);
    int roi_start_h = floorf(bottom_rois[2] * spatial_scale + 0.5 );
    int roi_end_w = floorf(bottom_rois[3] * spatial_scale + 0.5);
    int roi_end_h = floorf(bottom_rois[4] * spatial_scale + 0.5);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    const float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    const float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);
    for (int c = 0; c < channels; ++c) {
      const int data_inc = c * height * width;
      const int out_channel_inc = c * pooled_height * pooled_width;
      float* channel_data = bottom_data + data_inc;
      float* channel_out = top_data + out_channel_inc;
      float* channel_argmax = argmax_data + out_channel_inc;
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));
          // Add roi offsets and clip to input boundaries
          hstart = min(max(hstart + roi_start_h, 0), height);
          hend = min(max(hend + roi_start_h, 0), height);
          wstart = min(max(wstart + roi_start_w, 0), width);
          wend = min(max(wend + roi_start_w, 0), width);
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          const int pool_index = ph * pooled_width + pw;
          if (is_empty) {
            channel_out[pool_index] = 0;
            channel_argmax[pool_index] = -1;
          }
          // Define an empty pooling region to be zero
          // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int bottom_index = h * width + w;
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

  // Checking if contiguous
  if(!PyArray_ISCONTIGUOUS(data) || !PyArray_ISCONTIGUOUS(rois)){
    PyErr_Format(PyExc_ValueError, "RoIPoolGradOp: requires data to be C-contiguous");
    return 1;
  }
  if (*out != NULL || *argmaxes != NULL || (!vector_same_shape(data, *out)) || (!vector_same_shape(data, *argmaxes))){
    Py_XDECREF(*out);
    Py_XDECREF(*argmaxes);
    npy_intp dim[4];
    dim[0] = PyArray_DIMS(data)[0];
    dim[1] = num_rois;
    dim[2] = channels;
    dim[3] = POOLED_HEIGHT * POOLED_WIDTH;
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
    float* top_diff,
    float* argmax_data, int num_rois, float spatial_scale,
    int channels, int height, int width,
    int pooled_height, int pooled_width, float* bottom_diff,
    float* bottom_rois) {

  // (n, c, h, w) coords in bottom data
  float gradient = 0;
  // Accumulate gradient over all ROIs that pooled this element
  for (int roi_n = 0; roi_n < num_rois; ++roi_n) {

    const int out_inc = roi_n * channels * pooled_width * pooled_height;
    // Incrementing the pointers by respective ROI channel.
    top_diff += out_inc;
    argmax_data += out_inc;
    bottom_rois += roi_n * 5;
    for (int c = 0; c < channels; ++c) {
      const int data_inc = c * height * width;
      const int out_channel_inc = c * pooled_height * pooled_width;
      // incrementing the output dimension pointers
      float* channel_out = top_diff + out_channel_inc;
      float* channel_argmax = argmax_data + out_channel_inc;
      // increment input dimension pointers
      float* channel_grad = bottom_diff + data_inc;

      for (int h = 0; h < height; ++h){
        for(int w = 0; w < width; ++w){
          int roi_start_w = floorf(bottom_rois[1] * spatial_scale + 0.5);
          int roi_start_h = floorf(bottom_rois[2] * spatial_scale + 0.5 );
          int roi_end_w = floorf(bottom_rois[3] * spatial_scale + 0.5);
          int roi_end_h = floorf(bottom_rois[4] * spatial_scale + 0.5);
          int roi_width = max(roi_end_w - roi_start_w + 1, 1);
          int roi_height = max(roi_end_h - roi_start_h + 1, 1);

          // Skip if ROI doesn't include (h, w)
          const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                             h >= roi_start_h && h <= roi_end_h);
          if (!in_roi) {
            continue;
          }
          // Compute feasible set of pooled units that could have pooled
          // this bottom unit

          // Force malformed ROIs to be 1x1

          float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
          float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);
          int phstart = static_cast<int>(floor(static_cast<float>(h - roi_start_h) / bin_size_h));
          int phend = static_cast<int>(ceil(static_cast<float>(h - roi_start_h + 1) / bin_size_h));
          int pwstart = static_cast<int>(floor(static_cast<float>(w - roi_start_w) / bin_size_w));
          int pwend = static_cast<int>(ceil(static_cast<float>(w - roi_start_w + 1) / bin_size_w));

          phstart = min(max(phstart, 0), pooled_height);
          phend = min(max(phend, 0), pooled_height);
          pwstart = min(max(pwstart, 0), pooled_width);
          pwend = min(max(pwend, 0), pooled_width);
          for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
              if (channel_argmax[ph * pooled_width + pw] == 
                  (h * width + w)) {
                gradient += channel_out[ph * pooled_width + pw];
              }
            }
          }
        channel_grad[h * width + w] = gradient;
        }
      }
    }
  }
}


int APPLY_SPECIFIC(CPUBackward)(PyArrayObject* data,
                           PyArrayObject* rois,
                           PyArrayObject* argmaxes,
                           PyArrayObject* out_grad,
                           PyArrayObject** data_grad) {

  int num_rois = PyArray_DIMS(rois)[0];
  int channels = PyArray_DIMS(data)[1];
  int height = PyArray_DIMS(data)[2];
  int width = PyArray_DIMS(data)[3];
  int data_typenum = PyArray_ObjectType((PyObject*)(data), 0);

  int mem_nc;
  mem_nc = 0;
  int total_ndim = 4;
  //Checking Arrays are continious
  if(!PyArray_ISCONTIGUOUS(data) || !PyArray_ISCONTIGUOUS(rois) || !PyArray_ISCONTIGUOUS(argmaxes) || !PyArray_ISCONTIGUOUS(out_grad)){
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
      (float *)PyArray_DATA(out_grad), (float *)PyArray_DATA(argmaxes), num_rois , 
      SPATIAL_SCALE, channels, height, width, POOLED_HEIGHT, POOLED_WIDTH, 
      (float *)PyArray_DATA(*data_grad), (float *)PyArray_DATA(rois));

  return 0;
}