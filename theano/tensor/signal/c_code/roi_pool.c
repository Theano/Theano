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

bool CHECK_LT(float var1, float var2){
  (var1 < var2)?true:false;
}

bool CHECK_GE(float var1, float var2){
  (var1 >= var2)?true:false;
}

void APPLY_SPECIFIC(ROIPoolForward)(
    int batch_n, int num_rois, float* bottom_data,
    float spatial_scale, int channels, int height,
    int width, int pooled_height, int pooled_width,
    float* bottom_rois, float* top_data, float* argmax_data) {

  for (int index = 0; index < num_rois; ++index) {
    float* ind_roi = bottom_rois + index * 5;
    int roi_batch_ind = ind_roi[0];
    int roi_start_w = floorf(ind_roi[1] * spatial_scale + 0.5);
    int roi_start_h = floorf(ind_roi[2] * spatial_scale + 0.5);
    int roi_end_w = floorf(ind_roi[3] * spatial_scale + 0.5);
    int roi_end_h = floorf(ind_roi[4] * spatial_scale + 0.5);
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_n);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    const float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    const float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

    for (int c = 0; c < channels; ++c) {
      // Incrementing the data pointer to the required batch index
      float* batch_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
      float* batch_argmax = argmax_data + (index * channels + c) * pooled_height * pooled_width;
      float* batch_out = top_data + (index * channels + c) * pooled_height * pooled_width;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = static_cast<int>(floor(static_cast<float>(ph) * bin_size_h)) + roi_start_h;
          int wstart = static_cast<int>(floor(static_cast<float>(pw) * bin_size_w)) + roi_start_w;
          int hend = static_cast<int>(ceil(bin_size_h)) + hstart;
          int wend = static_cast<int>(ceil(bin_size_w)) + wstart;
          // Add roi offsets and clip to input boundaries
          hstart = min(max(hstart, 0), height);
          hend = min(max(hend, 0), height);
          wstart = min(max(wstart, 0), width);
          wend = min(max(wend, 0), width);
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          const int pool_index = ph * pooled_width + pw;
          // Define an empty pooling region to be zero
          // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
          if (is_empty) {
            batch_argmax[pool_index] = -1;
            batch_out[pool_index] = 0;
          }
          else{
            batch_out[pool_index]= -999999999.0;
          }
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int bottom_index = h * width + w;
              if (batch_data[bottom_index] > batch_out[pool_index]) {
                batch_out[pool_index] = batch_data[bottom_index];
                batch_argmax[pool_index] = bottom_index;
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
  int batch_n = PyArray_DIMS(data)[0];
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
  if (*out == NULL){
    Py_XDECREF(*out);
    Py_XDECREF(*argmaxes);
    npy_intp dim[4];
    dim[0] = num_rois;
    dim[1] = channels;
    dim[2] = POOLED_HEIGHT;
    dim[3] = POOLED_WIDTH;
    *out = (PyArrayObject*) PyArray_ZEROS(4, dim, data_typenum, 0);
    *argmaxes = (PyArrayObject*) PyArray_ZEROS(4, dim, data_typenum, 0);
    if (!*out || !*argmaxes) {
      PyErr_Format(PyExc_ValueError, "Could not allocate output storage");
      return 1;
    }
  }

  APPLY_SPECIFIC(ROIPoolForward)(
          batch_n, num_rois, (float *)PyArray_DATA(data), SPATIAL_SCALE, channels, height, width,
          POOLED_HEIGHT, POOLED_WIDTH, (float *)PyArray_DATA(rois), (float *)PyArray_DATA(*out), (float *)PyArray_DATA(*argmaxes));

  return 0;
}


void APPLY_SPECIFIC(ROIPoolBackward)(
    float* top_diff, float* argmax_data, float* bottom_diff, float* bottom_rois,
    int batch_n, int num_rois, float spatial_scale,
    int channels, int height, int width,
    int pooled_height, int pooled_width) {

  // (n, c, h, w) coords in bottom data
  // Accumulate gradient over all ROIs that pooled this element
  float gradient = 0;

  for (int roi_n = 0; roi_n < num_rois; ++roi_n) {

    // Incrementing the pointers by respective ROI channel.
    float* batch_roi = bottom_rois + roi_n * 5;
    int roi_batch_ind = static_cast<int> (batch_roi[0]);

    int roi_start_w = floorf(batch_roi[1] * spatial_scale + 0.5);
    int roi_start_h = floorf(batch_roi[2] * spatial_scale + 0.5);
    int roi_end_w = floorf(batch_roi[3] * spatial_scale + 0.5);
    int roi_end_h = floorf(batch_roi[4] * spatial_scale + 0.5);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
    float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

    for (int c = 0; c < channels; ++c) {
      float* batch_grad = bottom_diff + (roi_batch_ind * channels + c) * height * width;
      float* batch_argmax = argmax_data + (roi_n * channels + c) * pooled_height * pooled_width;
      float* batch_out = top_diff + (roi_n * channels + c) * pooled_height * pooled_width;

      for (int h = 0; h < height; ++h){
        for(int w = 0; w < width; ++w){

          int bottom_index = h * width + w;
          // Skip if ROI doesn't include (h, w)
          const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                             h >= roi_start_h && h <= roi_end_h);
          if (!in_roi) {
            continue;
          }
          // Compute feasible set of pooled units that could have pooled
          // this bottom unit

          // Force malformed ROIs to be 1x1

          int phstart = floor(static_cast<float>(h - roi_start_h) / bin_size_h);
          int phend = ceil(static_cast<float>(h - roi_start_h + 1) / bin_size_h);
          int pwstart = floor(static_cast<float>(w - roi_start_w) / bin_size_w);
          int pwend = ceil(static_cast<float>(w - roi_start_w + 1) / bin_size_w);

          phstart = min(max(phstart, 0), pooled_height);
          phend = min(max(phend, 0), pooled_height);
          pwstart = min(max(pwstart, 0), pooled_width);
          pwend = min(max(pwend, 0), pooled_width);

          for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
              int pool_index = ph * pooled_width + pw;
              if (static_cast<int>(batch_argmax[pool_index]) == bottom_index) {
                batch_grad[bottom_index] += batch_out[pool_index];
              }
            }
          }
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
  int data_typenum = PyArray_ObjectType((PyObject*)(out_grad), 0);
  int batch_n = PyArray_DIMS(data)[0];

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
      (float *)PyArray_DATA(out_grad), (float *)PyArray_DATA(argmaxes),(float *)PyArray_DATA(*data_grad), (float *)PyArray_DATA(rois),
      batch_n, num_rois, SPATIAL_SCALE, channels, height, width, POOLED_HEIGHT, POOLED_WIDTH);

  return 0;
}