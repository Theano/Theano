#section support_code_apply


#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>

#define max(a,b) (a>b?a:b)
#define min(a,b) (a<b?a:b)

void APPLY_SPECIFIC(ROIPoolForward)(
    const int nchannels, const float* bottom_data,
    const float spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const float* bottom_rois, float* top_data, float* argmax_data) {
    
    for (int index = 0; index < nchannels; ++index) {

        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c = (index / pooled_width / pooled_height) % channels;
        int n = index / pooled_width / pooled_height / channels;
        bottom_rois += n * 5;
        int roi_batch_index = bottom_rois[0];
        int roi_start_w = round(bottom_rois[1] * spatial_scale);
        int roi_start_h = round(bottom_rois[2] * spatial_scale);
        int roi_end_w = round(bottom_rois[3] * spatial_scale);
        int roi_end_h = round(bottom_rois[4] * spatial_scale);
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        float bin_size_h = (float)(roi_height)
                           / (float)(pooled_height);
        float bin_size_w = (float)(roi_width)
                           / (float)(pooled_width);

        int hstart = (int)(floor((float)ph * bin_size_h));
        int wstart = (int)(floor((float)pw * bin_size_w));
        int hend = (int)(ceil((float)(ph + 1)
                                         * bin_size_h));
        int wend = (int)(ceil((float)(pw + 1)
                                         * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        float maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        int maxidx = -1;
        bottom_data += (roi_batch_index * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
            int bottom_index = h * width + w;
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

int APPLY_SPECIFIC(GPUFwd)(PyGpuArrayObject* data,
                      PyGpuArrayObject* rois,
                      PyGpuArrayObject* out) {
  int batch_size = PyGpuArray_DIMS(rois)[0];
  int channels = PyGpuArray_DIMS(data)[1];
  int height = PyGpuArray_DIMS(data)[2];
  int width = PyGpuArray_DIMS(data)[3];

  // Prepare outputs.
  int dims[] = {0, 0, 0, 0};
  dims[0] = batch_size;
  dims[1] = channels;
  dims[2] = POOLED_HEIGHT;
  dims[3] = POOLED_WIDTH;

  int count = batch_size * channels * POOLED_HEIGHT * POOLED_WIDTH;


  APPLY_SPECIFIC(ROIPoolForward)(
          count, data->devdata, SPATIAL_SCALE, channels, height, width,
          POOLED_HEIGHT, POOLED_WIDTH, rois->devdata, (*out)->devdata);

  return 0;
}


void APPLY_SPECIFIC(ROIPoolBackward)(
    const int nchannels, const float* top_diff,
    const float* argmax_data, const int num_rois, const float spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, float* bottom_diff,
    const float* bottom_rois) {
    for (int index = 0; index < nchannels; ++index) {
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


int APPLY_SPECIFIC(GPUBackward)(PyGpuArrayObject* data,
                           PyGpuArrayObject* rois,
                           PyGpuArrayObject* out_grad,
                           PyGpuArrayObject** data_grad) {
    int count = gpuarray_get_elsize(data);
    int batch_size = PyGpuArray_DIMS(rois)[0];
    int channels = PyGpuArray_DIMS(data)[1];
    int height = PyGpuArray_DIMS(data)[2];
    int width = PyGpuArray_DIMS(data)[3];


    APPLY_SPECIFIC(ROIPoolBackward)(
        count, out_grad->devdata, batch_size, 
        SPATIAL_SCALE, channels, height, width, POOLED_HEIGHT, POOLED_WIDTH, 
        (*data_grad)->devdata, rois->devdata);

  return 0;
}