/*
Copyright (c) 2014, The Regents of the University of California (Regents)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met: 

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer. 
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

__kernel void im2col_kernel(const int num_kernels, __global float * data_im,
    const size_t height, const size_t width, const size_t kernel_h, const size_t kernel_w,
    const size_t pad_h, const size_t pad_w,
    const ptrdiff_t stride_h, const ptrdiff_t stride_w,
    const size_t height_col, const size_t width_col,
    __global float * data_col)
{
    for (int index = get_global_id(0); index < num_kernels; index += get_global_size(0)) 
    {
	int i = get_global_id(0);
	
    	int w_out = index % width_col;
    	int h_index = index / width_col;
    	int h_out = h_index % height_col;
    	int channel_in = h_index / height_col;
    	int channel_out = channel_in * kernel_h * kernel_w;
    	int h_in = h_out * stride_h - pad_h;
    	int w_in = w_out * stride_w - pad_w;

	float * data_col_ptr = data_col;
    	data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;

	float * data_im_ptr = data_im;
    	data_im_ptr += (channel_in * height + h_in) * width + w_in;

	// computation is done row-wise. Each work-item holds all elements of a row.
	if (i < kernel_h) 
	{
	    for (int j = 0; j < kernel_w; ++j)
	    {
        	int h = h_in + i;
        	int w = w_in + j;
        	* data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            	data_im_ptr[i * width + j] : 0;
        	data_col_ptr += height_col * width_col;
      	    }
    	}
    barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

