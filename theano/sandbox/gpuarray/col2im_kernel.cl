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

__kernel void col2im_kernel ( const int num_kernels, __global float * data_col,
    const size_t height, const size_t width, const size_t channels,
    const size_t patch_h, const size_t patch_w,
    const size_t pad_h, const size_t pad_w,
    const ptrdiff_t stride_h, const ptrdiff_t stride_w,
    const size_t height_col, const size_t width_col,
    __global float * data_im )
{
    for (int index = get_global_id(0); index < num_kernels; index += get_global_size(0))
    {
        float val = (float) 0.0;
    	int w = index % width + pad_w;
    	int h = (index / width) % height + pad_h;
    	int c = index / (width * height);
    	// compute the start and end of the output
    	int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    	int w_col_end = min(w / stride_w + 1, width_col);
    	int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    	int h_col_end = min(h / stride_h + 1, height_col);
	// equivalent implementation
    	int offset =
            (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
    	int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    	int coeff_w_col = (1 - stride_w * height_col * width_col);
    	for (int h_col = h_col_start; h_col < h_col_end; ++h_col) 
	{
     	    for (int w_col = w_col_start; w_col < w_col_end; ++w_col) 
	    {
        	val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      	    }
   	}    	
	data_im[index] = val;
	barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

