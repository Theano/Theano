// REMEMBER TO INCREASE c_code_cache_version when changing this file
//
//implement the valid convolution only

/*
for (int iter_m=0; iter_m < Os[0]; iter_m++) {
  // Reposition index into input image based on requested output size
  int pos_m = iter_m*%(self_dx)s;//The position of the patch in the image
  int new_m = (pos_m+dim_ker[0]-1);

  for (int iter_n=0; iter_n < Os[1]; iter_n++) {  // loop over columns
    int pos_n=iter_n*%(self_dy)s;
    %(type)s sum=0;

    // Sum over kernel, if index into image is out of bounds
    // fill with the value
    for (int j=0; j < dim_ker[0]; j++) {
      int inverse_row = (new_m-j);
      const %(type)s* idx_in=&in[inverse_row*dim_im[1]]; //JB: should be dim_im[1] right? (was dim_im[0])
      const %(type)s* idx_kern=&hvals[j*dim_ker[1]];
      int new_n = (pos_n+dim_ker[1]-1);
      for (int k=0,last=new_n; k < dim_ker[1]; k++,last--) {
        sum+=idx_kern[k]*idx_in[last];
      }
    }//for j
    out[iter_m*dim_zz[1]+iter_n] %(affectation)s sum;
  }//for n
 }//for m
*/
#ifndef CONV_KERNEL_CU
#define CONV_KERNEL_CU
#include <stdint.h>

/*
#define CHECK_BANK_CONFLICTS 0
#if CHECK_BANK_CONFLICTS
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif
*/

#define MIN(a, b) ((a) < (b) ? (a) : (b) )
#define MAX(a, b) ((a) < (b) ? (b) : (a) )

//Must be the same size as a ptr. We can't use unsigned long as on Windows 64
//bit, it is 32 bit.
const uintptr_t COALESCED_ALIGN = 0xFFFFFFFFFFFFFF00; // zero-out the trailing bits of pointers

__device__ void load_to_shared(float * dst, const float * src, const int thread_id, int nb_thread, const int N, const bool flipped=false){
  if (nb_thread < 64)
    {
      if(flipped)
        //TODO very slow on device before 1.3.
        //     make access to kern sequential and access to d_kern flipped.
        for(int i=thread_id;i<N;i+=nb_thread)
          dst[i]=src[N - 1 - i];
        //dst[N-1-i]=src[i];
      else
      {
        for(int i = thread_id; i < N; i += nb_thread)
        {
            dst[i] = src[i];
        }
      }
    }
  else
    {
      nb_thread = nb_thread & 0xFFFFFFE0; //make nb_thread a multiple of 32
      // Global memory:
      //  <-------------------------------------->
      //      A      A      A      A      A   // points of 256-byte alignment
      //         dddddddddddddddddddddd       // layout of src in global memory
      if (thread_id < nb_thread)
        {
          const float * my_src_ptr = (const float *)(
                  ((uintptr_t)src) & COALESCED_ALIGN);
          my_src_ptr += thread_id;
          while (my_src_ptr < src + N)
          {
              if (my_src_ptr >= src)
              {
                  int i = my_src_ptr - src;
                  if (flipped)
                  {
                      dst[N - 1 - i] = *my_src_ptr;
                  }
                  else
                  {
                      dst[i] = *my_src_ptr;
                  }
              }
              my_src_ptr += nb_thread;
          }
        }
    }
}

/*
 * We load from global memory to shared memory. The outer if is optimized away at compilation.
 */
__device__ void load_to_shared(float * dst, const float * src, const int thread_id,
                               int nb_thread, const int nb_col, const int nb_row,
                               const int stride_col, const int stride_row,
                               const bool flipped=false, const bool c_contiguous=true){
    if (c_contiguous)
    {
        load_to_shared(dst, src, thread_id, nb_thread, nb_col*nb_row, flipped);
    }
    else
    {
        if (flipped)
        {
            int LAST = nb_row * nb_col - 1;
            for(int i=thread_id;i<nb_row*nb_col;i+=nb_thread)
            {
                // XXX
                // THIS IS SLOW - use whatever blocks are in the the
                // threads to avoid division and modulo
                dst[LAST - i] \
                    = src[(i/nb_col)*stride_row+(i%nb_col)*stride_col];
            }
        }
        else
        {
            for(int i=thread_id;i<nb_row*nb_col;i+=nb_thread)
            {
                // XXX
                // THIS IS SLOW - use whatever blocks are in the the
                // threads to avoid division and modulo
                dst[i]=src[i/nb_col*stride_row+i%nb_col*stride_col];
            }
        }
    }
}

__device__ void fill(float * dst, int N, float value, int thread_id, int nb_thread){
  for(int i=thread_id;i<N;i+=nb_thread)
    dst[i]=value;
}

/*
 * We load from global memory to shared memory. The outer if is optimized away at compilation.
 * We put the image at the center of another one. Usefull to padd an image with 0.
 */
__device__ void load_padded_col_to_shared(float * dst, const float * src, 
                                          const int thread_id, const int nb_thread,
                                          const int nb_col, const int nb_row, 
                                          const int stride_col, const int stride_row,
                                          const int wid_pad, const bool c_contiguous=true){
  if(c_contiguous){//flipped==false
    for(int i=thread_id;i<nb_col*nb_row;i+=nb_thread){
      int col=i%nb_col;
      int row=i/nb_col;
      dst[row*(nb_col+2*wid_pad)+col+wid_pad]=src[i];
    }
    
  }else{
    for(int i=thread_id;i<nb_row*nb_col;i+=nb_thread){
      int col=i%nb_col;
      int row=i/nb_col;
      dst[row*(nb_col+2*wid_pad)+col+wid_pad]=src[row*stride_row+col*stride_col];
    }
  }

}

template<int i> __device__ float convolutionRowNoFlip(const float *data,
                                                      const float *kern){
    return convolutionRowNoFlip<i/2>(data, kern)+ convolutionRowNoFlip<(i+1)/2>(data+i/2, kern+i/2) ;
  //return data[i-1] * kern[i-1] + convolutionRowNoFlip<i - 1>(data,kern);
}

template<> __device__ float convolutionRowNoFlip<1>(const float *data,
                                                    const float *kern){
    return data[0]*kern[0];
}
template<> __device__ float convolutionRowNoFlip<0>(const float *data,
                                                    const float *kern){
    return 0;
}

template<int KERN_WIDTH>
__device__ void convolutionRowNoFlip(float& sum,
                                     const float *data,
                                     const float *kern, const int kern_wid){
  if(KERN_WIDTH>0)
    sum+=convolutionRowNoFlip<KERN_WIDTH>(data,kern);
  else
#pragma unroll 8
    for (int col=0; col < kern_wid; col++) {//loop over col
      sum+=data[col]*kern[col];
    }
}

template<bool accumulate>
__device__ void store_or_accumulate(float& dst,const float value ){
  if(accumulate){
    dst += value;
  }else
    dst = value;
}


/**
 * Implementation of the valid convolution that keep the full image and the full kernel in shared memory
 * Don't implement the stack.
 * each thread compute only one value for the output if split is false
 * thread block size=out_wid, out_len(or less then out_len if split is true)
 * grid block size=batch_id, nkern
 * dynamic shared memory: img_len*img_wid+kern_len*kern_wid
 * 
 * nkern: the number of kernel, used to compute the output image to store the result
 * nstack: the size of the stack, used to compute the image to load.
 * template flipped_kern: if true, we "flip" the kernel as in a real convolution, else we don't
 * template split: if true, each thread computes more than 1 output pixel
 *                 When true, allow for output image bigger then 512 pixel.
 *                 Use more registers.
 */
template<bool flipped_kern, int KERN_WIDTH, bool split>
__global__ void
conv_patch( float* img, float* kern, float* out,
            int img_len, int img_wid, int kern_len, int kern_wid,
            int nkern, int nstack)
{
  int __shared__ out_len, out_wid, nb_thread_id;
  out_len = img_len - kern_len + 1;
  out_wid = img_wid - kern_wid + 1;
  nb_thread_id = blockDim.z*blockDim.y*blockDim.x;

  extern __shared__ float s_data[];

    __shared__ int batch_id, kern_id;
    batch_id = blockIdx.x;
    kern_id = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int out_col = tx;//output col
    const int thread_id  = ty*blockDim.x + tx;

    float * d_img=&s_data[0];//size of [IMAGE_LEN * IMAGE_WID];
    float * d_kern=&s_data[img_len * img_wid];//size of [KERNEL_LEN * KERNEL_WID];

    kern+=kern_len*kern_wid*nstack*kern_id;

    img+=img_len*img_wid*(nstack*batch_id);

    load_to_shared(d_img, img, thread_id,nb_thread_id,img_len*img_wid);
    load_to_shared(d_kern, kern, thread_id,nb_thread_id,kern_len*kern_wid,flipped_kern);
    __syncthreads();

    if(!split){
      int out_row = ty;//output row
      float sum = 0.0f;
      for (int row=0; row < kern_len; row++) {//loop over row
        const float* idx_kern=&d_kern[row*kern_wid];
        const float* idx_in=&d_img[(row+out_row)*img_wid+out_col];
        convolutionRowNoFlip<KERN_WIDTH>(sum,idx_in,idx_kern,kern_wid);
      }
      out[batch_id*out_wid*out_len*nkern+//the good batch
          blockIdx.y*out_wid*out_len+//the output image
          out_row*out_wid+out_col] = sum;
    }else{
      for(int out_row=ty;out_row<out_len;out_row+=blockDim.y){
        float sum = 0.0f;
        for (int row=0; row < kern_len; row++) {//loop over row
          const float* idx_kern=&d_kern[row*kern_wid];
          const float* idx_in=&d_img[(row+out_row)*img_wid+out_col];
          convolutionRowNoFlip<KERN_WIDTH>(sum,idx_in,idx_kern,kern_wid);
        }
        out[batch_id*out_wid*out_len*nkern+//the good batch
            kern_id*out_wid*out_len+//the output image
            out_row*out_wid+out_col] = sum;
      }
    }
}

/**
 * As conv_patch, but implement the stack in the kernel.
 * I keep it separated from conv_patch as we take more registers and this could lower the occupency.
 * Implementation of the valid convolution that keep the full image and the full kernel in shared memory
 * each thread compute only one value for the output if split==false else it compute more than 1 values
 * thread block size=out_wid, out_len/X (X is any number, optimized value is ceil(out_len/N)
 * grid block size=batch_id, nkern
 * dynamic shared memory: img_len*img_wid+(preload_full_kern?KERNEL_LEN:1)*kern_wid
 * 
 * nkern: the number of kernel, used to compute the output image to store the result
 * nstack: the size of the stack, used to compute the image to load.
 * dx: patch stride rows(1 for normal convolution)
 * dy: patch stride cols(1 for normal convolution)
 * template flipped_kern: if true, we "flip" the kernel as in a real convolution, else we don't
 * template accumulate: if true, we add the result, else we override the result
 * template KERN_WIDTH: if 0, will work for any kern_wid, else it specialyse to this kern_wid as an optimization
 * template img_c_contiguous_2d: if true, the img have are collon and row contiguous
 * template kern_c_contiguous_2d: if true, the kernel have are collon and row contiguous
 * template split: if true, each thread generate more than 1 output pixel, but use more registers.
 * template preload_full_kern: if true, we load the full kernel in shared memory, else, we load 1 row at a time.
 * template subsample: if false, remove some computation needed when dx or dy!=1.
 */
template<bool flipped_kern, bool accumulate, int KERN_WIDTH, bool img_c_contiguous_2d, bool kern_c_contiguous_2d, bool split, bool preload_full_kern, bool subsample>
__global__ void
conv_patch_stack( float* img, float* kern, float* out,
                  int img_len, int img_wid, int kern_len, int kern_wid,
                  int out_len, int out_wid,
                  int nkern, int nstack, int img_stride_col,int img_stride_row,
                  int img_stride_stack, int img_stride_batch,
                  int kern_stride_col, int kern_stride_row,
                  int kern_stride_stack, int kern_stride_nkern, int dx, int dy)
{
  int __shared__ nb_thread_id;
  nb_thread_id = blockDim.z*blockDim.y*blockDim.x;

  extern __shared__ float s_data[];

    int batch_id = blockIdx.x;
    int kern_id = blockIdx.y;
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int out_col = tx;//output col
    int out_row = ty;//output row
    const int thread_id  = out_row*out_wid + out_col;

    float * d_img=&s_data[0];//size of [IMAGE_LEN * IMAGE_WID];
    float * d_kern=&s_data[img_len * img_wid];//size of [(preload_full_kern?KERNEL_LEN:1) * KERNEL_WID];

    if(!split){
      kern+=kern_stride_nkern*kern_id;//the good nkern
      img+=img_stride_batch*batch_id;//the good batch
      float sum = 0.0f;
      
      for (int stack = 0;stack<nstack;stack++,kern+=kern_stride_stack,
             img+=img_stride_stack){
        load_to_shared(d_img,img,thread_id,nb_thread_id,img_wid,img_len,
                       img_stride_col, img_stride_row, false, img_c_contiguous_2d);
        if(preload_full_kern)
          load_to_shared(d_kern, kern, thread_id, nb_thread_id, kern_wid,kern_len,
                         kern_stride_col, kern_stride_row, flipped_kern, kern_c_contiguous_2d);

        __syncthreads();
        
        for (int row=0; row < kern_len; row++) {//loop over row
          if(!preload_full_kern){
            __syncthreads();
            int idx2;
            if(flipped_kern) idx2=(kern_len-row-1)*kern_stride_row;
            else idx2=(row)*kern_stride_row;
            load_to_shared(d_kern, kern+idx2, thread_id, nb_thread_id, kern_wid,1,
                           kern_stride_col, kern_stride_row, flipped_kern, kern_c_contiguous_2d);
            __syncthreads();              
          }

          const float* idx_kern;
          if(preload_full_kern) idx_kern=&d_kern[row*kern_wid];
          else idx_kern=d_kern;
          const float* idx_in;
          if(subsample)
            idx_in=&d_img[(row+out_row*dx)*img_wid+out_col*dy];
          else
            idx_in=&d_img[(row+out_row)*img_wid+out_col];
          
          convolutionRowNoFlip<KERN_WIDTH>(sum,idx_in,idx_kern,kern_wid);
        }
        __syncthreads(); // ensure calculations have completed before any thread starts changing the shared memory
      }
      store_or_accumulate<accumulate>(
                                      out[batch_id*out_wid*out_len*nkern+//the good batch
                                          out_wid*out_len*kern_id+//the output image
                                          out_row*out_wid+out_col],sum);
    }else{

      float __shared__ *kern_, *img_;
      int __shared__ out_len_max;

      kern_=kern+kern_stride_nkern*kern_id;//the good nkern
      img_=img+img_stride_batch*batch_id;//the good batch
      //out_len_max must by higher then out_len as we need all thread when we load the image as the blockDim.y is not always a multiple of out_len.
      out_len_max = (out_len/blockDim.y+(out_len%blockDim.y==0?0:1))*blockDim.y;

      //TODO: inverse the out_row and stack loop to don't load the date as frequently!
      //TODO: do this happen elsewhere?
      for(;out_row<out_len_max;out_row+=blockDim.y){
        float sum = 0.0f;
        for (int stack = 0;stack<nstack;stack++){
          //TODO: load only the part of the image needed or put the partial result in shared memory
          int idx1=img_stride_stack*stack;
          load_to_shared(d_img,img_+idx1,thread_id,nb_thread_id,img_wid,img_len,
                         img_stride_col, img_stride_row, false, img_c_contiguous_2d);
          if(preload_full_kern){
            int idx2=kern_stride_stack*stack;
            load_to_shared(d_kern, kern_+idx2, thread_id, nb_thread_id, kern_wid,kern_len,
                           kern_stride_col, kern_stride_row, flipped_kern, kern_c_contiguous_2d);
          }
          __syncthreads();
          
          for (int row=0; row < kern_len; row++) {//loop over row
            if(!preload_full_kern){
              __syncthreads();
              int idx2=kern_stride_stack*stack;
              if(flipped_kern)
                idx2+=(kern_len-row-1)*kern_stride_row;
              else
                idx2+=(row)*kern_stride_row;
              load_to_shared(d_kern, kern_+idx2, thread_id, nb_thread_id, kern_wid,1,
                             kern_stride_col, kern_stride_row, flipped_kern, kern_c_contiguous_2d);
              __syncthreads();              
            }
            const float* idx_kern;
            if(preload_full_kern) idx_kern=&d_kern[row*kern_wid];
            else idx_kern=d_kern;
            const float* idx_in;
            if(subsample)
              idx_in=&d_img[(row+out_row*dx)*img_wid+out_col*dy];
            else
              idx_in=&d_img[(row+out_row)*img_wid+out_col];
            
            //if needed as on Fermi as reading out of bound index from shared memory generate an error.
            //Not needed on generation before as they worked anyway. Removing the if generate the good code
            //as we store the result of only the good thread.
            //This was with nvcc 3.0 on an GTX470 card.
            if(out_row<out_len)
              convolutionRowNoFlip<KERN_WIDTH>(sum,idx_in,idx_kern,kern_wid);
          }
          __syncthreads(); // ensure calculations have completed before any thread starts changing the shared memory
        }
        if(out_row<out_len)
          store_or_accumulate<accumulate>(
                                          out[batch_id*out_wid*out_len*nkern+//the good batch
                                              out_wid*out_len*kern_id+//the output image
                                              out_row*out_wid+out_col],sum);
      }

    }

}

/**
 * As conv_patch_stack, but kern_len thread for each output pixel
 * I keep it separated as use more register.
 * Implementation of the valid convolution that keep the full image and the full kernel in shared memory
 * thread block size=out_wid, out_len, ceil_intdiv(kern_len/nb_split)
 * grid block size=batch_id, nkern
 * dynamic shared memory: img_len*img_wid+kern_wid*(preload_full_kern?kern_len:thread_z)+out_size*thread_z
 * 
 * nkern: the number of kernel, used to compute the output image to store the result
 * nstack: the size of the stack, used to compute the image to load.
 * template flipped_kern: if true, we "flip" the kernel as in a real convolution, else we don't
 * template img_contiguous: if true, the img have are collon and row contiguous
 * template preload_full_kern: work only when split is true. We don't load the full kernel at once, but we load ceil_intdiv(kern_len/nb_split) kernel row at a time
 */
template<bool flipped_kern, int KERN_WIDTH, bool c_contiguous, bool split, bool preload_full_kern>
__global__ void
conv_patch_stack_reduce( float* img, float* kern, float* out,
                  int img_len, int img_wid, int kern_len, int kern_wid,
                  int nkern, int nstack, int img_stride_col,int img_stride_row,
                  int img_stride_stack, int img_stride_batch,
                  int kern_stride_col, int kern_stride_row,
                  int kern_stride_stack, int kern_stride_nkern)
{
  //int __shared__ out_len, out_wid, nb_thread_id;
  //out_len = img_len - kern_len + 1;
  //out_wid = img_wid - kern_wid + 1;
  const int out_wid = blockDim.x;
  const int out_len = blockDim.y;
  const int nb_thread_id = blockDim.z*blockDim.y*blockDim.x;

  extern __shared__ float s_data[];

    int batch_id = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int out_col = tx;//output col
    int out_row = ty;//output row
    const int thread_id  = tz*blockDim.y*blockDim.x+ty*blockDim.x+tx;

    //d_img size [IMAGE_LEN * IMAGE_WID];
    float * d_img=&s_data[0];

    //d_kern size[(preload_full_kern?KERNEL_LEN:blockDim.z) * KERNEL_WID]
    float * d_kern=&s_data[img_len * img_wid];

    //d_reduce size [n_threads]
    //N.B. this overlaps with d_img and d_kern!
    float * d_reduce=&s_data[0];

    float sum = 0.0f;

    kern+=kern_stride_nkern*blockIdx.y;//the good nkern
    img+=img_stride_batch*batch_id;//the good batch

    for (int stack = 0;stack<nstack;stack++,kern+=kern_stride_stack,
           img+=img_stride_stack){
      __syncthreads();
      load_to_shared(d_img, img, thread_id, nb_thread_id, img_wid, img_len,
                     img_stride_col, img_stride_row, false, c_contiguous);
      if(split && ! preload_full_kern){
        for(int first_row=0;first_row<kern_len;first_row+=blockDim.z){
            //N.B. - Jan 30, 2011 with CUDA 3.2 I found that without the explicit cast to
            // (int)blockDim.z, idx3 would sometimes be negative. I'm rusty on my signed vs. unsigned
            // details, but that seemed really weird. tricky bug to find too.
          int idx3 = flipped_kern
              ? max((kern_len - (int)blockDim.z - first_row),0)
              : first_row;
          int len3 = min(blockDim.z, kern_len - first_row);

          __syncthreads();
          load_to_shared(d_kern, kern+idx3*kern_stride_row, thread_id, nb_thread_id, kern_wid, len3,
                         kern_stride_col, kern_stride_row, flipped_kern, c_contiguous);
          __syncthreads();
          const float* idx_kern=&d_kern[tz*kern_wid];
          const float* idx_in=&d_img[(first_row+tz+out_row)*img_wid+out_col];
          float sum2 = 0;
          if(tz<len3)
            convolutionRowNoFlip<KERN_WIDTH>(sum2,idx_in,idx_kern,kern_wid);
          sum+=sum2;
        }
      }else if(split){
        load_to_shared(d_kern, kern, thread_id, nb_thread_id, kern_wid, kern_len,
                       kern_stride_col, kern_stride_row, flipped_kern, c_contiguous);
        __syncthreads();
        for(int row=tz;row<kern_len;row+=blockDim.z){
          const float* idx_kern=&d_kern[row*kern_wid];
          const float* idx_in=&d_img[(row+out_row)*img_wid+out_col];
          convolutionRowNoFlip<KERN_WIDTH>(sum,idx_in,idx_kern,kern_wid);
        }
      }else{
        int row = tz;//The row of the kernel.
        const float* idx_kern=&d_kern[row*kern_wid];
        const float* idx_in=&d_img[(row+out_row)*img_wid+out_col];
        load_to_shared(d_kern, kern, thread_id, nb_thread_id, kern_wid, kern_len,
                       kern_stride_col, kern_stride_row, flipped_kern, c_contiguous);
        __syncthreads();
        convolutionRowNoFlip<KERN_WIDTH>(sum,idx_in,idx_kern,kern_wid);
      }
        __syncthreads(); // ensure calculations have completed before any thread starts changing the shared memory
    }

    //reduce no sync because previous loop ends with sync
    d_reduce[thread_id]=sum;
    __syncthreads();
    if(thread_id<out_len*out_wid){ // blockDim.x==out_wid, blockDim.y==out_len
      //sum=0;
      for(int i=1;i<blockDim.z;i++){
        sum+=d_reduce[thread_id+i*out_wid*out_len];
      }
      out[batch_id*out_wid*out_len*nkern+//the good batch
          out_wid*out_len*blockIdx.y+//the output image
          out_row*out_wid+out_col] = sum;
    }
}

/**
 * WORK FOR IMAGE THAT DON'T FIT IN SHARED MEMORY
 * we store kern_len row of the image and the full kernel in the shared memory
 * each thread compute only one value for the output
 * Don't implement the stack and nkern in the kernel.
 * thread block size=out_wid
 * grid block size=out_len,batch_id
 * dynamic shared memory: kern_len*img_wid+kern_len*kern_wid
 * Diff with conv_patch: don't store the full image in the shared memory. 
 *    I.E. work for bigger image then conv_patch<split=true,...>.
 */
template<int KERN_WIDTH, bool c_contiguous>
__global__ void
conv_rows( float* img, float* kern, float* out,
           int img_len, int img_wid, int kern_len, int kern_wid,
           int nkern, int nstack,
           int img_stride_col, int img_stride_row,
           int img_stride_stack, int img_stride_batch,
           int kern_stride_col, int kern_stride_row,
           int kern_stride_stack, int kern_stride_nkern)
{
  int __shared__ out_len, out_wid, nb_thread_id, batch_id, kern_id;
  float __shared__ *d_img, *d_kern;
  out_len = img_len - kern_len + 1;
  out_wid = img_wid - kern_wid + 1;
  nb_thread_id = blockDim.z*blockDim.y*blockDim.x;
  batch_id= blockIdx.y/nkern;
  kern_id = blockIdx.y%nkern;

  extern __shared__ float s_data[];

    const int out_col = threadIdx.x;//output col
    const int out_row = blockIdx.x;;//output row
    const int thread_id = threadIdx.x;

    d_img=&s_data[0];//size of [KERN_LEN * IMAGE_WID];
    d_kern=&s_data[kern_len * img_wid];//size of [KERNEL_LEN * KERNEL_WID];
    
    img+=img_stride_batch*batch_id;//selection the good image from the batch
    img+=out_row*img_stride_row;//select the good top row.
    kern+=kern_stride_nkern*kern_id;//the good nkern

    load_to_shared(d_img,img,thread_id,nb_thread_id,img_wid,kern_len,
                   img_stride_col, img_stride_row, false, c_contiguous);
    load_to_shared(d_kern, kern, thread_id, nb_thread_id, kern_wid,kern_len,
                   kern_stride_col, kern_stride_row, true, c_contiguous);

    __syncthreads();
    float sum = 0.0f;

    for (int row=0; row < kern_len; row++) {//loop over row
      const float* idx_kern=&d_kern[row*kern_wid];
      const float* idx_in=&d_img[(row)*img_wid+out_col];
      convolutionRowNoFlip<KERN_WIDTH>(sum,idx_in,idx_kern,kern_wid);
    }

    out[batch_id*out_wid*out_len*nkern+//the good batch
        kern_id*out_wid*out_len+//the output image
        out_row*out_wid+out_col] = sum;
}

/**
 * WORK FOR IMAGE THAT DON'T FIT IN SHARED MEMORY
 * as conv_rows, but implement the stack. Separate as this use more register.
 * we store kern_len row of the image and the full kernel in the shared memory
 * each thread compute only one value for the output
 * thread block size=out_wid, block_len
 * grid block size=intceil(out_len/block_len),nb_batch*nb_kern
 * dynamic shared memory: (kern_len+block_len-1)*img_wid+kern_len*kern_wid
 * Diff with conv_patch: don't store the full image in the shared memory. 
 *    I.E. work for bigger image then conv_patch<split=true,...>.
 */
template<int KERN_WIDTH, bool c_contiguous>
__global__ void
conv_rows_stack( float* img, float* kern, float* out,
                 const int img_len, const int img_wid, const int kern_len, const int kern_wid,
                 const int nkern, const int nstack,
                 const int img_stride_col, const int img_stride_row,
                 const int img_stride_stack, const int img_stride_batch,
                 const int kern_stride_col, const int kern_stride_row,
                 const int kern_stride_stack, const int kern_stride_nkern)
{
  int __shared__ out_len, out_wid, nb_thread_id, batch_id, kern_id, nb_rows;
  float  __shared__ *d_img, *d_kern;
  out_len = img_len - kern_len + 1;
  out_wid = img_wid - kern_wid + 1;
  nb_thread_id = blockDim.z*blockDim.y*blockDim.x;
  batch_id= blockIdx.y/nkern;
  kern_id = blockIdx.y%nkern;
  nb_rows = blockDim.y;

  int rows_to_read = MIN(
          kern_len + nb_rows - 1,
          img_len - blockIdx.x * nb_rows);

  /**
   * Every thread ultimately computes one value in the output, at coordinates
   *   out[ batch_id, kern_id, out_row, out_col]
   *
   * The batch_id and kern_id are packed into blockIdx.y. out_row and out_col
   * are the threadIdx.x and threadIdx.y.
   *
   * Every thread block deals only with one image, and one filter kernel.
   */
  extern __shared__ float s_data[];

    const int out_col = threadIdx.x;//output col
    const int out_row = blockIdx.x*blockDim.y+threadIdx.y;//output row
    const int shared_row = threadIdx.y;
    const int thread_id = threadIdx.y*blockDim.x+threadIdx.x;

  /*
   * The kernel works by looping over channels (aka colours, aka the stack).
   * On each iteration, a thread block loads one channel of all the image rows that
   * it needs to use, and one channel slice of one kernel.
   */
    d_img=&s_data[0];//size of [(KERN_LEN+block_len-1) * IMAGE_WID];
    d_kern=&s_data[(kern_len+nb_rows-1) * img_wid];//size of [KERNEL_LEN * KERNEL_WID];

    float sum = 0.0f;
    for (int stack = 0; stack < nstack; stack++){

      int offset =
          img_stride_batch * batch_id
          + img_stride_stack * stack
          //blockIdx.x is which chunk of nb_rows this thread block deals with
          + img_stride_row * (blockIdx.x * nb_rows);

      load_to_shared(
              d_img,              // dst
              img+offset,         // src
              thread_id,          // linear position in block
              nb_thread_id,       // number of threads
              img_wid,            // cols in image to read
              rows_to_read,       // number of rows to read
              img_stride_col,     // img[i, j, k, l] to img[i, j, k, l + 1]
              img_stride_row,     // img[i, j, k, l] to img[i, j, k + 1, l]
              false,              // flip while reading
              c_contiguous);

      offset = kern_stride_nkern * kern_id + kern_stride_stack * stack;
      load_to_shared(d_kern, kern+offset, thread_id, nb_thread_id, kern_wid,kern_len,
                     kern_stride_col, kern_stride_row, true, c_contiguous);

      __syncthreads();

      for (int row=0; row < kern_len; row++) {//loop over row
        const float* idx_kern=&d_kern[row*kern_wid];
        const float* idx_in=&d_img[(row+shared_row)*img_wid+out_col];
        convolutionRowNoFlip<KERN_WIDTH>(sum,idx_in,idx_kern,kern_wid);
      }
      __syncthreads();//to be sure all thread have finished before we modif the shared memory.
    }
    if (out_row < out_len)
      out[batch_id*out_wid*out_len*nkern+//the good batch
          kern_id*out_wid*out_len+//the output image
          out_row*out_wid+out_col] = sum;
}

/**
 * WORK FOR IMAGE THAT DON'T FIT IN SHARED MEMORY
 * as conv_rows_stack, but load only block_len of the image at a time and 1 or all kern row.
 * we store block_len row of the image(at a time) and one or all kernel row in the shared memory
 * each thread compute only one value for the output
 * thread block size=out_wid, block_len
 * grid block size=intceil(out_len/block_len),nb_batch*nb_kern
 * dynamic shared memory: block_len * img_wid+(preload_full_kern?kern_len:1)*kern_wid
 * Diff with conv_patch: don't store the full image and kernel in the shared memory. 
 *    I.E. work for bigger image then conv_patch<split=true,...>.
 */
template<int KERN_WIDTH, bool c_contiguous, bool preload_full_kern>
__global__ void
conv_rows_stack2( float* img, float* kern, float* out,
                 const int img_len, const int img_wid, const int kern_len, const int kern_wid,
                 const int nkern, const int nstack,
                 const int img_stride_col, const int img_stride_row,
                 const int img_stride_stack, const int img_stride_batch,
                 const int kern_stride_col, const int kern_stride_row,
                 const int kern_stride_stack, const int kern_stride_nkern)
{
  int __shared__ out_len, out_wid, nb_thread_id, batch_id, kern_id, nb_rows;
  float  __shared__ *d_img, *d_kern;
  out_len = img_len - kern_len + 1;
  out_wid = img_wid - kern_wid + 1;
  nb_thread_id = blockDim.z*blockDim.y*blockDim.x;
  batch_id= blockIdx.y/nkern;
  kern_id = blockIdx.y%nkern;
  nb_rows = blockDim.y;

  extern __shared__ float s_data[];

    const int out_col = threadIdx.x;//output col
    const int out_row = blockIdx.x*blockDim.y+threadIdx.y;//output row
    const int shared_row = threadIdx.y;
    const int thread_id = threadIdx.y*blockDim.x+threadIdx.x;

    d_img=&s_data[0];//size of [nb_rows * IMAGE_WID];
    d_kern=&s_data[nb_rows*img_wid];//size of [(preload_full_kern?KERNEL_LEN:1) * KERNEL_WID];
    
    float sum = 0.0f;
    for (int stack = 0;stack<nstack;stack++){

      int _idx2=img_stride_batch*batch_id+img_stride_stack*stack;//selection the good image from the batch and stack
      _idx2+=(blockIdx.x*nb_rows)*img_stride_row;//select the good top row for the block of threads
    
      __syncthreads();
      load_to_shared(d_img,img+_idx2,thread_id,nb_thread_id,img_wid,nb_rows-1,
                           img_stride_col, img_stride_row, false, c_contiguous);
      if(preload_full_kern)
        load_to_shared(d_kern, kern+kern_stride_nkern*kern_id+kern_stride_stack*stack,
                       thread_id, nb_thread_id, kern_wid,kern_len,
                       kern_stride_col, kern_stride_row, true, c_contiguous);
      __syncthreads();

      for (int row=0; row < kern_len; row++) {//loop over row
        __syncthreads();
        if((blockIdx.x*nb_rows+row+nb_rows-1)<img_len){
          int _idx1=img_stride_batch*batch_id+img_stride_stack*stack;//selection the good image from the batch and stack
          _idx1+=(blockIdx.x*nb_rows)*img_stride_row;//select the good top row for the block of threads
          _idx1+=(row+nb_rows-1)*img_stride_row;//the current last row
          load_to_shared(d_img+((row+nb_rows-1)%nb_rows)*img_wid,
                         img+_idx1, thread_id, nb_thread_id, img_wid, 1,
                         img_stride_col, img_stride_row, false, c_contiguous);//we use d_img as a circular buffer.
        }

        if(!preload_full_kern){
          int _idx3=kern_stride_nkern*kern_id+kern_stride_stack*stack;//selection the good kern from the batch and stack
          _idx3+=(kern_len-row-1)*kern_stride_row;//the current last row flipped
          load_to_shared(d_kern, kern+_idx3,
                         thread_id, nb_thread_id, kern_wid,1,
                         kern_stride_col, kern_stride_row, true, c_contiguous);

        }
        __syncthreads();

        //if needed as on Fermi as reading out of bound index from shared memory generate an error.
        //Not needed on generation before as they worked anyway. Removing the if generate the good code
        //as we store the result of only the good thread.
        //This was with nvcc 3.0 on an GTX470 card.
        if(out_row<out_len){
          const float* idx_kern;
          if(preload_full_kern) idx_kern=&d_kern[row*kern_wid];
          else idx_kern=d_kern;
          const float* idx_in=&d_img[((shared_row+row)%nb_rows)*img_wid+out_col];
          float sum_ =0.0f;
          convolutionRowNoFlip<KERN_WIDTH>(sum_,idx_in,idx_kern,kern_wid);
          sum+=sum_;//We pass by an intermediate variable to have more precission.
        }
      }
    }
    __syncthreads();
    if(out_row<out_len)
      out[batch_id*out_wid*out_len*nkern+//the good batch
          kern_id*out_wid*out_len+//the output image
          out_row*out_wid+out_col] = sum;
}

/**
 * Implementation of 'valid' mode convolution that uses one block per output pixel, and uses a sum-reduce within each block to compute the
 * kernel-image inner-product in parallel.
 * 
 * This implementation uses shared memory for the reduce, so it is limited by the product of stacklen x kern_len
 *
 * template stack_loop: if true, we accept that blockDim.x < nstack and we add a loop for this(use 3 more registers, so lower occupency when true, but accept nstack*kern_len>512)
 * TODO: explain parameters, preconditions
 */
template<bool stack_loop>
__global__ void
conv_valid_row_reduce(int nB, int nK, int stacklen,
        int img_len, int img_wid, 
        int kern_len, int kern_wid,
        int out_len, int out_wid, //physical
        float *img, int img_str_B, int img_str_S, int img_str_R, int img_str_C,
        float *kern, int kern_str_K, int kern_str_S, int kern_str_R, int kern_str_C,
        float *out, int out_str_B, int out_str_K, int out_str_R, int out_str_C ,
        int subsample_rows, int subsample_cols,
        const int initial_reduce_boundary)
{
    const int outsize = nB * nK * out_len * out_wid;
    extern __shared__ float reducebuf[];
    for (int i = blockIdx.x; i < /*physical*/outsize; i += gridDim.x)
    {
        //figure out what output element we're in charge of computing
        int ii = i;
        int iB = ii % nB;      // output batch index
        ii = ii / nB;
        int iK = ii % nK;      // output kernel index
        ii = ii / nK;
        int iR_physical = ii % out_len; //output kernel row
        int iC_physical = ii / out_len; // output kernel column
        int iR_logical = iR_physical * subsample_rows;
        int iC_logical = iC_physical * subsample_cols;

        int ss = threadIdx.x;
        int rr = threadIdx.y;
        int img_rr = iR_logical + kern_len - 1 - rr;
        int reduceIdx = threadIdx.x * blockDim.y + threadIdx.y;
        float sum = 0.0f;
        if(stack_loop){
          for (; ss < stacklen; ss+=blockDim.x){
            float * kk_0 = kern + iK*kern_str_K + ss*kern_str_S + rr*kern_str_R;
            float * ii_0 = img + iB*img_str_B + ss*img_str_S + img_rr*img_str_R + (iC_logical + kern_wid - 1)*img_str_C;
            for (int cc = 0; cc < kern_wid; ++cc)
            {
                sum +=  kk_0[0] * ii_0[0];
                kk_0 += kern_str_C;
                ii_0 -= img_str_C;
            }
          }
        }else{
          float * kk_0 = kern + iK*kern_str_K + ss*kern_str_S + rr*kern_str_R;
          float * ii_0 = img + iB*img_str_B + ss*img_str_S + img_rr*img_str_R + (iC_logical + kern_wid - 1)*img_str_C;
          for (int cc = 0; cc < kern_wid; ++cc)
          {
            sum +=  kk_0[0] * ii_0[0];
            kk_0 += kern_str_C;
            ii_0 -= img_str_C;
          }
        }

        if (blockDim.x * blockDim.y == 1)
        {
            out[iB * out_str_B + iK * out_str_K + iR_physical * out_str_R + iC_physical * out_str_C] = sum;
        }
        else
        {
            reducebuf[reduceIdx] = sum;
            __syncthreads();
            int reduce_boundary = initial_reduce_boundary;

            // add in the terms above the reduce boundary
            if (reduceIdx + reduce_boundary < (blockDim.x * blockDim.y))
                reducebuf[reduceIdx] += reducebuf[reduce_boundary +reduceIdx];
            reduce_boundary >>= 1;
            // there are an equal number of terms above and below the reduce_boundary
            while (reduce_boundary)
            {
                __syncthreads();
                if (reduceIdx < reduce_boundary)
                {
                    reducebuf[reduceIdx] += reducebuf[reduce_boundary + reduceIdx];
                }
                reduce_boundary >>= 1;
            }
            if (reduceIdx == 0)
            {
                out[iB * out_str_B + iK * out_str_K + iR_physical * out_str_R + iC_physical * out_str_C] = reducebuf[0];
            }
        }
    }
}



/**
 * Reference implementation of 'valid' mode convolution (with stack)
 * 
 * This implementation works for any size of image and kernel.  It does not use shared memory.
 *
 * TODO: explain parameters, preconditions
 */
__global__ void
conv_reference_valid(int nB, int nK, int stacklen,
        int img_len, int img_wid, 
        int kern_len, int kern_wid,
        int out_len, int out_wid, //physical
        float *img, int img_str_B, int img_str_S, int img_str_R, int img_str_C,
        float *kern, int kern_str_K, int kern_str_S, int kern_str_R, int kern_str_C,
        float *out, int out_str_B, int out_str_K, int out_str_R, int out_str_C ,
        int subsample_rows, int subsample_cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int numThreads, outsize;
    numThreads = blockDim.x * gridDim.x;
    outsize = nB * nK * out_len * out_wid;

    for (int i = idx; i < outsize; i += numThreads)  //physical
    {
        //figure out what output element we're in charge of computing
        int ii = i;
        int iB = ii % nB;      // output batch index
        ii = ii / nB;
        int iK = ii % nK;      // output kernel index
        ii = ii / nK;
        int iR_physical = ii % out_len; //output kernel row
        int iC_physical = ii / out_len; // output kernel column
        int iR_logical = iR_physical * subsample_rows;
        int iC_logical = iC_physical * subsample_cols;

        float sum = 0.0f;
        for (int ss = 0; ss < stacklen; ++ss)
        {
            for (int rr = 0; rr < kern_len; ++rr)
            {
                int img_rr = iR_logical + kern_len - 1 - rr;
                for (int cc = 0; cc < kern_wid; ++cc)
                {
                    int img_cc = iC_logical + kern_wid-1-cc;
                    float k_0 = kern[iK*kern_str_K + ss*kern_str_S + rr*kern_str_R + cc*kern_str_C];
                    float i_0 = img[iB*img_str_B + ss*img_str_S + img_rr*img_str_R + img_cc*img_str_C];
                    sum +=  k_0 * i_0;
                }
            }
        }
        //coords[i*5+0] = iB;
        //coords[i*5+1] = iK;
        //coords[i*5+2] = iR;
        //coords[i*5+3] = iC;
        //coords[i*5+4] = iB * out_str_B + iK * out_str_K + iR * out_str_R + iC * out_str_C;
        out[iB * out_str_B + iK * out_str_K + iR_physical * out_str_R + iC_physical * out_str_C] = sum;
    }
}

/**
 * Reference implementation of 'full' mode convolution (with stack)
 * 
 * This implementation works for any size of image and kernel.  It does not use shared memory.
 *
 * TODO: explain parameters, preconditions
 */
__global__ void
conv_reference_full(int nB, int nK, int stacklen,
        int img_len, int img_wid, 
        int kern_len, int kern_wid,
        int out_len, int out_wid, //physical dimensions
        float *img, int img_str_B, int img_str_S, int img_str_R, int img_str_C,
        float *kern, int kern_str_K, int kern_str_S, int kern_str_R, int kern_str_C,
        float *out, int out_str_B, int out_str_K, int out_str_R, int out_str_C,
        int subsample_rows, int subsample_cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int numThreads, physical_outsize;
    numThreads = blockDim.x * gridDim.x;
    physical_outsize = nB * nK * out_len * out_wid;

    for (int i = idx; i < physical_outsize; i += numThreads) 
    {
        //figure out what output element we're in charge of computing
        int ii = i;
        int iB = ii % nB;      // output batch index
        ii = ii / nB;
        int iK = ii % nK;      // output kernel index
        ii = ii / nK;
        int iR_physical = ii % out_len; //output kernel row
        int iC_physical = ii / out_len; // output kernel column
        int iR_logical = iR_physical * subsample_rows;
        int iC_logical = iC_physical * subsample_cols;

        float sum = 0.0f;
        for (int ss = 0; ss < stacklen; ++ss)
        {
            for (int rr = 0; rr < kern_len; ++rr)
            {
                int img_rr = iR_logical - rr;
                if ((img_rr >= 0) && (img_rr < img_len))
                {
                    for (int cc = 0; cc < kern_wid; ++cc)
                    {
                        int img_cc = iC_logical - cc;
                        if ((img_cc >= 0) && (img_cc < img_wid))
                        {
                            float k_0 = kern[iK*kern_str_K + ss*kern_str_S + rr*kern_str_R + cc*kern_str_C];
                            float i_0 = img[iB*img_str_B + ss*img_str_S + img_rr*img_str_R + img_cc*img_str_C];
                            sum +=  k_0 * i_0;
                        }
                    }
                }
            }
        }
        out[iB * out_str_B + iK * out_str_K + iR_physical * out_str_R + iC_physical * out_str_C] = sum;
    }
}

#endif // #ifndef CONV_KERNEL_CU
/*
  Local Variables:
  mode:c++
  c-basic-offset:4
  c-file-style:"stroustrup"
  indent-tabs-mode:nil
  fill-column:79
  End:
*/
// vim: filetype=cpp:expandtab:shiftwidth=4:tabstop=8:softtabstop=4:textwidth=79 :
