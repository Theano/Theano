//we store the full image and the full kernel in the shared memory
//each thread compute only one value for the output
//thread block size=out_wid, out_len/nb_split
//grid block size=batch_id
//dynamic shared memory: img_len*img_wid+kern_len*kern_wid
__global__ void
conv_full_patch_split( float* img, float* kern, float* out, int img_len, int img_wid, int kern_len, int kern_wid, int nb_split)
{
  int __shared__ out_len, out_wid, nb_thread_id;
  out_len = img_len + kern_len - 1;
  out_wid = img_wid + kern_wid - 1;
  nb_thread_id = blockDim.z*blockDim.y*blockDim.x;

  extern __shared__ float s_data[];

    int batch_id = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int out_col = tx;//output col
    int out_row = ty;//output row
    const int thread_id  = out_row*out_wid + out_col;

    float * d_img=&s_data[0];//size of [IMAGE_LEN * IMAGE_WID];
    float * d_kern=&s_data[img_len * img_wid];//size of [KERNEL_LEN * KERNEL_WID];

    img+=img_len*img_wid*batch_id;//the good batch

    load_to_shared(d_img, img, thread_id, nb_thread_id, img_len*img_wid);
    load_to_shared(d_kern, kern, thread_id, nb_thread_id, kern_len*kern_wid);
    __syncthreads();

    for(int out_row=ty;out_row<out_len;out_row+=out_len/nb_split){
      float sum = 0.0f;
      int img_row = out_row;

      for (int row=0; row < kern_len; row++) {//loop over row
        int inverse_row = (img_row-row);
        if(inverse_row<0 ||inverse_row>=(img_len))continue;//row outside the image

        const float* idx_in=&d_img[inverse_row*img_wid];
        const float* idx_kern=&d_kern[row*kern_wid];
        int img_col = out_col;
        int col=0,last=0;
        for (col=0,last=img_col; col < kern_wid; col++,last--) {//loop over col
          if(last<0 ||last>=(img_wid))continue;//col outside the image        
          sum+=idx_in[last]*idx_kern[col];
        }
      }
      out[batch_id*out_len*out_wid+//the output image
          out_row*out_wid+out_col] = sum;
    }
}

//we store the full image and the full kernel in the shared memory
//each thread compute only one value for the output
//thread block size=out_wid, out_len
//grid block size=batch_id, nkern
//dynamic shared memory: img_len*img_wid+kern_len*kern_wid
__global__ void
conv_full_patch( float* img, float* kern, float* out,
                 int img_len, int img_wid,
                 int kern_len, int kern_wid, int nkern, int nstack)
{
  int __shared__ out_len, out_wid, nb_thread_id;
  out_len = img_len + kern_len - 1;
  out_wid = img_wid + kern_wid - 1;
  nb_thread_id = blockDim.z*blockDim.y*blockDim.x;

  extern __shared__ float s_data[];

    int batch_id = blockIdx.x;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int out_col = tx;//output col
    int out_row = ty;//output row
    const int thread_id  = out_row*out_wid + out_col;

    float * d_img=&s_data[0];//size of [IMAGE_LEN * IMAGE_WID];
    float * d_kern=&s_data[img_len * img_wid];//size of [KERNEL_LEN * KERNEL_WID];

    kern+=kern_len*kern_wid*nstack*blockIdx.y;//the good nkern
    img+=img_len*img_wid*batch_id;//the good batch

    load_to_shared(d_img, img, thread_id, nb_thread_id, img_len*img_wid);
    load_to_shared(d_kern, kern, thread_id, nb_thread_id, kern_len*kern_wid, true);
    __syncthreads();

    float sum = 0.0f;

    for (int row=0; row < kern_len; row++) {//loop over row
      if(row+out_row-kern_len+1<0 || row+out_row-kern_len+1>=img_len)continue;

      const float* idx_in=&d_img[(row+out_row-kern_len+1)*img_wid+out_col-kern_wid+1];
      const float* idx_kern=&d_kern[row*kern_wid];
      int col=0;
      int max_col=kern_wid;
      int img_col=out_col-kern_wid+1;
      max_col=min(max_col,img_wid-img_col);
      
      if(img_col<0){col=-img_col;img_col+=col;}
      for (; col < max_col; col++, img_col++) {//loop over col
        sum+=idx_in[col]*idx_kern[col];
      }
    }
    out[batch_id*out_wid*out_len*nkern+//the good batch
        out_wid*out_len*blockIdx.y+//the output image
        out_row*out_wid+out_col] = sum;
}

//we store the full image and the full kernel in the shared memory
//each thread compute only one value for the output
//thread block size=out_wid, out_len
//grid block size=batch_id, nkern
//dynamic shared memory: img_len*img_wid+kern_len*kern_wid
//template c_contiguous: if true, the img and kern have are column and row contiguous else we use the stride value from the param. The image need to be c_contiguous in the nbatch and nstack dimensions.

template<bool img_c_contiguous_2d, bool kern_c_contiguous_2d>
__global__ void
conv_full_patch_stack( float* img, float* kern, float* out,
                       int img_len, int img_wid,
                       int kern_len, int kern_wid, int nkern, int nstack,
                       int img_stride_col, int img_stride_row,
                       int kern_stride_col, int kern_stride_row, 
                       int kern_stride_stack, int kern_stride_nkern)
{
  int __shared__ out_len, out_wid, nb_thread_id;
  out_len = img_len + kern_len - 1;
  out_wid = img_wid + kern_wid - 1;
  nb_thread_id = blockDim.y*blockDim.x;//blockDim.z*
  float __shared__ *kern_, *img_;
  extern __shared__ float s_data[];

    const int batch_id = blockIdx.x;
    const int nkern_id = blockIdx.y;


    const int out_col = threadIdx.x;
    const int out_row = threadIdx.y;
    const int thread_id  = threadIdx.y*blockDim.x+ threadIdx.x;

    float* d_img=&s_data[0];//size of [IMAGE_LEN * IMAGE_WID];
    float* d_kern=&s_data[img_len * img_wid];//size of [KERNEL_LEN * KERNEL_WID];
    kern_=kern+kern_stride_nkern*nkern_id;//the good nkern
    img_=img+img_len*img_stride_row*(nstack*batch_id);//the good batch

    float sum = 0.0f;

    for (int stack = 0;stack<nstack;stack++){

      load_to_shared(d_img, img_+stack*img_len*img_stride_row, thread_id,nb_thread_id,img_wid,img_len,img_stride_col, img_stride_row,false,img_c_contiguous_2d);
      load_to_shared(d_kern, kern_+stack*kern_stride_stack, thread_id,nb_thread_id,kern_wid,kern_len,kern_stride_col,kern_stride_row,true,kern_c_contiguous_2d);
      __syncthreads();


      for (int row=0; row < kern_len; row++) {//loop over row
        if(row+out_row-kern_len+1<0 || row+out_row-kern_len+1>=img_len)continue;
        const float* idx_in=&d_img[(row+out_row-kern_len+1)*img_wid+out_col-kern_wid+1];
        const float* idx_kern=&d_kern[row*kern_wid];
        int col=0;
        int max_col=kern_wid;
        int img_col=out_col-kern_wid+1;
        max_col=min(max_col,img_wid-img_col);

        if(img_col<0){col=-img_col;img_col+=col;}
        for (; col < max_col; col++, img_col++) {//loop over col
          sum+=idx_in[col]*idx_kern[col];
        }
      }
      //Needed as not all thread finish at the same time the loop
      //And we don't want to overwrite the shared memory.
      __syncthreads();
    }
    out[batch_id*out_wid*out_len*nkern+//the good batch
        out_wid*out_len*blockIdx.y+//the output image
        out_row*out_wid+out_col] = sum;
}

/**
 * As conv_patch_stack, but used for the full convolution by padding the image in shared memory.
 * I keep it separated from conv_patch as we take 19-20 register which is more than the 10/16 max for each thread and thus this could lower the occupency.
 * Implementation of the valid convolution that keep the full image and the full kernel in shared memory
 * each thread compute only one value for the output if split is true. Otherwise compute ceil((float)out_len/N) pixel.
 * thread block size=out_wid, nb_rows (optimized value is ceil(out_len/N))
 * grid block size=batch_id, nkern
 * dynamic shared memory: full mem: (img_len+2*kern_len-2)*(img_wid+2*kern_wid-2)+kern_len*kern_wid
 * dynamic shared memory: low mem:((kern_len+nb_row-1)+2*kern_len-2)*(img_wid+2*kern_wid-2)+kern_len*kern_wid
 * 
 * nkern: the number of kernel, used to compute the output image to store the result
 * nstack: the size of the stack, used to compute the image to load.
 * template flipped_kern: if true, we "flip" the kernel as in a real convolution, else we don't
 * template c_contiguous: if true, the image and kernel have are c_contiguous.(use less registers)
 * template split: if true, each thread compute more than 1 output pixel.
 * template low_mem: if true, as split but with use less dynamic shared memory but use more registers.
 *          if you set split and low_mem to true, we will use the low_mem version!
 */
template<bool flipped_kern, int KERN_WIDTH, bool c_contiguous, bool split, bool low_mem >
__global__ void
conv_full_patch_stack_padded( float* img, float* kern, float* out,
                  const int img_len, const int img_wid,
                  const int kern_len, const int kern_wid,
                  const int nkern, const int nstack,
                  const int img_stride_col, const int img_stride_row,
                  const int img_stride_stack, const int img_stride_batch,
                  const int kern_stride_col, const int kern_stride_row,
                  const int kern_stride_stack, const int kern_stride_nkern)
{
  int __shared__ out_len, out_wid, nb_thread_id;
  out_len = img_len + kern_len - 1;
  out_wid = img_wid + kern_wid - 1;
  nb_thread_id = blockDim.z*blockDim.y*blockDim.x;

  extern __shared__ float s_data[];

    __shared__ int batch_id, kern_id, img_wid_valid, nb_rows;
    batch_id = blockIdx.x;
    kern_id = blockIdx.y;
    nb_rows = blockDim.y;

    // Thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    int out_col = tx;//output col
    const int thread_id  = ty*blockDim.x + tx;

    float * d_kern=&s_data[0];//size of [KERNEL_LEN * KERNEL_WID];
    float * d_img=&s_data[kern_len*kern_wid];//size of [see fct doc];

    kern+=kern_stride_nkern*kern_id;//the good nkern
    img+=img_stride_batch*batch_id;//the good batch

    img_wid_valid=img_wid+2*kern_wid-2;

    if(!split && !low_mem){
      fill(d_img,img_wid_valid*(img_len+2*kern_len-2), 0, thread_id, nb_thread_id);
      const int out_row = ty;//output row
      float sum = 0.0f;
      for (int stack = 0;stack<nstack;stack++,kern+=kern_stride_stack,
             img+=img_stride_stack){
          __syncthreads();
        load_padded_col_to_shared(d_img+img_wid_valid*(kern_len-1),img,
                                  thread_id,nb_thread_id,img_wid,img_len,
                                  img_stride_col, img_stride_row, kern_wid-1,
                                  c_contiguous);
        load_to_shared(d_kern, kern, thread_id, nb_thread_id, kern_wid,kern_len,
                       kern_stride_col, kern_stride_row, flipped_kern, c_contiguous);
        __syncthreads();

        for (int row=0; row < kern_len; row++) {//loop over row
          const float* idx_kern=&d_kern[row*kern_wid];
          const float* idx_in=&d_img[(row+out_row)*img_wid_valid+out_col];
          
          convolutionRowNoFlip<KERN_WIDTH>(sum, idx_kern, idx_in, kern_wid);
        }
      }
      out[batch_id*out_wid*out_len*nkern+//the good batch
          kern_id*out_wid*out_len+//the output image
          out_row*out_wid+out_col] = sum;
    }else if(split && !low_mem){
      fill(d_img,img_wid_valid*(img_len+2*kern_len-2), 0, thread_id, nb_thread_id);
      //out_len_max must by higher then out_len as we need all thread when we load the image as the nb_rows is not always a multiple of out_len.
      __shared__ int out_len_max;
      //TODO pass a parameter nb_split
      out_len_max = (out_len/blockDim.y+(out_len%blockDim.y==0?0:1))*blockDim.y;
      for(int out_row = ty;out_row<out_len_max;out_row+=nb_rows){
        float sum = 0.0f;
        for (int stack = 0;stack<nstack;stack++){
          __syncthreads();
          //TODO: load only the part of the image needed or put the partial result in shared memory
          load_padded_col_to_shared(d_img+img_wid_valid*(kern_len-1),
                                    img+img_stride_stack*stack,
                                    thread_id,nb_thread_id,img_wid,img_len,
                                    img_stride_col, img_stride_row, kern_wid-1,
                                    c_contiguous);
          load_to_shared(d_kern, kern+kern_stride_stack*stack,
                         thread_id, nb_thread_id, kern_wid,kern_len,
                         kern_stride_col, kern_stride_row, flipped_kern, c_contiguous);
          __syncthreads();
          //The if is needed as on Fermi as reading out of bound index from shared memory generate an error.
          //Not needed on generation before as they worked anyway. Removing the if generate the good code
          //as we store the result of only the good thread.
          //This was with nvcc 3.0 on an GTX470 card.
          if(out_row<out_len)
            for (int row=0; row < kern_len; row++) {//loop over row
              const float* idx_kern=&d_kern[row*kern_wid];
              const float* idx_in=&d_img[(row+out_row)*img_wid_valid+out_col];
              
              convolutionRowNoFlip<KERN_WIDTH>(sum, idx_kern, idx_in, kern_wid);
            }
          if(out_row<out_len)
            out[batch_id*out_wid*out_len*nkern+//the good batch
                out_wid*out_len*kern_id+//the output image
                out_row*out_wid+out_col] = sum;
        }
      }
    }else{//low_mem version
      //don't need to fill the last rows padding as this is done later.
      fill(d_img,img_wid_valid*((kern_len+nb_rows-1)+2*kern_len-2), 0, thread_id, nb_thread_id);
      //out_len_max must by higher then out_len as we need all thread when we load the image as the nb_rows is not always a multiple of out_len.
      __shared__ int out_len_max;
      //TODO pass a parameter nb_split
      if(thread_id==0)
        out_len_max = (out_len/nb_rows+(out_len%nb_rows==0?0:1))*nb_rows;
      __syncthreads();
      for(int out_row = ty, out_row_iter=0;out_row<out_len_max;
          out_row+=nb_rows, out_row_iter++){
        float sum = 0.0f;
        for (int stack = 0;stack<nstack;stack++){
          __syncthreads();
          const int len_to_load=min(kern_len+nb_rows,img_len-out_row_iter*nb_rows);//nb rows to load, min(nb_rows for this iter, nb rows left in the image)
          const int empty_row = max(kern_len-1-out_row_iter*nb_rows,0);//number of empty row at the start
          //we need to reload some row as when we change of out_row we lost the last load du to the stack.
          const int previous_row = min(out_row_iter*nb_rows,kern_len-1);//number of row from last out_row iteration to reload
          load_padded_col_to_shared(d_img+(kern_len-1-previous_row)*img_wid_valid,
                                    img+img_stride_stack*stack//the good stack image
                                    +(out_row_iter*nb_rows-previous_row)*img_stride_row,//the good split top row.
                                    thread_id,nb_thread_id,img_wid,
                                    len_to_load+previous_row,
                                    img_stride_col, img_stride_row, kern_wid-1,
                                    c_contiguous);
          //TODO: fill the last row padding only when needed.
          //We always fill the last rows padding event when not needed.
          int row_to_fill = 2*kern_len-2+nb_rows- empty_row - previous_row - len_to_load;
          row_to_fill = min(row_to_fill,kern_len-1);
          fill(d_img+(kern_len-1+len_to_load)*img_wid_valid,
               img_wid_valid*row_to_fill, 0, thread_id, nb_thread_id);
          load_to_shared(d_kern, kern+kern_stride_stack*stack,
                         thread_id, nb_thread_id, kern_wid,kern_len,
                         kern_stride_col, kern_stride_row, flipped_kern, c_contiguous);
          __syncthreads();

          for (int row=0; row < kern_len; row++) {//loop over row
            const float* idx_kern=&d_kern[row*kern_wid];
            const float* idx_in=&d_img[(row+out_row-out_row_iter*nb_rows)*img_wid_valid+out_col];
            
            convolutionRowNoFlip<KERN_WIDTH>(sum, idx_kern, idx_in, kern_wid);
          }
        }
        if(out_row<out_len)
          out[batch_id*out_wid*out_len*nkern+//the good batch
              out_wid*out_len*kern_id+//the output image
              out_row*out_wid+out_col] = sum;
      }
    }
}

template <int i> __device__ float everything_dot(const float * x, const int sx, const float * y, const int sy) 
{ 
    return everything_dot<i/2>(x, sx, y, sy) + everything_dot<(i+1)/2>(x+sy*(i/2), sx, y+sy*(i/2), sy) ;
    //return x[0] * y[0] + everything_dot<i-1>(x+sx, sx, y+sy, sy);
}
template <> __device__ float everything_dot<0>(const float * x, const int sx, const float * y, const int sy)
{ 
    return 0;
}
template <> __device__ float everything_dot<1>(const float * x, const int sx, const float * y, const int sy)
{ 
    return x[0] * y[0];
}
template<int NSTACK>
__global__ void
conv_full_load_everything( float* img, float* kern, float* out,
                 int img_len, int img_wid,
                 int kern_len, int kern_wid, int nkern, int nstack,
                 int img_stride_col, int img_stride_row,
                 int img_stride_stack, int img_stride_batch,
                 int kern_stride_col, int kern_stride_row, 
                 int kern_stride_stack, int kern_stride_nkern)
{
    int __shared__ out_len, out_wid, nb_thread_id;
    out_len = img_len + kern_len - 1;
    out_wid = img_wid + kern_wid - 1;
    nb_thread_id = blockDim.y*blockDim.x;

    extern __shared__ float s_data[];

    int batch_id = blockIdx.x;

    const int out_col = threadIdx.x;//output col
    const int out_row = threadIdx.y;//output row
    const int thread_id  = out_row*out_wid + out_col;

    float * d_img=&s_data[0]; //size [nstack * IMAGE_LEN * IMAGE_WID];
    float * d_kern=&s_data[nstack * img_len * img_wid];//size [nstack * KERNEL_LEN * KERNEL_WID];

    img += blockIdx.x * img_stride_batch;//the good batch

    // load the image to shared memory
    for (int i = thread_id; i < nstack * img_len * img_wid; i += nb_thread_id)
    {
        int stack = i / (img_wid*img_len);
        int row = (i % (img_wid*img_len)) / img_wid;
        int col = (i % (img_wid*img_len)) % img_wid;
        d_img[i] = img[stack*img_stride_stack +row*img_stride_row +col*img_stride_col];
    }

    for (int kern_idx = 0; kern_idx < nkern; ++kern_idx, kern += kern_stride_nkern)
    {
        // load the kernel into shared memory and flip it
        for (int i = thread_id; i < nstack * kern_len * kern_wid; i += nb_thread_id)
        {
            int stack = i / (kern_wid*kern_len);
            int row = (i % (kern_wid*kern_len)) / kern_wid;
            int col = (i % (kern_wid*kern_len)) % kern_wid;
            d_kern[stack*kern_len*kern_wid + (kern_len-1-row)*kern_wid + (kern_wid-1-col)]
               = kern[stack*kern_stride_stack +row*kern_stride_row +col*kern_stride_col];
        }
        __syncthreads();

        float sum = 0.0f;
        for (int row=0; row < kern_len; ++row)
        {
            int irow = out_row - kern_len+1+row;
            if (irow < 0 || irow > img_len) continue;
            for (int col = 0; col < kern_wid; ++col)
            {
                int icol = out_col - kern_wid+1+col;
                if (icol < 0 || icol > img_wid) continue;
                if (NSTACK > 0)
                {
                    sum += everything_dot<NSTACK>(d_img + irow*img_wid + icol, img_len*img_wid,
                            d_kern + row*kern_wid+col, kern_len*kern_wid);
                }
                else
                {
                    for (int stack = 0; stack < nstack; ++stack)
                    {
                        sum += d_img[stack*img_len*img_wid + irow*img_wid + icol] * d_kern[stack*kern_len*kern_wid+row*kern_wid+col];
                    }
                }
            }
        }
        out[batch_id*out_wid*out_len*nkern+//the good batch
            out_wid*out_len*kern_idx+//the output image
            out_row*out_wid+out_col] = sum;
        __syncthreads(); //don't start loading another kernel until we're done here
    }
}
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
