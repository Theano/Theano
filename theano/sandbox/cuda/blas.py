from theano import Op, Type, Apply, Variable, Constant
from theano import tensor, scalar
import StringIO

import cuda_ndarray
from theano.sandbox.cuda.type import CudaNdarrayType

class GpuDot22(Op):
    def __str__(self):
        return 'GpuDot22'
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x, y):
        if x.type.ndim != 2:
            raise TypeError(x)
        if y.type.ndim != 2:
            raise TypeError(y)
        return Apply(self, [x,y], [x.type()])

    def c_code_cache_version(self):
        return (1,0)

    def c_code(self, node, nodename, inputs, outputs, sub):
        x, y = inputs
        z, = outputs
        fail = sub['fail']
        return """
        if (%(x)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(x)==%%i must be 2", %(x)s->nd);
            %(fail)s;
        }
        if (%(y)s->nd != 2)
        {
            PyErr_Format(PyExc_TypeError, "rank(y)==%%i must be 2", %(y)s->nd);
            %(fail)s;
        }
        if ((NULL == %(z)s)
            || (CudaNdarray_HOST_DIMS(%(z)s)[0] != CudaNdarray_HOST_DIMS(%(x)s)[0])
            || (CudaNdarray_HOST_DIMS(%(z)s)[1] != CudaNdarray_HOST_DIMS(%(y)s)[1]))
        {
            //if (%(z)s) Py_DECREF(%(z)s);
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = CudaNdarray_HOST_DIMS(%(x)s)[0];
            dims[1] = CudaNdarray_HOST_DIMS(%(y)s)[1];
            %(z)s = (CudaNdarray*)CudaNdarray_new_null();
            if ((NULL == %(z)s) || CudaNdarray_alloc_contiguous(%(z)s, 2, dims))
            {
                if (%(z)s)
                {
                    Py_DECREF(%(z)s);
                    %(z)s = NULL;
                }
                %(fail)s;
            }
        }
        if (CudaNdarray_gemm(1.0f, %(x)s, %(y)s, 0.0f, %(z)s))
        {
            if (%(z)s)
            {
                Py_DECREF(%(z)s);
                %(z)s = NULL;
            }
            %(fail)s;
        }
        """ % locals()
gpu_dot22 = GpuDot22()

class GpuGemm(Op):
    destroy_map = {0:[0]}
    def __str__(self):
        return 'GpuGemm'
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, z, a, x, y, b):
        # the more complicated error checking performed by tensor.gemm is assumed to already
        # have been done
        return Apply(self, [z, a, x, y, b], [z.type()])

    def c_code_cache_version(self):
        return (1,0)

    def c_code(self, node, name, inputs, outputs, sub):
        z_in, a, x, y, b = inputs
        z_out, = outputs
        fail = sub['fail']
        return """

        #define REAL float
        float %(name)s_a = (%(a)s->descr->type_num == PyArray_FLOAT) 
        ? (REAL)(((float*)%(a)s->data)[0])
        : (REAL)(((double*)%(a)s->data)[0]);

        float %(name)s_b = (%(b)s->descr->type_num == PyArray_FLOAT) ?
        (REAL)(((float*)%(b)s->data)[0])
        : (REAL)(((double*)%(b)s->data)[0]);
        #undef REAL

        if (CudaNdarray_gemm(%(name)s_a, %(x)s, %(y)s, %(name)s_b, %(z_in)s))
        {
            %(fail)s;
        }
        %(z_out)s = %(z_in)s;
        Py_INCREF(%(z_out)s);
        """ % locals()
gpu_gemm = GpuGemm()

##
# Not really a BLAS operation, but whatever.
#
class GpuConv(Op):
    @staticmethod
    def logical_output_shape_2d(imshp, kshp, mode):
        if mode == 'valid':
            return imshp[0] - kshp[0] + 1, imshp[1] - kshp[1] + 1
        if mode == 'full':
            return imshp[0] + kshp[0] - 1, imshp[1] + kshp[1] - 1
        raise ValueError(mode)

    def __init__(self, border_mode, 
            subsample=(1,1), 
            logical_img_hw=None, 
            logical_kern_hw=None,
            logical_kern_align_top=True,
            version=-1,
            verbose=0):
        self.border_mode = border_mode
        self.subsample = subsample
        if logical_img_hw is not None:
            h,w = logical_img_hw
            #TODO: reconsider this... since shapes are not given in constructor,
            # maybe a multiplier + offset is a more appropriate way of passing this logical
            # grid
            logical_img_hw = tuple(logical_img_hw)
        self.logical_img_hw = logical_img_hw
        if logical_kern_hw is not None:
            h,w = logical_kern_hw
            #TODO: reconsider this... since shapes are not given in constructor,
            # maybe a multiplier + offset is a more appropriate way of passing this logical
            # grid
            logical_kern_hw = tuple(logical_kern_hw)
        self.logical_kern_hw = logical_kern_hw
        self.logical_kern_align_top = logical_kern_align_top
        self.version=version
        self.verbose=verbose

    def __eq__(self, other):
        return type(self) == type(other) \
            and self.border_mode == other.border_mode \
            and self.subsample == other.subsample \
            and self.logical_img_hw == other.logical_img_hw \
            and self.logical_kern_hw == other.logical_kern_hw \
            and self.logical_kern_align_top == other.logical_kern_align_top

    def __hash__(self):
        return hash(type(self)) \
            ^ hash(self.border_mode) \
            ^ hash(self.subsample) \
            ^ hash(self.logical_img_hw) \
            ^ hash(self.logical_kern_hw) \
            ^ hash(self.logical_kern_align_top)

    def __str__(self):
        return '%s{%s, %s, %s, %s, %s}' %(self.__class__.__name__,
                self.border_mode,
                str(self.subsample),
                str(self.logical_img_hw),
                str(self.logical_kern_hw),
                str(self.logical_kern_align_top))

    def make_node(self, img, kern):
        if img.type.ndim != 4:
            raise TypeError('img must be 4D tensor')
        if kern.type.ndim != 4:
            raise TypeError('kern must be 4D tensor')

        broadcastable = [img.type.broadcastable[0], kern.type.broadcastable[0], False, False]
        return Apply(self, [img, kern], [CudaNdarrayType(broadcastable)()])

    def perform(self, node, (img, kern), (out,)):
        out[0] = cuda_ndarray.conv(img, kern, 
                mode=self.border_mode,
                out=out[0],
                subsample=self.subsample,
                logical_img_shape=self.logical_img_hw,
                logical_kern_shape=self.logical_kern_hw,
                kern_align=self.logical_kern_align_top,
                version=self.version,
                verbose=self.verbose)
    def c_support_code_apply(self, node, nodename):
        if self.logical_img_hw is None or self.logical_kern_hw is None:
            return super(GpuConv,self).c_support_code_apply(node, nodename)
        img_wid = self.logical_img_hw[1]
        img_len = self.logical_img_hw[0]

        kern_wid=self.logical_kern_hw[1]
        kern_len=self.logical_kern_hw[0]
        return"""
const unsigned long int COALESCED_ALIGN = 0xFFFFFFFFFFFFFF00; // zero-out the trailing bits of pointers
#define MASKED_OFFSET(src) (((int)((unsigned long int)src - (((unsigned long int)src) & COALESCED_ALIGN))) / sizeof(float))
__device__ void load_to_shared(float * dst, const float * src, const int thread_id, int nb_thread, const int N, const bool flipped=false){
  if (nb_thread < 64)
    {
      if(flipped) 
        //TODO very slow on device before 1.3. make access to kern sequential and access to d_kern flipped.
        for(int i=thread_id;i<N;i+=nb_thread)
          dst[i]=src[N - 1 - i];
        //dst[N-1-i]=src[i];
      else
        for(int i=thread_id;i<N;i+=nb_thread)
          dst[i]=src[i];
    }
  else
    {
      nb_thread = nb_thread & 0xFFFFFFE0; //make nb_thread a multiple of 32
      // Global memory:
      //  <-------------------------------------->
      //      A      A      A      A      A   // points of 128-bit alignment
      //         dddddddddddddddddddddd       // layout of src in global memory
      //      |--|                            // masked_src_offset
      // 
      if (thread_id < nb_thread)
        {
          const int masked_src_offset = MASKED_OFFSET(src);
          for(int masked_i=thread_id; masked_i<N + masked_src_offset; masked_i+=nb_thread)
            {
              int i = masked_i - masked_src_offset;
              if (i >= 0)
                if (flipped)
                  dst[N-1-i] = src[i];
                else
                  dst[i]=src[i];
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
  if(flipped && ! c_contiguous){
    for(int i=thread_id;i<nb_row*nb_col;i+=nb_thread)
      dst[nb_row*nb_col-1-i]=src[i/nb_col*stride_row+i%%nb_col*stride_col];
  }else if(c_contiguous){
    load_to_shared(dst, src, thread_id, nb_thread, nb_col*nb_row, flipped);
  
  }else if(flipped){//c_contiguous==true
    //TODO very slow on device before 1.3. make access to kern sequential and access to d_kern flipped.
    int N=nb_col*nb_row;
    for(int i=thread_id;i<N;i+=nb_thread)
      dst[i]=src[N - 1 - i];
    //dst[N-1-i]=src[i];
  }else if(c_contiguous){//flipped==false
    for(int i=thread_id;i<nb_col*nb_row;i+=nb_thread)
      dst[i]=src[i];
  }else{ // !flipped && !c_contiguous
    /*
    for(int i=thread_id;i<nb_row;i+=nb_thread){
      float* s=&src[i*stride_row];
      float* d=&dst[i*nb_col];
      for(int j=thread_id;j<nb_col;i+=nb_thread)
	//	dst[i*nb_col+j]=src[i*stride_row+j*stride_col];//dst[i]=src[i];
	d[j]=s[j*stride_col];
	}*/
    /* We don't do this if as nvcc 2.3 take 2 more registers when we add the if
       Why it do this?
    if(stride_col==1 && stride_row==nb_col)
      for(int i=thread_id;i<nb_row*nb_col;i+=nb_thread)
	dst[i]=src[i];
	else*/
      for(int i=thread_id;i<nb_row*nb_col;i+=nb_thread)
	dst[i]=src[i/nb_col*stride_row+i%%nb_col*stride_col];

  }

}

__device__ void load_padded_col_to_shared(float * dst, const float * src, 
					  const int thread_id, const int nb_thread,
					  const int nb_col, const int nb_row, 
					  const int stride_col, const int stride_row,
					  const int wid_pad, const bool c_contiguous=true){
  if(c_contiguous){//flipped==false
    //template nb_col to have better performance!
    int row=0;
    int col=thread_id;
    for(int i=thread_id;i<nb_col*nb_row;i+=nb_thread, col+=nb_thread){
      col-=nb_col;row++;
      while(col>nb_col){
        col-=nb_col;row++;
      }
      dst[row*(nb_col+2*wid_pad)+col+wid_pad]=src[i];
    }
/*    
    for(int i=thread_id;i<nb_col*nb_row;i+=nb_thread){
      int col=i%%nb_col;
      int row=i/nb_col;
      dst[row*(nb_col+2*wid_pad)+col+wid_pad]=src[i];
    }
*/
  }else{
    for(int i=thread_id;i<nb_row*nb_col;i+=nb_thread){
      int col=i%%nb_col;
      int row=i/nb_col;
      dst[row*(nb_col+2*wid_pad)+col+wid_pad]=src[row*stride_row+col*stride_col];
    }
  }

}

template<int i> __device__ float convolutionRowNoFlip(const float *data,
						      const float *kern){
  return data[i-1] * kern[i-1] + convolutionRowNoFlip<i - 1>(data,kern);
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

__device__ void fill(float * dst, int N, float value, int thread_id, int nb_thread){
  for(int i=thread_id;i<N;i+=nb_thread)
    dst[i]=value;
}

template <typename T>
static T ceil_intdiv(T a, T b)
{
    return (a/b) + ((a %% b) ? 1: 0);
}


/**
 * As conv_patch_stack, but used for the full convolution by padding the image in shared memory.
 * I keep it separated from conv_patch as we take 19-20 register which is more then the 10/16 max for each thread and thus this could lower the occupency.
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
 * template split: if true, each thread compute more then 1 output pixel.
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
  out_len = %(img_len)s + %(kern_len)s - 1;
  out_wid = %(img_wid)s + %(kern_wid)s - 1;
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
    float * d_img=&s_data[%(kern_len)s*%(kern_wid)s];//size of [see fct doc];

    kern+=kern_stride_nkern*kern_id;//the good nkern
    img+=img_stride_batch*batch_id;//the good batch

    img_wid_valid=%(img_wid)s+2*%(kern_wid)s-2;

    if(!split && !low_mem){
      fill(d_img,img_wid_valid*(%(img_len)s+2*%(kern_len)s-2), 0, thread_id, nb_thread_id);
      const int out_row = ty;//output row
      float sum = 0.0f;
      for (int stack = 0;stack<nstack;stack++,kern+=kern_stride_stack,
	     img+=img_stride_stack){
	  __syncthreads();
	load_padded_col_to_shared(d_img+img_wid_valid*(%(kern_len)s-1),img,
				  thread_id,nb_thread_id,%(img_wid)s,%(img_len)s,
				  img_stride_col, img_stride_row, %(kern_wid)s-1,
				  c_contiguous);
	load_to_shared(d_kern, kern, thread_id, nb_thread_id, %(kern_wid)s,%(kern_len)s,
		       kern_stride_col, kern_stride_row, flipped_kern, c_contiguous);
	__syncthreads();

	for (int row=0; row < %(kern_len)s; row++) {//loop over row
	  const float* idx_kern=&d_kern[row*%(kern_wid)s];
	  const float* idx_in=&d_img[(row+out_row)*img_wid_valid+out_col];
	  
	  convolutionRowNoFlip<KERN_WIDTH>(sum, idx_kern, idx_in, %(kern_wid)s);
	}
      }
      out[batch_id*out_wid*out_len*nkern+//the good batch
	  kern_id*out_wid*out_len+//the output image
	  out_row*out_wid+out_col] = sum;
    }else if(split && !low_mem){
      fill(d_img,img_wid_valid*(%(img_len)s+2*%(kern_len)s-2), 0, thread_id, nb_thread_id);
      //out_len_max must by higher then out_len as we need all thread when we load the image as the nb_rows is not always a multiple of out_len.
      __shared__ int out_len_max;
      //TODO pass a parameter nb_split
      out_len_max = (out_len/blockDim.y+(out_len %% blockDim.y==0?0:1))*blockDim.y;
      for(int out_row = ty;out_row<out_len_max;out_row+=nb_rows){
	float sum = 0.0f;
	for (int stack = 0;stack<nstack;stack++){
	  __syncthreads();
	  //TODO: load only the part of the image needed or put the partial result in shared memory
	  load_padded_col_to_shared(d_img+img_wid_valid*(%(kern_len)s-1),
				    img+img_stride_stack*stack,
				    thread_id,nb_thread_id,%(img_wid)s,%(img_len)s,
				    img_stride_col, img_stride_row, %(kern_wid)s-1,
				    c_contiguous);
	  load_to_shared(d_kern, kern+kern_stride_stack*stack,
			 thread_id, nb_thread_id, %(kern_wid)s,%(kern_len)s,
			 kern_stride_col, kern_stride_row, flipped_kern, c_contiguous);
	  __syncthreads();

	  for (int row=0; row < %(kern_len)s; row++) {//loop over row
	    const float* idx_kern=&d_kern[row*%(kern_wid)s];
	    const float* idx_in=&d_img[(row+out_row)*img_wid_valid+out_col];
	    
	    convolutionRowNoFlip<KERN_WIDTH>(sum, idx_kern, idx_in, %(kern_wid)s);
	  }
	  if(out_row<out_len)
	    out[batch_id*out_wid*out_len*nkern+//the good batch
		out_wid*out_len*kern_id+//the output image
		out_row*out_wid+out_col] = sum;
	}
      }
    }else{//low_mem version
      //don't need to fill the last rows padding as this is done later.
      fill(d_img,img_wid_valid*((%(kern_len)s+nb_rows-1)+2*%(kern_len)s-2), 0, thread_id, nb_thread_id);
      //out_len_max must by higher then out_len as we need all thread when we load the image as the nb_rows is not always a multiple of out_len.
      __shared__ int out_len_max;
      //TODO pass a parameter nb_split
      if(thread_id==0)
	out_len_max = (out_len/nb_rows+(out_len %% nb_rows==0?0:1))*nb_rows;
      __syncthreads();
      for(int out_row = ty, out_row_iter=0;out_row<out_len_max;
	  out_row+=nb_rows, out_row_iter++){
	float sum = 0.0f;
	for (int stack = 0;stack<nstack;stack++){
	  __syncthreads();
	  const int len_to_load=min(%(kern_len)s+nb_rows,%(img_len)s-out_row_iter*nb_rows);//nb rows to load, min(nb_rows for this iter, nb rows left in the image)
	  const int empty_row = max(%(kern_len)s-1-out_row_iter*nb_rows,0);//number of empty row at the start
	  //we need to reload some row as when we change of out_row we lost the last load du to the stack.
	  const int previous_row = min(out_row_iter*nb_rows,%(kern_len)s-1);//number of row from last out_row iteration to reload
	  load_padded_col_to_shared(d_img+(%(kern_len)s-1-previous_row)*img_wid_valid,
				    img+img_stride_stack*stack//the good stack image
				    +(out_row_iter*nb_rows-previous_row)*img_stride_row,//the good split top row.
				    thread_id,nb_thread_id,%(img_wid)s,
				    len_to_load+previous_row,
				    img_stride_col, img_stride_row, %(kern_wid)s-1,
				    c_contiguous);
	  __syncthreads();
	  //TODO: fill the last row padding only when needed.
	  //We always fill the last rows padding event when not needed.
	  int row_to_fill = 2*%(kern_len)s-2+nb_rows- empty_row - previous_row - len_to_load;
	  row_to_fill = min(row_to_fill,%(kern_len)s-1);
	  fill(d_img+(%(kern_len)s-1+len_to_load)*img_wid_valid,
	       img_wid_valid*row_to_fill, 0, thread_id, nb_thread_id);
	  load_to_shared(d_kern, kern+kern_stride_stack*stack,
			 thread_id, nb_thread_id, %(kern_wid)s,%(kern_len)s,
			 kern_stride_col, kern_stride_row, flipped_kern, c_contiguous);
	  __syncthreads();

	  for (int row=0; row < %(kern_len)s; row++) {//loop over row
	    const float* idx_kern=&d_kern[row*%(kern_wid)s];
	    const float* idx_in=&d_img[(row+out_row-out_row_iter*nb_rows)*img_wid_valid+out_col];
	    
	    convolutionRowNoFlip<KERN_WIDTH>(sum, idx_kern, idx_in, %(kern_wid)s);
	  }
	}
	if(out_row<out_len)
	  out[batch_id*out_wid*out_len*nkern+//the good batch
	      out_wid*out_len*kern_id+//the output image
	      out_row*out_wid+out_col] = sum;
      }
    }
}

	void (*f_contig_3_flipped)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int)=conv_full_patch_stack_padded<true,%(kern_wid)s,true,false,false>;
	void (*f_contig_4_flipped)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int)=conv_full_patch_stack_padded<true,%(kern_wid)s,true,true,false>;
	void (*f_contig_5_flipped)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int)=conv_full_patch_stack_padded<true,%(kern_wid)s,true,false,true>;
	void (*f_3_flipped)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int)=conv_full_patch_stack_padded<true,%(kern_wid)s,false,false,false>;
	void (*f_4_flipped)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int)=conv_full_patch_stack_padded<true,%(kern_wid)s,false,true,false>;
	void (*f_5_flipped)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int)=conv_full_patch_stack_padded<true,%(kern_wid)s,false,false,true>;
	void (*f_contig_3)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int)=conv_full_patch_stack_padded<false,%(kern_wid)s,true,false,false>;
	void (*f_contig_4)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int)=conv_full_patch_stack_padded<false,%(kern_wid)s,true,true,false>;
	void (*f_contig_5)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int)=conv_full_patch_stack_padded<false,%(kern_wid)s,true,false,true>;
	void (*f_3)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int)=conv_full_patch_stack_padded<false,%(kern_wid)s,false,false,false>;
	void (*f_4)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int)=conv_full_patch_stack_padded<false,%(kern_wid)s,false,true,false>;
	void (*f_5)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int)=conv_full_patch_stack_padded<false,%(kern_wid)s,false,false,true>;

"""%locals()
    def c_code(self, node, nodename, (img, kern), (out,), sub):
        out_=node.outputs[0]
        img_=node.inputs[0]
        kern_=node.inputs[1]
        subsample_rows=self.subsample[0]
        subsample_cols=self.subsample[1]
        version=self.version
        verbose=self.verbose
        if self.logical_img_hw is None or self.logical_kern_hw is None:
            return super(GpuConv,self).c_code(node,nodename,(img, kern), (out,),sub)        
        #todo assert out is ccontiguous
        img_wid = self.logical_img_hw[1]
        img_len = self.logical_img_hw[0]
        kern_wid = self.logical_kern_hw[1]
        kern_len=self.logical_kern_hw[0]
        
        img_wid_padded=self.logical_img_hw[1]+2*self.logical_kern_hw[1]-2;
        img_len_padded=self.logical_img_hw[0]+2*self.logical_kern_hw[0]-2;
        img_size_padded=img_len_padded * img_wid_padded;
        out_dim_2, out_dim_3 = self.logical_output_shape_2d(self.logical_img_hw,self.logical_kern_hw,self.border_mode)

        fail=sub['fail']
        if False and self.subsample==(1,1) and self.border_mode=='full' and self.version in [3,4,5,-1] and out_dim_3<=512 and ((self.logical_kern_hw[0]+2*self.logical_kern_hw[0]-2)*img_wid_padded*4 + self.logical_kern_hw[0]*self.logical_kern_hw[1]*4<(16*1024-128)) and out_.dtype=='float32' and kern_.dtype=='float32' and img_.dtype=='float32':#-128 as this is the number of shared memory used statically
            return """

CudaNdarray* img = %(img)s;
CudaNdarray* kern = %(kern)s;
CudaNdarray* out_ = %(out)s;
CudaNdarray* out = out;
int version = %(version)s;
const int verbose = %(verbose)s;
    if (!img || img->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required img of 4D");
        return -1;
    }
    if (! kern || kern->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required kern of 4D");
        return -1;
    }

    int out_dim[4]={CudaNdarray_HOST_DIMS(img)[0], CudaNdarray_HOST_DIMS(kern)[0],
                    %(out_dim_2)s, %(out_dim_3)s};
    if(!(out && out->nd==4 && CudaNdarray_is_c_contiguous(out) 
	 && CudaNdarray_HOST_DIMS(out)[0]==out_dim[0]
	 && CudaNdarray_HOST_DIMS(out)[1]==out_dim[1]
	 && CudaNdarray_HOST_DIMS(out)[2]==out_dim[2]
	 && CudaNdarray_HOST_DIMS(out)[3]==out_dim[3])){
      if (out)
      {
          Py_DECREF(out);
          fprintf(stderr, "Warning: Conv is ignoring 'out' argument with wrong structure.\\n");
      }
      out = (CudaNdarray*)CudaNdarray_NewDims(4,out_dim);
      %(out)s = out;
    }


    if (! out || out->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required out of 4D");
        return -1;
    }
    if (%(subsample_rows)s==1 && %(subsample_cols)s==1)
    {
        //TODO: rethink these asserts in light of the difference between physical and logical dimensions
        assert (CudaNdarray_HOST_DIMS(out)[2] == CudaNdarray_HOST_DIMS(img)[2] + CudaNdarray_HOST_DIMS(kern)[2] - 1);
        assert (CudaNdarray_HOST_DIMS(out)[3] == CudaNdarray_HOST_DIMS(img)[3] + CudaNdarray_HOST_DIMS(kern)[3] - 1);
    }
    assert (CudaNdarray_HOST_DIMS(out)[0] == CudaNdarray_HOST_DIMS(img)[0]);
    assert (CudaNdarray_HOST_DIMS(out)[1] == CudaNdarray_HOST_DIMS(kern)[0]);
    assert (CudaNdarray_HOST_DIMS(img)[1] == CudaNdarray_HOST_DIMS(kern)[1]);

    //TODO: make separate version as if all fill this is slower. 
    //TODO: make a parameter the number of division
    //TODO: Should we make them in separate grid block instead?
 

    const int nstack=CudaNdarray_HOST_DIMS(kern)[1];
    const int nbatch=CudaNdarray_HOST_DIMS(img)[0];
    const int nkern=CudaNdarray_HOST_DIMS(kern)[0];
    const int img_wid=%(img_wid)s;
    const int img_len=%(img_len)s;
    const int kern_wid=%(kern_wid)s;
    const int kern_len=%(kern_len)s;
    const int out_wid=CudaNdarray_HOST_DIMS(out)[3];
    const int out_len=CudaNdarray_HOST_DIMS(out)[2];

    const int img_stride_col= CudaNdarray_HOST_STRIDES(img)[3];
    const int img_stride_row=CudaNdarray_HOST_STRIDES(img)[2];
    const int img_stride_stack= CudaNdarray_HOST_STRIDES(img)[1];
    const int img_stride_batch=CudaNdarray_HOST_STRIDES(img)[0];
    const int kern_stride_col= CudaNdarray_HOST_STRIDES(kern)[3];
    const int kern_stride_row=CudaNdarray_HOST_STRIDES(kern)[2];
    const int kern_stride_stack= CudaNdarray_HOST_STRIDES(kern)[1];
    const int kern_stride_nkern=CudaNdarray_HOST_STRIDES(kern)[0];

    const int img_size=img_len*img_wid;
    const int kern_size=%(kern_len)s*%(kern_wid)s;
    const int out_size=out_len*out_wid;
    const int img_size_byte = img_size*sizeof(float);
    const int kern_size_byte = kern_size*sizeof(float);
    const int out_size_byte = out_size*sizeof(float);

    bool subsample = %(subsample_rows)s!=1 || %(subsample_cols)s!=1;
    bool img_contiguous = CudaNdarray_is_c_contiguous(img);
    bool kern_contiguous = CudaNdarray_is_c_contiguous(kern);
    bool out_contiguous = CudaNdarray_is_c_contiguous(out);
    bool c_contiguous = img_contiguous &&  kern_contiguous && out_contiguous;

    bool img_contiguous_2d = (img_stride_col == 1) && (img_stride_row==img_wid);
    bool kern_contiguous_2d = (kern_stride_col == 1) && (kern_stride_row==%(kern_wid)s);

    //if the lower 2 dims are c_contiguous but flipped, unflipping the stride and not flipping the kernel in shared memroy
    //allow to use a version that use less registers(so is faster)
    //the unflipped version of variable haev the original value when we don't need to unflip it, but have the new value when we unflip it.
    bool kern_flipped=true;
    bool kern_contiguous_2d_unflipped = kern_contiguous_2d;
    float * kern_data_unflipped = kern->devdata;
    int kern_stride_col_unflipped=kern_stride_col;
    int kern_stride_row_unflipped=kern_stride_row;

   if (!subsample &&
	out_contiguous &&
	(version==3||version==4||version==5||version==-1) &&
	out_wid<512 &&//Maximum of 512 threads by block
	(%(kern_len)s+2*%(kern_len)s-2)*%(img_wid_padded)s*sizeof(float) + kern_size_byte<16*1024 //their is only 16k of shared memory
	) //conv_full_patch_stack_padded
    {
      //version 3 without split
      //version 4 with split (more registers)
      //version 5 with split (more registers) low mem version(some restriction and still more register)
        int nb_split=1;//The number of split (i.e. the number of output pixel each thread compute.)
	if((version==4 || version==5) && out_len>1) nb_split++;//to force the use of split=true when testing.
	if(%(kern_len)s==1 && version==5){
	  //version 5 don't support %(kern_len)s==1 as 1%%0 return -1.
	  version=-1;
	  if(verbose)printf("WARNING:conv full: Asking version 5 with %(kern_len)s==1. Combination not supported!\\n");
	}
	if(%(img_size_padded)s*4+kern_size_byte>16*1024) version=5;

	//we pass by ceil_intdiv in case the out_len is not a multiple of nb_split, we want nb_split the number of iteration.
	//Max of 16k of shared memory
	if(version==5)
	  while ((((%(kern_len)s+ceil_intdiv(out_len,nb_split)-1)+2*%(kern_len)s-2)*%(img_wid_padded)s*sizeof(float) + kern_size_byte)>16*1024) nb_split++;
	
	//327 as we use 25 register
	//version 5 will have only 1 block running at a time, so we can use 32 registers per threads, but their is some other stuff that for the limit to bu lower then 512.
	int max_thread = (version!=5?327:450);
	while (ceil_intdiv(out_len,nb_split)*out_wid>max_thread) nb_split++;
	if(version==-1 && out_size>512)version=4;
	if(version==-1)version=3;


	if(version==-1 && nb_split>1) version=4;
	else if(version==-1) version=3;
	else if(version==3 && nb_split!=1) version=4;//we force version 4 when we need more then 1 split as to be always execute.

	assert(version!=3 || nb_split==1);
	assert(version!=5 || %(kern_len)s>1);
	assert(version!=-1);

        dim3 threads(out_wid, ceil_intdiv(out_len,nb_split));
        dim3 grid(nbatch,nkern);

	int shared_size=%(img_size_padded)s*4 + kern_size_byte;
	if(version==5)
	  shared_size=((%(kern_len)s+threads.y-1)+2*%(kern_len)s-2)*%(img_wid_padded)s*sizeof(float) + kern_size_byte;

	void (*f)(float*, float*, float*,
		  int, int, int, int,
		  int, int, int, int,
		  int, int, int, int,
		  int, int);

        if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==3 && kern_flipped) f=f_contig_3_flipped;
	else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==4 && kern_flipped) f=f_contig_4_flipped;
	else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==5 && kern_flipped) f=f_contig_5_flipped;
	else if(version==3 && kern_flipped) f=f_3_flipped;
	else if(version==4 && kern_flipped) f=f_4_flipped;
	else if(version==5 && kern_flipped) f=f_5_flipped;
	else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==3) f=f_contig_3;
	else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==4) f=f_contig_4;
	else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==5) f=f_contig_5;
	else if(version==3) f=f_3;
	else if(version==4) f=f_4;
	else if(version==5) f=f_5;
	else assert(false);

	f<<< grid, threads, shared_size>>>
	     (img->devdata, kern_data_unflipped, out->devdata,
	      img_len, img_wid, kern_len, kern_wid, nkern, nstack,
	      img_stride_col, img_stride_row, img_stride_stack,
	      img_stride_batch, kern_stride_col_unflipped, kern_stride_row_unflipped,
	      kern_stride_stack, kern_stride_nkern);

        CNDA_THREAD_SYNC;
        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts) 
        {
            if (verbose>1) printf("threads.x=%%i, threads.y=%%i, grid.x=%%i, grid.y=%%i,shared_size=%%i, nb_threads=%%i, out_len=%%i, nb_split=%%i, version=%%i\\n", threads.x, threads.y, grid.x, grid.y, shared_size, threads.x * threads.y, out_len, nb_split, version);
            if (verbose) printf("INFO: used 'conv_full_patch_stack_padded' nb_split=%%d low_mem=%%s\\n",nb_split,(version==5?"true":"false"));
        }
        else
        {
            if (verbose) printf("threads.x=%%i, threads.y=%%i, grid.x=%%i, grid.y=%%i,shared_size=%%i, nb_threads=%%i, out_len=%%i, nb_split=%%i, version=%%i\\n", threads.x, threads.y, grid.x, grid.y, shared_size, threads.x * threads.y, out_len, nb_split, version);
            if (verbose) printf("INFO: impl 'conv_full_patch_stack_padded' %%s %%s failed (%%s), trying next implementation\\n",
				version==3?"no split": "split",(version==5?"low_mem":"not_low_mem"),
                                cudaGetErrorString(sts));
//TODO: raise an error!
            PyErr_Format(PyExc_RuntimeError, "INFO: impl 'conv_full_patch_stack_padded' %%s %%s failed (%%s), trying next implementation\\n",
				version==3?"no split": "split",(version==5?"low_mem":"not_low_mem"),
                                cudaGetErrorString(sts));
            %(fail)s;

        }                         
    }

"""%locals()
        else:
            super(GpuConv,self).c_code(node,nodename,(img, kern), (out,),sub)

class GpuDownsampleFactorMax(Op):
    def __init__(self, ds, ignore_border=False):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border

    def __eq__(self, other):
        return type(self) == type(other) and self.ds == other.ds and self.ignore_border == other.ignore_border

    def __hash__(self):
        return hash(type(self)) ^ hash(self.ds) ^ hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s}' % (self.__class__.__name__, self.ds, self.ignore_border)

    def make_node(self, x):
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError()
        if not x.type.ndim == 4:
            raise TypeError()
        return Apply(self, [x], [x.type()])
    #def perform(self, node, input_storage, output_storage):
        #raise NotImplementedError('only C is implemented')
    def c_code_cache_version(self):
        return ()
    def c_code(self, node, nodename, (x,), (z,), sub):
        fail = sub['fail']
        ds0, ds1 = self.ds
        ignore_border = int(self.ignore_border)
        return """
        int dims[4], xdim2, xdim3;
        if (%(x)s->nd != 4)
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        xdim2 = CudaNdarray_HOST_DIMS(%(x)s)[2];
        xdim3 = CudaNdarray_HOST_DIMS(%(x)s)[3];
        dims[0] = CudaNdarray_HOST_DIMS(%(x)s)[0];
        dims[1] = CudaNdarray_HOST_DIMS(%(x)s)[1];
        dims[2] = xdim2 / %(ds0)s;
        dims[3] = xdim3 / %(ds1)s;
        if (! %(ignore_border)s)
        {
            dims[2] += (xdim2%%(%(ds0)s)?1:0);
            dims[3] += (xdim3%%(%(ds1)s)?1:0);
        }
        if(dims[3]>512){
            PyErr_SetString(PyExc_ValueError, "last dimention bigger then 512. This case is not implemented.");
            %(fail)s;
        }

        if ((NULL == %(z)s)
            || (CudaNdarray_HOST_DIMS(%(z)s)[0] != dims[0])
            || (CudaNdarray_HOST_DIMS(%(z)s)[1] != dims[1])
            || (CudaNdarray_HOST_DIMS(%(z)s)[2] != dims[2])
            || (CudaNdarray_HOST_DIMS(%(z)s)[3] != dims[3]))
        {
            Py_XDECREF(%(z)s);
            %(z)s = (CudaNdarray*)CudaNdarray_new_null();
            if ((NULL == %(z)s)
                || CudaNdarray_alloc_contiguous(%(z)s, 4, dims))
            {
                Py_XDECREF(%(z)s);
                %(z)s = NULL;
                PyErr_SetString(PyExc_ValueError, "Was not able to allocate output!");
                %(fail)s;
            }
        }
        {
            dim3 grid(dims[0] * dims[1], dims[2]);
            //dim3 block(std::min(dims[3], 512)); //TODO: implement this by supporting more
            //outputs than threads
            dim3 block(dims[3]);
            if ((grid.x*grid.y) && dims[3])
            kMaxPool_%(nodename)s<%(ds0)s, %(ds1)s> <<<grid, block, xdim3*sizeof(float)>>>(
                dims[0], dims[1], dims[2], dims[3], xdim2, xdim3,
                CudaNdarray_DEV_DATA(%(x)s),
                CudaNdarray_HOST_STRIDES(%(x)s)[0],
                CudaNdarray_HOST_STRIDES(%(x)s)[1],
                CudaNdarray_HOST_STRIDES(%(x)s)[2],
                CudaNdarray_HOST_STRIDES(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(z)s));
            CNDA_THREAD_SYNC;
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kMaxPool_%(nodename)s",
                    cudaGetErrorString(err),
                    grid.x,
                    grid.y,
                    block.x,
                    block.y,
                    block.z);
                %(fail)s;
            }                         
        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        ignore_border = int(self.ignore_border)
        return """
        template<int pf2, int pf3>
        __global__ void kMaxPool_%(nodename)s(
           int D0, int D1, int D2, int D3, int xD2, int xD3,
           const float * x, int xS0, int xS1, int xS2, int xS3, 
           float *z)
        {
            float cur_max, cur_x;
            int i0 = blockIdx.x %% D0;
            int i1 = blockIdx.x / D0;
            int i2 = blockIdx.y;

            extern __shared__ float xbuf[]; //size [xD3]

            for (int r2 = 0; (r2 < pf2) && (%(ignore_border)s || (r2 + i2*pf2 < xD2)); ++r2)
            {
                __syncthreads();
                // load the current row of the image into shared memory
                for (int j = threadIdx.x; j < xD3; j += blockDim.x)
                {
                    xbuf[j] = x[i0*xS0 + i1*xS1 + (i2*pf2+r2)*xS2 + j*xS3];
                }
                __syncthreads();
                 
                // initialize our max if this is the first row we're loading
                cur_max = (r2 == 0) ? xbuf[threadIdx.x*pf3] : cur_max;

                // do a mini-reduction over the pf3 relevant elements in the current row
                if (%(ignore_border)s)
                {
                    for (int k = 0; k < pf3; ++k)
                    {
                        cur_x = xbuf[threadIdx.x*pf3+k];
                        cur_max = (cur_x > cur_max) ? cur_x : cur_max;
                    }
                }
                else
                {
                    for (int k = 0; k < pf3; ++k)
                    {
                        if (threadIdx.x*pf3 + k < xD3)
                        {
                            cur_x = xbuf[threadIdx.x*pf3+k];
                            cur_max = (cur_x > cur_max) ? cur_x : cur_max;
                        }
                    }
                }
            }

            //store the result to global memory
            z[i0 * D1*D2*D3 + i1*D2*D3 + i2*D3 + threadIdx.x] = cur_max;
        }
        """ % locals()

class GpuDownsampleFactorMaxGrad(Op):
    def __init__(self, ds, ignore_border):
        self.ds = tuple(ds)
        self.ignore_border = ignore_border

    def __eq__(self, other):
        return type(self) == type(other) and self.ds == other.ds and self.ignore_border == other.ignore_border

    def __hash__(self):
        return hash(type(self)) ^ hash(self.ds) ^ hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s}' % (self.__class__.__name__, self.ds, self.ignore_border)

    def make_node(self, x, z, gz):
        return Apply(self, [x, z, gz], [x.type()])
    def c_code_cache_version(self):
        return (1,)
    def c_code(self, node, nodename, (x, z, gz), (gx,), sub):
        fail = sub['fail']
        ds0, ds1 = self.ds
        ignore_border = int(self.ignore_border)
        return """
        if (%(x)s->nd != 4
            || %(z)s->nd != 4
            || %(gz)s->nd != 4)
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if ((NULL == %(gx)s)
            || (CudaNdarray_HOST_DIMS(%(gx)s)[0] != CudaNdarray_HOST_DIMS(%(x)s)[0])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[1] != CudaNdarray_HOST_DIMS(%(x)s)[1])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[2] != CudaNdarray_HOST_DIMS(%(x)s)[2])
            || (CudaNdarray_HOST_DIMS(%(gx)s)[3] != CudaNdarray_HOST_DIMS(%(x)s)[3]))
        {
            Py_XDECREF(%(gx)s);
            %(gx)s = (CudaNdarray*)CudaNdarray_new_null();
            if ((NULL == %(gx)s)
                || CudaNdarray_alloc_contiguous(%(gx)s, 4, CudaNdarray_HOST_DIMS(%(x)s)))
            {
                Py_XDECREF(%(gx)s);
                %(gx)s = NULL;
                %(fail)s;
            }
        }
        {
            //TODO: supporting more output columns than threads
            dim3 grid(CudaNdarray_HOST_DIMS(%(z)s)[0], CudaNdarray_HOST_DIMS(%(z)s)[2]);
            dim3 block(CudaNdarray_HOST_DIMS(%(x)s)[3]);
            kDownsampleMaxGrad_%(nodename)s<%(ds0)s, %(ds1)s> <<<grid, block>>>(
                CudaNdarray_HOST_DIMS(%(z)s)[0],
                CudaNdarray_HOST_DIMS(%(z)s)[1],
                CudaNdarray_HOST_DIMS(%(z)s)[2],
                CudaNdarray_HOST_DIMS(%(z)s)[3],
                CudaNdarray_HOST_DIMS(%(x)s)[2],
                CudaNdarray_HOST_DIMS(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(x)s),
                CudaNdarray_HOST_STRIDES(%(x)s)[0],
                CudaNdarray_HOST_STRIDES(%(x)s)[1],
                CudaNdarray_HOST_STRIDES(%(x)s)[2],
                CudaNdarray_HOST_STRIDES(%(x)s)[3],
                CudaNdarray_DEV_DATA(%(z)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1],
                CudaNdarray_HOST_STRIDES(%(z)s)[2],
                CudaNdarray_HOST_STRIDES(%(z)s)[3],
                CudaNdarray_DEV_DATA(%(gz)s),
                CudaNdarray_HOST_STRIDES(%(gz)s)[0],
                CudaNdarray_HOST_STRIDES(%(gz)s)[1],
                CudaNdarray_HOST_STRIDES(%(gz)s)[2],
                CudaNdarray_HOST_STRIDES(%(gz)s)[3],
                CudaNdarray_DEV_DATA(%(gx)s));
            CNDA_THREAD_SYNC;
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err) 
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s. (grid: %%i x %%i; block: %%i x %%i x %%i)\\n",
                    "kDownsampleMaxGrad_%(nodename)s",
                    cudaGetErrorString(err),
                    grid.x,
                    grid.y,
                    block.x,
                    block.y,
                    block.z);
                %(fail)s;
            }                         
        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        # This code is not sensitive to the ignore_border flag.
        # It runs for every position in the output z, and then computes the gradient for the
        # input pixels that were downsampled to that z-position.
        return """
        template<int ds0, int ds1> // ds0 is the downsampling factor in rows, ds1 in columns
        __global__ void kDownsampleMaxGrad_%(nodename)s(
           int D0, int D1, int D2, int D3, int xD2, int xD3,
           const float * x, int xS0, int xS1, int xS2, int xS3, 
           const float * z, int zS0, int zS1, int zS2, int zS3, 
           const float * gz, int gzS0, int gzS1, int gzS2, int gzS3, 
           float *gx)
        {
            float cur_max, cur_x, my_z, my_gz;
            int i0 = blockIdx.x;
            int i1 = 0;
            int i2 = blockIdx.y;       // row wrt z and/or gz
            int x_col = threadIdx.x;


            //TODO: raise occupancy.  Use threadIdx.y to run several iterations of this i1 loop
            //in parallel

            for (i1 = 0; i1 < D1; ++i1) // loop over images (same for z and x)
            {
                if (x_col >= ds1 * D3)
                {
                    // This happens only if x_col was ignored (via ignore_border)
                    // TODO: if ignore_border is False, this is impossible and we don't even
                    //       need to generate this code.

                    my_gz = 0.0f;
                    //any fp number suffices for my_z, so we don't even need to set it to
                    //anything in particular.
                }
                else
                {
                    my_gz = gz[i0 * gzS0 + i1 * gzS1 + i2 * gzS2 + (x_col/ds1)*gzS3];
                    my_z =   z[i0 *  zS0 + i1 *  zS1 + i2 *  zS2 + (x_col/ds1)* zS3];
                }

                for (int x_row = i2*ds0; (x_row < i2*ds0+ds0) && (x_row < xD2); ++x_row)
                {
                    gx[i0 * D1*xD2*xD3 + i1*xD2*xD3 + x_row*xD3 + x_col]
                       = (my_z == x[i0*xS0 + i1*xS1 + x_row*xS2 + x_col*xS3]) ? my_gz : 0.0f;
                }
            }
        }
        """ % locals()


