// REMEMBER TO INCREASE c_code_cache_version when changing this file
//
//TODO detect SHARED_SIZE dynamically
#define SHARED_SIZE (16*1024)

enum { ConvMode_FULL, ConvMode_VALID };
PyObject * PyGpuArray_Conv(PyGpuArrayObject *img, PyGpuArrayObject * kern,
                           PyGpuArrayObject * out, const int mode,
                           const size_t subsample_rows,
                           const size_t subsample_cols,
                           const int version, const int verbose);

template <typename T>
static T ceil_intdiv(T a, T b)
{
    return (a/b) + ((a % b) ? 1: 0);
}

/*
 * version: -1, autodetect, >=0 a specific version to use.
 *          If it can't be executed, we revert to the reference implementation
 */
int
PyGpuArray_conv_valid(const PyGpuArrayObject *img,
                      const PyGpuArrayObject * kern,
                      PyGpuArrayObject * out, size_t subsample_rows,
                      size_t subsample_cols,
                      int version = -1, int verbose=0,
                      int max_threads_dim0 = 512)
{
    int work_complete = 0;
    const int shared_avail = SHARED_SIZE-150;//144 is the biggest static shared size used with compiling this file.
    if (PyGpuArray_NDIM(img) != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required img of 4D");
        return -1;
    }
    if (PyGpuArray_NDIM(kern) != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required kern of 4D");
        return -1;
    }
    if (PyGpuArray_NDIM(out) != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required out of 4D");
        return -1;
    }
    
    if (verbose>1)
    {
        fprintf(stderr,
                "INFO: Running conv_valid version=%d,"
                " MACRO kern_width=%d with inputs:\n",
                version, THEANO_KERN_WID);
        fprintf(stderr,
                "INFO:   img  dim: %llu %llu %llu %llu  "
                "img  stride: %lld %lld %lld %lld\n",
                (unsigned long long)PyGpuArray_DIMS(img)[0],
                (unsigned long long)PyGpuArray_DIMS(img)[1],
                (unsigned long long)PyGpuArray_DIMS(img)[2],
                (unsigned long long)PyGpuArray_DIMS(img)[3],
                (long long)PyGpuArray_STRIDES(img)[0]/4,
                (long long)PyGpuArray_STRIDES(img)[1]/4,
                (long long)PyGpuArray_STRIDES(img)[2]/4,
                (long long)PyGpuArray_STRIDES(img)[3]/4);
        fprintf(stderr,
                "INFO:   kern dim: %llu %llu %llu %llu  "
                "kern stride: %lld %lld %lld %lld\n",
                (unsigned long long)PyGpuArray_DIMS(kern)[0],
                (unsigned long long)PyGpuArray_DIMS(kern)[1],
                (unsigned long long)PyGpuArray_DIMS(kern)[2],
                (unsigned long long)PyGpuArray_DIMS(kern)[3],
                (long long)PyGpuArray_STRIDES(kern)[0]/4,
                (long long)PyGpuArray_STRIDES(kern)[1]/4,
                (long long)PyGpuArray_STRIDES(kern)[2]/4,
                (long long)PyGpuArray_STRIDES(kern)[3]/4);
        fprintf(stderr,
                "INFO:   out dim: %llu %llu %llu %llu  "
                "out stride: %lld %lld %lld %lld\n",
                (unsigned long long)PyGpuArray_DIMS(out)[0],
                (unsigned long long)PyGpuArray_DIMS(out)[1],
                (unsigned long long)PyGpuArray_DIMS(out)[2],
                (unsigned long long)PyGpuArray_DIMS(out)[3],
                (long long)PyGpuArray_STRIDES(out)[0]/4,
                (long long)PyGpuArray_STRIDES(out)[1]/4,
                (long long)PyGpuArray_STRIDES(out)[2]/4,
                (long long)PyGpuArray_STRIDES(out)[3]/4);
        fprintf(stderr,
                "INFO:   subsample_rows=%llu, subsample_cols=%llu\n",
                (unsigned long long)subsample_rows,
                (unsigned long long)subsample_cols);
    }

    //Check the output size is valid
    assert (PyGpuArray_DIMS(out)[2] == ceil_intdiv(PyGpuArray_DIMS(img)[2]- PyGpuArray_DIMS(kern)[2] + 1, subsample_rows));
    assert (PyGpuArray_DIMS(out)[3] == ceil_intdiv(PyGpuArray_DIMS(img)[3]- PyGpuArray_DIMS(kern)[3] + 1, subsample_cols));

    assert (PyGpuArray_DIMS(out)[0] == PyGpuArray_DIMS(img)[0]);
    assert (PyGpuArray_DIMS(out)[1] == PyGpuArray_DIMS(kern)[0]);
    assert (PyGpuArray_DIMS(img)[1] == PyGpuArray_DIMS(kern)[1]);

    // we now search through a few implementations until one applies to our arguments.

    //TODO: make separate version as if all fill this is slower.
    //TODO: Make a switch with power of 2 max size as template
    //TODO: make a parameter the number of division
    //TODO: Should we make them in separate grid block instead?
 
    const int nstack=PyGpuArray_DIMS(kern)[1];
    const int nbatch=PyGpuArray_DIMS(img)[0];
    const int nkern=PyGpuArray_DIMS(kern)[0];
    const int img_wid=PyGpuArray_DIMS(img)[3];
    const int img_len=PyGpuArray_DIMS(img)[2];
    const int kern_wid=PyGpuArray_DIMS(kern)[3];
    const int kern_len=PyGpuArray_DIMS(kern)[2];
    const int out_wid=PyGpuArray_DIMS(out)[3];
    const int out_len=PyGpuArray_DIMS(out)[2];

    const int img_stride_col= PyGpuArray_STRIDES(img)[3]/4;
    const int img_stride_row=PyGpuArray_STRIDES(img)[2]/4;
    const int img_stride_stack= PyGpuArray_STRIDES(img)[1]/4;
    const int img_stride_batch=PyGpuArray_STRIDES(img)[0]/4;
    const int kern_stride_col= PyGpuArray_STRIDES(kern)[3]/4;
    const int kern_stride_row=PyGpuArray_STRIDES(kern)[2]/4;
    const int kern_stride_stack= PyGpuArray_STRIDES(kern)[1]/4;
    const int kern_stride_nkern=PyGpuArray_STRIDES(kern)[0]/4;

    const int img_size=img_len*img_wid;
    const int kern_size=kern_len*kern_wid;
    const int out_size=out_len*out_wid;
    const int img_size_byte = img_size*sizeof(float);
    const int kern_size_byte = kern_size*sizeof(float);
    const int out_size_byte = out_size*sizeof(float);
    if (!((THEANO_KERN_WID == PyGpuArray_DIMS(kern)[3]) || (THEANO_KERN_WID==0))){
     PyErr_Format(PyExc_ValueError, "ERROR: This GpuConv code was compiled for"
                   " %d kernel columns, but the kernel we received had %llud columns!",
                   THEANO_KERN_WID, (unsigned long long)PyGpuArray_DIMS(kern)[3]);
      return -1;
    }

    bool subsample = subsample_rows!=1 || subsample_cols!=1;
    bool img_contiguous = img->ga.flags & GA_C_CONTIGUOUS;
    bool kern_contiguous = kern->ga.flags & GA_C_CONTIGUOUS;
    bool out_contiguous = out->ga.flags & GA_C_CONTIGUOUS;
    bool c_contiguous = img_contiguous &&  kern_contiguous && out_contiguous;

    bool img_contiguous_2d = (img_stride_col == 1) && (img_stride_row==img_wid);
    bool kern_contiguous_2d = (kern_stride_col == 1) && (kern_stride_row==kern_wid);

    //if the lower 2 dims are c_contiguous but flipped, unflipping the
    // stride and not flipping the kernel in shared memroy
    //allow to use a version that use less registers(so is faster)
    //the unflipped version of variable have the original value when
    //we don't need to unflip it, but have the new value when we unflip it.
    bool kern_flipped=true;
    bool kern_contiguous_2d_unflipped = kern_contiguous_2d;
    const float * kern_data_unflipped = cuda_get_ptr(kern);
    int kern_stride_col_unflipped=kern_stride_col;
    int kern_stride_row_unflipped=kern_stride_row;
    if(kern_stride_col_unflipped==-1 && kern_stride_row_unflipped==-kern_wid){
      //the last two dimensions are c_contiguous but flipped!
      kern_stride_col_unflipped=1;
      kern_stride_row_unflipped=kern_wid;
      kern_flipped=false;
      kern_contiguous_2d_unflipped = true;
      kern_data_unflipped=&(cuda_get_ptr(kern)[(kern_wid-1)*kern_stride_col + (kern_len-1)*kern_stride_row]);
    }

    //if we remove the restriction
    //img_size_byte+kern_size_byte>8*1024, we can enter in condition where
    //we will lower the occupency due to shared memory and/or registers.
    if ((version == -1) &&
        (out_size<64 || img_size_byte+kern_size_byte>8*1024) &&
        out_size<=256){
      //condition for exec 
      if(!subsample &&
        out_contiguous &&
        out_size<=max_threads_dim0 &&//Maximum of X threads by block
         std::max(int(img_size_byte+2*kern_wid*sizeof(float)), out_size_byte*2)<shared_avail && //their is only 16k of shared memory and if we can't have the output at least twice in shared mem, we won't have any reduce!
        !work_complete)
        version = 7; //conv_patch_stack_reduce, switch to version 8/13 automatically if needed.
    }

    if (!subsample && c_contiguous &&
        (version==0||version==2||version==-1) &&
        out_wid<=max_threads_dim0 &&//Maximum of X threads for block.x
        nstack == 1 &&// don't implement the stack in the kernel.
        img_size_byte+kern_size_byte<shared_avail && //their is only 16k of shared memory
        !work_complete) //conv_patch
    {
        int nb_split=1;//The number of split (i.e. the number of output pixel each thread compute.)
        if(version==2 && out_len>1)nb_split++;//to force the use of split=true when testing.
        //we pass by ceil_intdiv in case the out_len is not a multiple of nb_split, we want nb_split the number of iteration.
        while (ceil_intdiv(out_len,nb_split)*out_wid>max_threads_dim0)
            nb_split++;
        dim3 threads(out_wid, ceil_intdiv(out_len,nb_split));

        dim3 grid(nbatch, nkern);
        int shared_size=(img_size + kern_size)*sizeof(float);
        void (*f)(const float*, const float*, float*,
                  int, int, int, int,
                  int, int);

#define CONV_PATCH_SPECIAL(kern_wid) \
            if(threads.y==out_len) f=conv_patch<true,kern_wid,false>;\
            else f=conv_patch<true,kern_wid,true>;

        CONV_PATCH_SPECIAL(THEANO_KERN_WID);

         f<<< grid, threads, shared_size>>>
             (cuda_get_ptr(img), cuda_get_ptr(kern), cuda_get_ptr(out),
              img_len, img_wid, kern_len, kern_wid, nkern, nstack);

        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts)
        {
            if (verbose)
              fprintf(stderr,
                      "INFO: used 'conv_patch' version %s nb_split=%d\n",
                      threads.y==out_len ? "no split": "split", nb_split);
            work_complete = true;
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "threads.x=%i, threads.y=%i, grid.x=%i, grid.y=%i,"
                      " shared_size=%i, nb_threads=%i, nb_split=%i\n",
                      threads.x, threads.y, grid.x, grid.y,
                      shared_size, threads.x * threads.y, nb_split);
            if (verbose)
              fprintf(stderr,
                      "INFO: impl 'conv_patch' failed (%s),"
                      " trying next implementation\n",
                      cudaGetErrorString(sts));
        }
    }

    if (out_contiguous &&
        (version==1||version==3||version==11||version==12||version==-1) &&
        (version!=1 || out_size<=max_threads_dim0) &&//Maximum of X threads by block.x
        out_wid<=max_threads_dim0 &&//Maximum of X threads by block.x
        img_size_byte+kern_wid*sizeof(float)<shared_avail && //their is only 16k of shared memory
        !work_complete) //conv_patch_stack
    {
      //version 1 is without split and preload the full kernel
      //version 3 is with split and preload the full kernel
      //version 11 is without split and load only 1 kernel row at a time.
      //version 12 is with split and load only 1 kernel row at a time.
        int nb_split=1;//The number of split (i.e. the number of output pixel each thread compute.)
        if((version==3||version==12) && out_len>1)nb_split++;//to force the use of split=true when testing.
        //we pass by ceil_intdiv in case the out_len is not a multiple of nb_split, we want nb_split the number of iteration.
        while (ceil_intdiv(out_len,nb_split)*out_wid>max_threads_dim0) nb_split++;
        dim3 threads(out_wid, ceil_intdiv(out_len,nb_split));

        bool preload_full_kernel = (img_size_byte + kern_size_byte) <shared_avail;
        if(version==11 || version==12) preload_full_kernel=false;
        dim3 grid(nbatch,nkern);
        int shared_size=(img_size + (preload_full_kernel?kern_size:kern_wid))*sizeof(float);

        void (*f)(const float*, const float*, float*,
                  int, int, int, int,
                  int, int, int, int,
                  int, int, int, int,
                  int, int, int, int,
                  int, int);

#define CONV_PATCH_STACK_SPECIAL(kern_wid) \
        if(preload_full_kernel && nb_split==1 && img_contiguous_2d && kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,true,true,false,true,true>;} \
        else if(preload_full_kernel && nb_split==1 && img_contiguous_2d && !kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,true,false,false,true,true>;} \
        else if(preload_full_kernel && nb_split==1 && !img_contiguous_2d && kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,false,true,false,true,true>;}\
        else if(preload_full_kernel && nb_split==1 && !img_contiguous_2d && !kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,false,false,false,true,true>;}\
        else if(preload_full_kernel && nb_split!=1 && img_contiguous_2d && kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,true,true,true,true,true>;}\
        else if(preload_full_kernel && nb_split!=1 && img_contiguous_2d && !kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,true,false,true,true,true>;}\
        else if(preload_full_kernel && nb_split!=1 && !img_contiguous_2d && kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,false,true,true,true,true>;}\
        else if(preload_full_kernel && nb_split!=1 && !img_contiguous_2d && !kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,false,false,true,true,true>;}\
        else if(!preload_full_kernel && nb_split==1 && img_contiguous_2d && kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,true,true,false,false,true>;}\
        else if(!preload_full_kernel && nb_split==1 && img_contiguous_2d && !kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,true,false,false,false,true>;}\
        else if(!preload_full_kernel && nb_split==1 && !img_contiguous_2d && kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,false,true,false,false,true>;}\
        else if(!preload_full_kernel && nb_split==1 && !img_contiguous_2d && !kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,false,false,false,false,true>;}\
        else if(!preload_full_kernel && nb_split!=1 && img_contiguous_2d && kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,true,true,true,false,true>;} \
        else if(!preload_full_kernel && nb_split!=1 && img_contiguous_2d && !kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,true,false,true,false,true>;} \
        else if(!preload_full_kernel && nb_split!=1 && !img_contiguous_2d && kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,false,true,true,false,true>;} \
        else if(!preload_full_kernel && nb_split!=1 && !img_contiguous_2d && !kern_contiguous_2d && subsample){ f=conv_patch_stack<true,false,kern_wid,false,false,true,false,true>;} \
        else if(preload_full_kernel && nb_split==1 && img_contiguous_2d && kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,true,true,false,true,false>;} \
        else if(preload_full_kernel && nb_split==1 && img_contiguous_2d && !kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,true,false,false,true,false>;} \
        else if(preload_full_kernel && nb_split==1 && !img_contiguous_2d && kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,false,true,false,true,false>;}\
        else if(preload_full_kernel && nb_split==1 && !img_contiguous_2d && !kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,false,false,false,true,false>;}\
        else if(preload_full_kernel && nb_split!=1 && img_contiguous_2d && kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,true,true,true,true,false>;}\
        else if(preload_full_kernel && nb_split!=1 && img_contiguous_2d && !kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,true,false,true,true,false>;}\
        else if(preload_full_kernel && nb_split!=1 && !img_contiguous_2d && kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,false,true,true,true,false>;}\
        else if(preload_full_kernel && nb_split!=1 && !img_contiguous_2d && !kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,false,false,true,true,false>;}\
        else if(!preload_full_kernel && nb_split==1 && img_contiguous_2d && kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,true,true,false,false,false>;}\
        else if(!preload_full_kernel && nb_split==1 && img_contiguous_2d && !kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,true,false,false,false,false>;}\
        else if(!preload_full_kernel && nb_split==1 && !img_contiguous_2d && kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,false,true,false,false,false>;}\
        else if(!preload_full_kernel && nb_split==1 && !img_contiguous_2d && !kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,false,false,false,false,false>;}\
        else if(!preload_full_kernel && nb_split!=1 && img_contiguous_2d && kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,true,true,true,false,false>;} \
        else if(!preload_full_kernel && nb_split!=1 && img_contiguous_2d && !kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,true,false,true,false,false>;} \
        else if(!preload_full_kernel && nb_split!=1 && !img_contiguous_2d && kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,false,true,true,false,false>;} \
        else if(!preload_full_kernel && nb_split!=1 && !img_contiguous_2d && !kern_contiguous_2d && !subsample){ f=conv_patch_stack<true,false,kern_wid,false,false,true,false,false>;}

        CONV_PATCH_STACK_SPECIAL(THEANO_KERN_WID);
        f<<< grid, threads, shared_size>>>
            (cuda_get_ptr(img), cuda_get_ptr(kern), cuda_get_ptr(out),
              img_len, img_wid, kern_len, kern_wid, 
              out_len, out_wid, nkern, nstack,
              img_stride_col, img_stride_row, img_stride_stack,
              img_stride_batch, kern_stride_col, kern_stride_row,
              kern_stride_stack, kern_stride_nkern, subsample_rows, subsample_cols);

        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts)
        {
            if (verbose>1)
              fprintf(stderr,
                      "threads.x=%i, threads.y=%i, grid.x=%i, grid.y=%i,"
                      " shared_size=%i, nb_threads=%i,"
                      " kern_flipped=true, accumulate=false, kern_width=%i,"
                      " img_c_contiguous_2d=%i,"
                      " kern_c_contiguous_2d=%i, nb_split=%i,"
                      " preload_full_kernel=%i,"
                      " subsample_rows=%llu, subsample_cols=%llu\n",
                      threads.x, threads.y, grid.x, grid.y,
                      shared_size, threads.x * threads.y,
                      THEANO_KERN_WID, img_contiguous_2d, kern_contiguous_2d,
                      nb_split, preload_full_kernel,
                      (unsigned long long)subsample_rows,
                      (unsigned long long)subsample_cols);
            if (verbose)
              fprintf(stderr,
                      "INFO: used 'conv_patch_stack' version with nb_split=%i"
                      " and preload_full_kernel=%i,"
                      " subsample_rows=%llu, subsample_cols=%llu\n",
                      nb_split, preload_full_kernel,
                      (unsigned long long)subsample_rows,
                      (unsigned long long)subsample_cols);
            work_complete = true;
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "threads.x=%i, threads.y=%i, grid.x=%i, grid.y=%i,"
                      " shared_size=%i, nb_threads=%i,"
                      " kern_flipped=true, accumulate=false,"
                      " kern_width=%i, img_c_contiguous_2d=%i,"
                      " kern_c_contiguous_2d=%i, nb_split=%i,"
                      " preload_full_kernel=%i,"
                      " subsample_rows=%llu, subsample_cols=%llu\n",
                      threads.x, threads.y, grid.x, grid.y,
                      shared_size, threads.x * threads.y,
                      THEANO_KERN_WID, img_contiguous_2d, kern_contiguous_2d,
                      nb_split, preload_full_kernel,
                      (unsigned long long)subsample_rows,
                      (unsigned long long)subsample_cols);
            if (verbose)
              fprintf(stderr,
                      "INFO: impl 'conv_patch_stack' failed (%s),"
                      " trying next implementation\n",
                      cudaGetErrorString(sts));
        }
    }

    if (!subsample && out_contiguous &&
        (version==4||version==-1) &&
        out_wid<=max_threads_dim0 &&//Maximum of X threads by block.x
        nstack == 1 &&// don't implement the stack in the kernel.
        kern_len*img_wid*sizeof(float)+kern_size_byte<shared_avail &&//their is only 16k of shared memory
        !work_complete) //conv_rows

    {
        dim3 threads(out_wid);
        dim3 grid(out_len, nbatch*nkern);
        int shared_size=(kern_len*img_wid + kern_size)*sizeof(float);
        void (*f)(const float*, const float*, float*,
                  int, int, int, int,
                  int, int, int, int,
                  int, int, int, int,
                  int, int);

#define CONV_ROWS_SPECIAL(kern_wid) \
        if(!img_contiguous_2d || !kern_contiguous_2d) f = conv_rows<kern_wid, false>;\
        else f = conv_rows<kern_wid, true>;\

        CONV_ROWS_SPECIAL(THEANO_KERN_WID);
        f<<< grid, threads, shared_size >>>
            (cuda_get_ptr(img), cuda_get_ptr(kern), cuda_get_ptr(out),
           img_len, img_wid, kern_len, kern_wid, nkern, nstack,
           img_stride_col, img_stride_row,
           img_stride_stack,img_stride_batch,
           kern_stride_col, kern_stride_row,
           kern_stride_stack, kern_stride_nkern);

        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts)
        {
            work_complete = true;
            if (verbose)
              fprintf(stderr, "INFO: used 'conv_rows' version\n");
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "threads.x=%i, threads.y=%i, grid.x=%i, grid.y=%i,"
                      " shared_size=%i, nb_threads=%i\n",
                      threads.x, threads.y, grid.x, grid.y,
                      shared_size, threads.x * threads.y);
            if (verbose)
              fprintf(stderr,
                      "INFO: impl 'conv_rows' failed (%s),"
                      " trying next implementation\n",
                      cudaGetErrorString(sts));
        }
    }
    if (!subsample && out_contiguous &&
        (version==5||version==-1) &&
        out_wid<=max_threads_dim0 &&//Maximum of X threads by block.x
        img_wid*kern_len*sizeof(float)+kern_size_byte<shared_avail && //their is only 16k of shared memory
        !work_complete) //conv_rows_stack

    {
        int nb_row=1;
        //TODO:if not c_contiguous, lower max_thread as we use 22
        //registers by thread and we won't execute 2 block in one MP.
        for(int i=2;i<=out_len;i++){
          if((i)*out_wid<=max_threads_dim0 && ((kern_len+i)*img_wid + kern_size)*sizeof(float)<shared_avail)
            nb_row=i;
        }

        dim3 threads(out_wid,nb_row);
        dim3 grid(ceil_intdiv(out_len,nb_row), nbatch*nkern);

        int shared_size=((kern_len+nb_row-1)*img_wid + kern_size)*sizeof(float);

        void (*f)(const float*, const float*, float*,
                  int, int, int, int,
                  int, int, int, int,
                  int, int, int, int,
                  int, int);

        if (0)
          fprintf(stderr,
                  "IMG CONTIG %i KERN_CONTIG %i (%i %i %i) (%i %i %i)\n",
                  img_contiguous_2d, kern_contiguous_2d,
                  threads.x, threads.y, threads.z,
                  grid.x, grid.y, grid.z);

        if(!img_contiguous_2d || !kern_contiguous_2d) {
            //fprintf(stderr, "using false version\n");
            f = conv_rows_stack<THEANO_KERN_WID, false>;
        } else {
            //fprintf(stderr, "using true version\n");
            f = conv_rows_stack<THEANO_KERN_WID, true>;
        }

        f<<< grid, threads, shared_size >>>
            (cuda_get_ptr(img),
             cuda_get_ptr(kern),
             cuda_get_ptr(out),
           img_len, img_wid, kern_len, kern_wid, nkern, nstack,
           img_stride_col, img_stride_row,
           img_stride_stack,img_stride_batch,
           kern_stride_col, kern_stride_row,
           kern_stride_stack, kern_stride_nkern);

        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts)
        {
            work_complete = true;
            if (verbose>1)
              fprintf(stderr,
                      "threads.x=%i, threads.y=%i, grid.x=%i, grid.y=%i,"
                      " shared_size=%i, nb_threads=%i\n",
                      threads.x, threads.y, grid.x, grid.y,
                      shared_size, threads.x * threads.y);
            if (verbose)
              fprintf(stderr, "INFO: used 'conv_rows_stack' version\n");
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "threads.x=%i, threads.y=%i, grid.x=%i, grid.y=%i,"
                      " shared_size=%i, nb_threads=%i\n",
                      threads.x, threads.y, grid.x, grid.y,
                      shared_size, threads.x * threads.y);
            if (verbose)
              fprintf(stderr,
                      "INFO: impl 'conv_rows_stack' failed (%s),"
                      " trying next implementation\n",
                      cudaGetErrorString(sts));
        }
    }

    if (!subsample && out_contiguous &&
        (version==9||version==10||version==-1) &&
        out_wid<=max_threads_dim0 &&//Maximum of X threads by block.x
        (img_wid+kern_wid)*sizeof(float)<shared_avail && //their is only 16k of shared memory
        (version != 9 || (img_wid+kern_len*kern_wid)*sizeof(float)<shared_avail) && //version 9 use more memory
        !work_complete) //conv_rows_stack2

    {
      // version 9:we preload the full kernel
      // version 10: load only a few row at a time.
        int nb_row=1;
        int version_back = version;
        //TODO:if not c_contiguous, lower max_thread as we use 22 registers by thread and we won't execute 2 block in one MP.
        if(version==-1 && (img_wid+kern_len*kern_wid)*sizeof(float)<shared_avail)
          version = 9;
        else if(version==-1)version = 10;

        int k_size = kern_size;
        if(version==10)
          k_size=kern_wid;

        for(int i=2;i<=out_len;i++){
          if(i*out_wid<=max_threads_dim0 && (i*img_wid + k_size)*sizeof(float)<shared_avail)
            nb_row=i;
        }

        //to test the case when we don't have a thread by output pixel.
        if((version_back!=-1)&& nb_row>1) nb_row--;

        dim3 threads(out_wid,nb_row);
        dim3 grid(ceil_intdiv(out_len,nb_row), nbatch*nkern);
          
        int shared_size=(threads.y*img_wid + k_size)*sizeof(float);

        void (*f)(const float*, const float*, float*,
                  int, int, int, int,
                  int, int, int, int,
                  int, int, int, int,
                  int, int);

#define CONV_ROWS_STACK2_SPECIAL(kern_wid) \
        if((!img_contiguous_2d || !kern_contiguous_2d)&&version==9) f = conv_rows_stack2<kern_wid, false,true>;\
        else if(version==9) f = conv_rows_stack2<kern_wid, true,true>;\
        else if(!img_contiguous_2d || !kern_contiguous_2d) f = conv_rows_stack2<kern_wid, false, false>;\
        else f = conv_rows_stack2<kern_wid, true, false>;

        CONV_ROWS_STACK2_SPECIAL(THEANO_KERN_WID);

        f<<< grid, threads, shared_size >>>
            (cuda_get_ptr(img),
             cuda_get_ptr(kern),
             cuda_get_ptr(out),
           img_len, img_wid, kern_len, kern_wid, nkern, nstack,
           img_stride_col, img_stride_row,
           img_stride_stack,img_stride_batch,
           kern_stride_col, kern_stride_row,
           kern_stride_stack, kern_stride_nkern);

        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts) 
        {
            work_complete = true;
            if (verbose>1)
              fprintf(stderr,
                      "threads.x=%i, threads.y=%i, grid.x=%i, grid.y=%i,"
                      " shared_size=%i, nb_threads=%i\n",
                      threads.x, threads.y, grid.x, grid.y,
                      shared_size, threads.x * threads.y);
            if (verbose)
              fprintf(stderr,
                      "INFO: used 'conv_rows_stack2' version %s with"
                      " %d row(s).\n",
                      (version==9?"'load full kernel'":
                       "'load 1 kern row at a time'"),nb_row);
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "threads.x=%i, threads.y=%i, grid.x=%i, grid.y=%i,"
                      " shared_size=%i, nb_threads=%i version=%d\n",
                      threads.x, threads.y, grid.x, grid.y,
                      shared_size, threads.x * threads.y,(version==9?2:3));
            if (verbose)
              fprintf(stderr,
                      "INFO: impl 'conv_rows_stack2' failed (%s),"
                      " trying next implementation\n",
                      cudaGetErrorString(sts));
        }
    }

    //version 8 is the same but we force the split.
    // The split is need in case we have too much threads.
    // This happen frequently if the kernel length is big.
    // Big kernel is frequent in the gradient.
    //version 8 need a minimum of kernel length as we force the split.
    //version 8 is needed to test more easily this kernel template parameter.
    //version 13 load only 1 kernel row at a time.
    if (!subsample &&
        out_contiguous &&
        out_size<=max_threads_dim0 &&//Maximum of X threads by block
        (version==7||version==8||version==13||version==-1) &&
        (version!=8||kern_len>1) && //version 8 need a minimal kernel length as big as the split.
        //version 13 need a minimal kernel length as big as the split.
        (version!=13||kern_len>1) &&
        !work_complete) //conv_patch_stack_reduce
    {
        int nb_split=1;
        int full_kern=true;

        if(version==8||version==13) nb_split++;//force the split.
        if(version==13)full_kern=false;

        //check if we can fit the full kernel in the shared memory
        if(sizeof(float)*std::max(img_size + kern_size, out_size*2) > shared_avail){
          full_kern = false;
        }

        //thread_z is going to be ceil_intdiv(kern_len, nb_split)
        // we need enough splits so that
        // a) thread_z fits in the 'z' threadIdx (i.e. is less than 64)
        // b) thread_z * out_len * out_wid fits in the thread count
        // c) the kernel doesn't need too much shared memory

        // constraint (a)
        // device 1.3 have a max of 64 thread in z
        while(ceil_intdiv(kern_len,nb_split)>64) nb_split++;

        // constraint (b)
        //  (TODO: read the number of threads per block from the device)
        while(out_size*ceil_intdiv(kern_len,nb_split)>max_threads_dim0)
            nb_split++;

        // tentative estimates (prior to contraint c)
        int thread_z=ceil_intdiv(kern_len,nb_split);
        int shared_size = sizeof(float)*(full_kern
                ? std::max(img_size + kern_size, out_size*thread_z)
                : std::max(img_size + thread_z*kern_wid, out_size*thread_z));

        // constraint (c)
        while ((shared_size >= shared_avail) && (nb_split <= kern_len)){
            //if we can't fit the kernel in shared memory, we must split it more.
            nb_split++;
            thread_z=ceil_intdiv(kern_len,nb_split);
            shared_size = sizeof(float)*(full_kern
                ? std::max(img_size + kern_size, out_size*thread_z)
                : std::max(img_size + thread_z*kern_wid, out_size*thread_z));
        }
        if (nb_split <= kern_len)
        {
            assert(thread_z>0);//should not happen, but in case...
            if(!full_kern) assert(thread_z!=kern_len);

            dim3 threads(out_wid, out_len, thread_z);
            dim3 grid(nbatch,nkern);

            void (*f)(const float*, const float*, float*,
                      int, int, int, int,
                      int, int, int, int,
                      int, int,
                      int, int,
                      int, int);

            const bool split=thread_z!=kern_len;
            const bool ccontig=img_contiguous_2d && kern_contiguous_2d_unflipped;

            //printf("kern_flipped=%d, ccontig=%d, split=%d, full_kern=%d\n",kern_flipped,ccontig,split,full_kern);
            //We will always be split when we don't load the full kernel
#define CONV_PATCH_STACK_REDUCE_SPECIAL(kern_wid) \
                if     (kern_flipped  && ccontig  && !split && full_kern) f=conv_patch_stack_reduce<true,kern_wid,true, false, true>;\
                else if(kern_flipped  && !ccontig && !split && full_kern) f=conv_patch_stack_reduce<true,kern_wid,false, false, true>;\
                else if(kern_flipped  && ccontig  && split && full_kern) f=conv_patch_stack_reduce<true,kern_wid,true, true, true>;\
                else if(kern_flipped  && !ccontig && split && full_kern) f=conv_patch_stack_reduce<true,kern_wid,false, true, true>;\
                else if(!kern_flipped && ccontig  && !split && full_kern) f=conv_patch_stack_reduce<false,kern_wid,true, false, true>;\
                else if(!kern_flipped && !ccontig && !split && full_kern) f=conv_patch_stack_reduce<false,kern_wid,false, false, true>;\
                else if(!kern_flipped && ccontig  && split && full_kern) f=conv_patch_stack_reduce<false,kern_wid,true, true, true>;\
                else if(!kern_flipped && !ccontig  && split && full_kern) f=conv_patch_stack_reduce<false,kern_wid,false, true, true>;\
                /*else if(kern_flipped  && ccontig  && !split && !full_kern) f=conv_patch_stack_reduce<true,kern_wid,true, false, false>;*/\
                /*else if(kern_flipped  && !ccontig && !split && !full_kern) f=conv_patch_stack_reduce<true,kern_wid,false, false, false>;*/\
                else if(kern_flipped  && ccontig  && split && !full_kern) f=conv_patch_stack_reduce<true,kern_wid,true, true, false>;\
                else if(kern_flipped  && !ccontig && split && !full_kern) f=conv_patch_stack_reduce<true,kern_wid,false, true, false>;\
                /*else if(!kern_flipped && ccontig  && !split && !full_kern) f=conv_patch_stack_reduce<false,kern_wid,true, false, false>;*/\
                /*else if(!kern_flipped && !ccontig && !split && !full_kern) f=conv_patch_stack_reduce<false,kern_wid,false, false, false>;*/\
                else if(!kern_flipped && ccontig  && split && !full_kern) f=conv_patch_stack_reduce<false,kern_wid,true, true, false>;\
                else if(!kern_flipped && !ccontig  && split && !full_kern) f=conv_patch_stack_reduce<false,kern_wid,false, true, false>;
            CONV_PATCH_STACK_REDUCE_SPECIAL(THEANO_KERN_WID);

            f<<< grid, threads, shared_size>>>(cuda_get_ptr(img), kern_data_unflipped, cuda_get_ptr(out),
                                               img_len, img_wid, kern_len, kern_wid,
                                               nkern, nstack,
                                               img_stride_col, img_stride_row, img_stride_stack, img_stride_batch,
                                               kern_stride_col_unflipped, kern_stride_row_unflipped,
                                               kern_stride_stack, kern_stride_nkern);

            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess == sts)
            {
                if (verbose>1)
                    fprintf(stderr,
                            "threads.x=%i, threads.y=%i, threads.z=%i, "
                            "grid.x=%i, grid.y=%i, shared_size=%i,"
                            " nb_threads=%i\n",
                            threads.x, threads.y, threads.z, grid.x, grid.y,
                            shared_size, threads.x * threads.y * threads.z);
                if (verbose)
                    fprintf(stderr,
                            "INFO: used 'conv_patch_stack_reduce' version"
                            " kern_flipped=%i ccontig=%i nb_split=%d,"
                            " preload_full_kern=%d\n",
                            kern_flipped, ccontig, nb_split, full_kern);
                work_complete = true;
            }
            else
            {
                if (verbose)
                  fprintf(stderr,
                          "threads.x=%i, threads.y=%i, threads.z=%i,"
                          " grid.x=%i, grid.y=%i,shared_size=%i,"
                          " nb_threads=%i\n",
                          threads.x, threads.y, threads.z,
                          grid.x, grid.y, shared_size,
                          threads.x * threads.y * threads.z);
                if (verbose)
                  fprintf(stderr,
                          "INFO: impl 'conv_patch_stack_reduce' failed (%s),"
                          " trying next implementation\n",
                          cudaGetErrorString(sts));
            }
        } // else no good nb_splits was found
    }

    if (1 && (version==6||version==-1) &&
        kern_len<=320 &&
        !work_complete) //conv_valid_row_reduce
    {
        int outsize = PyGpuArray_SIZE(out);
        int n_blocks = std::min(outsize, 4096);

        int block_nstack=nstack;
        //Max of 512 threads per blocks.
        //On old hardware, we have a max of 356 threads as we have only 
        //8k registers and the kernel use 23 register
        //TODO: check if we have 8k or 16k of register...
        while(block_nstack*kern_len>320)block_nstack--;
        dim3 n_threads(block_nstack, kern_len, 1);

        int n_reduce_buf = block_nstack * kern_len * sizeof(float);
        /* initial_reduce_boundary is the greatest power of two less than n_reduce_buf/ sizeof(float)
         *
         * if n_reduce_buf == sizeof(float), then initial_reduce_boundary == 0.
         * */
        int initial_reduce_boundary = (1 << (int)(log2((double)(n_reduce_buf/sizeof(float)))));
        if (initial_reduce_boundary == (n_reduce_buf / sizeof(float)))
            initial_reduce_boundary >>= 1;

        if (n_reduce_buf == sizeof(float))
            assert (initial_reduce_boundary == 0);
        else
        {
            assert (initial_reduce_boundary * 2 >= n_reduce_buf/sizeof(float));
            assert (initial_reduce_boundary < n_reduce_buf/sizeof(float));
        }


        void (*f)(int, int, int, int,
                  int, int, int, int, int,
                  const float*, int, int, int, int,
                  const float*, int, int, int, int,
                  float*, int, int, int, int,
                  int, int, int);

        //std::cerr << "initial_reduce_boundary " << initial_reduce_boundary << "\n";
        //std::cerr << "kerns " << nstack << " " << kern_len << "\n";
        //std::cerr << "n_reduce_buf/sizeof(float) " << n_reduce_buf / sizeof(float) << "\n";
        if(block_nstack==nstack)
          f=conv_valid_row_reduce<false>;
        else
          f=conv_valid_row_reduce<true>;
        f<<<n_blocks, n_threads, n_reduce_buf>>>(
                nbatch, nkern, PyGpuArray_DIMS(img)[1],
                img_len, img_wid,
                kern_len, kern_wid,
                out_len, out_wid,
                cuda_get_ptr(img),
                PyGpuArray_STRIDES(img)[0]/4, PyGpuArray_STRIDES(img)[1]/4, 
                img_stride_row, img_stride_col,
                cuda_get_ptr(kern),
                PyGpuArray_STRIDES(kern)[0]/4, PyGpuArray_STRIDES(kern)[1]/4,
                PyGpuArray_STRIDES(kern)[2]/4, PyGpuArray_STRIDES(kern)[3]/4,
                cuda_get_ptr(out),
                PyGpuArray_STRIDES(out)[0]/4, PyGpuArray_STRIDES(out)[1]/4,
                PyGpuArray_STRIDES(out)[2]/4, PyGpuArray_STRIDES(out)[3]/4,
                subsample_rows, subsample_cols, initial_reduce_boundary);

        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts) 
        {
            work_complete = true;
            if (verbose)
              fprintf(stderr, "INFO: used 'conv_valid_row_reduce' version\n");
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "threads.x=%i, threads.y=%i, grid.x=%i,"
                      " shared_size=%i, nb_threads=%i\n",
                      n_threads.x, n_threads.y, n_blocks,
                      n_reduce_buf, n_threads.x * n_threads.y);
            if (verbose)
              fprintf(stderr,
                      "INFO: impl 'conv_valid_row_reduce' failed (%s),"
                      " trying next implementation\n",
                      cudaGetErrorString(sts));
        }
    }

    if (1 && !work_complete) //conv_reference_valid
    {
        int outsize = PyGpuArray_SIZE(out);
        int n_blocks = std::min(outsize, 4096);
        int n_threads = std::min(ceil_intdiv(outsize, n_blocks),
                                 256);
        if (1)
        {
            if (verbose)
              fprintf(stderr, "INFO: launching conv_reference_valid\n");
            if (verbose>1)
              fprintf(stderr, "      img : %i %llu %i %i %p  "
                      "%lld %lld %lld %lld\n",
                      nbatch, (unsigned long long)PyGpuArray_DIMS(img)[1],
                      img_len, img_wid,
                      cuda_get_ptr(img),
                      (long long)PyGpuArray_STRIDES(img)[0]/4,
                      (long long)PyGpuArray_STRIDES(img)[1]/4,
                      (long long)PyGpuArray_STRIDES(img)[2]/4,
                      (long long)PyGpuArray_STRIDES(img)[3]/4);
            if (verbose>1)
              fprintf(stderr, "      kern: %i %i %i %i %p  "
                      "%lld %lld %lld %lld\n",
                      nkern, nstack, kern_len, kern_wid,
                      cuda_get_ptr(kern),
                      (long long)PyGpuArray_STRIDES(kern)[0]/4,
                      (long long)PyGpuArray_STRIDES(kern)[1]/4,
                      (long long)PyGpuArray_STRIDES(kern)[2]/4,
                      (long long)PyGpuArray_STRIDES(kern)[3]/4);
            if (verbose>1)
                fprintf(stderr, "      out : %llu %llu %i %i %p  "
                        "%lld %lld %lld %lld\n",
                      (unsigned long long)PyGpuArray_DIMS(out)[0],
                      (unsigned long long)PyGpuArray_DIMS(out)[1],
                      out_len, out_wid,
                      cuda_get_ptr(out),
                      (long long)PyGpuArray_STRIDES(out)[0]/4,
                      (long long)PyGpuArray_STRIDES(out)[1]/4,
                      (long long)PyGpuArray_STRIDES(out)[2]/4,
                      (long long)PyGpuArray_STRIDES(out)[3]/4);
            if (verbose>1)
              fprintf(stderr, "   launch params: %i %i %i\n",
                      outsize, n_blocks, n_threads);
        }
        conv_reference_valid<<<n_blocks, n_threads>>>(nbatch, nkern,
                PyGpuArray_DIMS(img)[1],
                img_len, img_wid,
                kern_len, kern_wid,
                out_len, out_wid,
                cuda_get_ptr(img),
                PyGpuArray_STRIDES(img)[0]/4,
                PyGpuArray_STRIDES(img)[1]/4,
                PyGpuArray_STRIDES(img)[2]/4,
                PyGpuArray_STRIDES(img)[3]/4,
                cuda_get_ptr(kern),
                PyGpuArray_STRIDES(kern)[0]/4,
                PyGpuArray_STRIDES(kern)[1]/4,
                PyGpuArray_STRIDES(kern)[2]/4,
                PyGpuArray_STRIDES(kern)[3]/4,
                cuda_get_ptr(out),
                PyGpuArray_STRIDES(out)[0]/4,
                PyGpuArray_STRIDES(out)[1]/4,
                PyGpuArray_STRIDES(out)[2]/4,
                PyGpuArray_STRIDES(out)[3]/4,
                subsample_rows, subsample_cols);

        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts)
        {
            work_complete = true;
            if (verbose)
              fprintf(stderr, "INFO: used 'conv_reference_valid' version\n");
        }
        else
        {
            if (verbose)
              fprintf(stderr, "INFO: 'conv_reference_valid' failed\n");
            PyErr_Format(PyExc_RuntimeError,
                         "ERROR: all implementations failed for"
                         " PyGpuArray_conv_valid! (%s)",
                         cudaGetErrorString(sts));
            return -1;
        }
    }
    if (!work_complete)
    {
      PyErr_Format(PyExc_RuntimeError,
                   "ERROR: no implementation(s) worked for"
                   " PyGpuArray_conv_valid!"
                   " Version asked(%d) (-1 mean use an heuristic)",
                   version);
        return -1;
    }
    return 0;
}

int
PyGpuArray_conv_full(const PyGpuArrayObject *img, const PyGpuArrayObject * kern,
                      PyGpuArrayObject * out, size_t subsample_rows,
                      size_t subsample_cols, int version = -1, int verbose=0,
                      int max_threads_dim0=512)
{
  //144 is the biggest static shared size used with compiling this file.
    const int shared_avail = SHARED_SIZE - 150;

    int work_complete = 0;
    if (PyGpuArray_NDIM(img) != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required img of 4D");
        return -1;
    }
    if (PyGpuArray_NDIM(kern) != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required kern of 4D");
        return -1;
    }
    if (PyGpuArray_NDIM(out) != 4)
    {
        PyErr_SetString(PyExc_ValueError, "required out of 4D");
        return -1;
    }
    // check the size of the output matrix
    assert (PyGpuArray_DIMS(out)[2] == ceil_intdiv(PyGpuArray_DIMS(img)[2] + PyGpuArray_DIMS(kern)[2] - 1, subsample_rows));
    assert (PyGpuArray_DIMS(out)[3] == ceil_intdiv(PyGpuArray_DIMS(img)[3] + PyGpuArray_DIMS(kern)[3] - 1, subsample_cols));

    assert (PyGpuArray_DIMS(out)[0] == PyGpuArray_DIMS(img)[0]);
    assert (PyGpuArray_DIMS(out)[1] == PyGpuArray_DIMS(kern)[0]);
    assert (PyGpuArray_DIMS(img)[1] == PyGpuArray_DIMS(kern)[1]);

    const int nstack=PyGpuArray_DIMS(kern)[1];
    const int nbatch=PyGpuArray_DIMS(img)[0];
    const int nkern=PyGpuArray_DIMS(kern)[0];
    const int img_wid=PyGpuArray_DIMS(img)[3];
    const int img_len=PyGpuArray_DIMS(img)[2];
    const int kern_wid=PyGpuArray_DIMS(kern)[3];
    const int kern_len=PyGpuArray_DIMS(kern)[2];
    const int out_wid=PyGpuArray_DIMS(out)[3];
    const int out_len=PyGpuArray_DIMS(out)[2];

    const int img_stride_col= PyGpuArray_STRIDES(img)[3]/4;
    const int img_stride_row=PyGpuArray_STRIDES(img)[2]/4;
    const int img_stride_stack=PyGpuArray_STRIDES(img)[1]/4;
    const int img_stride_batch=PyGpuArray_STRIDES(img)[0]/4;
    const int kern_stride_col= PyGpuArray_STRIDES(kern)[3]/4;
    const int kern_stride_row=PyGpuArray_STRIDES(kern)[2]/4;
    const int kern_stride_stack= PyGpuArray_STRIDES(kern)[1]/4;
    const int kern_stride_nkern=PyGpuArray_STRIDES(kern)[0]/4;

    const int img_size=img_len*img_wid;
    const int kern_size=kern_len*kern_wid;
    const int out_size=out_len*out_wid;
    const int img_size_byte = img_size*sizeof(float);
    const int kern_size_byte = kern_size*sizeof(float);
    //padded image sizes
    const int img_wid_padded=img_wid+2*kern_wid-2;
    const int img_len_padded=img_len+2*kern_len-2;
    const int img_size_padded=img_len_padded * img_wid_padded;
    const int img_size_padded_byte = img_size_padded*sizeof(float);
    
    //const int out_size_byte = out_size*sizeof(float); // unused 

    if (!((THEANO_KERN_WID == PyGpuArray_DIMS(kern)[3]) ||
          (THEANO_KERN_WID == 0))){
      PyErr_Format(PyExc_ValueError,
                   "ERROR: This GpuConv code was compiled for"
                   " %d kernel columns, but the kernel we received"
                   " had %llud columns!",
                   THEANO_KERN_WID, (unsigned long long)PyGpuArray_DIMS(kern)[3]);
      return -1;
    }
    bool subsample = subsample_rows!=1 || subsample_cols!=1;

    bool img_contiguous = img->ga.flags & GA_C_CONTIGUOUS;
    bool kern_contiguous = kern->ga.flags & GA_C_CONTIGUOUS;
    bool out_contiguous = out->ga.flags & GA_C_CONTIGUOUS;
    bool c_contiguous = img_contiguous &&  kern_contiguous && out_contiguous;

    bool img_contiguous_2d = (img_stride_col == 1) && (img_stride_row==img_wid);
    bool kern_contiguous_2d = (kern_stride_col == 1) && (kern_stride_row==kern_wid);

    bool img_batch_stack_contiguous = (img_stride_stack==img_stride_row*img_len) && (img_stride_batch==img_stride_stack*nstack);//don't support stride for nbatch and nstack

    //if the lower 2 dims are c_contiguous but flipped, unflipping the
    //stride and not flipping the kernel in shared memroy
    //allow to use a version that use less registers(so is faster)
    //the unflipped version of variable have the original value when
    //we don't need to unflip it, but have the new value when we unflip it.
    bool kern_flipped=true;
    bool kern_contiguous_2d_unflipped = kern_contiguous_2d;
    const float * kern_data_unflipped = cuda_get_ptr(kern);
    int kern_stride_col_unflipped=kern_stride_col;
    int kern_stride_row_unflipped=kern_stride_row;
    if(kern_stride_col_unflipped==-1 && kern_stride_row_unflipped==-kern_wid){
      //the last two dimensions are c_contiguous but flipped!
      kern_stride_col_unflipped=1;
      kern_stride_row_unflipped=kern_wid;
      kern_flipped=false;
      kern_contiguous_2d_unflipped = true;
      kern_data_unflipped=&(cuda_get_ptr(kern)[(kern_wid-1)*kern_stride_col + (kern_len-1)*kern_stride_row]);
    }

    if (verbose>1)
    {
        printf("INFO: Running conv_full version=%d,"
               " MACRO kern_width=%d with inputs:\n", version, THEANO_KERN_WID);
        printf("INFO:   img  dim: %llu %llu %llu %llu  "
               "img  stride: %lld %lld %lld %lld\n",
               (unsigned long long)PyGpuArray_DIMS(img)[0],
               (unsigned long long)PyGpuArray_DIMS(img)[1],
               (unsigned long long)PyGpuArray_DIMS(img)[2],
               (unsigned long long)PyGpuArray_DIMS(img)[3],
               (long long)PyGpuArray_STRIDES(img)[0]/4,
               (long long)PyGpuArray_STRIDES(img)[1]/4,
               (long long)PyGpuArray_STRIDES(img)[2]/4,
               (long long)PyGpuArray_STRIDES(img)[3]/4);
        printf("INFO:   kern dim: %llu %llu %llu %llu  "
               "kern stride: %lld %lld %lld %lld\n",
               (unsigned long long)PyGpuArray_DIMS(kern)[0],
               (unsigned long long)PyGpuArray_DIMS(kern)[1],
               (unsigned long long)PyGpuArray_DIMS(kern)[2],
               (unsigned long long)PyGpuArray_DIMS(kern)[3],
               (long long)PyGpuArray_STRIDES(kern)[0]/4,
               (long long)PyGpuArray_STRIDES(kern)[1]/4,
               (long long)PyGpuArray_STRIDES(kern)[2]/4,
               (long long)PyGpuArray_STRIDES(kern)[3]/4);
        printf("INFO:   out dim: %llu %llu %llu %llu  "
               "out stride: %lld %lld %lld %lld\n",
               (unsigned long long)PyGpuArray_DIMS(out)[0],
               (unsigned long long)PyGpuArray_DIMS(out)[1],
               (unsigned long long)PyGpuArray_DIMS(out)[2],
               (unsigned long long)PyGpuArray_DIMS(out)[3],
               (long long)PyGpuArray_STRIDES(out)[0]/4,
               (long long)PyGpuArray_STRIDES(out)[1]/4,
               (long long)PyGpuArray_STRIDES(out)[2]/4,
               (long long)PyGpuArray_STRIDES(out)[3]/4);
    }

    if (!subsample &&
        out_contiguous &&
        (version==3||version==4||version==5||version==-1) &&
        out_wid<=max_threads_dim0 &&//Maximum of X threads by block.x
        (kern_len+2*kern_len-2)*img_wid_padded*sizeof(float) + kern_size_byte<shared_avail && //their is only 16k of shared memory
        !work_complete) //conv_full_patch_stack_padded
    {
      //version 3 without split
      //version 4 with split (more registers)
      //version 5 with split (more registers) low mem version(some restriction and still more register)
        int nb_split=1;//The number of split (i.e. the number of output pixel each thread compute.)
        if((version==4 || version==5) && out_len>1) nb_split++;//to force the use of split=true when testing.
        if(kern_len==1 && version==5){
          //version 5 don't support kern_len==1 as 1%0 return -1.
          version=-1;
          if(verbose)fprintf(stderr, "WARNING:conv full: Asking version 5 with kern_len==1. Combination not supported!\n");
        }
        if(img_size_padded_byte+kern_size_byte>shared_avail) version=5;

        //we pass by ceil_intdiv in case the out_len is not a multiple
        //of nb_split, we want nb_split the number of iteration.
        //Max of 16k of shared memory
        if(version==5)
          while ((((kern_len+ceil_intdiv(out_len,nb_split)-1)+2*kern_len-2)*img_wid_padded*sizeof(float) + kern_size_byte)>shared_avail) nb_split++;
        
        //327 as we use 25 register
        //version 5 will have only 1 block running at a time, so we
        //can use 32 registers per threads, but their is some other stuff that
        //for the limit to bu lower then 512.
        int max_thread = (version!=5?327:450);
        while (ceil_intdiv(out_len,nb_split)*out_wid>max_thread) nb_split++;
        if(version==-1 && out_size>max_threads_dim0)version=4;
        if(version==-1)version=3;


        if(version==-1 && nb_split>1) version=4;
        else if(version==-1) version=3;
        //force version 4 when more than 1 split are needed to always execute.
        else if(version==3 && nb_split!=1) version=4;

        assert(version!=3 || nb_split==1);
        assert(version!=5 || kern_len>1);
        assert(version!=-1);

        dim3 threads(out_wid, ceil_intdiv(out_len,nb_split));
        dim3 grid(nbatch,nkern);

        int shared_size=img_size_padded_byte + kern_size_byte;
        if(version==5)
          shared_size=((kern_len+threads.y-1)+2*kern_len-2)*img_wid_padded*sizeof(float) + kern_size_byte;
        void (*f)(const float*, const float*, float*,
                  int, int, int, int,
                  int, int, int, int,
                  int, int, int, int,
                  int, int);

#define CONV_FULL_PATCH_STACK_PADDED_SPECIAL(kern_wid) \
             if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==3 && kern_flipped) f=conv_full_patch_stack_padded<true,kern_wid,true,false,false>;\
        else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==4 && kern_flipped) f=conv_full_patch_stack_padded<true,kern_wid,true,true,false>;\
        else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==5 && kern_flipped) f=conv_full_patch_stack_padded<true,kern_wid,true,false,true>;\
        else if(version==3 && kern_flipped) f=conv_full_patch_stack_padded<true,kern_wid,false,false,false>;\
        else if(version==4 && kern_flipped)f=conv_full_patch_stack_padded<true,kern_wid,false,true,false>;\
        else if(version==5 && kern_flipped)f=conv_full_patch_stack_padded<true,kern_wid,false,false,true>;\
        else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==3) f=conv_full_patch_stack_padded<false,kern_wid,true,false,false>;\
        else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==4) f=conv_full_patch_stack_padded<false,kern_wid,true,true,false>;\
        else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==5) f=conv_full_patch_stack_padded<false,kern_wid,true,false,true>;\
        else if(version==3) f=conv_full_patch_stack_padded<false,kern_wid,false,false,false>;\
        else if(version==4) f=conv_full_patch_stack_padded<false,kern_wid,false,true,false>;\
        else if(version==5) f=conv_full_patch_stack_padded<false,kern_wid,false,false,true>;\
        else assert(false);

        CONV_FULL_PATCH_STACK_PADDED_SPECIAL(THEANO_KERN_WID);

        f<<< grid, threads, shared_size>>>
            (cuda_get_ptr(img), kern_data_unflipped, cuda_get_ptr(out),
              img_len, img_wid, kern_len, kern_wid, nkern, nstack,
              img_stride_col, img_stride_row, img_stride_stack,
              img_stride_batch, kern_stride_col_unflipped, kern_stride_row_unflipped,
              kern_stride_stack, kern_stride_nkern);

        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts)
        {
          if (verbose>1)
            fprintf(stderr,
                    "threads.x=%i, threads.y=%i, threads.z=%i,"
                    " grid.x=%i, grid.y=%i, shared_size=%i, nb_threads=%i,"
                    " out_len=%i, nb_split=%i, version=%i\n",
                    threads.x, threads.y, threads.z,
                    grid.x, grid.y, shared_size,
                    threads.x * threads.y * threads.z,
                    out_len, nb_split, version);
            if (verbose)
              fprintf(stderr,
                      "INFO: used 'conv_full_patch_stack_padded'"
                      " nb_split=%d low_mem=%s\n",
                      nb_split, (version==5?"true":"false"));
            work_complete = true;
        }
        else
        {
          if (verbose)
            fprintf(stderr,
                    "threads.x=%i, threads.y=%i, threads.z=%i,"
                    " grid.x=%i, grid.y=%i,shared_size=%i, nb_threads=%i,"
                    " out_len=%i, nb_split=%i, version=%i\n",
                    threads.x, threads.y, threads.z,
                    grid.x, grid.y, shared_size,
                    threads.x * threads.y * threads.z,
                    out_len, nb_split, version);
          if (verbose)
            fprintf(stderr,
                    "INFO: impl 'conv_full_patch_stack_padded' %s %s"
                    " failed (%s), trying next implementation\n",
                    version==3?"no split": "split",
                    (version==5?"low_mem":"not_low_mem"),
                    cudaGetErrorString(sts));
        }                         
    }

    if (!subsample && c_contiguous &&
        (version==0||version==-1) &&
        out_size<=max_threads_dim0 &&//Maximum of X threads by block
        nstack == 1 &&// don't implement the stack in the kernel.
        img_size_byte+kern_size_byte<shared_avail && //their is only 16k of shared memory
        !work_complete) //conv_full_patch
    {
        dim3 threads(out_wid, out_len);
        dim3 grid(nbatch,nkern);
        int shared_size=(img_size + kern_size)*sizeof(float);
        //TODO assert c_continious for img, kern and out in the 2 inner dimensions.

        conv_full_patch<<< grid, threads, shared_size>>>
            (cuda_get_ptr(img),
             cuda_get_ptr(kern),
             cuda_get_ptr(out),
           img_len, img_wid,
           kern_len, kern_wid,
           nkern, nstack);

        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts) 
        {
            if (verbose) fprintf(stderr, "INFO: used 'conv_full_patch' version\n");
            work_complete = true;
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "threads.x=%i, threads.y=%i, grid.x=%i, grid.y=%i,"
                      " shared_size=%i, nb_threads=%i\n",
                      threads.x, threads.y, grid.x, grid.y, shared_size,
                      threads.x * threads.y);
            if (verbose)
              fprintf(stderr,
                      "INFO: impl 'conv_full_patch' failed (%s),"
                      " trying next implementation\n",
                      cudaGetErrorString(sts));
        }                         
    }
    if (false && !subsample && //disabled as test fail for this kernel
        (version==1||version==-1) &&
        out_size<=max_threads_dim0 &&//Maximum of X threads by block
        (nbatch > 20 || version==1) &&  // we only launch nbatch blocks, so make sure there is enough to be worth it, but if we specify the version, this check should not be done to allow testing.
        nstack*img_size_byte+nstack*kern_size_byte<shared_avail && //there is only 16k of shared memory
        !work_complete) //conv_full_load_everything
    {
        dim3 threads(out_wid, out_len);
        dim3 grid(nbatch);
        int shared_size=(img_size + kern_size)*nstack*sizeof(float);
        //TODO assert c_continious for img, kern and out in the 2 inner dimensions.

        //typeof(conv_full_load_everything<0>) f = ;
        void (*f)(const float*, const float*, float*,
                  int, int, int, int, int, int,
                  int, int, int, int, int, int, int, int) = conv_full_load_everything<0>;

        f = conv_full_load_everything<THEANO_KERN_WID>;

        f<<< grid, threads, shared_size>>>
            (cuda_get_ptr(img),
             cuda_get_ptr(kern),
             cuda_get_ptr(out),
           img_len, img_wid, 
           kern_len, kern_wid,
           nkern, nstack,
           PyGpuArray_STRIDES(img)[3]/4,
           PyGpuArray_STRIDES(img)[2]/4,
           PyGpuArray_STRIDES(img)[1]/4,
           PyGpuArray_STRIDES(img)[0]/4,
           PyGpuArray_STRIDES(kern)[3]/4,
           PyGpuArray_STRIDES(kern)[2]/4,
           PyGpuArray_STRIDES(kern)[1]/4,
           PyGpuArray_STRIDES(kern)[0]/4
           );

        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts) 
        {
            if (verbose) fprintf(stderr, "INFO: used 'conv_full_load_everything' version\n");
            work_complete = true;
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "threads.x=%i, threads.y=%i, grid.x=%i, grid.y=%i,"
                      " shared_size=%i, nb_threads=%i\n",
                      threads.x, threads.y, grid.x, grid.y, shared_size,
                      threads.x * threads.y);
            if (verbose)
              fprintf(stderr, "INFO: impl 'conv_full_load_everything'"
                      " failed (%s), trying next implementation\n",
                      cudaGetErrorString(sts));
        }
    }

    if (!subsample &&
        img_batch_stack_contiguous &&
        out_contiguous &&
        (version==2||version==-1) &&
        out_size<=max_threads_dim0 &&//Maximum of X threads by block
        img_size_byte+kern_size_byte<shared_avail && //their is only 16k of shared memory
        !work_complete) //conv_full_patch_stack
    {
        dim3 threads(out_wid, out_len);
        dim3 grid(nbatch,nkern);
        int shared_size=(img_size + kern_size)*sizeof(float);

        void (*f)(const float*, const float*, float*,
                  int, int, int, int,
                  int, int, int, int,
                  int, int, int, int);

        if(img_contiguous_2d && kern_contiguous_2d) f=conv_full_patch_stack<true,true>;\
        else if(img_contiguous_2d && !kern_contiguous_2d) f=conv_full_patch_stack<true,false>;\
        else if(!img_contiguous_2d && kern_contiguous_2d) f=conv_full_patch_stack<false,true>;\
        else if(!img_contiguous_2d && !kern_contiguous_2d) f=conv_full_patch_stack<false,false>;

        f<<< grid, threads, shared_size>>>(
                cuda_get_ptr(img),
                cuda_get_ptr(kern),
                cuda_get_ptr(out),
                img_len, img_wid,
                kern_len, kern_wid,
                nkern, nstack,img_stride_col, img_stride_row,
                kern_stride_col, kern_stride_row,
                kern_stride_stack, kern_stride_nkern);

        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts) 
        {
            if (verbose)
              fprintf(stderr, "INFO: used 'conv_full_patch_stack' version\n");
            work_complete = true;
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "threads.x=%i, threads.y=%i, grid.x=%i, grid.y=%i,"
                      " shared_size=%i, nb_threads=%i\n",
                      threads.x, threads.y, grid.x, grid.y,
                      shared_size, threads.x * threads.y);
            if (verbose)
              fprintf(stderr, "INFO: impl 'conv_full_patch_stack' failed (%s), trying next implementation\n",
                      cudaGetErrorString(sts));
        }                         
    }
    if (1 && !work_complete) //conv_reference_full
    {
        if(verbose>1) fprintf(stderr, "INFO: will start conv_reference_full\n");

        int outsize = PyGpuArray_SIZE(out);
        int n_blocks = std::min(outsize, 4096);
        int n_threads = std::min(ceil_intdiv(outsize, n_blocks),
                                 256);
        if (0)
        {
            if (verbose)
              fprintf(stderr, "INFO: launching conv_reference_valid\n");
            if (verbose)
              fprintf(stderr, "      img : %llu %llu %llu %llu %p  "
                      "%lld %lld %lld %lld\n",
                      (unsigned long long)PyGpuArray_DIMS(img)[0],
                      (unsigned long long)PyGpuArray_DIMS(img)[1],
                      (unsigned long long)PyGpuArray_DIMS(img)[2],
                      (unsigned long long)PyGpuArray_DIMS(img)[3],
                      cuda_get_ptr(img),
                      (long long)PyGpuArray_STRIDES(img)[0]/4,
                      (long long)PyGpuArray_STRIDES(img)[1]/4,
                      (long long)PyGpuArray_STRIDES(img)[2]/4,
                      (long long)PyGpuArray_STRIDES(img)[3]/4);
            if (verbose)
              fprintf(stderr, "      kern: %llu %llu %llu %llu %p  "
                      "%lld %lld %lld %lld\n",
                      (unsigned long long)PyGpuArray_DIMS(kern)[0],
                      (unsigned long long)PyGpuArray_DIMS(kern)[1],
                      (unsigned long long)PyGpuArray_DIMS(kern)[2],
                      (unsigned long long)PyGpuArray_DIMS(kern)[3],
                      cuda_get_ptr(kern),
                      (long long)PyGpuArray_STRIDES(kern)[0]/4,
                      (long long)PyGpuArray_STRIDES(kern)[1]/4,
                      (long long)PyGpuArray_STRIDES(kern)[2]/4,
                      (long long)PyGpuArray_STRIDES(kern)[3]/4
                        );
            if (verbose)
                fprintf(stderr, "      out : %llu %llu %llu %llu %p  "
                        "%lld %lld %lld %lld\n",
                      (unsigned long long)PyGpuArray_DIMS(out)[0],
                      (unsigned long long)PyGpuArray_DIMS(out)[1],
                      (unsigned long long)PyGpuArray_DIMS(out)[2],
                      (unsigned long long)PyGpuArray_DIMS(out)[3],
                      cuda_get_ptr(out),
                      (long long)PyGpuArray_STRIDES(out)[0]/4,
                      (long long)PyGpuArray_STRIDES(out)[1]/4,
                      (long long)PyGpuArray_STRIDES(out)[2]/4,
                      (long long)PyGpuArray_STRIDES(out)[3]/4);
            if (verbose)
              fprintf(stderr, "   launch params: %i %i %i\n",
                      outsize, n_blocks, n_threads);
            if (verbose)
                fprintf(stderr, "   subsample params: %llu %llu\n",
                        (unsigned long long)subsample_rows,
                        (unsigned long long)subsample_cols);
        }
        conv_reference_full<<<n_blocks, n_threads>>>(
                PyGpuArray_DIMS(img)[0], PyGpuArray_DIMS(kern)[0],
                PyGpuArray_DIMS(img)[1],
                PyGpuArray_DIMS(img)[2], PyGpuArray_DIMS(img)[3],
                PyGpuArray_DIMS(kern)[2], PyGpuArray_DIMS(kern)[3],
                PyGpuArray_DIMS(out)[2], PyGpuArray_DIMS(out)[3],
                cuda_get_ptr(img), PyGpuArray_STRIDES(img)[0]/4,
                PyGpuArray_STRIDES(img)[1]/4,
                PyGpuArray_STRIDES(img)[2]/4,
                PyGpuArray_STRIDES(img)[3]/4,
                cuda_get_ptr(kern), PyGpuArray_STRIDES(kern)[0]/4,
                PyGpuArray_STRIDES(kern)[1]/4,
                PyGpuArray_STRIDES(kern)[2]/4,
                PyGpuArray_STRIDES(kern)[3]/4,
                cuda_get_ptr(out), PyGpuArray_STRIDES(out)[0]/4,
                PyGpuArray_STRIDES(out)[1]/4,
                PyGpuArray_STRIDES(out)[2]/4,
                PyGpuArray_STRIDES(out)[3]/4,
                subsample_rows, subsample_cols);

        cudaError_t sts = cudaGetLastError();
        if (cudaSuccess == sts) 
        {
            if (verbose)
              fprintf(stderr, "INFO: used 'conv_reference_full' version"
                      " ishp(%d, %d) kshp(%d, %d) oshp(%d, %d) nbatch=%d"
                      " nkern=%d nstack=%d subsample=%d\n",
                      img_len,img_wid, kern_len, kern_wid,
                      out_len, out_wid, nbatch, nkern, nstack, subsample);
            work_complete = true;
        }
        else
        {
          if (verbose)
            fprintf(stderr, "threads.x=%i, threads.y=%i, grid.x=%i, grid.y=%i,"
                    " shared_size=%i, nb_threads=%i\n",
                    n_threads, 1, n_blocks, 1, 0, n_threads);
          if (verbose)
            fprintf(stderr, "INFO: impl 'conv_reference_full' failed (%s),"
                    " trying next implementation\n",
                    cudaGetErrorString(sts));
          PyErr_Format(PyExc_RuntimeError,
                       "ERROR: all implementations failed for"
                       " CudaNdarray_conv_full! (%s)",
                       cudaGetErrorString(sts));
          return -1;
        }
    }
    return 0;
}

PyObject *
PyGpuArray_Conv(PyGpuArrayObject *img, PyGpuArrayObject * kern,
                 PyGpuArrayObject * out, const int mode,
                 const size_t subsample_rows, const size_t subsample_cols,
                 const int version, const int verbose,
                 const int max_threads_dim0 = 512
                 )
{
    // Re-use the out object if possible.  If the out object it not used, then its refcount is not modified.
    //  If the out object is re-used then it is returned, and its refcount is incremented by 1.
    //
    if (PyGpuArray_NDIM(img) != 4)
    {
      PyErr_SetString(PyExc_ValueError, "PyGpuArray 4-D tensor required");
      return NULL;
    }
    if (PyGpuArray_NDIM(kern) != 4)
    {
      PyErr_SetString(PyExc_ValueError, "PyGpuArray 4-D tensor required");
      return NULL;
    }

    size_t out_dim[4];
    out_dim[0] = PyGpuArray_DIMS(img)[0];
    out_dim[1] = PyGpuArray_DIMS(kern)[0];
    size_t logical_rows, logical_cols;
    if (mode == ConvMode_VALID)
    {
        logical_rows = PyGpuArray_DIMS(img)[2] - PyGpuArray_DIMS(kern)[2] + 1;
        logical_cols = PyGpuArray_DIMS(img)[3] - PyGpuArray_DIMS(kern)[3] + 1;
    }
    else
    {
        logical_rows = PyGpuArray_DIMS(img)[2] + PyGpuArray_DIMS(kern)[2] - 1;
        logical_cols = PyGpuArray_DIMS(img)[3] + PyGpuArray_DIMS(kern)[3] - 1;
    }
    out_dim[2] = ceil_intdiv(logical_rows, subsample_rows);
    out_dim[3] = ceil_intdiv(logical_cols, subsample_cols);

    PyGpuArrayObject * rval = NULL;

    if ( out
         && PyGpuArray_NDIM(out)==4
         && out->ga.flags & GA_C_CONTIGUOUS
         && PyGpuArray_DIMS(out)[0]==out_dim[0]
         && PyGpuArray_DIMS(out)[1]==out_dim[1]
         && PyGpuArray_DIMS(out)[2]==out_dim[2]
         && PyGpuArray_DIMS(out)[3]==out_dim[3])
    {
      rval = out;
      Py_INCREF(rval);
      if (verbose)
        fprintf(stderr,
                "INFO: Conv is reusing the 'out' argument"
                " structure.\n");
    }
    else
    {
      if (out && verbose)
        fprintf(stderr,
                "INFO: Conv is ignoring 'out' argument with wrong"
                " structure.\n");
      else if(verbose)
        fprintf(stderr,
                "INFO: Conv don't have an 'out' argument"
                " structure.\n");

      rval = pygpu_zeros(4, out_dim,
                         img->ga.typecode, GA_C_ORDER,
                         pygpu_default_context(), Py_None);
      //rval might be null
    }
    if ((rval==NULL)
        || ((mode==ConvMode_VALID) && PyGpuArray_conv_valid(img, kern, rval,
                                                            subsample_rows,
                                                            subsample_cols,
                                                            version, verbose,
                                                            max_threads_dim0))
        || ((mode==ConvMode_FULL) && PyGpuArray_conv_full(img, kern, rval,
                                                          subsample_rows,
                                                          subsample_cols,
                                                          version, verbose,
                                                          max_threads_dim0))
            )
    {
        // if rval is something we just allocated,
        // and there was a problem, then we have to free it.
        Py_XDECREF(rval);
        return NULL;
    }
    return (PyObject*)rval;
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
