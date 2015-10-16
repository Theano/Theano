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
    if (!(PyGpuArray_DIMS(out)[2] == ceil_intdiv(PyGpuArray_DIMS(img)[2]- PyGpuArray_DIMS(kern)[2] + 1, subsample_rows) ||
          PyGpuArray_DIMS(out)[3] == ceil_intdiv(PyGpuArray_DIMS(img)[3]- PyGpuArray_DIMS(kern)[3] + 1, subsample_cols) ||
          PyGpuArray_DIMS(out)[0] == PyGpuArray_DIMS(img)[0] ||
          PyGpuArray_DIMS(out)[1] == PyGpuArray_DIMS(kern)[0] ||
          PyGpuArray_DIMS(img)[1] == PyGpuArray_DIMS(kern)[1])) {
        PyErr_SetString(PyExc_ValueError, "GpuConv: sizes don't match");
        return -1;
    }

    // we now search through a few implementations until one applies to our arguments.

    //TODO: make separate version as if all fill this is slower.
    //TODO: Make a switch with power of 2 max size as template
    //TODO: make a parameter the number of division
    //TODO: Should we make them in separate grid block instead?
 
    const int stack_len = PyGpuArray_DIMS(img)[1];
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
    const int out_stride_col = PyGpuArray_STRIDES(out)[3]/4;
    const int out_stride_row = PyGpuArray_STRIDES(out)[2]/4;
    const int out_stride_nkern = PyGpuArray_STRIDES(out)[1]/4;
    const int out_stride_batch = PyGpuArray_STRIDES(out)[0]/4;

    const int img_size=img_len*img_wid;
    const int kern_size=kern_len*kern_wid;
    const int out_size=out_len*out_wid;
    const int img_size_byte = img_size*sizeof(float);
    const int kern_size_byte = kern_size*sizeof(float);
    const int out_size_byte = out_size*sizeof(float);
    if (!((THEANO_KERN_WID == PyGpuArray_DIMS(kern)[3]) || (THEANO_KERN_WID==0))){
     PyErr_Format(PyExc_ValueError, "ERROR: This GpuConv code was compiled for"
                   " %d kernel columns, but the kernel we received had %llu columns!",
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
    if(kern_stride_col==-1 && kern_stride_row==-kern_wid){
      //the last two dimensions are c_contiguous but flipped!
      kern_flipped=false;
      kern_contiguous_2d_unflipped = true;
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
        size_t threads_per_block[3] = {(size_t)out_wid,
                               ceil_intdiv((size_t)out_len,(size_t)nb_split),
                               (size_t)1};
        size_t n_blocks[3] = {(size_t)nbatch, (size_t)nkern, (size_t)1};

        size_t shmem_sz = (img_size + kern_size)*sizeof(float);

        GpuKernel *k = NULL;
        if(threads_per_block[1]==out_len) k=&conv_patch_2_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else k=&conv_patch_3_node_<<<<HASH_PLACEHOLDER>>>>_0;

        void *kernel_params[] = {(void *)img->ga.data, (void *)&img->ga.offset,
                                 (void *)kern->ga.data, (void *)&kern->ga.offset,
                                 (void *)out->ga.data, (void *)&out->ga.offset,
                                 (void *)&img_len, (void *)&img_wid,
                                 (void *)&kern_len, (void *)&kern_wid,
                                 (void *)&nkern, (void *)&nstack};
        int err = GpuKernel_call(k, 3, threads_per_block, n_blocks, shmem_sz, kernel_params);

        if (err == GA_NO_ERROR)
        {
            if (verbose)
              fprintf(stderr,
                      "INFO: used 'conv_patch' version %s nb_split=%d\n",
                      threads_per_block[1]==out_len ? "no split": "split", nb_split);
            work_complete = true;
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "INFO: impl 'conv_patch' failed (%s),"
                      " trying next implementation\n",
                      GpuKernel_error(k, err));
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
        size_t threads_per_block[3] = {(size_t)out_wid,
                               (size_t)ceil_intdiv(out_len,nb_split),
                               (size_t)1};

        bool preload_full_kernel = (img_size_byte + kern_size_byte) <shared_avail;
        if(version==11 || version==12) preload_full_kernel=false;
        size_t n_blocks[3] = {(size_t)nbatch, (size_t)nkern, (size_t)1};
        size_t shmem_sz = (img_size + (preload_full_kernel?kern_size:kern_wid))*sizeof(float);

        GpuKernel *k = NULL;
        if(!preload_full_kernel && nb_split==1 && !img_contiguous_2d && !kern_contiguous_2d && !subsample){ k=&conv_patch_stack_64_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split==1 && !img_contiguous_2d && !kern_contiguous_2d && subsample){ k=&conv_patch_stack_65_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split==1 && !img_contiguous_2d && !kern_contiguous_2d && !subsample){ k=&conv_patch_stack_66_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split==1 && !img_contiguous_2d && !kern_contiguous_2d && subsample){ k=&conv_patch_stack_67_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split!=1 && !img_contiguous_2d && !kern_contiguous_2d && !subsample){ k=&conv_patch_stack_68_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split!=1 && !img_contiguous_2d && !kern_contiguous_2d && subsample){ k=&conv_patch_stack_69_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split!=1 && !img_contiguous_2d && !kern_contiguous_2d && !subsample){ k=&conv_patch_stack_70_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split!=1 && !img_contiguous_2d && !kern_contiguous_2d && subsample){ k=&conv_patch_stack_71_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split==1 && !img_contiguous_2d && kern_contiguous_2d && !subsample){ k=&conv_patch_stack_72_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split==1 && !img_contiguous_2d && kern_contiguous_2d && subsample){ k=&conv_patch_stack_73_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split==1 && !img_contiguous_2d && kern_contiguous_2d && !subsample){ k=&conv_patch_stack_74_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split==1 && !img_contiguous_2d && kern_contiguous_2d && subsample){ k=&conv_patch_stack_75_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split!=1 && !img_contiguous_2d && kern_contiguous_2d && !subsample){ k=&conv_patch_stack_76_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split!=1 && !img_contiguous_2d && kern_contiguous_2d && subsample){ k=&conv_patch_stack_77_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split!=1 && !img_contiguous_2d && kern_contiguous_2d && !subsample){ k=&conv_patch_stack_78_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split!=1 && !img_contiguous_2d && kern_contiguous_2d && subsample){ k=&conv_patch_stack_79_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split==1 && img_contiguous_2d && !kern_contiguous_2d && !subsample){ k=&conv_patch_stack_80_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split==1 && img_contiguous_2d && !kern_contiguous_2d && subsample){ k=&conv_patch_stack_81_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split==1 && img_contiguous_2d && !kern_contiguous_2d && !subsample){ k=&conv_patch_stack_82_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split==1 && img_contiguous_2d && !kern_contiguous_2d && subsample){ k=&conv_patch_stack_83_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split!=1 && img_contiguous_2d && !kern_contiguous_2d && !subsample){ k=&conv_patch_stack_84_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split!=1 && img_contiguous_2d && !kern_contiguous_2d && subsample){ k=&conv_patch_stack_85_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split!=1 && img_contiguous_2d && !kern_contiguous_2d && !subsample){ k=&conv_patch_stack_86_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split!=1 && img_contiguous_2d && !kern_contiguous_2d && subsample){ k=&conv_patch_stack_87_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split==1 && img_contiguous_2d && kern_contiguous_2d && !subsample){ k=&conv_patch_stack_88_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split==1 && img_contiguous_2d && kern_contiguous_2d && subsample){ k=&conv_patch_stack_89_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split==1 && img_contiguous_2d && kern_contiguous_2d && !subsample){ k=&conv_patch_stack_90_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split==1 && img_contiguous_2d && kern_contiguous_2d && subsample){ k=&conv_patch_stack_91_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split!=1 && img_contiguous_2d && kern_contiguous_2d && !subsample){ k=&conv_patch_stack_92_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(!preload_full_kernel && nb_split!=1 && img_contiguous_2d && kern_contiguous_2d && subsample){ k=&conv_patch_stack_93_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split!=1 && img_contiguous_2d && kern_contiguous_2d && !subsample){ k=&conv_patch_stack_94_node_<<<<HASH_PLACEHOLDER>>>>_0;}
        else if(preload_full_kernel && nb_split!=1 && img_contiguous_2d && kern_contiguous_2d && subsample){ k=&conv_patch_stack_95_node_<<<<HASH_PLACEHOLDER>>>>_0;}

        void *kernel_params[] = {(void *)img->ga.data, (void *)&img->ga.offset,
                                 (void *)kern->ga.data, (void *)&kern->ga.offset,
                                 (void *)out->ga.data, (void *)&out->ga.offset,
                                 (void *)&img_len, (void *)&img_wid,
                                 (void *)&kern_len, (void *)&kern_wid,
                                 (void *)&out_len, (void *)&out_wid,
                                 (void *)&nkern, (void *)&nstack,
                                 (void *)&img_stride_col, (void *)&img_stride_row,
                                 (void *)&img_stride_stack, (void *)&img_stride_batch,
                                 (void *)&kern_stride_col, (void *)&kern_stride_row,
                                 (void *)&kern_stride_stack, (void *)&kern_stride_nkern,
                                 (void *)&subsample_rows, (void *)&subsample_cols};

        int err = GpuKernel_call(k, 3, threads_per_block, n_blocks, shmem_sz, kernel_params);

        if (err == GA_NO_ERROR)
        {
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
                      "INFO: impl 'conv_patch_stack' failed (%s),"
                      " trying next implementation\n",
                      GpuKernel_error(k, err));
        }
    }

    if (!subsample && out_contiguous &&
        (version==4||version==-1) &&
        out_wid<=max_threads_dim0 &&//Maximum of X threads by block.x
        nstack == 1 &&// don't implement the stack in the kernel.
        kern_len*img_wid*sizeof(float)+kern_size_byte<shared_avail &&//their is only 16k of shared memory
        !work_complete) //conv_rows

    {
        size_t threads_per_block[3] = {(size_t)out_wid, (size_t)1, (size_t)1};
        size_t n_blocks[3] = {(size_t)out_len, (size_t)nbatch*nkern, (size_t)1};
        size_t shmem_sz = (kern_len*img_wid + kern_size)*sizeof(float);

        GpuKernel *k = NULL;
        if(!img_contiguous_2d || !kern_contiguous_2d) k=&conv_rows_0_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else k=&conv_rows_1_node_<<<<HASH_PLACEHOLDER>>>>_0;

        void *kernel_params[] = {
            (void *)img->ga.data, (void *)&img->ga.offset,
            (void *)kern->ga.data, (void *)&kern->ga.offset,
            (void *)out->ga.data, (void *)&out->ga.offset,
            (void *)&img_len, (void *)&img_wid,
            (void *)&kern_len, (void *)&kern_wid,
            (void *)&nkern, (void *)&nstack,
            (void *)&img_stride_col, (void *)&img_stride_row,
            (void *)&img_stride_stack, (void *)&img_stride_batch,
            (void *)&kern_stride_col, (void *)&kern_stride_row,
            (void *)&kern_stride_stack, (void *)&kern_stride_nkern};
        int err = GpuKernel_call(k, 3, threads_per_block, n_blocks, shmem_sz, kernel_params);

        if (err == GA_NO_ERROR)
        {
            work_complete = true;
            if (verbose)
              fprintf(stderr, "INFO: used 'conv_rows' version\n");
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "INFO: impl 'conv_rows' failed (%s),"
                      " trying next implementation\n",
                      GpuKernel_error(k, err));
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

        size_t threads_per_block[3] = {(size_t)out_wid, (size_t)nb_row, (size_t)1};
        size_t n_blocks[3] = {(size_t)ceil_intdiv(out_len,nb_row),
                              (size_t)nbatch*nkern, (size_t)1};

        size_t shmem_sz =((kern_len+nb_row-1)*img_wid + kern_size)*sizeof(float);

        GpuKernel *k = NULL;
        if(!img_contiguous_2d || !kern_contiguous_2d) {
            k=&conv_rows_stack_0_node_<<<<HASH_PLACEHOLDER>>>>_0;
        } else {
            k=&conv_rows_stack_1_node_<<<<HASH_PLACEHOLDER>>>>_0;
        }

        void *kernel_params[] = {
            (void *)img->ga.data, (void *)&img->ga.offset,
            (void *)kern->ga.data, (void *)&kern->ga.offset,
            (void *)out->ga.data, (void *)&out->ga.offset,
            (void *)&img_len, (void *)&img_wid,
            (void *)&kern_len, (void *)&kern_wid,
            (void *)&nkern, (void *)&nstack,
            (void *)&img_stride_col, (void *)&img_stride_row,
            (void *)&img_stride_stack, (void *)&img_stride_batch,
            (void *)&kern_stride_col, (void *)&kern_stride_row,
            (void *)&kern_stride_stack, (void *)&kern_stride_nkern};
        int err = GpuKernel_call(k, 3, threads_per_block, n_blocks, shmem_sz, kernel_params);

        if (err == GA_NO_ERROR)
        {
            work_complete = true;
            if (verbose)
              fprintf(stderr, "INFO: used 'conv_rows_stack' version\n");
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "INFO: impl 'conv_rows_stack' failed (%s),"
                      " trying next implementation\n",
                      GpuKernel_error(k, err));
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


        size_t threads_per_block[3] = {(size_t)out_wid, (size_t)nb_row, (size_t)1};
        size_t n_blocks[3] = {(size_t)ceil_intdiv(out_len,nb_row),
                              (size_t)nbatch*nkern, (size_t)1};
          
        size_t shmem_sz =((kern_len+nb_row-1)*img_wid + kern_size)*sizeof(float);

        GpuKernel *k = NULL;
        if((!img_contiguous_2d || !kern_contiguous_2d)&&version==9) k=&conv_rows_stack2_1_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(version==9) k=&conv_rows_stack2_3_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(!img_contiguous_2d || !kern_contiguous_2d) k=&conv_rows_stack2_0_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else k=&conv_rows_stack2_2_node_<<<<HASH_PLACEHOLDER>>>>_0;

        void *kernel_params[] = {
            (void *)img->ga.data, (void *)&img->ga.offset,
            (void *)kern->ga.data, (void *)&kern->ga.offset,
            (void *)out->ga.data, (void *)&out->ga.offset,
            (void *)&img_len, (void *)&img_wid,
            (void *)&kern_len, (void *)&kern_wid,
            (void *)&nkern, (void *)&nstack,
            (void *)&img_stride_col, (void *)&img_stride_row,
            (void *)&img_stride_stack, (void *)&img_stride_batch,
            (void *)&kern_stride_col, (void *)&kern_stride_row,
            (void *)&kern_stride_stack, (void *)&kern_stride_nkern};
        int err = GpuKernel_call(k, 3, threads_per_block, n_blocks, shmem_sz, kernel_params);

        if (err == GA_NO_ERROR)
        {
            work_complete = true;
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
                      "INFO: impl 'conv_rows_stack2' failed (%s),"
                      " trying next implementation\n",
                      GpuKernel_error(k, err));
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
        size_t thread_z=ceil_intdiv(kern_len,nb_split);
        size_t shmem_sz = sizeof(float)*(full_kern
                ? std::max((size_t)img_size + kern_size, out_size*thread_z)
                : std::max((size_t)img_size + thread_z*kern_wid, out_size*thread_z));

        // constraint (c)
        while ((shmem_sz >= shared_avail) && (nb_split <= kern_len)){
            //if we can't fit the kernel in shared memory, we must split it more.
            nb_split++;
            thread_z=ceil_intdiv(kern_len,nb_split);
            shmem_sz = sizeof(float)*(full_kern
                ? std::max((size_t)img_size + kern_size, out_size*thread_z)
                : std::max(img_size + thread_z*kern_wid, out_size*thread_z));
        }
        if (nb_split <= kern_len)
        {
            assert(thread_z>0);//should not happen, but in case...
            if(!full_kern) assert(thread_z!=kern_len);

            size_t threads_per_block[3] = {(size_t)out_wid,
                                   (size_t)out_len,
                                   (size_t)thread_z};
            size_t n_blocks[3] = {(size_t)nbatch, (size_t)nkern, (size_t)1};

            GpuKernel *k = NULL;

            const bool split=thread_z!=kern_len;
            const bool ccontig=img_contiguous_2d && kern_contiguous_2d_unflipped;

            //printf("kern_flipped=%d, ccontig=%d, split=%d, full_kern=%d\n",kern_flipped,ccontig,split,full_kern);
            //We will always be split when we don't load the full kernel

            /* if(!kern_flipped && !ccontig && !split && !full_kern) k=&conv_patch_stack_reduce_0_node_<<<<HASH_PLACEHOLDER>>>>_0;*/
            /*else*/ if(!kern_flipped && !ccontig && !split && full_kern) k=&conv_patch_stack_reduce_1_node_<<<<HASH_PLACEHOLDER>>>>_0;
            else if(!kern_flipped && !ccontig  && split && !full_kern) k=&conv_patch_stack_reduce_2_node_<<<<HASH_PLACEHOLDER>>>>_0;
            else if(!kern_flipped && !ccontig  && split && full_kern) k=&conv_patch_stack_reduce_3_node_<<<<HASH_PLACEHOLDER>>>>_0;
            /*else if(!kern_flipped && ccontig  && !split && !full_kern) k=&conv_patch_stack_reduce_4_node_<<<<HASH_PLACEHOLDER>>>>_0;*/
            else if(!kern_flipped && ccontig  && !split && full_kern) k=&conv_patch_stack_reduce_5_node_<<<<HASH_PLACEHOLDER>>>>_0;
            else if(!kern_flipped && ccontig  && split && !full_kern) k=&conv_patch_stack_reduce_6_node_<<<<HASH_PLACEHOLDER>>>>_0;
            else if(!kern_flipped && ccontig  && split && full_kern) k=&conv_patch_stack_reduce_7_node_<<<<HASH_PLACEHOLDER>>>>_0;
            /*else if(kern_flipped  && !ccontig && !split && !full_kern) k=&conv_patch_stack_reduce_8_node_<<<<HASH_PLACEHOLDER>>>>_0;*/
            else if(kern_flipped  && !ccontig && !split && full_kern) k=&conv_patch_stack_reduce_9_node_<<<<HASH_PLACEHOLDER>>>>_0;
            else if(kern_flipped  && !ccontig && split && !full_kern) k=&conv_patch_stack_reduce_10_node_<<<<HASH_PLACEHOLDER>>>>_0;
            else if(kern_flipped  && !ccontig && split && full_kern) k=&conv_patch_stack_reduce_11_node_<<<<HASH_PLACEHOLDER>>>>_0;
            /*else if(kern_flipped  && ccontig  && !split && !full_kern) k=&conv_patch_stack_reduce_12_node_<<<<HASH_PLACEHOLDER>>>>_0;*/
            else if(kern_flipped  && ccontig  && !split && full_kern) k=&conv_patch_stack_reduce_13_node_<<<<HASH_PLACEHOLDER>>>>_0;
            else if(kern_flipped  && ccontig  && split && !full_kern) k=&conv_patch_stack_reduce_14_node_<<<<HASH_PLACEHOLDER>>>>_0;
            else if(kern_flipped  && ccontig  && split && full_kern) k=&conv_patch_stack_reduce_15_node_<<<<HASH_PLACEHOLDER>>>>_0;

            void *kernel_params[] = {
                (void *)img->ga.data, (void *)&img->ga.offset,
                (void *)kern->ga.data, (void *)&kern->ga.offset,
                (void *)out->ga.data, (void *)&out->ga.offset,
                (void *)&img_len, (void *)&img_wid,
                (void *)&kern_len, (void *)&kern_wid,
                (void *)&nkern, (void *)&nstack,
                (void *)&img_stride_col, (void *)&img_stride_row,
                (void *)&img_stride_stack, (void *)&img_stride_batch,
                (void *)&kern_stride_col,
                (void *)&kern_stride_row,
                (void *)&kern_stride_stack, (void *)&kern_stride_nkern};
            int err = GpuKernel_call(k, 3, threads_per_block, n_blocks, shmem_sz, kernel_params);

            if (err == GA_NO_ERROR)
            {
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
                          "INFO: impl 'conv_patch_stack_reduce' failed (%s),"
                          " trying next implementation\n",
                          GpuKernel_error(k, err));
            }
        } // else no good nb_splits was found
    }

    if ((version==6||version==-1) &&
        kern_len<=320 &&
        !work_complete) //conv_valid_row_reduce
    {
        size_t outsize = PyGpuArray_SIZE(out);
        size_t n_blocks[3] = {std::min(outsize, (size_t)4096),
                              (size_t)1, (size_t)1};

        int block_nstack=nstack;
        //Max of 512 threads per blocks.
        //On old hardware, we have a max of 356 threads as we have only 
        //8k registers and the kernel use 23 register
        //TODO: check if we have 8k or 16k of register...
        while(block_nstack*kern_len>320)block_nstack--;
        size_t threads_per_block[3] = {(size_t)block_nstack, (size_t)kern_len, (size_t)1};

        size_t n_reduce_buf = block_nstack * kern_len * sizeof(float);
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

        GpuKernel *k = NULL;
        //std::cerr << "initial_reduce_boundary " << initial_reduce_boundary << "\n";
        //std::cerr << "kerns " << nstack << " " << kern_len << "\n";
        //std::cerr << "n_reduce_buf/sizeof(float) " << n_reduce_buf / sizeof(float) << "\n";
        if(block_nstack==nstack)
          k=&conv_valid_row_reduce_0_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else
          k=&conv_valid_row_reduce_1_node_<<<<HASH_PLACEHOLDER>>>>_0;

        void *kernel_params[] = {
            (void *)&nbatch, (void *)&nkern, (void *)&stack_len,
            (void *)&img_len, (void *)&img_wid,
            (void *)&kern_len, (void *)&kern_wid,
            (void *)&out_len, (void *)&out_wid,
            (void *)img->ga.data, (void *)&img->ga.offset,
            (void *)&img_stride_batch, (void *)&img_stride_stack,
            (void *)&img_stride_row, (void *)&img_stride_col,
            (void *)kern->ga.data, (void *)&kern->ga.offset,
            (void *)&kern_stride_nkern, (void *)&kern_stride_stack,
            (void *)&kern_stride_row, (void *)&kern_stride_col,
            (void *)out->ga.data, (void *)&out->ga.offset,
            (void *)&out_stride_batch, (void *)&out_stride_nkern,
            (void *)&out_stride_row, (void *)&out_stride_col,
            (void *)&subsample_rows, (void *)&subsample_cols,
            (void *)&initial_reduce_boundary};
        int err = GpuKernel_call(k, 3, threads_per_block, n_blocks, n_reduce_buf, kernel_params);

        if (err == GA_NO_ERROR)
        {
            work_complete = true;
            if (verbose)
              fprintf(stderr, "INFO: used 'conv_valid_row_reduce' version\n");
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "INFO: impl 'conv_valid_row_reduce' failed (%s),"
                      " trying next implementation\n",
                      GpuKernel_error(k, err));
        }
    }

    if (1 && !work_complete) //conv_reference_valid
    {
        size_t outsize = PyGpuArray_SIZE(out);
        size_t n_blocks[3] = {std::min(outsize, (size_t)4096),
                              (size_t)1, (size_t)1};
        size_t threads_per_block[3] = {std::min(ceil_intdiv(outsize, n_blocks[0]),
                                        (size_t)256),
                               (size_t)1, (size_t)1};

        if (verbose)
            fprintf(stderr, "INFO: launching conv_reference_valid\n");

        void *kernel_params[] = {
            (void *)&nbatch, (void *)&nkern, (void *)&stack_len,
            (void *)&img_len, (void *)&img_wid,
            (void *)&kern_len, (void *)&kern_wid,
            (void *)&out_len, (void *)&out_wid,
            (void *)img->ga.data, (void *)&img->ga.offset,
            (void *)&img_stride_batch, (void *)&img_stride_stack,
            (void *)&img_stride_row, (void *)&img_stride_col,
            (void *)kern->ga.data, (void *)&kern->ga.offset,
            (void *)&kern_stride_nkern, (void *)&kern_stride_stack,
            (void *)&kern_stride_row, (void *)&kern_stride_col,
            (void *)out->ga.data, (void *)&out->ga.offset,
            (void *)&out_stride_batch, (void *)&out_stride_nkern,
            (void *)&out_stride_row, (void *)&out_stride_col,
            (void *)&subsample_rows, (void *)&subsample_cols};
        int err = GpuKernel_call(&conv_reference_valid_node_<<<<HASH_PLACEHOLDER>>>>_0,
                                 3, threads_per_block, n_blocks, 0, kernel_params);

        if (err == GA_NO_ERROR)
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
                         GpuKernel_error(&conv_reference_valid_node_<<<<HASH_PLACEHOLDER>>>>_0, err));
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

    const int stack_len=PyGpuArray_DIMS(img)[1];
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
    const int out_stride_col = PyGpuArray_STRIDES(out)[3]/4;
    const int out_stride_row = PyGpuArray_STRIDES(out)[2]/4;
    const int out_stride_nkern = PyGpuArray_STRIDES(out)[1]/4;
    const int out_stride_batch = PyGpuArray_STRIDES(out)[0]/4;

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
    if(kern_stride_col==-1 && kern_stride_row==-kern_wid){
      //the last two dimensions are c_contiguous but flipped!
      kern_flipped=false;
      kern_contiguous_2d_unflipped = true;
    }

    if (verbose>1)
    {
        printf("INFO: Running conv_full version=%d,"
               " MACRO kern_width=%d with inputs:\n", version, THEANO_KERN_WID);
        printf("INFO:   img  dim: %llu %llu %llu %llu  "
               "img  stride: %lld %lld %lld %lld\n",
               (unsigned long long)nbatch,
               (unsigned long long)stack_len,
               (unsigned long long)img_len,
               (unsigned long long)img_wid,
               (long long)img_stride_batch,
               (long long)img_stride_stack,
               (long long)img_stride_row,
               (long long)img_stride_col);
        printf("INFO:   kern dim: %llu %llu %llu %llu  "
               "kern stride: %lld %lld %lld %lld\n",
               (unsigned long long)nkern,
               (unsigned long long)nstack,
               (unsigned long long)kern_len,
               (unsigned long long)kern_wid,
               (long long)kern_stride_nkern,
               (long long)kern_stride_stack,
               (long long)kern_stride_row,
               (long long)kern_stride_col);
        printf("INFO:   out dim: %llu %llu %llu %llu  "
               "out stride: %lld %lld %lld %lld\n",
               (unsigned long long)PyGpuArray_DIMS(out)[0],
               (unsigned long long)PyGpuArray_DIMS(out)[1],
               (unsigned long long)out_len,
               (unsigned long long)out_wid,
               (long long)out_stride_batch,
               (long long)out_stride_nkern,
               (long long)out_stride_row,
               (long long)out_stride_col);
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

        size_t threads_per_block[3] = {(size_t)out_wid,
                               ceil_intdiv((size_t)out_len,(size_t)nb_split),
                               (size_t)1};
        size_t n_blocks[3] = {(size_t)nbatch, (size_t)nkern, (size_t)1};

        size_t shmem_sz=img_size_padded_byte + kern_size_byte;
        if(version==5)
          shmem_sz=((kern_len+threads_per_block[1]-1)+2*kern_len-2)*img_wid_padded*sizeof(float) + kern_size_byte;

        GpuKernel *k = NULL;
        if(version==3) k=&conv_full_patch_stack_padded_0_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(version==5) k=&conv_full_patch_stack_padded_1_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(version==4) k=&conv_full_patch_stack_padded_2_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==3) k=&conv_full_patch_stack_padded_4_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==5) k=&conv_full_patch_stack_padded_5_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==4) k=&conv_full_patch_stack_padded_6_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(version==3 && kern_flipped) k=&conv_full_patch_stack_padded_8_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(version==5 && kern_flipped)k=&conv_full_patch_stack_padded_9_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(version==4 && kern_flipped)k=&conv_full_patch_stack_padded_10_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==3 && kern_flipped) k=&conv_full_patch_stack_padded_12_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==5 && kern_flipped) k=&conv_full_patch_stack_padded_13_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(img_contiguous_2d && kern_contiguous_2d_unflipped && version==4 && kern_flipped) k=&conv_full_patch_stack_padded_14_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else assert(false);

        void *kernel_params[] = {
            (void *)img->ga.data, (void *)&img->ga.offset,
            (void *)kern->ga.data, (void *)&kern->ga.offset,
            (void *)out->ga.data, (void *)&out->ga.offset,
            (void *)&img_len, (void *)&img_wid,
            (void *)&kern_len, (void *)&kern_wid,
            (void *)&nkern, (void *)&nstack,
            (void *)&img_stride_col, (void *)&img_stride_row,
            (void *)&img_stride_stack, (void *)&img_stride_batch,
            (void *)&kern_stride_col, (void *)&kern_stride_row,
            (void *)&kern_stride_stack, (void *)&kern_stride_nkern};
        int err = GpuKernel_call(k, 3, threads_per_block, n_blocks, shmem_sz, kernel_params);

        if (err == GA_NO_ERROR)
        {
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
                    "INFO: impl 'conv_full_patch_stack_padded' %s %s"
                    " failed (%s), trying next implementation\n",
                    version==3?"no split": "split",
                    (version==5?"low_mem":"not_low_mem"),
                    GpuKernel_error(k, err));
        }                         
    }

    if (!subsample && c_contiguous &&
        (version==0||version==-1) &&
        out_size<=max_threads_dim0 &&//Maximum of X threads by block
        nstack == 1 &&// don't implement the stack in the kernel.
        img_size_byte+kern_size_byte<shared_avail && //their is only 16k of shared memory
        !work_complete) //conv_full_patch
    {
        size_t threads_per_block[3] = {(size_t)out_wid, (size_t)out_len, (size_t)1};
        size_t n_blocks[3] = {(size_t)nbatch, (size_t)nkern, (size_t)1};
        size_t shmem_sz = (img_size + kern_size)*sizeof(float);
        //TODO assert c_continious for img, kern and out in the 2 inner dimensions.

        void *kernel_params[] = {
            (void *)img->ga.data, (void *)&img->ga.offset,
            (void *)kern->ga.data, (void *)&kern->ga.offset,
            (void *)out->ga.data, (void *)&out->ga.offset,
            (void *)&img_len, (void *)&img_wid,
            (void *)&kern_len, (void *)&kern_wid,
            (void *)&nkern, (void *)&nstack};
        int err = GpuKernel_call(&conv_full_patch_node_<<<<HASH_PLACEHOLDER>>>>_0,
                                 3, threads_per_block, n_blocks, shmem_sz, kernel_params);

        if (err == GA_NO_ERROR)
        {
            if (verbose) fprintf(stderr, "INFO: used 'conv_full_patch' version\n");
            work_complete = true;
        }
        else
        {
            if (verbose)
              fprintf(stderr,
                      "INFO: impl 'conv_full_patch' failed (%s),"
                      " trying next implementation\n",
                      GpuKernel_error(&conv_full_patch_node_<<<<HASH_PLACEHOLDER>>>>_0, err));
        }                         
    }
    if (false && !subsample && //disabled as test fail for this kernel
        (version==1||version==-1) &&
        out_size<=max_threads_dim0 &&//Maximum of X threads by block
        (nbatch > 20 || version==1) &&  // we only launch nbatch blocks, so make sure there is enough to be worth it, but if we specify the version, this check should not be done to allow testing.
        nstack*img_size_byte+nstack*kern_size_byte<shared_avail && //there is only 16k of shared memory
        !work_complete) //conv_full_load_everything
    {
        size_t threads_per_block[3] = {(size_t)out_wid, (size_t)out_len, (size_t)1};
        size_t n_blocks[3] = {(size_t)nbatch, (size_t)1, (size_t)1};
        size_t shmem_sz = (img_size + kern_size)*nstack*sizeof(float);
        //TODO assert c_continious for img, kern and out in the 2 inner dimensions.

        void *kernel_params[] = {
            (void *)img->ga.data, (void *)&img->ga.offset,
            (void *)kern->ga.data, (void *)&kern->ga.offset,
            (void *)out->ga.data, (void *)&out->ga.offset,
            (void *)&img_len, (void *)&img_wid,
            (void *)&kern_len, (void *)&kern_wid,
            (void *)&nkern, (void *)&nstack,
            (void *)&img_stride_col, (void *)&img_stride_row,
            (void *)&img_stride_stack, (void *)&img_stride_batch,
            (void *)&kern_stride_col, (void *)&kern_stride_row,
            (void *)&kern_stride_stack, (void *)&kern_stride_nkern};
        int err = GpuKernel_call(&conv_full_load_everything_node_<<<<HASH_PLACEHOLDER>>>>_0,
                                 3, threads_per_block, n_blocks, shmem_sz, kernel_params);

        if (err == GA_NO_ERROR)
        {
            if (verbose) fprintf(stderr, "INFO: used 'conv_full_load_everything' version\n");
            work_complete = true;
        }
        else
        {
            if (verbose)
              fprintf(stderr, "INFO: impl 'conv_full_load_everything'"
                      " failed (%s), trying next implementation\n",
                      GpuKernel_error(&conv_full_load_everything_node_<<<<HASH_PLACEHOLDER>>>>_0, err));
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
        size_t threads_per_block[3] = {(size_t)out_wid, (size_t)out_len, (size_t)1};
        size_t n_blocks[3] = {(size_t)nbatch, (size_t)nkern, (size_t)1};
        size_t shmem_sz = (img_size + kern_size)*sizeof(float);

        GpuKernel *k = NULL;
        if(!img_contiguous_2d && !kern_contiguous_2d) k=&conv_full_patch_stack_0_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(!img_contiguous_2d && kern_contiguous_2d) k=&conv_full_patch_stack_1_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(img_contiguous_2d && !kern_contiguous_2d) k=&conv_full_patch_stack_2_node_<<<<HASH_PLACEHOLDER>>>>_0;
        else if(img_contiguous_2d && kern_contiguous_2d) k=&conv_full_patch_stack_3_node_<<<<HASH_PLACEHOLDER>>>>_0;

        void *kernel_params[] = {
            (void *)img->ga.data, (void *)&img->ga.offset,
            (void *)kern->ga.data, (void *)&kern->ga.offset,
            (void *)out->ga.data, (void *)&out->ga.offset,
            (void *)&img_len, (void *)&img_wid,
            (void *)&kern_len, (void *)&kern_wid,
            (void *)&nkern, (void *)&nstack,
            (void *)&img_stride_col, (void *)&img_stride_row,
            (void *)&kern_stride_col, (void *)&kern_stride_row,
            (void *)&kern_stride_stack, (void *)&kern_stride_nkern};
        int err = GpuKernel_call(k, 3, threads_per_block, n_blocks, shmem_sz, kernel_params);

        if (err == GA_NO_ERROR)
        {
            if (verbose)
              fprintf(stderr, "INFO: used 'conv_full_patch_stack' version\n");
            work_complete = true;
        }
        else
        {
            if (verbose)
              fprintf(stderr, "INFO: impl 'conv_full_patch_stack' failed (%s), trying next implementation\n",
                      GpuKernel_error(k, err));
        }                         
    }
    if (1 && !work_complete) //conv_reference_full
    {
        if(verbose>1) fprintf(stderr, "INFO: will start conv_reference_full\n");

        size_t outsize = PyGpuArray_SIZE(out);
        size_t n_blocks[3] = {std::min(outsize, (size_t)4096),
                              (size_t)1, (size_t)1};
        size_t threads_per_block[3] = {std::min(ceil_intdiv(outsize, n_blocks[0]),
                                        (size_t)256),
                               (size_t)1, (size_t)1};

        void *kernel_params[] = {
            (void *)&nbatch, (void *)&nkern, (void *)&stack_len,
            (void *)&img_len, (void *)&img_wid,
            (void *)&kern_len, (void *)&kern_wid,
            (void *)&out_len, (void *)&out_wid,
            (void *)img->ga.data, (void *)&img->ga.offset,
            (void *)&img_stride_batch, (void *)&img_stride_stack,
            (void *)&img_stride_row, (void *)&img_stride_col,
            (void *)kern->ga.data, (void *)&kern->ga.offset,
            (void *)&kern_stride_nkern, (void *)&kern_stride_stack,
            (void *)&kern_stride_row, (void *)&kern_stride_col,
            (void *)out->ga.data, (void *)&out->ga.offset,
            (void *)&out_stride_batch, (void *)&out_stride_nkern,
            (void *)&out_stride_row, (void *)&out_stride_col,
            (void *)&subsample_rows, (void *)&subsample_cols};
        int err = GpuKernel_call(&conv_reference_full_node_<<<<HASH_PLACEHOLDER>>>>_0,
                                 3, threads_per_block, n_blocks, 0, kernel_params);

        if (err == GA_NO_ERROR)
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
            fprintf(stderr, "INFO: impl 'conv_reference_full' failed (%s),"
                    " trying next implementation\n",
                    GpuKernel_error(&conv_reference_full_node_<<<<HASH_PLACEHOLDER>>>>_0, err));
          PyErr_Format(PyExc_RuntimeError,
                       "ERROR: all implementations failed for"
                       " CudaNdarray_conv_full! (%s)",
                       GpuKernel_error(&conv_reference_full_node_<<<<HASH_PLACEHOLDER>>>>_0, err));
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
                         img->context, Py_None);
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
