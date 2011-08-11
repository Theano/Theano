"""
This code can only work if gnumpy and theano are initialized on the
same gpu as theano.
"""

try:
    import gnumpy
    import cudamat
    gnumpy_available = True

    ___const_garray = gnumpy.rand(1)

    import theano.sandbox.cuda as cuda
    if cuda.cuda_available == False:
        raise ImportError('Optional theano package cuda disabled')

    def cudandarray_to_garray(x, copyif=False):
        """ take a CudaNdarray and return a gnumpy.garray object.

        :type x: CudaNdarray
        :param x: The array to transform to gnumpy.garray.
        :type copyif: bool
        :param copyif: If False, raise an error if x is not c contiguous.
                       If it is c contiguous, we return a GPUArray that share
                       the same memory region as x.
                       If True, copy x if it is no c contiguous, so the return won't
                       shape the same memory region. If c contiguous, the return
                       will share the same memory region.

                       We need to do this as GPUArray don't fully support strided memory.

        :return type: cudamat.CUDAMatrix
        """
        if not isinstance(x, cuda.CudaNdarray):
            raise ValueError("We can transfer only CudaNdarray to cudamat.CUDAMatrix")
        else:
            # Check if it is c contiguous
            size = 1
            c_contiguous = True
            for i in range(x.ndim-1, -1, -1):
                if x.shape[i] == 1:
                    continue
                if x._strides[i] != size:
                    c_contiguous = False
                    break
                size *= x.shape[i]
            if not c_contiguous:
                if copyif:
                    x = x.copy()
                else:
                    raise ValueError("We where asked to don't copy memory, but the memory is not c contiguous.")

            # Now x is always c contiguous.



            # the next step is to create a CUDAMatrix object. We do so by first creating
            # a cudamat object with no data_host.
            cm_mat = cudamat.cudamat()

            cm_mat.size[0] = reduce(lambda x,y:x*y, x.shape, 1)
            cm_mat.size[1] = 1
            cm_mat.on_host = 0
            cm_mat.on_device = 1
            cm_mat.is_trans = 0
            cm_mat.owns_data = 0 # <-- note: cm_mat dosen't owe the data; x does. So x will delete it.


            # x.gpudata is a long. We need a pointer to a float. cast.
            import ctypes
            cm_mat.data_device = ctypes.cast(x.gpudata, ctypes.POINTER(ctypes.c_float))

            px = cudamat.CUDAMatrix(cm_mat)

            px._base = x # x won't be freed if the cudamat object isn't freed.




            px.mat_on_host = False # let cudamat know that we don't have a numpy
                                   # array attached.

            # Note how gnumpy tracks its cudamat objects: it moves things to the
            # _cmsReuseCache when the gnumpy array is deleted, thus the arrays
            # returned by theano will never be deleted.
            # However, if the garray thinks that the object is a view, then it won't
            # move the _base to the _cmsResueCache; so the cudamat object will be deleted,
            # and we won't overpump the world with memory.
            _is_alias_of = ___const_garray

            ans = gnumpy.garray(px,
                                x.shape,
                                _is_alias_of)

            return ans

    def garray_to_cudandarray(x):
        """ take a gnumpy.garray and make a CudaNdarray that point to its memory
        """
        if not isinstance(x, gnumpy.garray):
            raise ValueError("We can transfer only gnumpy.garray to CudaNdarray")
        # elif x.dtype != "float32":
        #     raise ValueError("CudaNdarray support only float32")
        # We don't need this, because cudamat is always float32.
        else:
            strides = [1]
            for i in x.shape[::-1][:-1]:
                strides.append(strides[-1]*i)
            strides = tuple(strides[::-1])


            import ctypes
            ptr_long = long(ctypes.cast(x._base.mat.data_device, ctypes.c_void_p).value)

            # seems legit.
            z = cuda.from_gpu_pointer(ptr_long, x.shape, strides, x._base)
            return z

except (ImportError, OSError):
    gnumpy_available = False
