// modified from pytorch
// https://github.com/pytorch/pytorch/master/blob/torch/lib/THC/THCTensorTopK.cuh
//
// Converts a type (maybe float) to an integer representation with the same
// sorting; i.e., for floats f1, f2:
// if f1 < f2 then convert(f1) < convert(f2)
// We use this to enable radix selection of floating-point values.
// This also gives a relative order for NaNs, but that's ok, as they
// will all be adjacent

template <typename T>
struct RadixConfig {};

template <>
struct RadixConfig<float> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (x ^ mask);
  }

  static inline __device__ float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

    return __int_as_float(v ^ mask);
  }
};

template <>
struct RadixConfig<unsigned char> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(unsigned char v) {
    return v;
  }

  static inline __device__ unsigned char deconvert(RadixType v) {
    return v;
  }
};

template <>
struct RadixConfig<char> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(char v) {
    return 128u + v;
  }

  static inline __device__ char deconvert(RadixType v) {
    return v - 128;
  }
};

template <>
struct RadixConfig<short> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(short v) {
    assert(sizeof(short) == 2);
    return 32768u + v;
  }

  static inline __device__ short deconvert(RadixType v) {
    return v - 32768;
  }
};

template <>
struct RadixConfig<int> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(int v) {
    assert(sizeof(int) == 4);
    return 2147483648u + v;
  }

  static inline __device__ int deconvert(RadixType v) {
    return v - 2147483648u;
  }
};

template <>
struct RadixConfig<long> {
  typedef unsigned long long int RadixType;

  static inline __device__ RadixType convert(long v) {
    assert(sizeof(long) == 8);
    return 9223372036854775808ull + v;
  }

  static inline __device__ long deconvert(RadixType v) {
    return v - 9223372036854775808ull;
  }
};

template <>
struct RadixConfig<double> {
  typedef unsigned long long int RadixType;

  static inline __device__ RadixType convert(double v) {
    RadixType x = __double_as_longlong(v);
    RadixType mask = -((x >> 63)) | 0x8000000000000000;
    return (x ^ mask);
  }

  static inline __device__ double deconvert(RadixType v) {
    RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __longlong_as_double(v ^ mask);
  }
};

#ifdef USE_HALF
template <>
struct RadixConfig<half> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(half v) {
#if defined(__CUDACC_VER__) && __CUDACC_VER__ >= 80000
    RadixType x = __half_as_ushort(v);
    RadixType mask = -((x >> 15)) | 0x8000;
    return (x ^ mask);
#else
    assert(false);
    return 0u;
#endif
  }

  static inline __device__ half deconvert(RadixType v) {
#if defined(__CUDACC_VER__) && __CUDACC_VER__ >= 80000
    RadixType mask = ((v >> 15) - 1) | 0x8000;
    return __ushort_as_half(v ^ mask);
#else
    assert(false);
    return ScalarConvert<int, half>::to(0);
#endif
  }
};
#endif

// $$inp_t should be replaced in c_code
// we cannot use templated __global__ because gpuarray API does not support it yet
#define NDIM            $ndim
#define INPUT_TYPE      $inp_t
#define INDEX_TYPE     $out_t
#define bitsof(T)       (sizeof(T)*8)
#define RADIX_BITS      2
#define RADIX_SIZE      (1<<RADIX_BITS)
#define RADIX_MASK(n)   ((RADIX_SIZE-1) << (n*RADIX_BITS))
#define RADIX_DIGITS(T) (bitsof(T)/RADIX_BITS)
#define radix_t         RadixConfig<INPUT_TYPE>::RadixType

#if RADIX_SIZE > 32
#error "RADIX_SIZE must be smaller than warp size (32)"
#endif

template <typename T>
static inline __device__ T binary_cumsum(int idx, int warp_id, int lane_id, T* smem, bool value) {
    // cumsum within 1D thread block, which adds up `value` of all threads whose id is *no greater than* the current thread
    // cumsum within warp
    unsigned int warp_bits = __ballot(value);
    T warp_sum = __popc(((2<<lane_id)-1) & warp_bits);

    if (lane_id == 0)
        smem[warp_id] = __popc(warp_bits);

    __syncthreads();

    // cumsum across warps in one thread
    if (idx == 0) {
        int current = 0;
        for (int i = 0; i < blockDim.x / 32; ++i) {
            T v = smem[i];
            smem[i] = smem[i]+current;
            current = current+v;
        }
    }

    __syncthreads();

    // load the carry from the preceding warp
    if (warp_id >= 1) {
        warp_sum = warp_sum+smem[warp_id - 1];
    }

    return warp_sum;
}

template <typename T>
static inline __device__ T binary_cumsum_exclusive(
    int idx, int warp_id, int lane_id, T* smem, bool value) {
    // cumsum within 1D thread block, which adds up `value` of all threads
    // whose id is *less than* the current thread

    // cumsum within warp
    unsigned int warp_bits = __ballot(value);
    T warp_sum = __popc(((1<<lane_id)-1) & warp_bits);

    if (lane_id == 0)
        smem[warp_id] = __popc(warp_bits);

    __syncthreads();

    // cumsum across warps in one thread
    if (idx == 0) {
        int current = 0;
        for (int i = 0; i < blockDim.x / 32; ++i) {
            T v = smem[i];
            smem[i] = smem[i]+current;
            current = current+v;
        }
    }

    __syncthreads();

    // load the carry from the preceding warp
    if (warp_id >= 1)
        warp_sum += smem[warp_id - 1];

    return warp_sum;
}

// apply raw(byte) offset to pointer
template <typename T>
static __device__ inline T* ptr_add(T *ptr, ga_ssize offset) {
    return (T*)((char*)ptr + offset);
}

// get array element using raw(byte) offset
template <typename T>
static __device__ inline T& ptr_at(T *ptr, ga_ssize offset) {
    return *((T*)((char*)ptr + offset));
}

KERNEL void k_topk_dense(
        $dims
        // ga_size dims_1, ga_ssize dims_2, ... , dims_$${NDIM}
        $dstv
        // INPUT_TYPE *dstv
        $dstv_strides
        // ga_ssize dstv_strides_0, ga_ssize dstv_strides_1, ... , dstv_strides_$${NDIM}
        $dsti
        // INDEX_TYPE *dsti
        $dsti_strides
        // ga_ssize dsti_strides_0, ga_ssize dsti_strides_1, ... , dsti_strides_$${NDIM}
        ga_ssize k,
        INPUT_TYPE* src,
        $src_strides
        // ga_ssize src_strides_0, ga_ssize src_strides_1, ... , src_strides_$${NDIM}
        size_t size) {
    /*
    extern __shared__ radix_t smem[];
    ga_ssize __shared__ bins[RADIX_SIZE]; // TODO: does using 32-bit gives speedup?
    bool is_topk = true;
    bool is_topkth = true; // exactly k-th largest
    radix_t out_idx;

    const size_t idx = threadIdx.x;
    size_t __shared__ k2, exceed;
    const ga_uint warp_id = idx / 32;
    const ga_uint lane_id = idx % 32;
    radix_t *wmem = (radix_t*)(smem) + warp_id * 32;
    const bool in_range = (idx < size);
    is_topk &= in_range;
    const INPUT_TYPE xval = in_range ? ptr_at(src, idx*src_strides_0) : (INPUT_TYPE)0;
    radix_t x = in_range ? RadixConfig<INPUT_TYPE>::convert(xval) : 0;

    // resolve negative k
    if (k<0) { x = ~x; k = -k; }
    if (idx==0) k2 = k;

    // 0. get the slice for thread block to work on
    size_t gid = blockIdx.x, gidx;
    $set_slice
    //for(int i=0; i<NDIM; i++) {
        //gidx = gid % dims_$${i};
        //gid /= dims_$${i};
        //dsti = ptr_add(dsti, gidx*dsti_strides_$${i+1};
        //dstv = ptr_add(dstv, gidx*dstv_strides_$${i+1};
        //src = ptr_add(src, gidx*src_strides_$${i+1});
    //}

    // 1. filter is_topk and is_topkth using radix select

    #pragma unroll
    for (int i=bitsof(INPUT_TYPE)-RADIX_BITS; i>=0; i-=RADIX_BITS) {
        smem[idx] = 0;
        int digit = (x>>i) & (RADIX_SIZE-1);
        // count within warp
        #pragma unroll
        for (int bin=0; bin<RADIX_SIZE; ++bin) {
            bool incr_bin = (bin == digit) && is_topkth && in_range;
            unsigned int incr_bin_warp = __ballot(incr_bin);
            if (lane_id==0)
                wmem[bin] += __popc(incr_bin_warp);
        }
        __syncthreads();
        // sum counts across all warps
        // TODO: test in-block parallel sum?
        if (idx < RADIX_SIZE) {
            for(int w=32; w<blockDim.x; w+=32)
                smem[idx] += smem[idx + w];
        }
        __syncthreads();

        // calculate k minus cumsum(count)
        if (idx<RADIX_SIZE)
            bins[idx] = 0;
        if (idx == 0) {
            exceed = k; // how many the number of is_topk exceeds k
            bins[RADIX_SIZE-1] = k2 - smem[RADIX_SIZE-1];
            if (bins[RADIX_SIZE-1] > 0)
                k2 = bins[RADIX_SIZE-1];
            else
                exceed = min(exceed, bins[RADIX_SIZE-1]);
            #pragma unroll
            for(int bin=RADIX_SIZE-1; bin; --bin) {
                bins[bin-1] = bins[bin] - smem[bin-1];
                if (bins[bin-1] > 0)
                    k2 = bins[bin-1];
                else
                    exceed = min(exceed, bins[bin-1]);
            }
        }
        __syncthreads();


        // smem -> count
        // bins -> k2 - cumsum(count)
        if (is_topk && is_topkth) {
            ga_ssize icount = bins[digit];
            if (icount > 0) {
                is_topkth = false;
            } else if (icount < 0) {
                if (digit+1!=RADIX_SIZE) {
                    if (bins[digit+1] <= 0) {
                        is_topk = false;
                        is_topkth = false;
                    }
                }
            }
        }
    }

    // 2. find the index of output array, if exists

    if (exceed != 0) {
        // top_kth value may not be unique, so we need to
        // perform binary cumsum on is_topkth to drop exceeding top-kth values
        out_idx = binary_cumsum_exclusive<radix_t>(idx, warp_id, lane_id, smem, is_topkth);
        is_topk &= (out_idx < exceed);
    }

    // perform binary cumsum on is_topk to determine the indices to put result
    out_idx = binary_cumsum_exclusive<radix_t>(idx, warp_id, lane_id, smem, is_topk);
    __syncthreads();

    if (is_topk) {
        $write_value;
        // ptr_at(dstv, out_idx * dstv_strides_0) = xval;
        $write_index;
        // ptr_at(dsti, out_idx * dsti_strides_0) = (INDEX_TYPE)idx;
    }
    */
}
