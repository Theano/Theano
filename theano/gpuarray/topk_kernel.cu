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

#define bitsof(T)       (sizeof(T)*8)
#define RADIX_BITS      2
#define RADIX_SIZE      (1<<RADIX_BITS)
#define RADIX_MASK(n)   ((RADIX_SIZE-1) << (n*RADIX_BITS))
#define RADIX_DIGITS(T) (bitsof(T)/RADIX_BITS)
#define radix_t         RadixConfig<T>::RadixType

#if RADIX_SIZE > 32
#error "RADIX_SIZE must be smaller than warp size (32)"
#endif

template <typename T>
inline __device__ T binary_cumsum(int idx, int warp_id, int lane_id, T* smem, bool value) {
    // cumsum within 1D thread block, which adds up `value` of all threads whose id is *no greater than* the current thread
    // cumsum within warp
    unsigned int warp_bits = __ballot(in);
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
    if (warp >= 1) {
        warp_sum = warp_sum+smem[warp - 1];
    }

    return warp_sum;
}

template <typename T>
inline __device__ T binary_cumsum_exclusive(
    int idx, int warp_id, int lane_id, T* smem, bool value) {
    // cumsum within 1D thread block, which adds up `value` of all threads
    // whose id is *less than* the current thread
    // cumsum within warp
    unsigned int warp_bits = __ballot(in);
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
    if (warp >= 1) {
        warp_sum = warp_sum+smem[warp - 1];
    }

    return warp_sum;
}


template <typename T>
void __global__ topk_1d_contig_kernel(T* dst, T* src, size_t size, size_t k) {
    extern radix_t smem[];
    ssize_t bins[RADIX_SIZE]; // TODO: does using 32-bit gives speedup?
    bool is_topk = true;
    bool is_topkth = true; // exactly k-th largest

    size_t idx = threadIdx.x;
    size_t k2 = k, exceed;
    int warp_id = idx / 32;
    int lane_id = idx % 32;
    radix_t wmem = smem + warp_id * 32;
    bool in_range = (idx < size);
    RadixConfig<T>::RadixType x = in_range ? RadixConfig<T>::convert(src[idx]) : 0;
    // 1. find the kth largest value using radix select

    // 1.1 for each radix mask, count
    smem[threadIdx.x] = 0;
    #pragma unroll
    for (int i=bitsof(T)-RADIX_BITS; i; i-=RADIX_BITS) {
        radix_t mask = (RADIX_SIZE-1)<<i;
        int digit = (x>>i) & (RADIX_SIZE-1);
        // count within warp
        #pragma unroll
        for (int bin=0; bin<RADIX_SIZE; ++bin) {
            bool incr_bin = (bin == digit) && is_topkth && in_range;
            unsigned int incr_bin_warp = __ballot(incr_bin);
            if (lane_id==0)
                wmem[bin] += __popc(bin_warp);
        }
        __syncthreads();
        // sum counts across all warps
        // TODO: test in-block parallel sum?
        if (idx<RADIX_SIZE)
            bins[idx] = 0;
        if (idx==0) {
            for(int w=1; w<blockDim.x/32; ++w) {
                #pragma unroll
                for(int bin=0; bin<RADIX_SIZE; ++bin) {
                    smem[bin] += wmem[bin];
                }
            }
        }
        __syncthreads();

        // broadcast sum result
        if (idx < RADIX_SIZE)
            smem[idx] = bins[idx];
        __syncthreads();

        // calculate k minus cumsum(count)
        exceed = -k; // how many the number of is_topk exceeds k
        if (idx == 0) {
            bins[0] = k2 - smem[0];
            if (bins[0] > 0)
                k2 = bins[0];
            else if (bins[0] < 0)
                exceed = max(exceed, bins[0]);
            #pragma unroll
            for(int bin=1; bin<RADIX_SIZE; ++bin) {
                bins[bin] = bins[bin-1] - smem[bin];
                if (bins[bin] > 0)
                    k2 = bins[bin];
                else if (bins[bin] < 0)
                    exceed = max(exceed, bins[bin]);
            }
        }
        __syncthreads();


        // smem -> count
        // bins -> k2 - cumsum(count)
        if (is_topk && is_topkth) {
            ssize_t icount = bins[digit];
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
    //
    // top_kth value may not be unique, so we need to
    // count how many is needed

    // perform binary cumsum on is_topkth to drop exceeding top-kth values
    radix_t topkth_idx = binary_cumsum_exclusive<radix_t>(idx, warp_id, lane_id, smem, is_topkth);
    if (topkth_idx >= exceed)
        is_topk = false;

    // perform binary cumsum on is_topk to determine idx to put result
    topkth_idx = binary_cumsum_exclusive<radix_t>(idx, warp_id, lane_id, smem, is_topkth);
    if (is_topk)
        dst[topkth_idx] = idx;
}
