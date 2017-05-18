// modified from pytorch
// https://github.com/pytorch/pytorch/master/blob/torch/lib/THC/THCTensorTopK.cuh
//
// Converts a type (maybe float) to an integer representation with the same
// sorting; i.e., for floats f1, f2:
// if f1 < f2 then convert(f1) < convert(f2)
// We use this to enable radix selection of floating-point values.
// This also gives a relative order for NaNs, but that's ok, as they
// will all be adjacent

#if __CUDA_ARCH__ < 350
#define __ldg(ptr) (*(ptr))
#endif


template <typename T>
struct RadixConfig {
    typedef T RadixType;
  static inline __device__ RadixType convert(T v) {
      return v;
  }

  static inline __device__ float deconvert(RadixType v) {
      return v;
  }
};

template <>
struct RadixConfig<ga_float> {
  typedef ga_uint RadixType;

  static inline __device__ RadixType convert(ga_float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (x ^ mask);
  }

  static inline __device__ ga_float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

    return __int_as_float(v ^ mask);
  }
};

template <>
struct RadixConfig<ga_double> {
  typedef unsigned long long int RadixType;

  static inline __device__ RadixType convert(ga_double v) {
    RadixType x = __double_as_longlong(v);
    RadixType mask = -((x >> 63)) | 0x8000000000000000;
    return (x ^ mask);
  }

  static inline __device__ ga_double deconvert(RadixType v) {
    RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
    return __longlong_as_double(v ^ mask);
  }
};


template <>
struct RadixConfig<ga_ubyte> {
  typedef ga_uint RadixType;

  static inline __device__ RadixType convert(ga_ubyte v) {
    return v;
  }

  static inline __device__ ga_ubyte deconvert(RadixType v) {
    return v;
  }
};

template <>
struct RadixConfig<ga_byte> {
  typedef ga_uint RadixType;

  static inline __device__ RadixType convert(ga_byte v) {
    return 128u + v;
  }

  static inline __device__ ga_byte deconvert(RadixType v) {
    return v - 128;
  }
};

template <>
struct RadixConfig<ga_short> {
  typedef ga_uint RadixType;

  static inline __device__ RadixType convert(ga_short v) {
    assert(sizeof(ga_short) == 2);
    return 32768u ^ v;
  }

  static inline __device__ ga_short deconvert(RadixType v) {
    return v - 32768;
  }
};

template <>
struct RadixConfig<int> {
  typedef ga_uint RadixType;

  static inline __device__ RadixType convert(int v) {
    assert(sizeof(int) == 4);
    return (1u << 31) ^ v;
  }

  static inline __device__ int deconvert(RadixType v) {
    return (1u << 31) ^ v;
  }
};

template <>
struct RadixConfig<ga_long> {
  typedef unsigned long long int RadixType;

  static inline __device__ RadixType convert(ga_long v) {
    assert(sizeof(long) == 8);
    return (1ull << 63) ^ v;
  }

  static inline __device__ ga_long deconvert(RadixType v) {
    return (1ull << 63) ^ v;
  }
};

#ifdef USE_HALF
// TODO: make this work
template <>
struct RadixConfig<half> {
  typedef ga_uint RadixType;

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
// we cannot use templated kernel because gpuarray API does not support it
#define NDIM            $ndim
#define INPUT_TYPE      $inp_t
#define INDEX_TYPE      $out_t
#define bitsof(T)       (sizeof(T)*8)
#define RADIX_BITS      2
#define RADIX_SIZE      (1<<RADIX_BITS)
#define RADIX_MASK(n)   ((RADIX_SIZE-1) << (n*RADIX_BITS))
#define RADIX_DIGITS(T) (bitsof(T)/RADIX_BITS)
#define radix_t         RadixConfig<INPUT_TYPE>::RadixType
#define WRITE_VALUE     $write_value
#define WRITE_INDEX     $write_index

#if RADIX_SIZE > 32
#error "RADIX_SIZE must be smaller than warp size (32)"
#endif

template <typename T>
static inline __device__ T binary_cumsum(
        int idx, int warp_id, int lane_id, T* smem, bool value) {
    // cumsum within 1D thread block, which adds up `value` of all threads 
    // whose id is *no greater than* the current thread
    // binary_cumsum(1, 0, 1, 0, 1) -> (1, 1, 2, 2, 3)

    // cumsum within warp
    ga_uint warp_bits = __ballot(value);
    T warp_sum = __popc(((2<<lane_id)-1) & warp_bits);

    if (lane_id == 0)
        smem[warp_id] = __popc(warp_bits);

    local_barrier();

    // cumsum across warps in one thread
    if (idx == 0) {
        int current = 0;
        for (int i = 0; i < LDIM_0 / GA_WARP_SIZE; ++i) {
            T v = smem[i];
            smem[i] = smem[i]+current;
            current = current+v;
        }
    }

    local_barrier();

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
    // binary_cumsum(1, 0, 1, 0, 1) -> (0, 1, 1, 2, 2)

    // cumsum within warp
    ga_uint warp_bits = __ballot(value);
    T warp_sum = __popc(((1<<lane_id)-1) & warp_bits);

    if (lane_id == 0)
        smem[warp_id] = __popc(warp_bits);

    local_barrier();

    // cumsum across warps in one thread
    if (idx == 0) {
        int current = 0;
        for (int i = 0; i < LDIM_0 / GA_WARP_SIZE; ++i) {
            T v = smem[i];
            smem[i] = smem[i]+current;
            current = current+v;
        }
    }

    local_barrier();

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

// read array element using raw(byte) offset
template <typename T>
static __device__ inline T ptr_read(T *ptr, ga_ssize offset) {
    return __ldg(((T*)((char*)ptr + offset)));
}

