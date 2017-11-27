// modified from pytorch
// https://github.com/pytorch/pytorch/master/blob/torch/lib/THC/THCTensorTopK.cuh
// original license below:
/*
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/


#if __CUDA_ARCH__ < 350
#define __ldg(ptr) (*(ptr))
#endif


typedef ptrdiff_t ssize_t;


__device__ __forceinline__ int lane_id() {
  int id;
  asm("mov.s32 %0, %laneid;" : "=r"(id) );
  return id;
}

__device__ __forceinline__ unsigned lane_mask_lt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned lane_mask_le() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned lane_mask_gt() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_gt;" : "=r"(mask));
  return mask;
}

__device__ __forceinline__ unsigned lane_mask_ge() {
  unsigned mask;
  asm("mov.u32 %0, %%lanemask_ge;" : "=r"(mask));
  return mask;
}

template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
  static __device__ __forceinline__
  unsigned int get(unsigned int val, int pos, int len) {
    unsigned int ret;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
    return ret;
  }

  static __device__ __forceinline__
  unsigned int set(unsigned int val, unsigned int toInsert, int pos, int len) {
    unsigned int ret;
    asm("bfi.b32 %0, %1, %2, %3, %4;" :
        "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
    return ret;
  }
};

template <>
struct Bitfield<unsigned long long int> {
  static __device__ __forceinline__
  unsigned long long int get(unsigned long long int val, int pos, int len) {
    unsigned long long int ret;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
    return ret;
  }

  static __device__ __forceinline__
  unsigned long long int set(unsigned long long int val, unsigned long long int toInsert, int pos, int len) {
    unsigned long long int ret;
    asm("bfi.b64 %0, %1, %2, %3, %4;" :
        "=l"(ret) : "l"(toInsert), "l"(val), "r"(pos), "r"(len));
    return ret;
  }
};


template <typename T>
struct RadixConfig {
// Converts a type (maybe float) to an integer representation with the same
// sorting; i.e., for floats f1, f2:
// if f1 < f2 then convert(f1) < convert(f2)
// We use this to enable radix selection of floating-point values.
// This also gives a relative order for NaNs, but that's ok, as they
// will all be adjacent
  typedef unsigned int RadixType;
  static inline __device__ RadixType convert(T v) {
      return (RadixType)v;
  }

  static inline __device__ float deconvert(RadixType v) {
      return (T)v;
  }
};

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
struct RadixConfig<double> {
  typedef unsigned long long RadixType;

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
struct RadixConfig<char> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(char v) {
    return 128u + v;
  }

  static inline __device__ char deconvert(RadixType v) {
    return v - 128;
  }
};

// g++ makes difference between 'signed char' (ga_byte, int8) and 'char'.
// Same code as for char.
template <>
struct RadixConfig<ga_byte> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(ga_byte v) {
    return 128u + v;
  }

  static inline __device__ ga_byte deconvert(RadixType v) {
    return v - 128;
  }
};

template <>
struct RadixConfig<short> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(short v) {
    assert(sizeof(short) == 2);
    return 32768u ^ v;
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
struct RadixConfig<long long> {
  typedef unsigned long long RadixType;

  static inline __device__ RadixType convert(long long v) {
    assert(sizeof(long long) == 8);
    return 9223372036854775808ull + v;
  }

  static inline __device__ long long deconvert(RadixType v) {
    return v - 9223372036854775808ull;
  }
};

/* NB: This specialization for ga_half does know that ga_half is a struct with only one member of type ga_ushort.
 * So, if ga_half implementation changes, this code should change too.
 * TODO: Maybe should gpuarray provide abstract functions to manipulate ga_half internal structure? e.g:
 *   unsigned short ga_half2bits(ga_half value);
 *   ga_half ga_bits2half(unsigned short bits);
 */
template <>
struct RadixConfig<ga_half> {
  typedef unsigned int RadixType;

  static inline __device__ RadixType convert(ga_half v) {
    RadixType mask = -(((RadixType)v.data >> 15)) | 0x8000;
    return (v.data ^ mask);
  }

  static inline __device__ ga_half deconvert(RadixType v) {
    RadixType mask = ((v >> 15) - 1) | 0x8000;
    ga_half out = {(unsigned short)(v ^ mask)};
    return out;
  }
};

// $$inp_t should be replaced in c_code
// we cannot use templated kernel because gpuarray API does not support it
#define NDIM            $ndim
#define INPUT_TYPE      $inp_t
#define INDEX_TYPE      $out_t
#define bitsof(T)       (sizeof(T)*8)
#define radix_t         RadixConfig<INPUT_TYPE>::RadixType
#define WRITE_VALUE     $write_value
#define WRITE_INDEX     $write_index

#if RADIX_SIZE > 32
#error "RADIX_SIZE must be smaller than warp size (32)"
#endif

void __device__ atomicAdd(long long *dst, long long &src) {
    atomicAdd(
        reinterpret_cast<unsigned long long*>(dst),
        reinterpret_cast<unsigned long long&>(src));
}

template <typename T>
static inline __device__ T binary_cumsum(
    int idx, int warp_id, T* smem, bool value) {
    // cumsum within 1D thread block, which adds up `value` of all threads
    // whose id is *no greater than* the current thread
    // binary_cumsum(1, 0, 1, 0, 1) -> (1, 1, 2, 2, 3)

    // cumsum within warp
    unsigned int warp_bits = __ballot(value);
    T warp_sum = __popc(lane_mask_le() & warp_bits);

    if (lane_id() == 0)
        smem[warp_id] = __popc(warp_bits);

    local_barrier();

    // cumsum across warps in one thread
    if (idx == 0) {
        T sum = smem[0];
        for (int i = 1; i < blockDim.x / GA_WARP_SIZE; ++i) {
            sum += smem[i];
            smem[i] = sum;
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
    int idx, int warp_id, T* smem, bool value) {
    // cumsum within 1D thread block, which adds up `value` of all threads
    // whose id is *less than* the current thread
    // binary_cumsum_excl(1, 0, 1, 0, 1) -> (0, 1, 1, 2, 2)

    // cumsum within warp
    unsigned int warp_bits = __ballot(value);
    T warp_sum = __popc(lane_mask_lt() & warp_bits);

    if (lane_id() == 0)
        smem[warp_id] = __popc(warp_bits);

    local_barrier();

    // cumsum across warps in one thread
    if (idx == 0) {
        T sum = smem[0];
        for (int i = 1; i < blockDim.x / GA_WARP_SIZE; ++i) {
            sum += smem[i];
            smem[i] = sum;
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
static __device__ inline T* ptr_add(T *ptr, ssize_t offset) {
    return (T*)((char*)ptr + offset);
}

// get array element using raw(byte) offset
template <typename T>
static __device__ inline T& ptr_at(T *ptr, ssize_t offset) {
    return *((T*)((char*)ptr + offset));
}

// read array element using raw(byte) offset
template <typename T>
static __device__ inline T ptr_read_cached(T *ptr, ssize_t offset) {
    return __ldg(((T*)((char*)ptr + offset)));
}

/* NB: __ldg is not defined for ga_half, so we must specialize ptr_read_cached.
 * To do it, I try to use a built-in type that should have the same size as ga_half.
 * Based on current ga_half implementation (2017/11/27), it should be ga_ushort.
 * This code must be updated every time ga_half implementation size changes,
 * until a better code be provided. */
#define GA_HALF_STD_TYPE ga_ushort
static __device__ inline ga_half ptr_read_cached(ga_half *ptr, ssize_t offset) {

    int check_ga_half_std_type[ ( ( sizeof(GA_HALF_STD_TYPE) - sizeof(ga_half) ) ? -1 : 1 ) ];

    GA_HALF_STD_TYPE out = __ldg(((GA_HALF_STD_TYPE*)((char*)ptr + offset)));
    ga_half real_out;
    *(GA_HALF_STD_TYPE*)(&real_out) = out;
    return real_out;

}
#undef GA_HALF_STD_TYPE

/* Comparisons involving ga_half and conversions from integers (e.g. 0, 1) to ga_half lead to compilation errors.
 * Following functions are provided to bypass these issues. */

template<typename T>
static __device__ inline T theano_zero() {return 0;}
template<>
__device__ inline ga_half theano_zero() {return ga_float2half(0);}

template<typename T>
static __device__ inline T theano_one() {return 1;}
template<>
__device__ inline ga_half theano_one() {return ga_float2half(1);}

template<typename A, typename B> static __device__ inline bool theano_eq(const A& a, const B& b) {return a == b;}
template<typename A, typename B> static __device__ inline bool theano_ne(const A& a, const B& b) {return a != b;}
template<typename A, typename B> static __device__ inline bool theano_lt(const A& a, const B& b) {return a < b;}
template<typename A, typename B> static __device__ inline bool theano_gt(const A& a, const B& b) {return a > b;}
template<typename A, typename B> static __device__ inline bool theano_le(const A& a, const B& b) {return a <= b;}
template<typename A, typename B> static __device__ inline bool theano_ge(const A& a, const B& b) {return a >= b;}

template<typename T> static __device__ inline bool theano_eq(const ga_half& a, const T& b) {return ga_half2float(a) == b;}
template<typename T> static __device__ inline bool theano_ne(const ga_half& a, const T& b) {return ga_half2float(a) != b;}
template<typename T> static __device__ inline bool theano_lt(const ga_half& a, const T& b) {return ga_half2float(a) < b;}
template<typename T> static __device__ inline bool theano_gt(const ga_half& a, const T& b) {return ga_half2float(a) > b;}
template<typename T> static __device__ inline bool theano_le(const ga_half& a, const T& b) {return ga_half2float(a) <= b;}
template<typename T> static __device__ inline bool theano_ge(const ga_half& a, const T& b) {return ga_half2float(a) >= b;}

template<typename T> static __device__ inline bool theano_eq(const T& a, const ga_half& b) {return a == ga_half2float(b);}
template<typename T> static __device__ inline bool theano_ne(const T& a, const ga_half& b) {return a != ga_half2float(b);}
template<typename T> static __device__ inline bool theano_lt(const T& a, const ga_half& b) {return a < ga_half2float(b);}
template<typename T> static __device__ inline bool theano_gt(const T& a, const ga_half& b) {return a > ga_half2float(b);}
template<typename T> static __device__ inline bool theano_le(const T& a, const ga_half& b) {return a <= ga_half2float(b);}
template<typename T> static __device__ inline bool theano_ge(const T& a, const ga_half& b) {return a >= ga_half2float(b);}

static __device__ inline bool theano_eq(const ga_half& a, const ga_half& b) {return ga_half2float(a) == ga_half2float(b);}
static __device__ inline bool theano_ne(const ga_half& a, const ga_half& b) {return ga_half2float(a) != ga_half2float(b);}
static __device__ inline bool theano_lt(const ga_half& a, const ga_half& b) {return ga_half2float(a) < ga_half2float(b);}
static __device__ inline bool theano_gt(const ga_half& a, const ga_half& b) {return ga_half2float(a) > ga_half2float(b);}
static __device__ inline bool theano_le(const ga_half& a, const ga_half& b) {return ga_half2float(a) <= ga_half2float(b);}
static __device__ inline bool theano_ge(const ga_half& a, const ga_half& b) {return ga_half2float(a) >= ga_half2float(b);}
