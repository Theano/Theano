#define RADIX_BITS 2
#define RADIX_SIZE      (1<<RADIX_BITS)
#define RADIX_DIGITS(T) (bitsof(T)/RADIX_BITS)

#define COUNT_TYPE $count_t
#define KERNEL_NAME $kname

// if count_t is int, work for array size within [1025, 2^31-1]
// if count_t is long long, work for array size within [2^31, 2^63-1]
template <typename DataType, typename RadixType, typename CountType>
__device__ DataType find_pattern(DataType* smem,
                             DataType* data,
                             CountType slice_size,
                             CountType stride,
                             RadixType known_bits,
                             RadixType known_bits_mask) {
    if (threadIdx.x < 32)
        smem[threadIdx.x] = theano_zero<DataType>();

    local_barrier();

    // All threads participate in the loop, in order to sync on the flag
    for (CountType i = threadIdx.x; i < (slice_size + (CountType)blockDim.x-1); i += blockDim.x) {
        bool in_range = (i < slice_size);
        DataType v = in_range ? ptr_read_cached(data, i*stride) : theano_zero<DataType>();

        if (in_range && ((RadixConfig<DataType>::convert(v) & known_bits_mask) == known_bits)) {
            // There should not be conflicts if we are using find_pattern,
            // since the result is unique
            smem[0] = theano_one<DataType>();
            smem[1] = v; // can't use val as the flag, since it could be 0
        }

        local_barrier();

        DataType found = smem[0];
        DataType val = smem[1];

        local_barrier();

        // Check to see if a thread found the value
        if (theano_ne(found, 0))
            return val;
    }
    return theano_zero<DataType>();
}

// This function counts the distribution of all input values in a
// slice we are selecting by radix digit at `radix_digit_pos`, but only
// those that pass the filter `((v & known_bits_mask) == known_bits)`.
// This produces and broadcasts the seen counts for a single block only.
// `smem` must have at least `RADIX_SIZE` elements.
template <typename DataType, typename RadixType, typename CountType>
__device__ void count_radix_masked(CountType counts[RADIX_SIZE],
                                    CountType* smem,
                                    RadixType known_bits,
                                    RadixType known_bits_mask,
                                    int radix_digit_pos,
                                    CountType slice_size,
                                    CountType stride,
                                    DataType* data) {
    // Clear out per-thread counts from a previous round
#pragma unroll
    for (int i = 0; i < RADIX_SIZE; ++i)
        counts[i] = 0;

    if (threadIdx.x < RADIX_SIZE)
        smem[threadIdx.x] = 0;

    local_barrier();

    // Scan over all the data. Upon a read, the warp will accumulate
    // counts per each digit in the radix using warp voting.
    for (CountType i = threadIdx.x; i < slice_size; i += blockDim.x) {
        RadixType val = RadixConfig<DataType>::convert(ptr_read_cached(data, i*stride));

        bool has_val = ((val & known_bits_mask) == known_bits);
        RadixType digit_in_radix = Bitfield<RadixType>::get(val, radix_digit_pos, RADIX_BITS);

        #pragma unroll
        for (int j = 0; j < RADIX_SIZE; ++j) {
            bool vote = has_val && (digit_in_radix == j);
            counts[j] += __popc(__ballot(vote));
        }
    }

    // Now, for each warp, sum values
    if (lane_id() == 0) {
        for (int i=0; i<RADIX_SIZE; ++i)
            atomicAdd(&smem[i], counts[i]);
    }
    /*
    // not sure why, but this just give wrong results
    if (lane_id() < RADIX_SIZE)
        atomicAdd(&smem[lane_id()], counts[lane_id()]);
        */

    local_barrier();

    // For each thread, read in the total counts
    #pragma unroll
    for (unsigned int i = 0; i < RADIX_SIZE; ++i)
        counts[i] = smem[i];

    local_barrier();
}

template <typename DataType, typename RadixType, typename CountType>
__device__ void radix_select(DataType* data,
                            CountType k,
                            bool order,
                            CountType slice_size,
                            CountType stride,
                            CountType* smem,
                            DataType* top_kth) {
    // Per-thread buckets into which we accumulate digit counts in our
    // radix
    register CountType counts[RADIX_SIZE];

    // We only consider elements x such that (x & known_bits_mask) == known_bits
    // Initially, we consider all elements of the array, so the above
    // statement is true regardless of input.
    RadixType known_bits = 0, known_bits_mask = 0;

    // We are looking for the top k_to_find-th element when iterating over
    // digits; this count gets reduced by elimination when counting
    // successive digits
    CountType k_to_find = abs(k);

    // We start at the most significant digit in our radix, scanning
    // through to the least significant digit
    #pragma unroll
    for (int digit_pos = bitsof(DataType) - RADIX_BITS;
            digit_pos >= 0; digit_pos -= RADIX_BITS) {

        // Count radix distribution for the current position and reduce
        // across all threads
        count_radix_masked<DataType, RadixType, CountType>(
                    counts, smem,
                    known_bits, known_bits_mask, digit_pos,
                    slice_size, stride, data);

        // All threads participate in the comparisons below to know the
        // final result

        #define CHECK_RADIX(i) \\
            int count = counts[i]; \\
            /* All threads have the same value in counts here, so all */  \\
            /* threads will return from the function. */  \\
            if (count == 1 && k_to_find == 1) {  \\
                /* There is a unique answer. */  \\
                known_bits = Bitfield<RadixType>::set(  \\
                    known_bits, i, digit_pos, RADIX_BITS);  \\
                known_bits_mask = Bitfield<RadixType>::set(  \\
                    known_bits_mask, RADIX_SIZE-1, digit_pos, RADIX_BITS);  \\
                /* The answer is now the unique element v such that: */  \\
                /* (v & known_bits_mask) == known_bits */  \\
                /* However, we do not yet know what the actual element is. We */  \\
                /* need to perform a search through the data to find the */  \\
                /* element that matches this pattern. */  \\
                *top_kth = find_pattern<DataType, RadixType, CountType>(  \\
                        (DataType*) smem, data, slice_size,  \\
                        stride, known_bits, known_bits_mask);  \\
                return;  \\
            }  \\
            if (count >= k_to_find) {  \\
                known_bits = Bitfield<RadixType>::set(known_bits, i, digit_pos, RADIX_BITS);  \\
                known_bits_mask = Bitfield<RadixType>::set(  \\
                    known_bits_mask, RADIX_SIZE-1, digit_pos, RADIX_BITS);  \\
                /* The top-Kth element v must now be one such that: */  \\
                /* (v & known_bits_mask == known_bits) */  \\
                /* but we haven't narrowed it down; we must check the next */  \\
                /* least-significant digit */  \\
                break;  \\
            }  \\
            k_to_find -= count

        if (order) {
            #pragma unroll
            for (int i=RADIX_SIZE - 1; i >= 0; --i) {
                CHECK_RADIX(i);
            }
        } else {
            #pragma unroll
            for (int i=0; i < RADIX_SIZE; ++i) {
                CHECK_RADIX(i);
            }
        }
        #undef CHECK_RADIX
    } // end digit_pos for

    // There is no unique result, but there is a non-unique result
    // matching `known_bits` exactly
    *top_kth = RadixConfig<DataType>::deconvert(known_bits);
}

extern "C" __global__ void KERNEL_NAME(
        $dims
        // size_t dims_1, ssize_t dims_2, ... , dims_$${NDIM}
        $dstv
        // INPUT_TYPE *dstv
        $dstv_offset
        // size_t offset
        $dstv_strides
        // ssize_t dstv_strides_0, ssize_t dstv_strides_1, ... , dstv_strides_$${NDIM}
        $dsti
        // INDEX_TYPE *dsti
        $dsti_offset
        // size_t offset
        $dsti_strides
        // ssize_t dsti_strides_0, ssize_t dsti_strides_1, ... , dsti_strides_$${NDIM}
        ssize_t k,
        INPUT_TYPE* src,
	size_t src_offset,
        $src_strides
        // ssize_t src_strides_0, ssize_t src_strides_1, ... , src_strides_$${NDIM}
        size_t size) {
    __shared__ COUNT_TYPE smem[32];
    INPUT_TYPE topkth_value;

    const bool order = (k>0);
    k = (order ? k : -k);
    const int idx = threadIdx.x;
    const int warp_id = idx / GA_WARP_SIZE;

    // get the slice for thread block to work on
    // size <- the axis to work on
    // dims_1+ <- batched dimensions
    unsigned int gid = blockIdx.x, gidx;
    $set_slice
    // $$set_slice expands into:
    //for(int i=1; i<NDIM; i++) {
        // gidx = gid % dims_$${i};
        // gid /= dims_$${i};
        // dsti = ptr_add(dsti, gidx*dsti_strides_$${i});
        // dstv = ptr_add(dstv, gidx*dstv_strides_$${i});
        // src = ptr_add(src, gidx*src_strides_$${i});
    //}

    radix_select<INPUT_TYPE, radix_t, COUNT_TYPE>(
        src, k, order, size, src_strides_0,
        smem, &topkth_value);

    // Every value that is strictly less/greater than `pattern`
    // (depending on sort dir) in sorted int format is in the top-K.
    // The top-K value itself might not be unique.
    //
    // Since there are a variable number of elements that we see that
    // are within the top-k, we don't know at what index to write out
    // the resulting values.
    // In order to get this, we perform an exclusive cumsum of
    // `has_topk`. This will return the resulting index into which we
    // need to write the result, if a thread has a result.

    // All threads need to participate in the loop and the cumsum
    // but not necessarily in the load; hence loop bounds being rounded
    // up to a multiple of the block dim.
    COUNT_TYPE iter_bound = size + blockDim.x-1;
    INDEX_TYPE write_base = 0;

    for (int i = idx; i < iter_bound; i += blockDim.x) {
        bool in_range = (i < size);
        INPUT_TYPE v = in_range ? ptr_read_cached(src, i*src_strides_0) : theano_zero<INPUT_TYPE>();
        bool has_topk;
        if (order) {
            has_topk = in_range && (theano_gt(v, topkth_value));
        } else {
            has_topk = in_range && (theano_lt(v, topkth_value));
        }

        int index = binary_cumsum_exclusive(idx, warp_id, smem, has_topk);
        int carry = smem[blockDim.x / 32 - 1];

        if (has_topk) {
            COUNT_TYPE write_idx = write_base + index;
#if WRITE_VALUE == 1
            ptr_at(dstv, write_idx * dstv_strides_0) = v;
#endif
#if WRITE_INDEX == 1
            ptr_at(dsti, write_idx * dsti_strides_0) = (INDEX_TYPE)i;
#endif
        }

        write_base += carry;
    }

    COUNT_TYPE topk_remaining = (k - write_base);

    for (COUNT_TYPE i = idx; i < iter_bound; i += blockDim.x) {
        bool in_range = (i < size);
        INPUT_TYPE v = in_range ? ptr_read_cached(src, i*src_strides_0) : theano_zero<INPUT_TYPE>();
        bool has_topk = in_range && (theano_eq(v, topkth_value));

        int index = binary_cumsum_exclusive(idx, warp_id, smem, has_topk);
        int carry = smem[blockDim.x / 32 - 1];

        if (has_topk && index < topk_remaining) {
            COUNT_TYPE write_idx = write_base + index;
#if WRITE_VALUE == 1
            ptr_at(dstv, write_idx * dstv_strides_0) = v;
#endif
#if WRITE_INDEX == 1
            ptr_at(dsti, write_idx * dsti_strides_0) = (INDEX_TYPE)i;
#endif
        }

        if (carry >= topk_remaining)
            break;

        topk_remaining -= carry;
        write_base += carry;
    }
}

