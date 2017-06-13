#define RADIX_BITS 2
#define RADIX_SIZE      (1<<RADIX_BITS)
#define RADIX_MASK(n)   ((RADIX_SIZE-1) << (n*RADIX_BITS))
#define RADIX_DIGITS(T) (bitsof(T)/RADIX_BITS)

// works when length on axis is in [1025, 2^31-1]
KERNEL void k_topk_dense_large(
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
        ga_size size, ga_ushort inp_per_thread) {
    LOCAL_MEM ga_int smem[32];
    LOCAL_MEM radix_t known_bits;
    LOCAL_MEM ga_uint k2;
    int counts[RADIX_SIZE];
    unsigned out_idx;
    INPUT_TYPE xval;
    radix_t x;
    bool in_range, is_topk;

    const ga_uint idx = LID_0;
    const ga_uint inp_idx = idx * inp_per_thread;
    const ga_int warp_id = idx / GA_WARP_SIZE;

    // 0. get the slice for thread block to work on
    // TODO if ndim <= 3, use native indexing ? (blockIdx.[xyz])
    ga_uint gid = GID_0, gidx;
    $set_slice
    //for(int i=1; i<NDIM; i++) {
        // gidx = gid % dims_$${i};
        // gid /= dims_$${i};
        // dsti = ptr_add(dsti, gidx*dsti_strides_$${i};
        // dstv = ptr_add(dstv, gidx*dstv_strides_$${i};
        // src = ptr_add(src, gidx*src_strides_$${i});
    //}
    src = ptr_add(src, idx*inp_per_thread*src_strides_0);

    if (idx==0) {
        known_bits = 0;
        k2 = (k>=0) ? k : -k;
    }
    const radix_t inv_bits = (k>=0) ? 0 : ~0;
    if (k<0) { k = -k; }

    local_barrier();

    // 1. find bits of top-k-th value using radix select
    #pragma unroll
    for (int i=bitsof(INPUT_TYPE)-RADIX_BITS; i>=0; i-=RADIX_BITS) {
        #pragma unroll
        for (int j=0; j<RADIX_SIZE; ++j)
            counts[j] = 0;
        if (warp_id == 0)
            smem[idx] = 0;
        local_barrier();

        // count within warp
        for (int j=0; j<inp_per_thread; ++j) {
            in_range = (inp_idx+j) < size;
            xval = in_range ? ptr_read(src, j*src_strides_0) : (INPUT_TYPE)0;
            x = inv_bits^RadixConfig<INPUT_TYPE>::convert(xval);
            ga_int digit = (int)((x>>i) & (RADIX_SIZE-1));

            #pragma unroll
            for (int bin=0; bin<RADIX_SIZE; ++bin) {
                bool incr_bin = (
                    (bin == digit) &&
                    ((x >> (i+RADIX_BITS)) == known_bits) && in_range);
                counts[bin] += __popc(__ballot(incr_bin));
            }
        }
        local_barrier();

        // sum counts across all warps
        if (lane_id() < RADIX_SIZE) {
            atomicAdd(&smem[lane_id()], counts[lane_id()]);
        }
        local_barrier();

        // update known bits
        if (idx==0) {
            #pragma unroll
            for (int bin=RADIX_SIZE-1; bin>=0; --bin) {
                if (smem[bin] >= k2) {
                    known_bits = (known_bits << RADIX_BITS) | bin;
                    break;
                } else
                    k2 -= smem[bin];
            }
        }
        local_barrier();
    }

    // now we use k2 for base index to write output
    if (idx == 0)
        k2 = 0;
    local_barrier();

    // 2. write values smaller than top-kth
    for (int i=0; i<inp_per_thread; ++i) {
        in_range = (inp_idx+i) < size;
        xval = in_range ? ptr_read(src, i*src_strides_0) : (INPUT_TYPE)0;
        x = inv_bits ^ RadixConfig<INPUT_TYPE>::convert(xval);
        is_topk = (x > known_bits) && in_range;
        out_idx = binary_cumsum(idx, warp_id, smem, is_topk);
        if (is_topk) {
#if WRITE_VALUE == 1
            ptr_at(dstv, (out_idx+k2-1) * dstv_strides_0) = xval;
#endif
#if WRITE_INDEX == 1
            ptr_at(dsti, (out_idx+k2-1) * dsti_strides_0) = (INDEX_TYPE)(idx*inp_per_thread + i);
#endif
        }
        local_barrier();

        if (idx == blockDim.x - 1)
            k2 += out_idx;
        local_barrier();
    }
    // 3. write values equal to top-kth
    for (int i=0; i<inp_per_thread; ++i) {
        in_range = (inp_idx+i) < size;
        xval = in_range ? ptr_read(src, i*src_strides_0) : (INPUT_TYPE)0;
        x = inv_bits ^ RadixConfig<INPUT_TYPE>::convert(xval);
        is_topk = (x == known_bits) && in_range;
        out_idx = binary_cumsum(idx, warp_id, smem, is_topk);
        is_topk &= (out_idx+k2) <= k;
        if (is_topk) {
#if WRITE_VALUE == 1
        ptr_at(dstv, (out_idx+k2-1) * dstv_strides_0) = xval;
#endif
#if WRITE_INDEX == 1
        ptr_at(dsti, (out_idx+k2-1) * dsti_strides_0) = (INDEX_TYPE)(inp_idx+ i);
#endif
        }
        local_barrier();

        if (idx == blockDim.x - 1)
            k2 += out_idx;
        local_barrier();

        if(k2 >= k)
            break;
    }
}

