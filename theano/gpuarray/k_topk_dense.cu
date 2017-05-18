// works when length on axis is within max allowed threads in block (1024)
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
        ga_size size) {
    LOCAL_MEM radix_t smem[32 * RADIX_SIZE];
    ga_ssize LOCAL_MEM bins[RADIX_SIZE+1]; // TODO: does using 32-bit gives good speedup?
    bool is_topk=true, is_topkth=true;
    radix_t out_idx;

    const ga_ushort idx = LID_0;
    ga_size LOCAL_MEM k2, exceed;
    const ga_ubyte warp_id = idx / GA_WARP_SIZE;
    const ga_ubyte lane_id = idx % GA_WARP_SIZE;
    const bool in_range = (idx < size);
    is_topk &= in_range;


    // 0. get the slice for thread block to work on
    // TODO if ndim <= 3, use native indexing ? (blockIdx.[xyz])
    ga_size gid = GID_0, gidx;
    $set_slice
    //for(int i=1; i<NDIM; i++) {
        // gidx = gid % dims_$${i};
        // gid /= dims_$${i};
        // dsti = ptr_add(dsti, gidx*dsti_strides_$${i};
        // dstv = ptr_add(dstv, gidx*dstv_strides_$${i};
        // src = ptr_add(src, gidx*src_strides_$${i});
    //}

    // get input and its radix friendly form
    const INPUT_TYPE xval = in_range ? ptr_at(src, idx*src_strides_0) : (INPUT_TYPE)0;
    radix_t x = in_range ? RadixConfig<INPUT_TYPE>::convert(xval) : 0;

    // resolve negative k
    if (k<0) { x = ~x; k = -k; }
    if (idx==0) {
        k2 = k;
        bins[RADIX_SIZE] = 1;
    }

    // 1. filter is_topk and is_topkth using radix select

    #pragma unroll
    for (int i=bitsof(INPUT_TYPE)-RADIX_BITS; i>=0; i-=RADIX_BITS) {
        int digit = (x>>i) & (RADIX_SIZE-1);
        // count within warp
        #pragma unroll
        for (int bin=0; bin<RADIX_SIZE; ++bin) {
            bool incr_bin = (bin == digit) && is_topkth && in_range;
            ga_uint incr_bin_warp = __ballot(incr_bin);
            if (lane_id==0)
                smem[bin + RADIX_SIZE*warp_id] = __popc(incr_bin_warp);
        }
        local_barrier();
        // sum counts across all warps
        // TODO: test in-block parallel sum?
        if (idx < RADIX_SIZE) {
            for(int w=RADIX_SIZE; w<LDIM_0*RADIX_SIZE / GA_WARP_SIZE; w+=RADIX_SIZE)
                smem[idx] += smem[idx + w];
        }
        local_barrier();

        // bins = k - cumsum(smem[:RADIX_SIZE])
        if (idx == 0) {
            bins[RADIX_SIZE-1] = k2 - smem[RADIX_SIZE-1];
            if (bins[RADIX_SIZE-1] > 0)
                k2 = bins[RADIX_SIZE-1];
            #pragma unroll
            for (int bin=RADIX_SIZE-1; bin; --bin) {
                bins[bin-1] = bins[bin] - smem[bin-1];
                if (bins[bin-1] > 0)
                    k2 = bins[bin-1];
            }
        }
        local_barrier();


        // smem -> count
        // bins -> k2 - cumsum(count)
        if (is_topk && is_topkth) {
            ga_ssize icount = bins[digit];
            if (icount > 0) {
                is_topkth = false;
            } else if (bins[digit+1] <= 0) {
                is_topk = false;
                is_topkth = false;
            }
        }
    }

    if (idx==0) {
        #pragma unroll
        for (int bin=RADIX_SIZE-1; bin>=0; --bin) {
            if (bins[bin] <= 0) {
                exceed = -bins[bin];
                break;
            }
        }
    }
    local_barrier();


    // 2. find the index of output array, if exists

    if (exceed != 0) {
        // top_kth value may not be unique, so we need to
        // perform binary cumsum on is_topkth to drop exceeding top-kth values
        out_idx = binary_cumsum_exclusive<radix_t>(idx, warp_id, lane_id, smem, is_topkth);
        is_topk &= ((!is_topkth) || out_idx>=exceed);
    }

    // perform binary cumsum on is_topk to determine the indices to put result
    out_idx = binary_cumsum_exclusive<radix_t>(idx, warp_id, lane_id, smem, is_topk);
    local_barrier();

    if (is_topk) {
#if WRITE_VALUE == 1
        ptr_at(dstv, out_idx * dstv_strides_0) = xval;
#endif
#if WRITE_INDEX == 1
        ptr_at(dsti, out_idx * dsti_strides_0) = (INDEX_TYPE)idx;
#endif
    }
}

