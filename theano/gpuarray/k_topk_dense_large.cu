// works when length on axis is larger than max allowed threads in block (1024)
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
    LOCAL_MEM radix_t smem[32 * RADIX_SIZE];
    LOCAL_MEM radix_t known_bits, known_bits_mask;
    radix_t out_idx;
    ga_size LOCAL_MEM write_base;
    INPUT_TYPE xval;
    radix_t x;
    ga_int i;
    bool in_range, is_topk;

    const ga_size idx = LID_0;
    ga_size LOCAL_MEM k2;
    const ga_ushort warp_id = idx / GA_WARP_SIZE;
    const ga_ushort lane_id = idx % GA_WARP_SIZE;

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
    src = ptr_add(src, idx*inp_per_thread*src_strides_0);

    LOCAL_MEM radix_t inv_bits;
    if (idx==0) {
        known_bits = known_bits_mask = 0;
        k2 = abs(k);
        inv_bits = (k>=0) ? 0 : (~0);
        write_base = 0;
    }
    if (k<0) { k = -k; }

    local_barrier();

    // 1. find bits of top-k-th value using radix select
    #pragma unroll
    for (i=bitsof(INPUT_TYPE)-RADIX_BITS; i>=0; i-=RADIX_BITS) {
    /*for (i=bitsof(INPUT_TYPE)-RADIX_BITS; i>=0; i*=-1) {*/
        if (lane_id == 0) {
            #pragma unroll
            for (int bin=0; bin<RADIX_SIZE; ++bin) {
                smem[bin + warp_id*RADIX_SIZE] = 0;
            }
        }
        local_barrier();

        for (int j=0; j<inp_per_thread; ++j) {
            in_range = (idx*inp_per_thread+j) < size;
            xval = in_range ? ptr_read(src, j*src_strides_0) : (INPUT_TYPE)0;
            x = inv_bits^RadixConfig<INPUT_TYPE>::convert(xval);
            ga_int digit = (int)((x>>i) & (RADIX_SIZE-1));

            // count within warp
            #pragma unroll
            for (int bin=0; bin<RADIX_SIZE; ++bin) {
                bool incr_bin = (
                    (bin == digit) &&
                    ((x&known_bits_mask) == known_bits) &&
                    in_range);
                ga_uint incr_bin_warp = __ballot(incr_bin);
                if (lane_id==0)
                    smem[bin + RADIX_SIZE*warp_id] += __popc(incr_bin_warp);
            }
        }
        local_barrier();
        // sum counts across all warps
        // TODO: test in-block parallel sum?
        if (idx < RADIX_SIZE) {
            for(int w=RADIX_SIZE;
                w<(LDIM_0/ GA_WARP_SIZE)*RADIX_SIZE;
                w+=RADIX_SIZE)
                smem[idx] += smem[idx + w];
        }
        local_barrier();

        // update known bits
        if (idx==0) {
            #pragma unroll
            for (int bin=RADIX_SIZE-1; bin>=0; --bin) {
                if (smem[bin] >= k2) {
                    known_bits |= (bin << i);
                    known_bits_mask |= ((RADIX_SIZE-1) << i);
                    break;
                } else
                    k2 -= smem[bin];
            }
        }
        local_barrier();
    }

    /*
    if (idx < RADIX_SIZE) {
        ptr_at(dstv, idx*dstv_strides_0) = known_bits;
        ptr_at(dstv, idx*dstv_strides_0) = smem[idx];
    }
    return;
    */

    // 2. write values smaller than top-kth
    for (i=0; i<inp_per_thread; ++i) {
        in_range = (idx*inp_per_thread+i) < size;
        xval = in_range ? ptr_read(src, i*src_strides_0) : (INPUT_TYPE)0;
        x = inv_bits ^ RadixConfig<INPUT_TYPE>::convert(xval);
        is_topk = (x > known_bits) && in_range;
        out_idx = binary_cumsum<radix_t>(idx, warp_id, lane_id, smem, is_topk);
        if (is_topk) {
#if WRITE_VALUE == 1
            ptr_at(dstv, (out_idx+write_base-1) * dstv_strides_0) = xval;
#endif
#if WRITE_INDEX == 1
            ptr_at(dsti, (out_idx+write_base-1) * dsti_strides_0) = (INDEX_TYPE)(idx*inp_per_thread + i);
#endif
        }
        local_barrier();

        if (idx == blockDim.x - 1)
            write_base += out_idx;
        local_barrier();
    }
    // 3. write values equal to top-kth
    for (i=0; i<inp_per_thread; ++i) {
        in_range = (idx*inp_per_thread+i) < size;
        xval = in_range ? ptr_read(src, i*src_strides_0) : (INPUT_TYPE)0;
        x = inv_bits ^ RadixConfig<INPUT_TYPE>::convert(xval);
        is_topk = (x == known_bits) && in_range;
        out_idx = binary_cumsum<radix_t>(idx, warp_id, lane_id, smem, is_topk);
        is_topk = ((out_idx+write_base) <= abs(k)) && is_topk;
        if (is_topk) {
#if WRITE_VALUE == 1
        ptr_at(dstv, (out_idx+write_base-1) * dstv_strides_0) = xval;
#endif
#if WRITE_INDEX == 1
        ptr_at(dsti, (out_idx+write_base-1) * dsti_strides_0) = (INDEX_TYPE)(idx*inp_per_thread + i);
#endif
        }
        local_barrier();

        if (idx == blockDim.x - 1)
            write_base += out_idx;
        local_barrier();

        if(write_base >= abs(k))
            break;
    }
}

