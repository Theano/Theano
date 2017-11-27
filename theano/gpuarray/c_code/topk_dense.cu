#define RADIX_BITS 4
#define RADIX_SIZE      (1<<RADIX_BITS)
#define RADIX_MASK(n)   ((RADIX_SIZE-1) << (n*RADIX_BITS))
#define RADIX_DIGITS(T) (bitsof(T)/RADIX_BITS)

// works when length on axis is within max allowed threads in block (1024)
extern "C" __global__ void k_topk_dense(
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
    __shared__ int smem[32 * RADIX_SIZE];
    __shared__ int k2;
    const unsigned int idx = threadIdx.x;
    bool is_topk= (idx < size);
    bool is_topkth = is_topk;
    size_t out_idx;

    const unsigned char warp_id = idx / GA_WARP_SIZE;
    // 0. get the slice for thread block to work on

    size_t gid = blockIdx.x, gidx;
    $set_slice
    // $$set_slice expands into:
    //for(int i=1; i<NDIM; i++) {
        // gidx = gid % dims_$${i};
        // gid /= dims_$${i};
        // dsti = ptr_add(dsti, gidx*dsti_strides_$${i};
        // dstv = ptr_add(dstv, gidx*dstv_strides_$${i};
        // src = ptr_add(src, gidx*src_strides_$${i});
    //}

    // get input and its radix friendly form
    const INPUT_TYPE xval = is_topk ? ptr_at(src, idx*src_strides_0) : theano_zero<INPUT_TYPE>();
    radix_t x = RadixConfig<INPUT_TYPE>::convert(xval);

    // resolve negative k
    if (k<0) { x = ~x; k = -k; }
    if (idx==0)
        k2 = k;

    // 1. filter is_topk and is_topkth using radix select

    #pragma unroll
    for (int i=bitsof(INPUT_TYPE)-RADIX_BITS; i>=0; i-=RADIX_BITS) {
        const int digit = Bitfield<radix_t>::get(x, i, RADIX_BITS);
        /*int digit = (x>>i) & (RADIX_SIZE-1);*/
        // count within warp
        #pragma unroll
        for (int bin=0; bin<RADIX_SIZE; ++bin) {
            bool vote = (bin == digit) && is_topkth;
            unsigned int votes = __ballot(vote);
            if (lane_id()==0)
                smem[bin + RADIX_SIZE*warp_id] = __popc(votes);
        }
        local_barrier();
        // sum counts across all warps
        if (idx < RADIX_SIZE) {
            int sum = smem[idx];
            #pragma unroll
            for(int w=RADIX_SIZE; w<blockDim.x*RADIX_SIZE / GA_WARP_SIZE; w+=RADIX_SIZE)
                sum += smem[idx + w];
            smem[idx] = sum;
        }
        local_barrier();

        // find the bucket and update k2
        // smem[:RADIX_SIZE:-1] = k2 - cumsum(smem[:RADIX_SIZE-1:-1])
        if (idx == 0) {
            int sum = k2;
            #pragma unroll
            for (int bin=RADIX_SIZE-1; bin>=0; --bin) {
                sum -= smem[bin];
                smem[bin] = sum;
                k2 = (sum > 0) ? sum : k2;
            }
            smem[RADIX_SIZE] = 1;
        }
        local_barrier();

        if (is_topkth) {
            is_topk &= (smem[digit+1] > 0);
            is_topkth &= (smem[digit] <= 0) && (smem[digit+1] > 0);
        }
        local_barrier();
    }

    // set k2 as number of exceeding values
    if (idx==0) {
        #pragma unroll
        for (int bin=RADIX_SIZE-1; bin>=0; --bin) {
            if (smem[bin] <= 0)
                break;
            k2 = smem[bin];
        }
    }
    local_barrier();

    // 2. find the index of output array, if exists

    if (k2 != 0) {
        // top_kth value may not be unique, so we need to
        // perform binary cumsum on is_topkth to drop exceeding top-kth values
        out_idx = binary_cumsum_exclusive(idx, warp_id, smem, is_topkth);
        if ((out_idx >= k2) && is_topkth)
            is_topk = false;
        local_barrier();
    }

    // perform binary cumsum on is_topk to determine the indices to put result
    out_idx = binary_cumsum_exclusive(idx, warp_id, smem, is_topk);

    if (is_topk) {
#if WRITE_VALUE == 1
        ptr_at(dstv, out_idx * dstv_strides_0) = xval;
#endif
#if WRITE_INDEX == 1
        ptr_at(dsti, out_idx * dsti_strides_0) = (INDEX_TYPE)idx;
#endif
    }
}
