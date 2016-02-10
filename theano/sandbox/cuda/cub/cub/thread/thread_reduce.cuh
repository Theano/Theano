/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * Thread utilities for sequential reduction over statically-sized array types
 */

#pragma once

#include "../thread/thread_operators.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 * \addtogroup UtilModule
 * @{
 */

/**
 * \name Sequential reduction over statically-sized array types
 * @{
 */


template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReduce(
    T*                  input,                  ///< [in] Input array
    ReductionOp         reduction_op,           ///< [in] Binary reduction operator
    T                   prefix,                 ///< [in] Prefix to seed reduction with
    Int2Type<LENGTH>    length)
{
    T addend = *input;
    prefix = reduction_op(prefix, addend);

    return ThreadReduce(input + 1, reduction_op, prefix, Int2Type<LENGTH - 1>());
}

template <
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReduce(
    T*                  input,                  ///< [in] Input array
    ReductionOp         reduction_op,           ///< [in] Binary reduction operator
    T                   prefix,                 ///< [in] Prefix to seed reduction with
    Int2Type<0>         length)
{
    return prefix;
}


/**
 * \brief Perform a sequential reduction over \p LENGTH elements of the \p input array, seeded with the specified \p prefix.  The aggregate is returned.
 *
 * \tparam LENGTH     LengthT of input array
 * \tparam T          <b>[inferred]</b> The data type to be reduced.
 * \tparam ScanOp     <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
 */
template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReduce(
    T*          input,                  ///< [in] Input array
    ReductionOp reduction_op,           ///< [in] Binary reduction operator
    T           prefix)                 ///< [in] Prefix to seed reduction with
{
    return ThreadReduce(input, reduction_op, prefix, Int2Type<LENGTH>());
}


/**
 * \brief Perform a sequential reduction over \p LENGTH elements of the \p input array.  The aggregate is returned.
 *
 * \tparam LENGTH     LengthT of input array
 * \tparam T          <b>[inferred]</b> The data type to be reduced.
 * \tparam ScanOp     <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
 */
template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReduce(
    T*          input,                  ///< [in] Input array
    ReductionOp reduction_op)           ///< [in] Binary reduction operator
{
    T prefix = input[0];
    return ThreadReduce<LENGTH - 1>(input + 1, reduction_op, prefix);
}


/**
 * \brief Perform a sequential reduction over the statically-sized \p input array, seeded with the specified \p prefix.  The aggregate is returned.
 *
 * \tparam LENGTH     <b>[inferred]</b> LengthT of \p input array
 * \tparam T          <b>[inferred]</b> The data type to be reduced.
 * \tparam ScanOp     <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
 */
template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReduce(
    T           (&input)[LENGTH],       ///< [in] Input array
    ReductionOp reduction_op,           ///< [in] Binary reduction operator
    T           prefix)                 ///< [in] Prefix to seed reduction with
{
    return ThreadReduce<LENGTH>(input, reduction_op, prefix);
}


/**
 * \brief Serial reduction with the specified operator
 *
 * \tparam LENGTH     <b>[inferred]</b> LengthT of \p input array
 * \tparam T          <b>[inferred]</b> The data type to be reduced.
 * \tparam ScanOp     <b>[inferred]</b> Binary reduction operator type having member <tt>T operator()(const T &a, const T &b)</tt>
 */
template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReduce(
    T           (&input)[LENGTH],       ///< [in] Input array
    ReductionOp reduction_op)           ///< [in] Binary reduction operator
{
    return ThreadReduce<LENGTH>((T*) input, reduction_op);
}


//@}  end member group

/** @} */       // end group UtilModule

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
