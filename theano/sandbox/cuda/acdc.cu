/* Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.  Users and possessors of this source code
* are hereby granted a nonexclusive, royalty-free license to use this code
* in individual and commercial software.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/


/* NOTES:.
* 1) The tmp variables must each have space for length * batchSize * groupSize * sizeof(complexType).
* 2) Templated types must be (cufftReal, cufftComplex) or (cufftDoubleReal, cufftDoubleComplex)
* 3) Length must be even.
* 4) DCT maps to a type-2 DCT. Inverse DCT maps to a type-3 DCT. IDCT(DCT(x)) == x.
*/

#include <stdio.h>
#include <cufft.h>

// Useful to have
#define ROOT2 1.4142135623730951f

// This is quite system dependent. Slower systems would benefit from a smaller value here.
#define R2C_SWITCH_SIZE (1 << 19)

template<typename realType, typename complexType, bool forward, bool R2C>
__global__ void DCT_setup(int length,
                          int batchSize,
                          int groupSize,
                          const realType * __restrict__ A,
                          const realType * __restrict__ Ab,
                          const realType * __restrict__ in,
                          realType * __restrict__ out) {
   int element = blockIdx.x * blockDim.x + threadIdx.x;
   if (element >= length) return;
   
   int groupID = blockIdx.y;

   realType Alocal;
   realType Ablocal;
   
   int index;
   if (element < length / 2) {
      index = element * 2;
   }
   else {
      index = length - 2 * (element - length / 2) - 1;
   }
   
   if (A != NULL) {           
      Alocal = A[groupID * length + index];
      if (Ab != NULL) {
        Ablocal = Ab[groupID * length + index];
      }
   }      
   
   for (int batchID = blockIdx.z; batchID < batchSize; batchID += gridDim.z) {    
      realType val;
      
      if (forward) val = ((realType*)(in))[length * batchID + index];
      else         val = ((realType*)(in))[length * (batchID * groupSize + groupID) + index];
            
      if (A != NULL) {           
         val *= Alocal;
         if (Ab != NULL) {
           val += Ablocal;
         }
      }      

      if (R2C) {
         ((realType*)(out))[element + length * (batchID * groupSize + groupID)] = (realType)val;            
      }
      else {
         complexType outVal;
         outVal.x = val;
         outVal.y = 0.f;
         ((complexType*)(out))[element + length * (batchID * groupSize + groupID)] = outVal; 
      }
   }
}


template<typename realType, typename complexType, bool R2C>
__global__ void DCT_final(int length,
                          int batchSize,
                          int groupSize,
                          const realType * __restrict__ A,
                          const realType * __restrict__ Ab,
                          const realType * __restrict__ in,
                          realType * __restrict__ out) {
   int element = blockIdx.x * blockDim.x + threadIdx.x;
   if (element >= length) return;
   
   int groupID = blockIdx.y;

   realType Alocal;
   realType Ablocal;
   
   if (A != NULL) {           
      Alocal = A[groupID * length + element];
      if (Ab != NULL) {
        Ablocal = Ab[groupID * length + element];
      }
   }      
   
   for (int batchID = blockIdx.z; batchID < batchSize; batchID += gridDim.z) {    
      complexType val;
      if (R2C) {
         if (element <= length / 2) {
            val = ((complexType*)(in))[length * (batchID * groupSize + groupID) + element];
         }
         else {
            val = ((complexType*)(in))[length * (batchID * groupSize + groupID) + length - element];
            val.y = -val.y;         
         }
      }
      else {
         val = ((complexType*)(in))[length * (batchID * groupSize + groupID) + element];
      }
      complexType val2;
      complexType ret;

      sincospi(element / (2.f * (length)), &(val2.y), &(val2.x));

      val2.y = -val2.y;

      ret.x = val.x * val2.x - val.y * val2.y;

      // Normalisation
      if (element == 0) {
         ret.x *= rsqrt((realType)length);
      }
      else {
         ret.x *= ROOT2 * rsqrt((realType)length);
      }

      if (A != NULL) {
         ret.x *= Alocal;
         if (Ab != NULL) {
           ret.x += Ablocal;
         }
      }

      ((realType*)(out))[length * (batchID * groupSize + groupID) + element] = ret.x;
   }
}

template<typename realType, typename complexType>
__global__ void IDCT_final(int length,
                           int batchSize,
                           int groupSize,
                           const realType * __restrict__ A,
                           const realType * __restrict__ Ab,
                           const realType * __restrict__ in,
                           realType * __restrict__ out) {
   int element = blockIdx.x * blockDim.x + threadIdx.x;
   if (element >= length) return;
   
   int groupID = blockIdx.y;

   realType Alocal;
   realType Ablocal;
   
   int index;
   if (element < length / 2) {
      index = element * 2;
   }
   else {
      index = length - 2 * (element - length / 2) - 1;
   }
   
   if (A != NULL) {           
      Alocal = A[groupID * length + index];
      if (Ab != NULL) {
        Ablocal = Ab[groupID * length + index];
      }
   }      
   
   for (int batchID = blockIdx.z; batchID < batchSize; batchID += gridDim.z) {    
      complexType val = ((complexType*)(in))[length * (batchID * groupSize + groupID) + element];
      
      // "A" for backward pass
      if (A != NULL) {
         val.x *= Alocal;
         if (Ab != NULL) {
           val.x += Ablocal;
         }
      }

      ((realType*)(out))[length * (batchID * groupSize + groupID) + index] = val.x;
      
   }
}

template<typename realType, typename complexType, bool R2C>
__global__ void DCT_final_IDCT_setup(int length,
                                     int batchSize,
                                     int groupSize,
                                     const realType * __restrict__ D,
                                     const realType * __restrict__ Db,
                                     const realType * __restrict__ in,
                                     realType * __restrict__ out,
                                     realType * __restrict__ deltaMid) {
   int element = blockIdx.x * blockDim.x + threadIdx.x;
   if (element >= length) return;
   
   int groupID = blockIdx.y;
   
   realType dlocal;
   realType dblocal;
   
   if (D != NULL) {
     dlocal = D[groupID * length + element];
     if (Db != NULL) {
       dblocal = Db[groupID * length + element];
     }
   }
      
      
   for (int batchID = blockIdx.z; batchID < batchSize; batchID += gridDim.z) {          
      complexType val;
      if (R2C) {
         if (element <= length / 2) {
            val = ((complexType*)(in))[length * (batchID * groupSize + groupID) + element];
         }
         else {
            val = ((complexType*)(in))[length * (batchID * groupSize + groupID) + length - element];
            val.y = -val.y;         
         }
      }
      else {
         val = ((complexType*)(in))[length * (batchID * groupSize + groupID) + element];
      }
      
      complexType val2;
      complexType ret;

      sincospi(element / (2.f * (length)), &(val2.y), &(val2.x));

      val2.y = -val2.y;

      ret.x = val.x * val2.x - val.y * val2.y;

      // Normalisation
      if (element == 0) {
         ret.x *= rsqrt((realType)length);
      }
      else {
         ret.x *= ROOT2 * rsqrt((realType)length);
      }

      realType re_in = ret.x;

      if (D != NULL) {
        re_in *= dlocal;
        if (Db != NULL) {
          re_in += dblocal;
        }
      }

      if (deltaMid) {
         deltaMid[element + length * (batchID * groupSize + groupID)] = re_in;
      }

      // Un-normalisation
      if (element == 0) {
         re_in *= rsqrtf((realType)length);
      }
      else {
         re_in *= ROOT2 * rsqrtf((realType)length);
      }

      sincospi(element / (2.f * length), &(val2.y), &(val2.x));

      val.x = re_in * val2.x;
      val.y = -re_in * val2.y;

      ((complexType*)(out))[length * (batchID * groupSize + groupID) + element] = val;
   }
}

template<typename realType>
__global__ void updateWeights(int length, 
                              int batchSize,
                              int groupSize,
                              const realType * __restrict__ D,
                              const realType * __restrict__ in,
                              const realType * __restrict__ gradOutput,
                              realType * __restrict__ delta_D,
                              realType * __restrict__ delta_Db) {
   int element = blockIdx.x * blockDim.x + threadIdx.x;
   if (element >= length) return;

   int groupID = blockIdx.y;
   
   D += length * groupID;
   delta_D += length * groupID;
   delta_Db += length * groupID;
   
   realType recp_localD = 1.f / D[element];
   realType localDeltaD = 0.f;
   realType localDeltaDb = 0.f;
   
   for (int batchID = 0; batchID < batchSize; batchID++) {
      realType val = gradOutput[length * (batchID * groupSize + groupID) + element] * recp_localD;
      
      localDeltaD += val * in[length * batchID + element];
      localDeltaDb += val;         
   }
   
   delta_D[element] += localDeltaD;  
   delta_Db[element] += localDeltaDb;
   
}

template<typename realType, typename complexType>
int acdc_fp(cudaStream_t stream,
             int length, int batchSize, int groupSize,
             cufftHandle planR2C, cufftHandle planC2C,
             const realType * __restrict__ in,
             const realType * __restrict__ A,
             const realType * __restrict__ Ab,
             const realType * __restrict__ D,
             const realType * __restrict__ Db,
             realType * __restrict__ out,
             realType * __restrict__ tmp1,
             realType * __restrict__ tmp2) {

   if (length & 1) {
      printf("acdc_fp: length must be even (%d passed)\n", length);
      return 1;
   }
   
   cufftSetStream(planR2C, stream);
   cufftSetStream(planC2C, stream);
   
   dim3 blockDim;
   dim3 gridDim;

   gridDim.y = groupSize;

   blockDim.x = 128;
   gridDim.x = (length + blockDim.x - 1) / blockDim.x;
   gridDim.z = (batchSize + 1) / 2;
   
   // Two DCTs required. Inverse is handled in the custom setup.
   // R2C is only faster for longer sequences (launch latency vs bandwidth)
   if (length * batchSize * groupSize >= R2C_SWITCH_SIZE) {
      DCT_setup<realType, complexType, true, true> <<< gridDim, blockDim, 0, stream >>> (
        length, batchSize, groupSize, A, Ab, in, tmp1);
        
      cufftExecR2C(planR2C, (realType*)tmp1, (complexType*)tmp2);

      DCT_final_IDCT_setup<realType, complexType, true> <<< gridDim, blockDim, 0, stream >>> (
        length, batchSize, groupSize, D, Db, tmp2, tmp1, NULL);
   }
   else {
      DCT_setup<realType, complexType, true, false> <<< gridDim, blockDim, 0, stream >>> (
        length, batchSize, groupSize, A, Ab, in, tmp1);
        
      cufftExecC2C(planC2C, (complexType*)tmp1, (complexType*)tmp2, CUFFT_FORWARD);

      DCT_final_IDCT_setup<realType, complexType, false> <<< gridDim, blockDim, 0, stream >>> (
        length, batchSize, groupSize, D, Db, tmp2, tmp1, NULL);    
   }
   
   cufftExecC2C(planC2C, (complexType*)tmp1, (complexType*)tmp2, CUFFT_FORWARD);
      
   IDCT_final<realType, complexType> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, NULL, NULL, tmp2, out);
     
   return 0;
}



// NOTE: For the backward pass "in" is bottom, "out" is top, so we write to in.
template<typename realType, typename complexType>
int acdc_bp(cudaStream_t stream,
             int length, 
             int batchSize,
             int groupSize,
             cufftHandle planR2C, cufftHandle planC2C,
             realType * __restrict__ delta_in,
             const realType * __restrict__ A,
             const realType * __restrict__ Ab,
             const realType * __restrict__ D,
             const realType * __restrict__ Db,
             const realType * __restrict__ delta_out,
             realType * __restrict__ delta_mid,
             realType * __restrict__ tmp1,
             realType * __restrict__ tmp2) {
   
   if (length & 1) {
      printf("acdc_bp: length must be even (%d passed)\n", length);
      return 1;
   }
   
   cufftSetStream(planR2C, stream);
   cufftSetStream(planC2C, stream);

   dim3 blockDim;
   dim3 gridDim;
   
   gridDim.y = groupSize;
   

   blockDim.x = 128;
   gridDim.x = (length + blockDim.x - 1) / blockDim.x;
   gridDim.z = (batchSize + 1) / 2;
         
   // Backward through CD
   // R2C is only faster for longer sequences (launch latency vs bandwidth)
   if (length * batchSize * groupSize >= R2C_SWITCH_SIZE) {
      DCT_setup<realType, complexType, false, true> <<< gridDim, blockDim, 0, stream >>> (
        length, batchSize, groupSize, NULL, NULL, delta_out, tmp1);
        
      cufftExecR2C(planR2C, (realType*)tmp1, (complexType*)tmp2);
     
      DCT_final_IDCT_setup<realType, complexType, true> <<< gridDim, blockDim, 0, stream >>> (
        length, batchSize, groupSize, D, NULL, tmp2, tmp1, delta_mid);
   }
   else {
      DCT_setup<realType, complexType, false, false> <<< gridDim, blockDim, 0, stream >>> (
        length, batchSize, groupSize, NULL, NULL, delta_out, tmp1);
        
      cufftExecC2C(planC2C, (complexType*)tmp1, (complexType*)tmp2, CUFFT_FORWARD);

      DCT_final_IDCT_setup<realType, complexType, false> <<< gridDim, blockDim, 0, stream >>> (
        length, batchSize, groupSize, D, NULL, tmp2, tmp1, delta_mid);
   }
      
   // Backward through CA
   cufftExecC2C(planC2C, (complexType*)tmp1, (complexType*)tmp2, CUFFT_FORWARD);

   IDCT_final<realType, complexType> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, A, NULL, tmp2, delta_in);
     
   return 0;
}

template<typename realType, typename complexType>
int acdc_bp_acc(cudaStream_t stream,
                 int length, 
                 int batchSize,
                 int groupSize,
                 cufftHandle planR2C, cufftHandle planC2C,                  
                 realType * __restrict__ delta_in,
                 realType * __restrict__ delta_mid,
                 const realType * __restrict__ A,
                 const realType * __restrict__ Ab,
                 const realType * __restrict__ D,
                 const realType * __restrict__ inputA,
                 realType * __restrict__ inputD,
                 realType * __restrict__ delta_A,
                 realType * __restrict__ delta_Ab,
                 realType * __restrict__ delta_D,
                 realType * __restrict__ delta_Db,
                 realType * __restrict__ tmp1,
                 realType * __restrict__ tmp2) {

   if (length & 1) {
      printf("acdc_bp_acc length must be even (%d passed)\n", length);
      return 1;
   }

   cufftSetStream(planR2C, stream);
   cufftSetStream(planC2C, stream);

   dim3 blockDim;
   dim3 gridDim;
   
   gridDim.y = groupSize;   

   blockDim.x = 32;
   gridDim.x = (length + blockDim.x - 1) / blockDim.x;
   
   updateWeights<realType> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, A, inputA, delta_in, delta_A, delta_Ab);

   blockDim.x = 128;
   gridDim.x = (length + blockDim.x - 1) / blockDim.x;
   gridDim.z = (batchSize + 1) / 2;


   // Forward through AC to calculate input going into D
   // R2C is only faster for longer sequences (launch latency vs bandwidth)
   if (length * batchSize * groupSize >= R2C_SWITCH_SIZE) {      
      DCT_setup<realType, complexType, true, true> <<< gridDim, blockDim, 0, stream >>> (
        length, batchSize, groupSize, A, Ab, inputA, tmp1);
        
      cufftExecR2C(planR2C, (realType*)tmp1, (complexType*)tmp2);      
      
      DCT_final<realType, complexType, true> <<< gridDim, blockDim, 0, stream >>> (
        length, batchSize, groupSize, NULL, NULL, tmp2, inputD);
   }
   else {
      DCT_setup<realType, complexType, true, false> <<< gridDim, blockDim, 0, stream >>> (
        length, batchSize, groupSize, A, Ab, inputA, tmp1);
        
      cufftExecC2C(planC2C, (complexType*)tmp1, (complexType*)tmp2, CUFFT_FORWARD);
      
      DCT_final<realType, complexType, false> <<< gridDim, blockDim, 0, stream >>> (
        length, batchSize, groupSize, NULL, NULL, tmp2, inputD);
   }     
   
   blockDim.x = 32;
   gridDim.x = (length + blockDim.x - 1) / blockDim.x;
   gridDim.z = 1;
     
   updateWeights<realType> <<< gridDim, blockDim, 0, stream >>> (
     length, batchSize, groupSize, D, inputD, delta_mid, delta_D, delta_Db);
     
   return 0;
}


