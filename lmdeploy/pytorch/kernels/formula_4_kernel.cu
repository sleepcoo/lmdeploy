#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdint>

// Row Major
// (32, 32, 1) (mat_row / 32)
__global__ void ffn_4(__half *mat, __half *vec, __half *res,
                      unsigned int mat_row, unsigned int mat_col) {
  float sum = 0;
  // __half sum = __float2half(0.0f);
  __shared__ float warp_sum[32];
  warp_sum[threadIdx.x] = 0.0f;

  unsigned int col_id =
      blockIdx.y * 32 + threadIdx.x;  // (0,512) (0,32), max:32*511+32=16384
  __half *res_p = &res[col_id];
  unsigned int warp_id = threadIdx.y;  // (0,32)
  unsigned int row_id = warp_id;
  __half *vec_p = &vec[row_id];
  __half *mat_p = &mat[row_id * mat_col + col_id];
  __half mat_val = __float2half(0.0f);
#pragma unroll 32
  for (int iter = 0; iter < mat_row; iter = iter + 32) {
    __half vec_val = vec_p[iter];
    if (__half2float(vec_val) == 0.0f)
      continue;
    else
      mat_val = mat_p[iter * mat_col];
    sum += __half2float(vec_val) * __half2float(mat_val);
  }
  atomicAdd(&warp_sum[threadIdx.x], sum);

  __syncthreads();
  if (warp_id == 0) {
    // Write final result
    float sum = warp_sum[threadIdx.x];
    *res_p = __float2half(sum);
  }
}

void launch_ffn_4(const cudaStream_t stream, __half *mat, __half *vec,
                  __half *res, unsigned int mat_row, unsigned int mat_col) {
  dim3 grid_dim(1, mat_col / 32);
  dim3 block_dim(32, 32, 1);

  ffn_4<<<grid_dim, block_dim, 0, stream>>>(mat, vec, res, mat_row, mat_col);
}
