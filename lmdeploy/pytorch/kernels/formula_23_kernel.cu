#include <cuda.h>
#include <cuda_fp16.h>
// #include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdint>

// Col major
__global__ void ffn_fuse_23(__half *vec_sparse, __half *vec_input,
                            __half *mat_up, __half *res, unsigned int mat_row,
                            unsigned int mat_col, float threshold) {
  int col_id = blockIdx.y * 32 + threadIdx.y;
  int num_per_threadx = mat_row / 32;
  int row_chunk_id = threadIdx.x;
  int row_id = row_chunk_id * num_per_threadx;

  __half *vec_sparse_p = &vec_sparse[col_id];  // per thread
  __half *vec_input_p = &vec_input[row_id];    // per thread
  __half *mat_up_p =
      &mat_up[col_id * mat_row + row_id];  // per thread, col-major
  __half *res_p = &res[col_id];            // per thread

  float4 *vec_input_f4 = reinterpret_cast<float4 *>(vec_input_p);
  float4 vec_input_f_val;
  float4 *mat_up_f4 = reinterpret_cast<float4 *>(mat_up_p);
  float4 mat_up_f_val;

  float sum = 0;
  __half vec_sparse_val = *vec_sparse_p;
  if (__half2float(vec_sparse_val) <= threshold)
    ;
  else {
#pragma unroll
    for (int i = 0; i < (num_per_threadx / 8) /*8个half*/; i++) {
      vec_input_f_val = vec_input_f4[i];
      const __half2 *vec_input_h1 = (__half2 *)&vec_input_f_val.x;
      const __half2 *vec_input_h2 = (__half2 *)&vec_input_f_val.y;
      const __half2 *vec_input_h3 = (__half2 *)&vec_input_f_val.z;
      const __half2 *vec_input_h4 = (__half2 *)&vec_input_f_val.w;

      mat_up_f_val = mat_up_f4[i];
      const __half2 *mat_up_h1 = (__half2 *)&mat_up_f_val.x;
      const __half2 *mat_up_h2 = (__half2 *)&mat_up_f_val.y;
      const __half2 *mat_up_h3 = (__half2 *)&mat_up_f_val.z;
      const __half2 *mat_up_h4 = (__half2 *)&mat_up_f_val.w;

      sum += __half2float(vec_input_h1->x) * __half2float(mat_up_h1->x);
      sum += __half2float(vec_input_h1->y) * __half2float(mat_up_h1->y);
      sum += __half2float(vec_input_h2->x) * __half2float(mat_up_h2->x);
      sum += __half2float(vec_input_h2->y) * __half2float(mat_up_h2->y);
      sum += __half2float(vec_input_h3->x) * __half2float(mat_up_h3->x);
      sum += __half2float(vec_input_h3->y) * __half2float(mat_up_h3->y);
      sum += __half2float(vec_input_h4->x) * __half2float(mat_up_h4->x);
      sum += __half2float(vec_input_h4->y) * __half2float(mat_up_h4->y);
    }
  }

  __shared__ float warp_sum[32];
  warp_sum[threadIdx.y] = 0.0f;
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  if (threadIdx.x == 0) {
    float sum = sum;
    if (__half2float(vec_sparse_val) > threshold) {
      sum = sum * __half2float(vec_sparse_val);
    }
    *res_p = __float2half(sum);
  }
}

void launch_ffn_fuse_23(const cudaStream_t stream, __half *vec_sparse,
                        __half *vec_input, __half *mat_up, __half *res,
                        unsigned int mat_row, unsigned int mat_col,
                        float threshold) {
  dim3 grid_dim(1, mat_col / 32);
  dim3 block_dim(32, 32, 1);

  ffn_fuse_23<<<grid_dim, block_dim, 0, stream>>>(
      vec_sparse, vec_input, mat_up, res, mat_row, mat_col, threshold);
}
