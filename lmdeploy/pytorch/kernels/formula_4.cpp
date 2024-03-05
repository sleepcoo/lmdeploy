#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
void launch_ffn_4(cudaStream_t stream, __half *mat, __half *vec, __half *res,
                  unsigned int mat_row, unsigned int mat_col);

void torch_launch_ffn_4(torch::Tensor &mat, torch::Tensor &vec,
                        torch::Tensor &res, int mat_row, int mat_col) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  launch_ffn_4(stream, (__half *)mat.data_ptr(), (__half *)vec.data_ptr(),
               (__half *)res.data_ptr(), mat_row, mat_col);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_launch_ffn_4", &torch_launch_ffn_4, "ffn_4 kernel warpper");
}
