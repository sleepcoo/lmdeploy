#include <cuda_fp16.h>
#include <torch/extension.h>

void launch_ffn_4(__half *mat, __half *vec, __half *res, unsigned int mat_row,
                  unsigned int mat_col);

void torch_launch_ffn_4(torch::Tensor &mat, torch::Tensor &vec,
                        torch::Tensor &res, int mat_row, int mat_col) {
  launch_ffn_4((__half *)mat.data_ptr(), (__half *)vec.data_ptr(),
               (__half *)res.data_ptr(), mat_row, mat_col);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_launch_ffn_4", &torch_launch_ffn_4, "ffn_4 kernel warpper");
}
