#include <cuda_fp16.h>
#include <torch/extension.h>

void launch_ffn_fuse_23(__half *vec_sparse, __half *vec_input, __half *mat_up,
                        __half *res, unsigned int mat_row, unsigned int mat_col,
                        float threshold = 0);

void torch_launch_ffn_fuse_23(torch::Tensor &vec_sparse,
                              torch::Tensor &vec_input, torch::Tensor &mat_up,
                              torch::Tensor &res, int mat_row, int mat_col,
                              float threshold = 0.) {
  launch_ffn_fuse_23((__half *)vec_sparse.data_ptr(),
                     (__half *)vec_input.data_ptr(),
                     (__half *)mat_up.data_ptr(), (__half *)res.data_ptr(),
                     mat_row, mat_col, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_launch_ffn_fuse_23", &torch_launch_ffn_fuse_23,
        "ffn_fuse_23 kernel warpper");
}
