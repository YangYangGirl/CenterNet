#pragma once
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor> ConvboxMatch_forward_cuda( const at::Tensor& pred, const at::Tensor& label, const int num_box, const int num_class, const int batch_size, const int rows, const int cols, const int channels);

at::Tensor ConvboxMatch_backward_cuda();
