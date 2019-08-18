#pragma once

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
std::tuple<at::Tensor, at::Tensor, at::Tensor> ConvboxMatch_forward(
		const at::Tensor& pred,
                const at::Tensor& label,
		const int num_box,
		const int num_class,
		const int batch_size,
		const int rows,
		const int cols,
		const int channels 
		) {
  if (pred.type().is_cuda()) {
#ifdef WITH_CUDA
    return ConvboxMatch_forward_cuda(pred, label, num_box, num_class, batch_size, rows, cols, channels);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor ConvboxMatchLoss_backward(
        const at::Tensor& pred
        ) {
  if (pred.type().is_cuda()) {
#ifdef WITH_CUDA
    return ConvboxMatch_backward_cuda();
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

