#pragma once
#include <torch/extension.h>

at::Tensor distance_loss_forward_cuda(const at::Tensor& input, const at::Tensor& point);
at::Tensor distance_loss_backward_cuda(const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& point);
