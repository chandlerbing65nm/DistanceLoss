#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

TORCH_API at::Tensor distance_loss_forward_cuda(const at::Tensor& input, const at::Tensor& point);
TORCH_API at::Tensor distance_loss_backward_cuda(const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& point);


template <typename scalar_t>
__global__ void distance_loss_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ point,
    scalar_t* __restrict__ output,
    const size_t num_pixels) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_pixels) {
    const scalar_t x_diff = input[index * 2] - point[0];
    const scalar_t y_diff = input[index * 2 + 1] - point[1];
    output[index] = sqrt(x_diff * x_diff + y_diff * y_diff);
  }
}

template <typename scalar_t>
__global__ void distance_loss_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ point,
    scalar_t* __restrict__ grad_point, // Change the name of this variable to reflect its purpose
    const size_t num_pixels) {
  // Add two variables to store the accumulated gradients for the reference point
  __shared__ scalar_t shared_grad_x;
  __shared__ scalar_t shared_grad_y;

  if (threadIdx.x == 0) {
    shared_grad_x = 0;
    shared_grad_y = 0;
  }
  __syncthreads();

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_pixels) {
    const scalar_t x_diff = input[index * 2] - point[0];
    const scalar_t y_diff = input[index * 2 + 1] - point[1];
    const scalar_t epsilon = 1e-8;
    const scalar_t distance = sqrt(x_diff * x_diff + y_diff * y_diff) + epsilon;
    if (distance > epsilon) {
      // Calculate gradients with respect to the reference point's x and y coordinates
      scalar_t grad_x = -grad_output[index] * x_diff / distance;
      scalar_t grad_y = -grad_output[index] * y_diff / distance;

      // Atomic addition to accumulate gradients across threads
      atomicAdd(&shared_grad_x, grad_x);
      atomicAdd(&shared_grad_y, grad_y);
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    grad_point[0] = shared_grad_x;
    grad_point[1] = shared_grad_y;
  }
}


at::Tensor distance_loss_forward_cuda(const at::Tensor& input, const at::Tensor& point) {
  const auto num_pixels = input.size(0);
  auto output = at::zeros({num_pixels}, input.options());
  const int threads = 1024;
  const int blocks = (num_pixels + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "distance_loss_forward_cuda", ([&] {
    distance_loss_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.data_ptr<scalar_t>(),
        point.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        num_pixels);
  }));

  cudaDeviceSynchronize();

  // Calculate the average distance by summing up the distances and dividing by the number of pixels.
  auto average_distance = output.sum() / num_pixels;

  return average_distance;
}


at::Tensor distance_loss_backward_cuda(const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& point) {
  const auto num_pixels = input.size(0);
  auto grad_point = at::zeros({2}, input.options()); // Change the tensor size to 2 (for x and y coordinates of the point)
  const int threads = 1024;
  const int blocks = (num_pixels + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "distance_loss_backward_cuda", ([&] {
    distance_loss_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        point.data_ptr<scalar_t>(),
        grad_point.data_ptr<scalar_t>(), // Pass the updated grad_point tensor
        num_pixels);
  }));

  cudaDeviceSynchronize();
  return grad_point;
}

