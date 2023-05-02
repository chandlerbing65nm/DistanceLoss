#include "distance_loss.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Remove the DistanceLoss class binding

  // Bind the forward and backward functions directly
  m.def("forward", &distance_loss_forward_cuda, "Distance Loss Forward (CUDA)");
  m.def("backward", &distance_loss_backward_cuda, "Distance Loss Backward (CUDA)");
}
