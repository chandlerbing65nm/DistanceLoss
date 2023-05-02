import torch
from torch.autograd import Function

import distance_loss as _C

class DistanceLossFunction(Function):
    @staticmethod
    def forward(ctx, input, point):
        input = input.contiguous()
        point = point.contiguous()

        output = _C.forward(input, point)
        ctx.save_for_backward(input, point, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, point, output = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        grad_point = _C.backward(grad_output, input, point)  # Assuming this is the updated CUDA code to return gradients w.r.t point
        return None, grad_point  # Return None for the input gradients and grad_point for the point gradients

_distance_loss = DistanceLossFunction.apply

class DistanceLoss(torch.nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, input, point):
        return _distance_loss(input, point)
