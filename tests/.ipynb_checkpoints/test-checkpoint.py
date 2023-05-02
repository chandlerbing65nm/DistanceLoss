import torch
from python.distance_loss import DistanceLoss

input_tensor = torch.randn(1, 3, 256, 256).cuda()
input_tensor.requires_grad = True
point = torch.tensor([128.0, 128.0], dtype=torch.float).cuda()

dl = DistanceLoss().cuda()

loss = dl(input_tensor, point)
print("Loss:", loss)

# Sum the loss tensor elements to make it a scalar
scalar_loss = torch.sum(loss)
print("Scalar Loss:", scalar_loss)

scalar_loss.backward()

# Print gradients for the input tensor
print("Gradients:", input_tensor.grad)
