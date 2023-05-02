import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from python.distance_loss import DistanceLoss
from PIL import Image
import numpy as np

num_iterations = 5000
learning_rate = 1e-2

input_tensor = torch.randn(10, 2).cuda()

# Target point to be optimized
target_point = torch.tensor([10, 10], dtype=torch.float).cuda()
target_point.requires_grad = True  # Add this line to enable gradient calculations for the target_point

dl = DistanceLoss().cuda()
optimizer = Adam([target_point], lr=learning_rate)  # Change the optimizer to optimize the target_point instead of the input_tensor

for i in range(num_iterations):
    optimizer.zero_grad()
    
    loss = dl(input_tensor, target_point)
    
    loss.backward()
    
    # Normalize gradients
    grad_norm = target_point.grad.norm().item()
    normalized_grad = target_point.grad / grad_norm

    # Clip gradients
    clip_grad_norm_(normalized_grad, 1.0)

    # Manually update the target_point using the clipped, normalized gradient
    with torch.no_grad():
        target_point -= learning_rate * normalized_grad
    
    if i % 25 == 24:
        print(f"Iteration {i + 1}, Loss: {loss.item()}, Grad norm: {grad_norm}")

print("Final target point:\n", target_point)
