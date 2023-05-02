import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from python.distance_loss import DistanceLoss

num_iterations = 1000
learning_rate = 1e-2

input_tensor = torch.randn(1, 3, 256, 256).cuda()
input_tensor.requires_grad = True

# Target point to minimize the loss
target_point = torch.tensor([128.0, 128.0], dtype=torch.float).cuda()

dl = DistanceLoss().cuda()
optimizer = Adam([input_tensor], lr=learning_rate)

for i in range(num_iterations):
    optimizer.zero_grad()
    
    scalar_loss = dl(input_tensor, target_point)
    
    scalar_loss.backward()
    
    # Normalize gradients
    grad_norm = input_tensor.grad.norm().item()
    normalized_grad = input_tensor.grad / grad_norm

    # Clip gradients
    clip_grad_norm_(normalized_grad, 1.0)

    # Manually update the input_tensor using the clipped, normalized gradient
    with torch.no_grad():
        input_tensor -= learning_rate * normalized_grad
    
    if i % 25 == 24:
        print(f"Iteration {i + 1}, Loss: {scalar_loss.item()}, Grad norm: {grad_norm}")

print("Final input tensor gradients:", input_tensor.grad)
