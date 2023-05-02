import torch
from torch.optim import SGD
from python.distance_loss import DistanceLoss

num_iterations = 1000
learning_rate = 1e-4

input_tensor = torch.randn(1, 3, 256, 256).cuda()
input_tensor.requires_grad = True

# Target point to minimize the loss
target_point = torch.tensor([128.0, 128.0], dtype=torch.float).cuda()

dl = DistanceLoss().cuda()
optimizer = SGD([input_tensor], lr=learning_rate)

for i in range(num_iterations):
    optimizer.zero_grad()
    
    loss = dl(input_tensor, target_point)
    scalar_loss = torch.sum(loss)
    
    scalar_loss.backward()
    optimizer.step()
    
    if i%25==24:
        print(f"Iteration {i + 1}, Loss: {scalar_loss.item()}")

print("Final input tensor gradients:", input_tensor.grad)
