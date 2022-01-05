import torch
from torch import nn
loss = nn.CrossEntropyLoss()
pred = torch.tensor([[0.5, 0.5]], requires_grad=True)
y = torch.tensor([1])
print(loss(pred, y))
