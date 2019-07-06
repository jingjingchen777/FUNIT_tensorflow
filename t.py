import torch
x = torch.rand(5, 3)
print x.shape

print x.contiguous().view(-1).shape

# print x.size(0)
# print x.size(1)