import torch

t1 = torch.tensor([[1,2,3]],dtype=float)
t2 = torch.tensor([[7,5,9]],dtype=float)
print((t1+t2)/2)