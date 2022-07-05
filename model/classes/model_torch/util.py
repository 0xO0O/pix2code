from torch.nn import Module

class ToTensor(Module):
    def forward(self,x):
        tensor, _ = x
        return tensor