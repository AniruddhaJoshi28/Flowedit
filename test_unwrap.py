import torch
from torch import inference_mode

class MyModel(torch.nn.Module):
    @inference_mode()
    def infer(self):
        x = torch.ones(1, requires_grad=True)
        return (x * 2).requires_grad

m = MyModel()
print("Before:", m.infer())

if hasattr(m.infer, "__func__"):
    func = m.infer.__func__
    if hasattr(func, "__wrapped__"):
        unwrapped = func.__wrapped__.__get__(m, m.__class__)
        print("Unwrapped bound method:", unwrapped())

