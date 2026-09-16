import torch

# requires_grad=True demek: "bu sayının nasıl değiştiğini takip et"
x = torch.tensor(3.0, requires_grad=True)

y = x ** 2   # y = x'in karesi

y.backward()   # PyTorch'a "geriye doğru hesapla" de

print("x'in değeri:", x.item())
print("y'nin değeri:", y.item())
print("y'nin x'e göre değişim hızı (gradyan):", x.grad.item())