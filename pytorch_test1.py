import torch
import numpy as np

# Numpy array (bildiğin şey)
numpy_dizi = np.array([1, 2, 3, 4])
print("Numpy array:", numpy_dizi)
print("Tipi:", type(numpy_dizi))

print()

# PyTorch tensor (yeni şey)
tensor = torch.tensor([1, 2, 3, 4])
print("PyTorch tensor:", tensor)
print("Tipi:", type(tensor))