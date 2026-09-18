import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 10)
)

model.eval()

sahte_girdi = torch.randn(1, 1, 28, 28)

torch.onnx.export(model, sahte_girdi, "model.onnx")

print("Model ONNX formatına dönüştürüldü: model.onnx")