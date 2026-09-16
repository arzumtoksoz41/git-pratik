import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

donusum = transforms.ToTensor()
egitim_verisi = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=donusum)
test_verisi = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=donusum)

egitim_yukleyici = DataLoader(egitim_verisi, batch_size=64, shuffle=True)
test_yukleyici = DataLoader(test_verisi, batch_size=64, shuffle=False)

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

kayip_fonksiyonu = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("CNN modeli hazır, eğitime başlıyoruz...")

for epoch in range(3):
    for resimler, etiketler in egitim_yukleyici:
        tahminler = model(resimler)
        kayip = kayip_fonksiyonu(tahminler, etiketler)
        optimizer.zero_grad()
        kayip.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} tamamlandı, kayıp: {kayip.item():.4f}")

dogru_sayisi = 0
toplam_sayisi = 0
with torch.no_grad():
    for resimler, etiketler in test_yukleyici:
        tahminler = model(resimler)
        _, tahmin_edilen = torch.max(tahminler, 1)
        toplam_sayisi += etiketler.size(0)
        dogru_sayisi += (tahmin_edilen == etiketler).sum().item()

print(f"\nCNN modelinin TEST doğruluğu: %{100 * dogru_sayisi / toplam_sayisi:.2f}")