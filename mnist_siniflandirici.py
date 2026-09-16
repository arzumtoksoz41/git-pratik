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
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

kayip_fonksiyonu = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3 epoch eğit (1 yerine biraz daha fazla, daha iyi öğrensin)
for epoch in range(3):
    for resimler, etiketler in egitim_yukleyici:
        tahminler = model(resimler)
        kayip = kayip_fonksiyonu(tahminler, etiketler)
        optimizer.zero_grad()
        kayip.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} tamamlandı, kayıp: {kayip.item():.4f}")

# Şimdi TEST verisiyle doğruluğu ölçelim (model bu veriyi hiç görmedi)
dogru_sayisi = 0
toplam_sayisi = 0

with torch.no_grad():
    for resimler, etiketler in test_yukleyici:
        tahminler = model(resimler)
        _, tahmin_edilen = torch.max(tahminler, 1)
        toplam_sayisi += etiketler.size(0)
        dogru_sayisi += (tahmin_edilen == etiketler).sum().item()

print(f"\nModelin TEST verisi üzerindeki doğruluğu: %{100 * dogru_sayisi / toplam_sayisi:.2f}")