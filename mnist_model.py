import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Veriyi hazırla
donusum = transforms.ToTensor()
egitim_verisi = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=donusum)
egitim_yukleyici = DataLoader(egitim_verisi, batch_size=64, shuffle=True)

# Modeli tanımla - resmi alıp 0-9 arası bir rakam tahmin eden basit bir ağ
model = nn.Sequential(
    nn.Flatten(),           # 28x28 resmi düz bir çizgiye aç (784 sayı)
    nn.Linear(784, 128),    # 784 girdi -> 128 nörona bağla
    nn.ReLU(),              # aktivasyon fonksiyonu
    nn.Linear(128, 10)      # 128 -> 10 çıktı (0'dan 9'a kadar rakamlar)
)

kayip_fonksiyonu = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Model hazır, eğitime başlıyoruz...")

# Eğitim döngüsü - 1 epoch
for resimler, etiketler in egitim_yukleyici:
    tahminler = model(resimler)
    kayip = kayip_fonksiyonu(tahminler, etiketler)

    optimizer.zero_grad()
    kayip.backward()
    optimizer.step()

print(f"1 epoch tamamlandı. Son batch kaybı: {kayip.item():.4f}")